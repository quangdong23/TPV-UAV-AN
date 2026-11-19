import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from uav_pipeline.events import EventOptions,ViolationMonitor
from uav_pipeline.video.evidence import LiveEvidence


class EarlyUploadTests(unittest.TestCase):
    def test_confirmation_queues_before_postroll_and_analysis_can_continue(self):
        for vfr in (False,True):
            with self.subTest(vfr=vfr),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);monitor=ViolationMonitor(EventOptions(pre_seconds=2,post_seconds=8))
                started=threading.Event();release=threading.Event();captured=[]
                class Uploader:
                    def upload_events(self,output,events):
                        captured.extend(events);started.set()
                        release.wait(5)
                        return {'ready':len(events),'already_ready':0,'failed':0}
                live=LiveEvidence(root,monitor,10,Uploader(),vfr=vfr,upload_on_confirmation=True)
                frame=np.zeros((32,32,3),np.uint8)
                try:
                    for index in range(3):live.write(frame,index/10)
                    self.assertFalse(started.is_set())
                    monitor.events.append({'event_id':'e1','kind':'sidewalk_encroachment',
                        'first_frame':1,'first_timestamp_seconds':.1,'confirmation_frame':3,
                        'confirmation_timestamp_seconds':.3,'track_ids':[1],'class_ids':[4],
                        'detected_at':'2026-11-18T12:00:00+07:00','images':[]})
                    live.write(frame,.3)
                    self.assertTrue(started.wait(3))
                    self.assertEqual(captured[0]['clip']['upload_trigger'],'confirmation')
                    self.assertTrue(captured[0]['clip']['post_roll_omitted'])
                    self.assertLess(captured[0]['clip']['end_output_seconds'],1)
                    self.assertTrue((root/captured[0]['clip']['path']).is_file())
                    # Network still blocked; processing the next frame must work.
                    live.write(frame,.4)
                    self.assertEqual(live.buffer_frames,5)
                finally:
                    release.set();live.close()

    def test_legacy_postroll_remains_selectable(self):
        with tempfile.TemporaryDirectory() as tmp:
            monitor=ViolationMonitor(EventOptions(post_seconds=8))
            monitor.events.append({'event_id':'e1','kind':'sidewalk_encroachment',
                'first_frame':0,'track_ids':[1],'class_ids':[4],'images':[]})
            class Uploader:
                def upload_events(self,*args):raise AssertionError('Must wait for post-roll')
            live=LiveEvidence(tmp,monitor,10,Uploader(),upload_on_confirmation=False)
            try:
                live.write(np.zeros((32,32,3),np.uint8),0)
                self.assertFalse(live.ready(1))
                self.assertTrue(monitor.events)
            finally:live.close()

    def test_offline_pipeline_uploads_before_input_is_exhausted(self):
        from tests.test_video import FakeDetector
        from tests.test_combined_video import FakeSidewalk
        from uav_pipeline.segmentation.session import SidewalkSession
        from uav_pipeline.segmentation.media.video import VideoPreviewOptions
        from uav_pipeline.video.pipeline import VideoOptions,process_video
        class Detector(FakeDetector):
            def predict(self,image,**kwargs):
                return {'object_detections':[{'class_id':4,'bbox_xyxy':[20,20,50,70],'score':.9,'status':'predicted'},
                    {'class_id':1,'bbox_xyxy':[50,10,65,70],'score':.95,'status':'predicted'}],'model_calls':1}
        uploaded=threading.Event();callbacks=[];progress={'frames':0,'eof':False}
        class Uploader:
            def upload_events(self,output,events):
                callbacks.append((progress.copy(),events));uploaded.set()
                return {'ready':len(events),'already_ready':0,'failed':0}
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);source=root/'input.avi'
            frame=np.random.default_rng(15).integers(0,255,(120,160,3),np.uint8)
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),10,(160,120))
            for _ in range(30):writer.write(frame)
            writer.release()
            factory=cv2.VideoCapture
            class Capture:
                def __init__(self,path,*args):
                    self.capture=factory(path,*args);self.source=str(path)==str(source);self.count=0
                def __getattr__(self,name):return getattr(self.capture,name)
                def read(self):
                    if self.source and self.count==29 and not uploaded.wait(3):
                        raise RuntimeError('Upload did not start before input EOF')
                    ok,image=self.capture.read()
                    if self.source:
                        self.count+=int(ok);progress.update(frames=self.count,eof=not ok)
                    return ok,image
            session=SidewalkSession(FakeSidewalk(),VideoPreviewOptions(keyframe_interval=1),sample_step=0)
            with patch('cv2.VideoCapture',Capture):
                process_video(source,root/'output',Detector(),VideoOptions(min_hits=1,keyframe_interval=1,
                    min_detection_interval=1,segment_frames=60,upload_on_confirmation=True),sidewalk=session,
                    events=ViolationMonitor(EventOptions(encroachment_frames=3)),event_uploader=Uploader(),
                    recording_started_at='2026-11-18T12:00:00+07:00')
            self.assertTrue(callbacks)
            self.assertFalse(callbacks[0][0]['eof'])
            self.assertLess(callbacks[0][0]['frames'],30)
            self.assertEqual(callbacks[0][1][0]['clip']['upload_trigger'],'confirmation')
            self.assertEqual(list((root/'output/pending_uploads').glob('*.json')),[])
            summary=json.loads((root/'output/summary.json').read_text())
            self.assertGreater(summary['event_uploads']['ready'],0)
