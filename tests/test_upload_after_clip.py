import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np
from uav_pipeline.events import ViolationMonitor,EventOptions
from uav_pipeline.video.evidence import LiveEvidence
from uav_pipeline.video.pipeline import VideoOptions


def event(name,frame,fps=10):
    return {'event_id':name,'kind':'sidewalk_encroachment','first_frame':frame,
            'first_timestamp_seconds':frame/fps,'track_ids':[frame],'class_ids':[4],
            'images':[],'detected_at':'2026-11-18T12:00:00+07:00'}


class UploadAfterClipTests(unittest.TestCase):
    def test_default_keeps_full_postroll(self):
        self.assertFalse(VideoOptions().upload_on_confirmation)

    def test_later_pending_event_cannot_delay_ready_clip(self):
        for vfr in (False,True):
            with self.subTest(vfr=vfr),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);monitor=ViolationMonitor(EventOptions(pre_seconds=.2,post_seconds=.4))
                early=event('early',1);late=event('late',6)
                monitor.events.extend([early,late]);started=threading.Event();sent=[]
                class Uploader:
                    def upload_events(self,output,events):
                        sent.extend(events);started.set()
                        return {'ready':len(events),'already_ready':0,'failed':0}
                live=LiveEvidence(root,monitor,10,Uploader(),vfr=vfr)
                (root/'source.mp4').write_bytes(b'source')
                segments=[{'video':'source.mp4','frames':9,'start_frame':0,
                           'start_timestamp':0.,'end_timestamp':.9}]
                def cut(source,target,*args):Path(target).write_bytes(b'clip')
                try:
                    with patch('uav_pipeline.video.evidence.consolidate_recording'),patch.object(monitor,'_write_clip',side_effect=cut),patch('uav_pipeline.video.vfr.cut_vfr',side_effect=cut):
                        live.flush(segments,9,.8)
                    self.assertTrue(started.wait(3))
                    self.assertEqual([row['event_id'] for row in sent],['early'])
                    self.assertEqual([row['event_id'] for row in monitor.events],['late'])
                    self.assertAlmostEqual(sent[0]['clip']['end_output_seconds'],.5)
                    self.assertFalse(sent[0]['clip']['post_roll_omitted'])
                finally:live.close()

    def test_first_clip_upload_starts_before_second_clip_finishes_cutting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);monitor=ViolationMonitor(EventOptions(pre_seconds=.1,post_seconds=.3))
            monitor.events.extend([event('first',1),event('second',10)])
            started=threading.Event();sent=[]
            class Uploader:
                def upload_events(self,output,events):
                    assert (output/'events.json').exists()
                    sent.extend(events);started.set()
                    return {'ready':len(events),'already_ready':0,'failed':0}
            live=LiveEvidence(root,monitor,10,Uploader())
            (root/'source.mp4').write_bytes(b'source')
            segments=[{'video':'source.mp4','frames':20,'start_frame':0}]
            test=self
            def cut(source,target,*args):
                if target.name=='second.mp4':
                    test.assertTrue(started.wait(3),'First upload waited for second clip')
                target.write_bytes(b'clip')
            try:
                with patch('uav_pipeline.video.evidence.consolidate_recording'),patch.object(monitor,'_write_clip',side_effect=cut):
                    live.flush(segments,20,1.9)
            finally:live.close()
            self.assertEqual(len(sent),2)

    def test_full_eight_second_clip_uploads_at_deadline_without_eof(self):
        for vfr in (False,True):
            with self.subTest(vfr=vfr),tempfile.TemporaryDirectory() as tmp:
                monitor=ViolationMonitor(EventOptions(pre_seconds=2,post_seconds=8))
                started=threading.Event();sent=[]
                class Uploader:
                    def upload_events(self,output,events):
                        sent.extend(events);started.set()
                        return {'ready':len(events),'already_ready':0,'failed':0}
                live=LiveEvidence(tmp,monitor,10,Uploader(),vfr=vfr)
                frame=np.zeros((32,32,3),np.uint8)
                try:
                    for index in range(20):live.write(frame,index/10)
                    monitor.events.append(event('full',10))
                    for index in range(20,89):live.write(frame,index/10)
                    self.assertFalse(started.is_set(),'Post-roll was cut short')
                    live.write(frame,8.9)
                    self.assertTrue(started.wait(3),'Clip not sent at evidence deadline')
                    clip=sent[0]['clip']
                    self.assertAlmostEqual(clip['end_output_seconds'],9.)
                    self.assertFalse(clip['post_roll_omitted'])
                    self.assertEqual(clip['upload_trigger'],'post_roll')
                    # Continue analysis without finish()/EOF.
                    live.write(frame,9.)
                finally:live.close()
