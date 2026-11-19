import contextlib
from datetime import datetime,timedelta,timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import requests
from uav_pipeline.uploads import EvidenceUploader,UploadError,behavior_name
from uav_pipeline.events import ViolationMonitor,EventOptions
from tests.test_events import observation


class Response:
    def __init__(self,data=None,code=200,headers=None):
        self.data=data;self.status_code=code;self.headers=headers or {}
    def json(self):return self.data
    def raise_for_status(self):
        if self.status_code>=400:
            raise requests.HTTPError(response=self)



class API:
    def __init__(self):
        self.calls=[];self.puts=[];self.fail_parts=False;self.uncertain=False
    def request(self,method,url,**kwargs):
        self.calls.append((method,url,kwargs))
        if url.endswith('/abort'):
            if self.uncertain:raise requests.Timeout()
            return Response({'success':True})
        if url.endswith('/initiate'):
            size=kwargs['json']['file_size']
            return Response({'violation_id':'remote1','part_size':5,'total_parts':(size+4)//5},201)
        if url.endswith('/part-urls'):
            return Response({'parts':[{'part_number':n,'url':'https://minio/part/'+str(n)} for n in kwargs['json']['part_numbers']]})
        if url.endswith('/complete'):
            if self.uncertain:raise requests.Timeout('secret URL must never be logged')
            return Response({'success':True,'status':'ready','violation_id':'remote1','video_key':'videos/evidence.mp4'})
        raise AssertionError(url)
    def put(self,url,**kwargs):
        self.puts.append((url,kwargs))
        return Response(code=503 if self.fail_parts else 200,headers={'ETag':'"etag"'})


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/'event_clips').mkdir();(self.root/'event_clips/clip_000001.mp4').write_bytes(b'0123456789ab')
        self.event={'event_id':'event_000001','kind':'sidewalk_encroachment','class_ids':[7],
                    'detected_at':'2026-11-16T18:30:00.123+07:00',
                    'clip':{'status':'saved','path':'event_clips/clip_000001.mp4'}}
        self.api=API();self.client=EvidenceUploader('https://api.example','secret-key',location='20, 105',http=self.api,sleep=lambda _:None)
        self.client._client.ensure_browser_compatible_video=lambda path:None
    def tearDown(self):self.tmp.cleanup()

    def test_business_fields_demo_confidence_and_ready_upload_deduplication(self):
        self.assertEqual(self.client.upload_one(self.root,self.event),'ready')
        payload=self.api.calls[0][2]['json']
        self.assertEqual(set(payload),{'filename','file_size','content_type','detected_at','location','behavior_name','confidence','camera_id','model_version','violators','images'})
        self.assertEqual(payload['detected_at'],self.event['detected_at'])
        self.assertEqual(payload['location'],'20, 105')
        self.assertEqual(payload['behavior_name'],'đặt biển quảng cáo sai quy định')
        self.assertEqual(b''.join(kwargs['data'] for _,kwargs in sorted(self.api.puts)),b'0123456789ab')
        self.assertTrue(all('headers' not in kwargs for _,kwargs in self.api.puts))
        calls=len(self.api.calls)
        self.assertEqual(self.client.upload_one(self.root,self.event),'already_ready')
        self.assertEqual(len(self.api.calls),calls)
        ledger=list((self.root/'uploads').glob('*.json'))[0].read_text()
        self.assertNotIn('secret-key',ledger);self.assertNotIn('https://minio',ledger)

    def test_utc_occurrence_is_converted_to_vietnam_without_shifting_instant(self):
        event={**self.event,'detected_at':'2026-11-16T12:30:00.123+00:00'}
        self.client.upload_one(self.root,event)
        payload=self.api.calls[0][2]['json']
        self.assertEqual(payload['detected_at'],'2026-11-16T19:30:00.123+07:00')
        self.assertEqual(datetime.fromisoformat(payload['detected_at']).timestamp(),
                         datetime.fromisoformat(event['detected_at']).timestamp())

    def test_part_failure_uses_be_abort_before_new_attempt(self):
        self.api.fail_parts=True
        with self.assertRaises(UploadError):self.client.upload_one(self.root,self.event)
        self.assertTrue((self.root/self.event['clip']['path']).exists())
        self.api.fail_parts=False
        self.assertEqual(self.client.upload_one(self.root,self.event),'ready')
        self.assertEqual(sum(url.endswith('/initiate') for _,url,_ in self.api.calls),2)
        self.assertEqual(sum(url.endswith('/abort') for _,url,_ in self.api.calls),1)

    def test_ambiguous_complete_never_creates_duplicate(self):
        self.api.uncertain=True
        with self.assertRaises(UploadError):self.client.upload_one(self.root,self.event)
        ledger=list((self.root/'uploads').glob('*.json'))[0].read_text()
        self.assertNotIn('secret URL',ledger)
        self.assertEqual(json.loads(ledger)['state'],'aborting')
        calls=len(self.api.calls)
        with self.assertRaises(UploadError):self.client.upload_one(self.root,self.event)
        self.assertEqual(len(self.api.calls),calls)

    def test_missing_occurrence_time_or_clip_never_uploads(self):
        for value in (None,'2026-11-16T18:30:00'):
            with self.assertRaises(UploadError):self.client.upload_one(self.root,{**self.event,'detected_at':value})
        with self.assertRaises(UploadError):self.client.upload_one(self.root,{**self.event,'clip':{'status':'error'}})
        self.assertEqual(self.api.calls,[])

    def test_event_uses_first_observation_not_confirmation_or_upload_time(self):
        monitor=ViolationMonitor(EventOptions(encroachment_frames=3))
        mask=np.full((100,100),2,np.uint8)
        motion={'reliable':True,'matrix':np.array([[1.,0.,0.],[0.,1.,0.]]),'scene_cut':False}
        moments=['2026-11-16T18:30:00+07:00','2026-11-16T18:30:01+07:00','2026-11-16T18:30:02+07:00']
        for i in range(3):monitor.update(i,i/10,[observation()],mask,motion,detected_at=moments[i])
        self.assertEqual(monitor.events[0]['detected_at'],moments[0])
        self.assertEqual(monitor.events[0]['confirmation_frame'],2)

    def test_live_pipeline_exports_clips_before_automatic_upload(self):
        from tests.test_live import Source
        from tests.test_video import FakeDetector
        from tests.test_combined_video import FakeSidewalk
        from uav_pipeline.segmentation.session import SidewalkSession
        from uav_pipeline.segmentation.media.video import VideoPreviewOptions
        from uav_pipeline.video.pipeline import VideoOptions,process_video
        from uav_pipeline.video.live import RTMPCapture
        class Detector(FakeDetector):
            def predict(self,image,**kwargs):
                return {'object_detections':[{'class_id':4,'bbox_xyxy':[20,20,50,70],'score':.9,'status':'predicted'}, {'class_id':1,'bbox_xyxy':[50,10,65,70],'score':.95,'status':'predicted'}],'model_calls':1}
        frame=np.random.default_rng(15).integers(0,255,(120,160,3),np.uint8)
        cap=RTMPCapture('rtmp://host/live/source',factory=lambda *args:Source([frame.copy() for _ in range(12)]))
        session=SidewalkSession(FakeSidewalk(),VideoPreviewOptions(keyframe_interval=1),sample_step=0)
        monitor=ViolationMonitor(EventOptions(encroachment_frames=3))
        opts=VideoOptions(min_hits=1,keyframe_interval=1,min_detection_interval=1,segment_frames=4)
        completed=[]
        class Uploader:
            def upload_events(self,output,events):
                assert (output/'events.json').exists()
                assert events
                for e in events:
                    assert datetime.fromisoformat(e['detected_at']).utcoffset()==timedelta(hours=7)
                    assert len(e['images'])==2
                    assert e['group']['rendered'] is True
                    for image in e['images']:assert (output/image['path']).is_file()
                    assert e['clip']['status']=='saved'
                    assert (output/e['clip']['path']).exists()
                completed.extend(events)
                return {'ready':len(events),'already_ready':0,'failed':0}
        with contextlib.redirect_stdout(io.StringIO()):
            cap.open()
            process_video('rtmp://host/live/source',self.root/'session',Detector(),opts,max_frames=12,
                sidewalk=session,events=monitor,live_capture=cap,event_uploader=Uploader())
        self.assertTrue(completed)
        journal=[json.loads(x) for x in (self.root/'session/event_journal.jsonl').read_text().splitlines()]
        self.assertEqual(journal[0]['detected_at'],completed[0]['detected_at'])
        summary=json.loads((self.root/'session/summary.json').read_text())
        self.assertEqual(summary['event_uploads']['ready'],len(completed))
