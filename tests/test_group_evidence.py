import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from uav_pipeline.evidence_groups import EvidenceGroups
from uav_pipeline.uploads import EvidenceUploader,UploadError
from tests.test_uploads import API,Response


def track(tid,cid,box):
    return {'track_id':tid,'class_id':cid,'bbox_xyxy':box,'position_source':'detection','track_confirmed':True}


class GroupEvidenceTests(unittest.TestCase):
    def test_stable_person_group_crop_is_source_resolution_and_invisible(self):
        groups=EvidenceGroups()
        observations=[track(1,4,[20,20,40,50]),track(2,1,[40,10,55,50])]
        for timestamp in (0,.2,.4):groups.update(observations,timestamp)
        event={'event_id':'event_000001','track_ids':[1],'confirmation_frame':4}
        image=np.full((200,200,3),100,np.uint8);before=image.copy()
        with tempfile.TemporaryDirectory() as directory:
            groups.capture(event,observations,image,Path(directory),(100,100,3),'actual.pt')
            self.assertEqual(event['group']['track_ids'],[1,2])
            self.assertFalse(event['group']['rendered'])
            self.assertEqual(len(event['images']),2)
            crop=cv2.imread(str(Path(directory)/event['images'][0]['path']))
            x1,y1,x2,y2=event['group']['bbox_source_pixels']
            self.assertEqual(crop.shape[:2],(y2-y1,x2-x1))
            self.assertGreaterEqual(x2,110)
            np.testing.assert_array_equal(image,before)

    def test_ambiguous_people_are_not_assigned_and_state_expires(self):
        groups=EvidenceGroups()
        obs=[track(1,4,[20,20,40,50]),track(2,1,[40,10,55,50]),track(3,1,[5,10,20,50])]
        for timestamp in (0,.2,.4):groups.update(obs,timestamp)
        self.assertEqual(groups.links,{})
        groups.update(obs[:2],1)
        self.assertTrue(groups.links)
        groups.update([],1.2)
        self.assertFalse(groups.links)


class ImageAPI(API):
    def __init__(self):
        super().__init__();self.timeline=[];self.fail_image=False
    def request(self,method,url,**kwargs):
        self.timeline.append(url.rsplit('/',1)[-1])
        if url.endswith('/abort'):return Response({'success':True})
        response=super().request(method,url,**kwargs)
        if url.endswith('/initiate'):
            response.data['images']=[{'upload_url':'https://minio/image'} for _ in kwargs['json'].get('images',[])]
        return response
    def put(self,url,**kwargs):
        if url.endswith('/image'):
            self.timeline.append('image')
            assert kwargs['headers']['Content-Type']=='image/jpeg'
            assert kwargs['headers']['Content-Length']==str(len(kwargs['data']))
            assert kwargs['data']
            return Response(code=503 if self.fail_image else 200)
        self.timeline.append('video')
        return super().put(url,**kwargs)


class ImageUploadTests(unittest.TestCase):
    def test_image_and_video_complete_together_and_ready_is_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'video.mp4').write_bytes(b'123456')
            cv2.imwrite(str(root/'crop.jpg'),np.zeros((20,20,3),np.uint8))
            event={'event_id':'event_1','kind':'sidewalk_encroachment','class_ids':[4],
                'detected_at':'2026-11-17T12:00:00+07:00','model_version':'actual.pt',
                'images':[{'path':'crop.jpg'}],'clip':{'status':'saved','path':'video.mp4'}}
            api=ImageAPI();client=EvidenceUploader('https://example','hidden',location='20, 105',http=api,sleep=lambda _:None)
            client._client.ensure_browser_compatible_video=lambda path:None
            self.assertEqual(client.upload_one(root,event),'ready')
            payload=api.calls[0][2]['json']
            self.assertEqual(payload['images'][0]['content_type'],'image/jpeg')
            self.assertEqual(payload['model_version'],'actual.pt')
            self.assertLess(api.timeline.index('image'),api.timeline.index('complete'))
            self.assertLess(api.timeline.index('video'),api.timeline.index('complete'))
            self.assertEqual(client.upload_one(root,event),'already_ready')
            self.assertEqual(api.timeline.count('initiate'),1)
            for path in (root/'uploads').iterdir():self.assertNotIn('https://minio',path.read_text())

    def test_image_failure_does_not_complete_and_retries_known_incomplete_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'video.mp4').write_bytes(b'123456')
            cv2.imwrite(str(root/'crop.jpg'),np.zeros((20,20,3),np.uint8))
            event={'event_id':'event_1','kind':'sidewalk_encroachment','class_ids':[4],
                'detected_at':'2026-11-17T12:00:00+07:00','images':[{'path':'crop.jpg'}],
                'clip':{'status':'saved','path':'video.mp4'}}
            api=ImageAPI();api.fail_image=True
            client=EvidenceUploader('https://example','hidden',location='20, 105',http=api,sleep=lambda _:None)
            client._client.ensure_browser_compatible_video=lambda path:None
            with self.assertRaises(UploadError):client.upload_one(root,event)
            self.assertNotIn('complete',api.timeline)
            api.fail_image=False
            self.assertEqual(client.upload_one(root,event),'ready')
            self.assertIn('abort',api.timeline)

class OfflinePipelineUploadTests(unittest.TestCase):
    def test_offline_images_and_clip_reach_uploader_with_recording_time(self):
        import contextlib
        import io
        from tests.test_video import FakeDetector
        from tests.test_combined_video import FakeSidewalk
        from uav_pipeline.events import ViolationMonitor,EventOptions
        from uav_pipeline.segmentation.session import SidewalkSession
        from uav_pipeline.segmentation.media.video import VideoPreviewOptions
        from uav_pipeline.video.pipeline import VideoOptions,process_video
        class Detector(FakeDetector):
            def predict(self,image,**kwargs):
                return {'object_detections':[{'class_id':4,'bbox_xyxy':[20,20,50,70],'score':.9,'status':'predicted'}, {'class_id':1,'bbox_xyxy':[50,10,65,70],'score':.95,'status':'predicted'}],'model_calls':1}
        captured=[]
        test=self
        class Uploader:
            def upload_events(self,output,events):
                for event in events:
                    test.assertEqual(len(event['images']),2)
                    test.assertEqual(event['clip']['status'],'saved')
                    test.assertTrue(event['detected_at'].startswith('2026-11-17T12:00:00'))
                    for image in event['images']:test.assertTrue((output/image['path']).is_file())
                    test.assertTrue((output/event['clip']['path']).is_file())
                    captured.append(event)
                return {'ready':len(events),'failed':0,'already_ready':0}
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            root=Path(directory);source=root/'source.avi'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),10,(160,120))
            frame=np.random.default_rng(15).integers(0,255,(120,160,3),np.uint8)
            for _ in range(12):writer.write(frame)
            writer.release()
            session=SidewalkSession(FakeSidewalk(),VideoPreviewOptions(keyframe_interval=1),sample_step=0)
            process_video(source,root/'output',Detector(),VideoOptions(min_hits=1,keyframe_interval=1,
                min_detection_interval=1,segment_frames=4,save_tracking=False),sidewalk=session,
                events=ViolationMonitor(EventOptions(encroachment_frames=3)),event_uploader=Uploader(),
                recording_started_at='2026-11-17T12:00:00+07:00')
            self.assertFalse((root/'output/tracks.jsonl').exists())
            self.assertFalse(list((root/'output/segments').glob('*.jsonl')))
            self.assertEqual(len(list((root/'output/segments').glob('previous_*.png'))),1)
            self.assertFalse((root/'output/clip_staging').exists())
        self.assertTrue(captured)

class StrictGroupTests(unittest.TestCase):
    def test_object_only_saves_crop_without_person(self):
        groups=EvidenceGroups();obs=[track(1,4,[20,20,40,50])]
        for timestamp in (0,.2,.4,.6):groups.update(obs,timestamp)
        event={'event_id':'event_1','track_ids':[1],'confirmation_frame':6}
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(groups.capture(event,obs,np.zeros((100,100,3),np.uint8),Path(directory),(100,100,3),'test'))
            self.assertEqual(event['group']['person_track_ids'],[])
            self.assertEqual(event['evidence_status'],'ready')

    def test_adjacent_person_must_persist_before_photo(self):
        groups=EvidenceGroups();obs=[track(1,4,[20,20,40,50]),track(2,1,[40,10,55,50])]
        event={'event_id':'event_1','track_ids':[1],'confirmation_frame':1}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);image=np.zeros((100,100,3),np.uint8)
            groups.update(obs,0)
            self.assertTrue(groups.capture(event,obs,image,root,image.shape,'test'))
            self.assertEqual(event['group']['person_track_ids'],[])
            groups.update(obs,.2);groups.update(obs,.4)
            self.assertTrue(groups.capture(event,obs,image,root,image.shape,'test',image_frame=4))
            self.assertEqual(event['group']['image_frame'],4)

class VideoWithoutPhotoTests(unittest.TestCase):
    def test_video_uploads_when_photo_group_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'video.mp4').write_bytes(b'123456')
            event={'event_id':'event_1','kind':'sidewalk_encroachment','class_ids':[5],
                   'evidence_status':'skipped_no_related_person','images':[],
                   'detected_at':'2026-11-17T12:00:00+07:00',
                   'clip':{'status':'saved','path':'video.mp4'}}
            api=ImageAPI();client=EvidenceUploader('https://example','hidden',location='20, 105',http=api,sleep=lambda _:None)
            client._client.ensure_browser_compatible_video=lambda path:None
            result=client.upload_events(root,[event])
            self.assertEqual(result['ready'],1)
            self.assertEqual(api.calls[0][2]['json']['images'],[])
            self.assertIn('video',api.timeline)
            self.assertIn('complete',api.timeline)

    def test_person_next_to_motorcycle_does_not_validate_distant_chair_or_sign(self):
        groups=EvidenceGroups()
        obs=[track(1,1,[40,30,55,70]),track(2,2,[50,35,75,70]),
             track(3,5,[5,5,20,30]),track(4,7,[65,0,150,90])]
        for timestamp in (0,.2,.4):groups.update(obs,timestamp)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);image=np.zeros((160,160,3),np.uint8)
            for tid,cid in [(3,5),(4,7)]:
                event={'event_id':f'event_{tid}','track_ids':[tid],'class_ids':[cid],'confirmation_frame':4}
                self.assertTrue(groups.capture(event,obs,image,root,image.shape,'test'))
                self.assertEqual(event['group']['person_track_ids'],[])
            repair={'event_id':'event_repair','track_ids':[1,2],'class_ids':[1,2],'confirmation_frame':4}
            self.assertTrue(groups.capture(repair,obs,image,root,image.shape,'test'))
            self.assertEqual(repair['group']['track_ids'],[1,2])
