from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from uav_pipeline.activities import ActivityClassifier, CLASSES
from uav_pipeline.events import ViolationMonitor


def observations(support_class=2):
    return [dict(track_id=1,class_id=1,bbox_xyxy=[20,20,60,90],position_source='detection',
                 detection_age_seconds=0.,track_confirmed=True),
            dict(track_id=2,class_id=support_class,bbox_xyxy=[50,40,110,95],position_source='detection',
                 detection_age_seconds=0.,track_confirmed=True)]


class ActivityTests(unittest.TestCase):
    def setUp(self):
        self.frame=np.full((120,160,3),114,np.uint8)
        self.mask=np.full((120,160),2,np.uint8)
        self.motion={'scene_cut':False,'reliable':True,'matrix':np.array([[1.,0.,0.],[0.,1.,0.]])}

    def test_rgb_letterbox_and_imagenet_normalization(self):
        frame=np.zeros((20,40,3),np.uint8);frame[:,:,2]=255
        tensor=ActivityClassifier.preprocess(frame,[0,0,40,20])
        self.assertEqual(tuple(tensor.shape),(3,224,224))
        np.testing.assert_allclose(tensor[:,112,112],[(1-.485)/.229,-.456/.224,-.406/.225],rtol=1e-5)
        np.testing.assert_allclose(tensor[:,0,0],(np.full(3,114/255)-[.485,.456,.406])/[.229,.224,.225],rtol=1e-5)
        with self.assertRaises(ValueError):ActivityClassifier.preprocess(frame,[0,0,0,10])

    def test_real_checkpoint_loads_strictly_and_runs_without_download(self):
        classifier=ActivityClassifier(device='cpu')
        classifier.load_model()
        self.assertFalse(classifier.model.training)
        with torch.inference_mode():
            probs=classifier.model(torch.zeros(1,3,224,224)).softmax(1)
        self.assertEqual(tuple(probs.shape),(1,3))
        self.assertTrue(torch.isfinite(probs).all())
        self.assertEqual(classifier.provenance()['classes'],CLASSES)

    def test_only_confirmed_fresh_sidewalk_interactions_become_crops(self):
        classifier=ActivityClassifier()
        rows=observations()
        self.assertEqual(len(classifier.candidates(rows,self.mask,self.frame.shape)),1)
        self.assertEqual(classifier.candidates(rows,np.zeros_like(self.mask),self.frame.shape),[])
        rows[0]['position_source']='prediction'
        self.assertEqual(classifier.candidates(rows,self.mask,self.frame.shape),[])
        rows=observations();rows[1]['detection_age_seconds']=.6
        self.assertEqual(classifier.candidates(rows,self.mask,self.frame.shape),[])
        self.assertEqual(classifier.candidates(observations(11),self.mask,self.frame.shape),[])

    def test_persistence_negative_label_cache_and_scene_reset(self):
        classifier=ActivityClassifier()
        repair=np.array([[.01,.98,.01]],np.float32)
        with patch.object(classifier,'_classify',return_value=repair) as infer:
            for i in range(5):
                rows=classifier.update(i,i*.25,self.frame,observations(),self.mask,self.motion,True)
                self.assertEqual(rows[0]['confirmed'],i==4)
            self.assertEqual(infer.call_count,5)
            rows=classifier.update(5,1.1,self.frame,observations(),self.mask,self.motion,False)
            self.assertTrue(rows[0]['confirmed']);self.assertEqual(infer.call_count,5)
            rows=classifier.update(6,1.3,self.frame,observations(),self.mask,{'scene_cut':True},True)
            self.assertFalse(rows[0]['confirmed'])
        with patch.object(classifier,'_classify',return_value=np.array([[.99,.005,.005]],np.float32)):
            rows=classifier.update(7,1.6,self.frame,observations(),self.mask,self.motion,True)
            self.assertFalse(rows[0]['confirmed'])
            self.assertEqual(rows[0]['activity'],'other_activity')
        self.assertEqual(classifier.update(8,1.7,self.frame,[],self.mask,self.motion,False),[])
        self.assertEqual(classifier.states,{})

    def test_three_class_model_only_classifies_vehicle_interactions(self):
        for label,support,expected in [(1,2,True),(2,2,True),(1,3,True)]:
            classifier=ActivityClassifier();p=np.full((1,3),.01,np.float32);p[0,label]=.98
            with patch.object(classifier,'_classify',return_value=p):
                for i in range(5):rows=classifier.update(i,i*.25,self.frame,observations(support),self.mask,self.motion,True)
            self.assertEqual(rows[0]['confirmed'],expected)
        classifier=ActivityClassifier()
        for support in (4,6,10):
            self.assertEqual(classifier.candidates(observations(support),self.mask,self.frame.shape),[])

    def test_stale_and_tracking_only_images_do_not_accumulate_model_evidence(self):
        classifier=ActivityClassifier()
        with patch.object(classifier,'_classify',return_value=np.array([[.01,.98,.01]])) as infer:
            for i in range(8):
                rows=classifier.update(i,i*.25,self.frame,observations(),self.mask,self.motion,i==0)
                if rows:self.assertFalse(rows[0]['confirmed'])
            self.assertEqual(infer.call_count,1)
            self.assertEqual(rows,[])

    def test_empty_model_evidence_disables_proximity_only_repair_alerts(self):
        monitor=ViolationMonitor()
        for i in range(90):
            monitor.update(i,i/30,observations(),self.mask,self.motion,activity_regions=[])
        self.assertEqual(monitor.events,[])
        self.assertEqual(monitor.repair_regions,[])

    def test_model_events_export_evidence_without_sticky_activity_boxes(self):
        for kind in ('motorcycle_repair_suspected','street_vending_suspected','vehicle_washing_suspected'):
            monitor=ViolationMonitor()
            region=dict(kind=kind,confirmed=True,model='mobilenet_v3_small',track_ids=[1,2],
                        class_ids=[1,2],bbox_xyxy=[20,20,110,95],evidence_age_seconds=0.,
                        label='activity',score=.95,independent_model_samples=5)
            monitor.update(0,0.,observations(),self.mask,self.motion,source_size=[320,240],activity_regions=[region])
            created=monitor.update(1,.1,observations(),self.mask,self.motion,source_size=[320,240],activity_regions=[region])
            self.assertEqual(created[0]['kind'],kind)
            self.assertEqual(created[0]['evidence']['combined_bbox_source_pixels'],[40.,40.,220.,190.])
            self.assertEqual(monitor.confirmed_track_ids,set())
            monitor.update(2,.2,observations(),self.mask,self.motion,activity_regions=[])
            self.assertEqual(monitor.repair_regions,[])

    def test_pipeline_writes_activity_predictions_in_source_pixels_without_events(self):
        from tests.test_combined_video import FakeSidewalk,make_video
        from tests.test_video import FakeDetector
        from uav_pipeline.video.pipeline import process_video,VideoOptions
        from uav_pipeline.segmentation.session import SidewalkSession
        from uav_pipeline.segmentation.media.video import VideoPreviewOptions
        class FakeActivity:
            model_calls=images_evaluated=0
            def load_model(self):pass
            def provenance(self):return {'model':'fake_activity'}
            def update(self,index,timestamp,frame,observations,mask,motion,detector_ran):
                self.model_calls+=1;self.images_evaluated+=1
                return [dict(bbox_xyxy=[5.,5.,15.,20.],confirmed=True,activity='street_vending',score=.9)]
            def render(self,frame,predictions):return frame
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'input.avi';make_video(source)
            session=SidewalkSession(FakeSidewalk(),VideoPreviewOptions(keyframe_interval=1),sample_step=0)
            options=VideoOptions(work_max_side=80,min_hits=1,keyframe_interval=1,min_detection_interval=1,warmup_runs=0)
            with redirect_stdout(io.StringIO()):
                process_video(source,root/'out',FakeDetector(),options,sidewalk=session,activity=FakeActivity())
            record=json.loads((root/'out/tracks.jsonl').read_text().splitlines()[0])
            self.assertEqual(record['activity_predictions'][0]['bbox_xyxy'],[20.,20.,60.,80.])
            self.assertEqual(record['activity_regions'][0]['coordinate_space'],'source_pixels')
            summary=json.loads((root/'out/summary.json').read_text())
            self.assertEqual(summary['counts']['activity_model_calls'],4)
            self.assertFalse(summary['activity_classifier']['events_enabled'])

    def test_repair_confirmation_override_is_honored(self):
        classifier=ActivityClassifier();classifier.config['repair_confirmation_seconds']=2.
        with patch.object(classifier,'_classify',return_value=np.array([[.01,.98,.01]])):
            for i in range(9):
                rows=classifier.update(i,i*.25,self.frame,observations(),self.mask,self.motion,True)
                self.assertEqual(rows[0]['confirmed'],i==8)

    def test_three_class_manifest_and_probability_mapping(self):
        self.assertEqual(CLASSES,['other_activity','repairing_vehicle','washing_vehicle'])
        classifier=ActivityClassifier()
        with patch.object(classifier,'_classify',return_value=np.array([[.01,.01,.98]])):
            rows=classifier.update(0,0.,self.frame,observations(),self.mask,self.motion,True)
        self.assertEqual(rows[0]['activity'],'washing_vehicle')
        self.assertEqual(set(rows[0]['probabilities']),set(CLASSES))
        self.assertEqual(rows[0]['kind'],'vehicle_washing_suspected')

    def test_frame_cap_samples_even_when_time_interval_has_not_elapsed(self):
        classifier=ActivityClassifier()
        with patch.object(classifier,'_classify',return_value=np.array([[.01,.98,.01]])) as infer:
            for i in range(16):
                classifier.update(i,i/60,self.frame,observations(),self.mask,self.motion,i%5==0)
            self.assertEqual(infer.call_count,4)

    def test_participant_boxes_are_normalized_and_follow_crop_participants(self):
        classifier=ActivityClassifier()
        candidate=classifier.candidates(observations(3),self.mask,self.frame.shape)[0]
        self.assertEqual([p['class_id'] for p in candidate['participant_boxes_normalized']],[1,3])
        for participant,original in zip(candidate['participant_boxes_normalized'],observations(3)):
            np.testing.assert_allclose(np.array(participant['bbox_xyxy'])*[160,120,160,120],original['bbox_xyxy'])

    def test_preview_progress_keeps_temporal_confirmation_and_resets_on_label_change(self):
        classifier=ActivityClassifier()
        classifier.config['ema_alpha']=1.
        with patch.object(classifier,'_classify',return_value=np.array([[.01,.98,.01]])):
            for i in range(5):
                rows=classifier.update(i,i*.25,self.frame,observations(),self.mask,self.motion,True)
                self.assertEqual(rows[0]['confirmed'],i==4)
                self.assertEqual(rows[0]['confirmation_progress']['samples'],i+1)
        with patch.object(classifier,'_classify',return_value=np.array([[.01,.01,.98]])):
            rows=classifier.update(5,1.25,self.frame,observations(),self.mask,self.motion,True)
        self.assertFalse(rows[0]['confirmed'])
        self.assertEqual(rows[0]['activity'],'washing_vehicle')
        self.assertEqual(rows[0]['confirmation_progress']['samples'],1)
        self.assertEqual(rows[0]['confirmation_progress']['seconds'],0.)
        self.assertEqual(classifier.update(6,1.5,self.frame,[],self.mask,self.motion,True),[])

    def test_other_and_low_confidence_remain_visible_without_alerts(self):
        for probabilities in (np.array([[.98,.01,.01]]),np.array([[.2,.6,.2]])):
            classifier=ActivityClassifier()
            with patch.object(classifier,'_classify',return_value=probabilities):
                for i in range(5):
                    rows=classifier.update(i,i*.25,self.frame,observations(),self.mask,self.motion,True)
            self.assertFalse(rows[0]['confirmed'])
            self.assertEqual(rows[0]['status'],'other_or_uncertain')
            image=np.zeros_like(self.frame)
            with patch('uav_pipeline.video.pipeline._put_label') as label:
                classifier.render(image,rows)
            label.assert_called_once()
            np.testing.assert_array_equal(image[20,20],[255,200,0])
            np.testing.assert_array_equal(image[95,100],[0,220,0])
            np.testing.assert_array_equal(image[12,30],[0,180,255])
