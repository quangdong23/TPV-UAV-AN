import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import torch
from uav_pipeline.detection.detector import YOLOEDetector
from uav_pipeline.detection.classes import ClassId
from uav_pipeline.detection.postprocess import annotate_encroachment
from uav_pipeline.segmentation.session import keep_on_sidewalk
from uav_pipeline.events import ViolationMonitor
from uav_pipeline.tracking.objects import ObjectTracker


class ConstructionTests(unittest.TestCase):
    def test_new_classes_are_positive_and_static(self):
        d=YOLOEDetector(device='cpu')
        self.assertEqual(set(d.config['model']['supplementary_detector']['class_ids']),{10,11,12,13,14})
        for phrase,cid,group in d.vocabulary:
            if phrase in ('pile of bricks','sand pile','concrete blocks'):
                self.assertEqual(cid,11);self.assertIsNone(group)
        self.assertIn(('paving slabs', None, 'installed_paving'), d.vocabulary)
        for cid in (11,12):
            self.assertTrue(ObjectTracker().is_static({'class_id':cid}))
            idx=next(i for i,v in enumerate(d.vocabulary) if v[1]==cid)
            predicted,review,_=d.decode(np.array([[10,10,30,30,.9,idx]]),100,100)
            self.assertIn(cid,[v.class_id for v in predicted])

    def test_overlap_screen_and_temporal_event_agree(self):
        mask=np.zeros((100,100),np.uint8);mask[20:60,10:40]=2
        for cid in (11,12):
            row=dict(track_id=1,class_id=cid,bbox_xyxy=[10,20,90,80],position_source='detection',detection_age_seconds=0.,track_confirmed=True)
            # Bottom centre is outside sidewalk, but the material pile overlaps it.
            annotated=annotate_encroachment(row,mask)
            self.assertTrue(annotated['is_encroaching'])
            self.assertEqual(len(keep_on_sidewalk([row],mask)),1)
            monitor=ViolationMonitor()
            for i in range(9):self.assertEqual(monitor.update(i,i/30,[row],mask,{'reliable':False}),[])
            alerts=monitor.update(9,.3,[row],mask,{'reliable':False})
            self.assertEqual(len(alerts),1)
            self.assertEqual(alerts[0]['class_ids'],[cid])
            self.assertEqual(alerts[0]['evidence']['contact_rule'],'bbox_sidewalk_overlap_at_least_0.1')
            empty=np.zeros_like(mask)
            self.assertEqual(keep_on_sidewalk([row],empty),[])
            self.assertFalse(annotate_encroachment(row,empty)['is_encroaching'])
            monitor=ViolationMonitor()
            for i in range(12):self.assertEqual(monitor.update(i,i/30,[row],empty,{'reliable':False}),[])

    def test_merge_remaps_legacy_and_extra_channels_for_batches(self):
        d=YOLOEDetector(device='cpu')
        def result(cls):
            return SimpleNamespace(boxes=SimpleNamespace(data=torch.tensor([[1.,2.,10.,20.,.8,float(cls)]])),speed={'inference':1.})
        d.engine=Mock();d.engine.predict.return_value=[result(0),result(0)]
        d._primary_indices=[3]
        d.supplementary_engine=Mock();d._supplementary_indices=[7]
        with patch('ultralytics.engine.model.Model.predict',return_value=[result(0),result(0)]):
            output=d._predict_boxes(source=[np.zeros((32,32,3),np.uint8)]*2,imgsz=640,half=False)
        self.assertEqual(len(output),2)
        for row in output:
            self.assertEqual(row.boxes.data[:,5].tolist(),[3.,7.])
            self.assertEqual(row.speed['inference'],2.)

    def test_paving_hypothesis_cannot_become_an_alert_without_review(self):
        d=YOLOEDetector(device='cpu')
        idx=next(i for i,v in enumerate(d.vocabulary) if v[0]=='stacked paving slabs')
        predicted,review,_=d.decode(np.array([[1,1,90,90,.9,idx]]),100,100)
        self.assertEqual(predicted,[])
        self.assertEqual(review[0].class_id,11)
        self.assertIn('installed_pavement',review[0].review_reason)


def test_new_prompt_channels_decode_without_inventing_water_or_garbage_violations():
    import numpy as np
    from uav_pipeline.detection.detector import YOLOEDetector
    from uav_pipeline.detection.classes import ClassId
    from uav_pipeline.detection.postprocess import ENCROACHMENT_CLASSES
    from uav_pipeline.detection.registry import ClassRegistry
    from uav_pipeline.events import ViolationMonitor
    detector=YOLOEDetector(device='cpu')
    expected={'electric drill':12,'handheld power drill':12,'angle grinder':12,
              'aluminum window frame':12,'stack of loose bricks':11,
              'cardboard box':10,'plastic bucket':13,'garbage truck':14,
              'garbage collection handcart':14}
    for phrase,cid in expected.items():
        index=next(i for i,v in enumerate(detector.vocabulary) if v[0]==phrase)
        assert detector.vocabulary[index][1]==cid
        positive,review,_=detector.decode(np.array([[20,20,60,80,.95,index]]),100,100)
        assert any(d.class_id==cid for d in positive+review)
    for cid in (13,14):
        assert ClassRegistry.get(cid).support_status=='candidate_verification'
        assert cid not in ENCROACHMENT_CLASSES
        monitor=ViolationMonitor()
        observation={'track_id':1,'class_id':cid,'bbox_xyxy':[20,20,60,80],
                     'position_source':'detection','detection_age_seconds':0.,'track_confirmed':True}
        motion={'reliable':True,'scene_cut':False,'matrix':np.eye(2,3)}
        for i in range(20):
            assert monitor.update(i,i/30,[observation],np.full((100,100),2,np.uint8),motion)==[]


def test_legacy_vocabulary_is_preserved_and_supplement_excludes_new_cart_sign_prompts():
    import json
    from uav_pipeline.paths import MODELS_DIR
    from uav_pipeline.detection.detector import YOLOEDetector
    d=YOLOEDetector(device='cpu')
    legacy=json.loads((MODELS_DIR/'yoloe-v8s-single.json').read_text())
    lookup={row[0]:row for row in d.vocabulary}
    assert all(row[0] in lookup for row in legacy['vocabulary'])
    extension=d.config['model']['supplementary_detector']
    assert 6 not in extension['class_ids'] and 7 not in extension['class_ids']
    manifest=json.loads((MODELS_DIR/'yoloe-sidewalk-extra-v5.json').read_text())
    expected=[phrase for phrase,cid,_ in d.vocabulary if cid in extension['class_ids'] or phrase in extension['counter_phrases']]
    assert manifest['names']==expected
    assert len(expected)==71


def test_unvalidated_garbage_vehicle_is_review_only_and_never_enters_tracker():
    import numpy as np
    from uav_pipeline.detection.detector import YOLOEDetector
    from uav_pipeline.tracking.objects import ObjectTracker
    d=YOLOEDetector(device='cpu')
    index=next(i for i,v in enumerate(d.vocabulary) if v[0]=='garbage truck')
    positive,review,_=d.decode(np.array([[20,20,80,80,.99,index]]),100,100)
    assert not positive
    assert review and review[0].class_id==14
    assert 'new_class_requires_labelled_validation' in review[0].review_reason
    movement={'scene_cut':False,'reliable':True,'matrix':np.eye(2,3),'local_boxes':{}}
    assert ObjectTracker(min_hits=1).update(0,[v.to_dict() for v in review],movement)==[]
