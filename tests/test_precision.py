import unittest
import numpy as np
from uav_pipeline.paths import PROJECT_ROOT
from uav_pipeline.detection.detector import YOLOEDetector
from uav_pipeline.detection.postprocess import Detection,mark_exclusive_overlaps
import torch


class PrecisionTests(unittest.TestCase):
    def test_ambiguous_high_confidence_machine_stays_review_only(self):
        d=YOLOEDetector(device='cpu',config_path=PROJECT_ROOT/'configs/detection_precision.yaml')
        index=next(i for i,v in enumerate(d.vocabulary) if v[0]=='portable generator')
        positive,review,_=d.decode(np.array([[10,10,30,30,.99,index]]),100,100)
        self.assertEqual(positive,[])
        self.assertTrue(any('ambiguous_phrase_requires_visual_verification' in x.review_reason for x in review))

    def test_oversized_material_scope_is_reviewed_but_small_brick_target_remains(self):
        d=YOLOEDetector(device='cpu',config_path=PROJECT_ROOT/'configs/detection_precision.yaml')
        index=next(i for i,v in enumerate(d.vocabulary) if v[0]=='pile of bricks')
        positive,review,_=d.decode(np.array([[0,0,90,90,.95,index]]),100,100)
        self.assertEqual(positive,[])
        self.assertTrue(any('oversized_object_scope_requires_review' in x.review_reason for x in review))
        positive,review,_=d.decode(np.array([[10,10,30,30,.95,index]]),100,100)
        self.assertTrue(any(x.class_id==11 for x in positive))

    def test_bucket_and_chair_same_object_is_not_two_confident_labels(self):
        rows=[Detection(str(cid),cid,str(cid),str(cid),'example','test','full',.9,[10,10,30,30]) for cid in (5,13)]
        report=mark_exclusive_overlaps(rows,[(5,13)])
        self.assertTrue(report)
        self.assertTrue(all(x.status=='review' for x in rows))

