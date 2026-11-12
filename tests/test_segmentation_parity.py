import math
import unittest

import numpy as np

from uav_pipeline.segmentation.core.config import DecisionConfig
from uav_pipeline.segmentation.inference.predictor import decide_mask
from uav_pipeline.segmentation.postprocessing.visualization import CLASS_COLORS_RGB, colorize_mask


class SegmentationParityTests(unittest.TestCase):
    def test_lut_preserves_every_uint8_label_and_other_dtype(self):
        for dtype in (np.uint8,np.int32):
            mask=np.arange(256,dtype=dtype).reshape(16,16)
            expected=np.zeros((16,16,3),np.uint8)
            for cls,color in CLASS_COLORS_RGB.items():expected[mask==cls]=color
            np.testing.assert_array_equal(colorize_mask(mask),expected)

    def test_three_group_decisions_preserve_sort_implementation(self):
        rng=np.random.default_rng(193)
        scores=rng.random((3,100,100),dtype=np.float32)
        scores/=scores.sum(axis=0,keepdims=True)
        scores[:,:2,:]=1/3
        config=DecisionConfig()
        sorted_scores=np.sort(scores,axis=0)
        confidence=sorted_scores[-1]
        margin=confidence-sorted_scores[-2]
        entropy=-(scores*np.log(np.maximum(scores,1e-8))).sum(axis=0)/math.log(3)
        expected=np.where(scores.argmax(axis=0)==2,2,0).astype(np.uint8)
        expected[(confidence<config.min_confidence)|(margin<config.min_margin)|
                 (entropy>config.max_normalized_entropy)]=config.unknown_id
        mask,actual_confidence=decide_mask(scores,config)
        np.testing.assert_array_equal(mask,expected)
        np.testing.assert_array_equal(actual_confidence,confidence)
