"""Box results from a segmentation checkpoint without reconstructing pixel masks.

Keep the upstream segmentation output unpacking and class-channel count. The
checkpoint still computes its segmentation head; this skips CPU/GPU mask export.
Imported only when loading an actual segmentation checkpoint.
"""
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.yolo.segment.predict import SegmentationPredictor


class BoxesOnlySegmentationPredictor(SegmentationPredictor):
    def construct_results(self, preds, img, orig_imgs, protos=None):
        return DetectionPredictor.construct_results(self, preds, img, orig_imgs)

    def construct_result(self, pred, img, orig_img, img_path, proto=None):
        return DetectionPredictor.construct_result(self, pred, img, orig_img, img_path)
