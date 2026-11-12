"""Offline COCO tests with controlled labels, not a UAV accuracy claim."""
import json
from pathlib import Path
import tempfile
import unittest
from importlib.util import find_spec

from uav_pipeline.evaluation.coco import evaluate_coco, export_detector_frames


class CocoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gt = self.root/'annotations.json'
        self.gt.write_text(json.dumps({'info': {}, 'images': [{'id': 5, 'width': 100, 'height': 100}],
                                      'categories': [{'id': 17, 'name': 'person'}],
                                      'annotations': [{'id': 1, 'image_id': 5, 'category_id': 17,
                                                       'bbox': [10, 20, 30, 40], 'area': 1200, 'iscrowd': 0}]}))

    @unittest.skipUnless(find_spec("pycocotools") is not None, "Install optional evaluation dependencies")
    def test_perfect_ap_and_empty_predictions(self):
        predictions = self.root/'predictions.json'
        predictions.write_text(json.dumps([{'image_id': 5, 'category_id': 17,
                                           'bbox': [10, 20, 30, 40], 'score': .9}]))
        perfect = evaluate_coco(self.gt, predictions, self.root/'report.json')
        self.assertAlmostEqual(perfect['metrics']['mAP_50_95'], 1.)
        self.assertIsNone(perfect['metrics']['AP_small'])
        predictions.write_text('[]')
        empty = evaluate_coco(self.gt, predictions, self.root/'empty.json')
        self.assertEqual(empty['metrics']['mAP_50_95'], 0.)

    @unittest.skipUnless(find_spec("pycocotools") is not None, "Install optional evaluation dependencies")
    def test_invalid_prediction_is_rejected(self):
        predictions = self.root/'predictions.json'
        for changes in ({'bbox': [10, 20, -1, 40]}, {'score': float('nan')}, {'category_id': 99}):
            row = {'image_id': 5, 'category_id': 17, 'bbox': [10, 20, 30, 40], 'score': .9, **changes}
            predictions.write_text(json.dumps([row]))
            with self.assertRaises(ValueError):
                evaluate_coco(self.gt, predictions, self.root/'report.json')

    def test_export_uses_observed_source_detections_and_explicit_mappings(self):
        candidate = self.root/'candidate'
        candidate.mkdir()
        (candidate/'summary.json').write_text(json.dumps({'source_sha256': 'a'*64}))
        frame_map = self.root/'frames.json'
        frame_map.write_text(json.dumps({'source_sha256': 'a'*64, 'image_to_frame': {'5': 10}}))
        category_map = self.root/'classes.json'
        category_map.write_text(json.dumps({'1': 17}))
        record = {'frame_index': 10, 'coordinate_space': 'source_pixels', 'source_size': [100, 100],
                  'detector_ran': True, 'tracks': [{'bbox_xyxy': [90, 90, 99, 99]}],
                  'detector_evidence': {'coordinate_space': 'source_pixels', 'object_detections': [
                      {'class_id': 1, 'bbox_xyxy': [10, 20, 40, 60], 'score': .9}]}}
        path = candidate/'tracks.jsonl'
        path.write_text(json.dumps(record)+'\n')
        output = self.root/'predictions.json'
        self.assertEqual(export_detector_frames(candidate, self.gt, frame_map, category_map, output), [17])
        self.assertEqual(json.loads(output.read_text())[0]['bbox'], [10, 20, 30, 40])
        record['detector_ran'] = False
        path.write_text(json.dumps(record)+'\n')
        with self.assertRaisesRegex(ValueError, 'actual source-pixel detector'):
            export_detector_frames(candidate, self.gt, frame_map, category_map, output)
        frame_map.write_text(json.dumps({'source_sha256': 'b'*64, 'image_to_frame': {'5': 10}}))
        with self.assertRaisesRegex(ValueError, 'hash'):
            export_detector_frames(candidate, self.gt, frame_map, category_map, output)


if __name__ == '__main__':
    unittest.main()
