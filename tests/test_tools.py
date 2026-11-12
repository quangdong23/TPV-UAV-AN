"""Run with the separate TrackEval environment: python -m unittest ...test_tools."""
import json
import copy
from pathlib import Path
import tempfile
import unittest
from importlib.util import find_spec

from uav_pipeline.evaluation.accept import accept_runs, quality_check, speed_check
from uav_pipeline.evaluation.regression import compare_runs
from uav_pipeline.evaluation.tracking import evaluate_tracking


class EvaluationTests(unittest.TestCase):
    @unittest.skipUnless(find_spec("trackeval") is not None, "Install optional evaluation dependencies")
    def test_events_enabled_need_explicit_event_ground_truth(self):
        a, b = self.candidate('before', switch=True), self.candidate('after')
        for folder in (a, b):
            path = folder/'summary.json'
            summary = json.loads(path.read_text())
            summary['event_rules'] = {}
            path.write_text(json.dumps(summary))
            (folder/'events.json').write_text(json.dumps({'events': []}))
        result = accept_runs([a], [b], [self.annotation])
        self.assertFalse(result['passed'])
        self.assertIn('Runs with events enabled require explicit event ground truth', result['videos'][0]['reasons'])
        self.gt['events'] = []
        self.annotation.write_text(json.dumps(self.gt))
        self.assertTrue(accept_runs([a], [b], [self.annotation])['passed'])

    @unittest.skipUnless(find_spec("trackeval") is not None, "Install optional evaluation dependencies")
    def test_class_regression_and_different_runtime_are_rejected(self):
        a = evaluate_tracking(self.candidate('a'), self.annotation)
        b = copy.deepcopy(a)
        b['by_class']['4']['IDF1'] = .5
        self.assertTrue(quality_check(a, b, 'preserve', .001))
        summary = {'completed': True, 'all_source_frames_processed': True, 'frame_skipping_enabled': False,
                   'source_sha256': 'a'*64, 'gpu': 'test', 'processed_frames': 4,
                   'steady_pipeline_fps_this_run': 16., 'runtime': {'device': 'cuda', 'versions': {'torch': 'x'}}}
        changed = copy.deepcopy(summary)
        changed['runtime']['versions']['torch'] = 'y'
        self.assertTrue(speed_check(summary, changed, 15., 'steady', 0.))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gt = {'schema_version': 1, 'coordinate_space': 'source_pixels',
                   'human_verified': True, 'source_sha256': 'a'*64, 'source_size': [100, 100],
                   'clips': [{'name': 'test', 'frames': [{'frame_index': i, 'tracks': [self.track(7)]}
                                                        for i in range(4)]}]}
        self.annotation = self.root/'gt.json'
        self.annotation.write_text(json.dumps(self.gt))

    @staticmethod
    def track(identity, cid=4):
        return {'track_id': identity, 'class_id': cid, 'bbox_xyxy': [10, 20, 30, 40]}

    def candidate(self, name, *, switch=False, wrong_class=False, fps=16.):
        folder = self.root/name
        folder.mkdir()
        summary = {'source_sha256': 'a'*64, 'completed': True, 'all_source_frames_processed': True,
                   'frame_skipping_enabled': False, 'processed_frames': 4, 'gpu': 'test_gpu',
                   'steady_pipeline_fps_this_run': fps, 'processed_fps': fps}
        (folder/'summary.json').write_text(json.dumps(summary))
        rows = [{'frame_index': i, 'coordinate_space': 'source_pixels', 'source_size': [100, 100],
                 'timestamp_seconds': i/10, 'tracks': [self.track(2 if switch and i >= 2 else 1,
                                                                5 if wrong_class else 4)]}
                for i in range(4)]
        (folder/'tracks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return folder

    @unittest.skipUnless(find_spec("trackeval") is not None, "Install optional evaluation dependencies")
    def test_official_metrics_perfect_and_id_switch(self):
        perfect = evaluate_tracking(self.candidate('perfect'), self.annotation)
        switched = evaluate_tracking(self.candidate('switched', switch=True), self.annotation)
        self.assertEqual(perfect['overall']['IDF1'], 1.)
        self.assertEqual(perfect['overall']['HOTA'], 1.)
        self.assertEqual(switched['overall']['IDF1'], .5)
        self.assertAlmostEqual(switched['overall']['HOTA'], 2**-.5)
        self.assertEqual(switched['overall']['ID_switches'], 1)

    @unittest.skipUnless(find_spec("trackeval") is not None, "Install optional evaluation dependencies")
    def test_wrong_class_is_false_positive_and_negative(self):
        actual = evaluate_tracking(self.candidate('wrong', wrong_class=True), self.annotation)['overall']
        self.assertEqual(actual['true_positives'], 0)
        self.assertEqual(actual['false_positives'], 4)
        self.assertEqual(actual['false_negatives'], 4)

    def test_unverified_gt_and_wrong_source_are_rejected(self):
        folder = self.candidate('candidate')
        self.gt['human_verified'] = False
        self.annotation.write_text(json.dumps(self.gt))
        with self.assertRaisesRegex(ValueError, 'human_verified'):
            evaluate_tracking(folder, self.annotation)
        self.gt['human_verified'] = True
        self.gt['source_sha256'] = 'b'*64
        self.annotation.write_text(json.dumps(self.gt))
        with self.assertRaisesRegex(ValueError, 'hash'):
            evaluate_tracking(folder, self.annotation)

    @unittest.skipUnless(find_spec("trackeval") is not None, "Install optional evaluation dependencies")
    def test_quality_gain_passes_but_degradation_and_no_gain_fail(self):
        bad = evaluate_tracking(self.candidate('bad', switch=True), self.annotation)
        good = evaluate_tracking(self.candidate('good'), self.annotation)
        self.assertEqual(quality_check(bad, good, 'improve', .001), [])
        self.assertTrue(quality_check(good, bad, 'improve', .001))
        self.assertTrue(quality_check(good, good, 'improve', .001))

    @unittest.skipUnless(find_spec("trackeval") is not None, "Install optional evaluation dependencies")
    def test_acceptance_checks_speed_and_quality_for_each_video(self):
        reference = self.candidate('reference', switch=True)
        fast = self.candidate('fast')
        report = accept_runs([reference], [fast], [self.annotation])
        self.assertTrue(report['passed'])
        slow = self.candidate('slow', fps=14.9)
        self.assertFalse(accept_runs([reference], [slow], [self.annotation])['passed'])
        with self.assertRaises(ValueError):
            accept_runs([reference], [fast], None)
        with self.assertRaises(ValueError):
            accept_runs([reference, reference], [fast, fast], [self.annotation]*2)

    def test_preserve_mode_uses_regression_without_pretending_accuracy(self):
        a, b = self.candidate('a'), self.candidate('b')
        result = accept_runs([a], [b], None, mode='preserve')
        self.assertTrue(result['passed'])
        self.assertFalse(result['accuracy_measured'])
        self.assertTrue(compare_runs(a, b)['semantic_regression_passed'])
        changed = self.candidate('changed', switch=True)
        self.assertFalse(accept_runs([a], [changed], None, mode='preserve')['passed'])

    def test_partial_skipping_missing_speed_and_other_gpu_fail_closed(self):
        a = {'completed': True, 'all_source_frames_processed': True, 'frame_skipping_enabled': False,
             'source_sha256': 'a'*64, 'gpu': 'GPU A', 'processed_frames': 100,
             'processed_fps': 16., 'steady_pipeline_fps_this_run': 16.}
        for changes in ({'completed': False}, {'all_source_frames_processed': False},
                        {'frame_skipping_enabled': True}, {'steady_pipeline_fps_this_run': None},
                        {'steady_pipeline_fps_this_run': float('nan')}, {'gpu': 'GPU B'},
                        {'processed_frames': 99}, {'source_sha256': 'b'*64}):
            self.assertTrue(speed_check(a, {**a, **changes}, 15., 'steady', 0.))
        self.assertEqual(speed_check(a, a, 15., 'steady', 0.), [])

    def test_native_mask_path_cannot_escape_candidate_directory(self):
        folder = self.candidate('candidate')
        self.gt['clips'][0]['frames'][0]['sidewalk_mask'] = 'truth.png'
        self.annotation.write_text(json.dumps(self.gt))
        import cv2
        import numpy as np
        cv2.imwrite(str(self.root/'truth.png'), np.zeros((100, 100), np.uint8))
        path = folder/'tracks.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]['sidewalk'] = {'mask_path': '../truth.png'}
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        with self.assertRaisesRegex(ValueError, 'inside candidate'):
            evaluate_tracking(folder, self.annotation)


if __name__ == '__main__':
    unittest.main()
