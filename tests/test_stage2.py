"""Decode ordering, bounded telemetry and detection-backed ROI recovery."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
import torch

from uav_pipeline.detection.roi import grid_regions, select_recovery_regions, track_recovery_regions
from uav_pipeline.video.capture import PrefetchCapture
from uav_pipeline.video.telemetry import LatencyHistogram
from uav_pipeline.video.pipeline import VideoOptions, process_video
from uav_pipeline.tracking.objects import ObjectTracker
from uav_pipeline.detection.detector import YOLOEDetector
from tests.test_tracking_trunks import detection, motion
from tests.test_combined_video import make_video
from tests.test_video import FakeDetector
from uav_pipeline.evaluation.regression import difference


class Capture:
    def __init__(self, times=(0., 37., 99., 140.), fail=False):
        self.times, self.index, self.fail = times, 0, fail
        self.closed = False

    def isOpened(self):
        return not self.closed

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.index
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.times[max(0, self.index-1)]
        return {cv2.CAP_PROP_FPS: 30., cv2.CAP_PROP_FRAME_COUNT: len(self.times)}.get(prop, 1.)

    def read(self):
        if self.fail and self.index == 1:
            raise ValueError('decoder failure')
        if self.index == len(self.times):
            return False, None
        self.index += 1
        return True, np.full((2, 3, 3), self.index, np.uint8)

    def release(self):
        self.closed = True


class Stage2Tests(unittest.TestCase):
    def test_prefetch_consumer_positions_and_vfr_timestamps(self):
        for capacity in (1, 2):
            raw = Capture()
            capture = PrefetchCapture(raw, capacity)
            try:
                self.assertEqual(capture.get(cv2.CAP_PROP_POS_FRAMES), 0)
                for index, timestamp in enumerate(raw.times):
                    ok, frame = capture.read()
                    self.assertTrue(ok)
                    self.assertEqual(int(frame[0, 0, 0]), index+1)
                    self.assertEqual(capture.get(cv2.CAP_PROP_POS_MSEC), timestamp)
                    self.assertEqual(capture.get(cv2.CAP_PROP_POS_FRAMES), index+1)
                self.assertFalse(capture.read()[0])
                self.assertFalse(capture.read()[0])
            finally:
                capture.release()
            self.assertTrue(raw.closed)
            self.assertFalse(capture.worker.is_alive())

    def test_prefetch_error_and_early_release_of_full_queue(self):
        capture = PrefetchCapture(Capture(fail=True), 1)
        try:
            self.assertTrue(capture.read()[0])
            with self.assertRaisesRegex(RuntimeError, 'decoder failed'):
                capture.read()
        finally:
            capture.release()
        capture = PrefetchCapture(Capture(tuple(range(100))), 1)
        capture.release()
        capture.release()
        self.assertFalse(capture.worker.is_alive())
        self.assertLessEqual(capture.queue.qsize(), 1)

    def test_prefetch_keeps_video_records_and_resumed_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'input.avi'
            make_video(source, 8)
            rows = []
            for prefetch in (0, 2):
                folder = root/str(prefetch)
                options = VideoOptions(min_hits=1, keyframe_interval=2, min_detection_interval=2,
                                       segment_frames=2, decode_prefetch=prefetch)
                with redirect_stdout(io.StringIO()), patch('uav_pipeline.video.io.shutil.which', return_value=None):
                    process_video(source, folder, FakeDetector(), options, max_frames=4)
                    process_video(source, folder, FakeDetector(), options, resume=True)
                rows.append([json.loads(line) for line in (folder/'tracks.jsonl').read_text().splitlines()])
            self.assertEqual(len(rows[0]), 8)
            self.assertEqual(len(rows[1]), 8)
            for before, after in zip(*rows):
                self.assertIsNone(difference(before, after, 1e-4))

    def test_histogram_quantiles_have_documented_upper_bound_and_bounded_storage(self):
        histogram = LatencyHistogram()
        self.assertIsNone(histogram.quantile(.95))
        values = np.linspace(.00001, .5, 20000)
        for value in values:
            histogram.add(float(value))
        for fraction in (.5, .95):
            exact = values[max(0, int(np.ceil(len(values)*fraction))-1)]
            self.assertGreaterEqual(histogram.quantile(fraction), exact-1e-12)
            self.assertLessEqual(histogram.quantile(fraction), exact*1.01+1e-12)
        bins = len(histogram.bins)
        for _ in range(100000):
            histogram.add(.1)
        self.assertLessEqual(len(histogram.bins), bins+1)
        self.assertEqual(histogram.count, 120000)
        for invalid in (float('nan'), float('inf'), -1):
            with self.assertRaises(ValueError):
                histogram.add(invalid)

    def test_recovery_retains_grid_coverage_without_adding_crop_budget(self):
        grid = grid_regions(1920, 1080, 640)
        cursor, seen = 0, set()
        for index in range(len(grid)*2):
            focused = [grid[index % len(grid)]] if index % 3 else [(1200, 400, 1840, 1040)]
            selected, advance = select_recovery_regions(grid, 2, cursor, focused)
            self.assertLessEqual(len(selected), 2)
            self.assertTrue(any(region in grid for region in selected))
            seen.update(grid[(cursor+i) % len(grid)] for i in range(advance))
            cursor += advance
        self.assertEqual(seen, set(grid))
        self.assertEqual(select_recovery_regions(grid, 1, 0, [grid[-1]])[0], [grid[0]])
        self.assertEqual(select_recovery_regions(grid, 0, 0, [grid[-1]])[0], [])

    def test_recovery_requires_confirmed_recent_geometry_and_resets_on_scene_cut(self):
        tracker = ObjectTracker(min_hits=2)
        d = detection([1500, 850, 1530, 920], cid=1, source='roi_1200_440')
        tracker.update(0, [d], motion())
        args = dict(shape=(1080, 1920, 3), patch_size=640, min_hits=2,
                    dynamic_limit=.4, static_limit=1.5)
        local = motion({1: d['bbox_xyxy']})
        self.assertFalse(track_recovery_regions(tracker.tracks, .3, local, **args))
        tracker.update(.05, [d], motion())
        self.assertTrue(track_recovery_regions(tracker.tracks, .3, local, **args))
        self.assertFalse(track_recovery_regions(tracker.tracks, 2., local, **args))
        self.assertFalse(track_recovery_regions(tracker.tracks, .3, motion(), **args))
        self.assertFalse(track_recovery_regions(tracker.tracks, .3, {**local, 'scene_cut': True}, **args))

    def test_targeted_crop_recovers_controlled_detector_output_and_same_id_with_same_model_calls(self):
        detector = YOLOEDetector(device='cpu')
        detector.engine = Mock()
        image = np.zeros((1080, 1920, 3), np.uint8)
        image[850:920, 1500:1530, 0] = 255

        def predict(**kwargs):
            crops = kwargs['source']
            if not isinstance(crops, list):
                return [SimpleNamespace(boxes=SimpleNamespace(data=torch.empty((0, 6))))]
            output = []
            for crop in crops:
                y, x = np.nonzero(crop[:, :, 0])
                data = (torch.tensor([[int(x.min()), int(y.min()), int(x.max())+1, int(y.max())+1, .8, 0]])
                        if len(x) else torch.empty((0, 6)))
                output.append(SimpleNamespace(boxes=SimpleNamespace(data=data)))
            return output

        detector.engine.predict.side_effect = predict
        old = detector.predict_video_frame(image, color_order='BGR', roi_patch_budget=2, roi_offset=0)
        tracker = ObjectTracker(min_hits=1, max_prediction_seconds=.4)
        d = detection([1500, 850, 1530, 920], cid=1, source='roi_1200_440')
        tracker.update(0, [d], motion())
        movement = motion({1: d['bbox_xyxy']})
        hints = track_recovery_regions(tracker.tracks, .3, movement, image.shape, 640,
                                      min_hits=1, dynamic_limit=.4, static_limit=1.5)
        new = detector.predict_video_frame(image, color_order='BGR', roi_patch_budget=2,
                                          roi_offset=0, roi_recovery_regions=hints)
        self.assertFalse(old['object_detections'])
        self.assertEqual(old['model_calls'], new['model_calls'])
        self.assertEqual(old['images_evaluated'], new['images_evaluated'])
        self.assertEqual(new['roi_recovery_patches'], 1)
        self.assertEqual(new['object_detections'][0]['bbox_xyxy'], [1500., 850., 1530., 920.])
        tracks = tracker.update(.3, new['object_detections'], movement,
                                observed_regions=new['roi_regions'], observed_classes=new['observed_class_ids'])
        self.assertEqual(tracks[0]['track_id'], 1)
        self.assertEqual(tracks[0]['position_source'], 'detection')
        self.assertTrue(tracker.update(.5, None, motion()))

    def test_stage2_options_are_bounded(self):
        for options in ({'decode_prefetch': -1}, {'decode_prefetch': 9}, {'decode_prefetch': True},
                        {'roi_recovery_budget': 2}, {'roi_recovery_budget': True}):
            with self.assertRaises(ValueError):
                VideoOptions(**options).validate()
