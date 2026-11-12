"""Behaviour and numeric regressions for the TPV-to-UAV upgrade."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tests.test_combined_video import FakeSidewalk, make_video
from tests.test_video import FakeDetector, detection
from tests.test_tracking_trunks import motion
from uav_pipeline.tracking.motion import warp_box, warp_boxes
from uav_pipeline.video.pipeline import VideoOptions, process_video
from uav_pipeline.tracking.objects import ObjectTracker
from uav_pipeline.segmentation.core.config import DecisionConfig, PipelineConfig, PolygonConfig
from uav_pipeline.segmentation.core.pipeline import SidewalkPipeline
from uav_pipeline.segmentation.inference.predictor import InferenceResult, decide_mask
from uav_pipeline.segmentation.inference.preprocessing import luminance_score
from uav_pipeline.segmentation.media.video import VideoPreviewOptions
from uav_pipeline.segmentation.postprocessing.completion import close_ellipse
from uav_pipeline.segmentation.postprocessing.visualization import (
    colorize_mask, create_overlay, draw_feature_boundaries,
)
from uav_pipeline.segmentation.session import SidewalkSession
from uav_pipeline.events import ViolationMonitor


class UpgradeTests(unittest.TestCase):
    def test_benchmark_passes_configs_and_events_and_checks_target_per_video(self):
        from uav_pipeline.cli.benchmark import run_pipeline
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for fps in (16., 14.):
                def run(command, **kwargs):
                    folder = Path(command[command.index('--output')+1])
                    folder.mkdir()
                    summary = {'processed_frames': 2, 'completed': True,
                               'all_source_frames_processed': True, 'frame_skipping_enabled': False,
                               'steady_pipeline_fps_this_run': fps, 'processed_fps': fps,
                               'source_sha256': 'a'*64}
                    (folder/'summary.json').write_text(json.dumps(summary))
                    (folder/'tracks.jsonl').write_text('\n'.join(json.dumps({'frame_index': i, 'tracks': []})
                                                                  for i in range(2))+'\n')
                    self.assertIn('--events', command)
                    self.assertIn('--video-config', command)
                    self.assertIn('--sidewalk-config', command)
                    return SimpleNamespace(returncode=0)

                with patch('uav_pipeline.cli.benchmark.subprocess.run', side_effect=run), redirect_stdout(io.StringIO()):
                    code = run_pipeline(['--output', str(root/str(fps)), '--videos', '1', '--events',
                                         '--require-target-fps', '--baseline', str(root/'absent.json')])
                self.assertEqual(code, 0 if fps >= 15 else 1)

    def test_benchmark_rejects_truncated_records_and_partial_fps_acceptance(self):
        from uav_pipeline.cli.benchmark import summarize, run_pipeline
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'summary.json').write_text(json.dumps({'processed_frames': 2}))
            (root/'tracks.jsonl').write_text(json.dumps({'frame_index': 0, 'tracks': []})+'\n')
            with self.assertRaises(ValueError):
                summarize(root)
        with redirect_stdout(io.StringIO()), patch('sys.stderr', new=io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                run_pipeline(['--require-target-fps', '--max-frames', '5'])
            self.assertEqual(error.exception.code, 2)

    def test_parallel_clips_preserve_every_decoded_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.avi'
            make_video(source, frames=12)
            for use_ffmpeg in (False, True):
                prefix = 'ffmpeg' if use_ffmpeg else 'opencv'
                jobs = [(root/f'{prefix}_{i}.mp4', first, last)
                        for i, (first, last) in enumerate(((0, 6), (2, 10)))]
                context = (patch('uav_pipeline.events.shutil.which', return_value=None)
                           if not use_ffmpeg else patch('uav_pipeline.events.shutil.which', wraps=__import__('shutil').which))
                with context:
                    for i, (_, first, last) in enumerate(jobs):
                        ViolationMonitor._write_clip(source, root/f'{prefix}_reference_{i}.mp4', first, last, 10.)
                    self.assertEqual(ViolationMonitor._write_clips(source, jobs, 10.), {})
                for i, (target, first, last) in enumerate(jobs):
                    a, b = cv2.VideoCapture(str(root/f'{prefix}_reference_{i}.mp4')), cv2.VideoCapture(str(target))
                    try:
                        for _ in range(last-first):
                            ok_a, frame_a = a.read()
                            ok_b, frame_b = b.read()
                            self.assertTrue(ok_a and ok_b)
                            np.testing.assert_array_equal(frame_a, frame_b)
                        self.assertFalse(a.read()[0])
                        self.assertFalse(b.read()[0])
                    finally:
                        a.release()
                        b.release()

    def test_parallel_clip_error_does_not_lose_other_clip_or_leave_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good, bad = root/'good.mp4', root/'bad.mp4'

            def write(source, target, first, last, fps):
                target.write_bytes(b'partial' if first == 0 else b'complete')
                if first == 0:
                    raise RuntimeError('controlled clip error')

            with patch.object(ViolationMonitor, '_write_clip', side_effect=write):
                errors = ViolationMonitor._write_clips(root/'source.mp4', [(bad, 0, 2), (good, 2, 4)], 10.)
            self.assertEqual(errors, {'event_clips/bad.mp4': 'controlled clip error'})
            self.assertEqual(good.read_bytes(), b'complete')
            self.assertFalse(bad.exists())
            self.assertFalse(bad.with_suffix('.tmp.mp4').exists())

    def test_luminance_matches_median_for_odd_even_and_constant_images(self):
        rng = np.random.default_rng(3)
        for shape in ((1, 1), (1, 2), (37, 51), (520, 960), (521, 961)):
            for value in (None, 0, 127, 255):
                image = (rng.integers(0, 256, (*shape, 3), dtype=np.uint8) if value is None
                         else np.full((*shape, 3), value, np.uint8))
                self.assertEqual(luminance_score(image),
                                 float(np.median(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))/255.))

    def test_batch_camera_warp_matches_scalar_for_rotation_scale_and_empty(self):
        rng = np.random.default_rng(33)
        for count in (0, 1, 70, 300):
            boxes = rng.uniform(0, 800, (count, 4))
            boxes[:, 2:] += boxes[:, :2]
            for angle, scale in ((0, 1), (12, .95), (-6, 1.1)):
                matrix = cv2.getRotationMatrix2D((240, 160), angle, scale)
                matrix[:, 2] += [3, -5]
                expected = np.asarray([warp_box(box, matrix) for box in boxes], np.float32).reshape(-1, 4)
                np.testing.assert_array_equal(warp_boxes(boxes, matrix), expected)

    def test_overlay_rgb_and_bgr_match_full_image_reference_and_do_not_mutate(self):
        rng = np.random.default_rng(52)
        rgb = rng.integers(0, 256, (95, 137, 3), dtype=np.uint8)
        polygon = {'features': [{'properties': {'class_id': 2}, 'geometry': {
            'type': 'Polygon', 'coordinates': [[[5, 5], [80, 7], [80, 75], [5, 5]]]}}]}
        for kind in ('empty', 'full', 'sparse', 'unknown'):
            mask = np.zeros(rgb.shape[:2], np.uint8)
            if kind == 'full':
                mask[:] = 2
            elif kind == 'sparse':
                mask[30:50, 50:60] = 2
            elif kind == 'unknown':
                mask[5:15, 10:20] = 255
            for unknown in (False, True):
                for alpha in (0., .45, 1.):
                    active = mask != 0 if unknown else mask == 2
                    expected = rgb.copy()
                    blended = cv2.addWeighted(rgb, 1-alpha, colorize_mask(mask), alpha, 0)
                    expected[active] = blended[active]
                    np.testing.assert_array_equal(create_overlay(rgb, mask, alpha, unknown), expected)
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    actual = create_overlay(bgr, mask, alpha, unknown, color_order='bgr')
                    np.testing.assert_array_equal(actual, expected[:, :, ::-1])
                    out = draw_feature_boundaries(actual, polygon, PolygonConfig(), color_order='bgr', copy=False)
                    self.assertIs(out, actual)
                    reference = draw_feature_boundaries(expected, polygon, PolygonConfig())
                    np.testing.assert_array_equal(out, reference[:, :, ::-1])
                    np.testing.assert_array_equal(bgr, rgb[:, :, ::-1])

    def test_mask_argmax_ties_and_nan_match_legacy_decision(self):
        scores = np.array([[.5, .1, .2, np.nan, .2, .2],
                           [.5, .1, .2, .2, np.nan, .2],
                           [0., .8, .2, .8, .8, np.nan]], np.float32).reshape(3, 1, 6)
        mask, _ = decide_mask(scores, DecisionConfig(min_confidence=0, min_margin=0, max_normalized_entropy=1))
        expected = np.where(scores.argmax(axis=0) == 2, 2, 0).astype(np.uint8)
        np.testing.assert_array_equal(mask, expected)

    def test_deferred_mask_decision_handles_smoothing_and_shape_change(self):
        config = PipelineConfig(device='cpu')
        config.preprocessing.low_light_mode = 'disabled'
        config.completion.enabled = False
        config.completion.envelope_enabled = False
        image = np.zeros((48, 64, 3), np.uint8)
        scores = np.zeros((3, 48, 64), np.float32)
        scores[0] = .9
        scores[2, :, 32:] = .99
        scores[0, :, 32:] = .01
        scores /= scores.sum(0)
        calls = []

        def predict(variants, **kwargs):
            calls.append(kwargs)
            mask, confidence = (None, None) if kwargs.get('defer_decision') else decide_mask(scores, config.decision)
            return InferenceResult(mask, confidence, scores.copy(), [], [], [])

        pipeline = SidewalkPipeline(config)
        pipeline._predictor = SimpleNamespace(predict=predict)
        for previous in (scores.copy(), np.zeros((3, 12, 16), np.float32)):
            with patch('uav_pipeline.segmentation.core.pipeline.decide_mask', wraps=decide_mask) as decision:
                actual = pipeline.process_frame(image, prev_scores=previous, temporal_smoothing=.3)
                self.assertEqual(decision.call_count, 1)
                self.assertTrue(calls[-1]['defer_decision'])
                np.testing.assert_array_equal(actual.inference.mask, decide_mask(scores, config.decision)[0])

    def test_exact_rowwise_morphology_including_borders_and_small_images(self):
        rng = np.random.default_rng(9)
        for shape in ((1, 1), (7, 13), (64, 91), (416, 768)):
            for size in (3, 33, 61, 121):
                for image in (rng.integers(0, 256, shape, np.uint8),
                              np.zeros(shape, np.uint8), np.full(shape, 255, np.uint8)):
                    expected = cv2.morphologyEx(image, cv2.MORPH_CLOSE,
                                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
                    np.testing.assert_array_equal(close_ellipse(image, size, 'rowwise'), expected)
                    np.testing.assert_array_equal(close_ellipse(image, size), expected)

    def test_full_frame_association_keeps_id_through_long_mask_gap(self):
        class Detector(FakeDetector):
            def predict(self, image, **kwargs):
                return {'object_detections': [detection((20, 20, 40, 60), 1)], 'model_calls': 1}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.avi'
            make_video(source, frames=18)
            results = {}
            for full_frame in (False, True):
                session = SidewalkSession(FakeSidewalk(['left']*2 + ['empty']*12 + ['left']*4),
                                          VideoPreviewOptions(keyframe_interval=1), sample_step=0)
                options = VideoOptions(min_hits=2, keyframe_interval=1, min_detection_interval=1,
                                       max_lost_seconds=.5, max_prediction_seconds=.4,
                                       track_full_frame=full_frame)
                folder = root/str(full_frame)
                with redirect_stdout(io.StringIO()), patch('uav_pipeline.video.io.shutil.which', return_value=None):
                    process_video(source, folder, Detector(), options, sidewalk=session)
                rows = [json.loads(s) for s in (folder/'tracks.jsonl').read_text().splitlines()]
                self.assertTrue(all(not row['tracks'] for row in rows[2:14]))
                self.assertTrue(all(not row['detector_evidence']['object_detections'] for row in rows[2:14]))
                self.assertEqual(rows[1]['tracks'][0]['track_id'], 1)
                results[full_frame] = rows
            self.assertEqual(results[True][14]['tracks'][0]['track_id'], 1)
            self.assertEqual(results[False][15]['tracks'][0]['track_id'], 2)

    def test_dynamic_cart_is_opt_in_and_survives_state_roundtrip(self):
        for dynamic in (False, True):
            tracker = ObjectTracker(min_hits=1, dynamic_cart_tracking=dynamic)
            for index in range(6):
                box = [20+3*index, 20, 70+3*index, 70]
                tracker.update(index/10, [detection(box, 6)], motion())
            self.assertEqual(tracker.is_static(tracker.tracks[0]), not dynamic)
            restored = ObjectTracker(min_hits=1, dynamic_cart_tracking=dynamic)
            restored.load_state_dict(tracker.state_dict())
            self.assertEqual(restored.is_static(restored.tracks[0]), not dynamic)
            self.assertEqual(restored.next_id, tracker.next_id)

    def test_invalid_upgrade_options_are_rejected(self):
        for values in ({'track_full_frame': 1}, {'dynamic_cart_tracking': 'yes'},
                       {'kalman_size_ratio': float('nan')}, {'kalman_process_noise': 0},
                       {'kalman_class_scales': []}, {'kalman_class_scales': {'11': 1}},
                       {'kalman_class_scales': {'2': True}}):
            with self.assertRaises(ValueError):
                VideoOptions(**values).validate()
        config = PipelineConfig()
        config.completion.morphology_backend = 'unmeasured-auto'
        with self.assertRaises(ValueError):
            config.validate()
