import json
import io
from contextlib import redirect_stdout
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from uav_pipeline.tracking.motion import CameraMotion
from uav_pipeline.video.pipeline import VideoOptions, process_video
from uav_pipeline.tracking.objects import ObjectTracker
from uav_pipeline.video.scheduler import detection_reason


def movement(shift=0, local=None, cut=False):
    return {'matrix': np.array([[1., 0, shift], [0, 1., 0]]), 'reliable': True,
            'local_boxes': local or {}, 'scene_cut': cut, 'reason': 'test'}


def detection(box=(20, 20, 50, 50), cid=1, status='predicted'):
    return {'class_id': cid, 'bbox_xyxy': list(box), 'score': .8,
            'class_name_vi': 'Người', 'status': status}


class TrackerTests(unittest.TestCase):
    def test_sparse_detector_schedule_survives_one_weak_or_lost_track(self):
        options = VideoOptions()
        tracks = [{'id': i, 'last_detection': 0., 'missed_detection': False} for i in range(10)]
        last_index, last_time = -1, -1.
        calls = []
        for frame in range(60):
            motion = movement(local={i: [0, 0, 10, 10] for i in range(9)})
            reason = detection_reason(frame, frame/30, tracks, motion, last_index, last_time, options)
            if reason:
                calls.append(frame)
                last_index, last_time = frame, frame/30
                for track in tracks:
                    track['last_detection'] = last_time
        self.assertEqual(calls, list(range(0, 60, options.keyframe_interval)))
        # A stale lost track previously triggered another detector call on every frame.
        tracks.append({'id': 100, 'last_detection': -100., 'missed_detection': True})
        self.assertIsNone(detection_reason(55, 55/30, tracks, movement(), 54, 54/30, options))

    def test_bad_camera_flow_is_rate_limited_but_cuts_are_immediate(self):
        options = VideoOptions()
        motion = movement()
        motion['reliable'] = False
        last_index, last_time, calls = -1, -1., []
        for frame in range(60):
            reason = detection_reason(frame, frame/30, [], motion, last_index, last_time, options)
            if reason:
                calls.append(frame)
                last_index, last_time = frame, frame/30
        self.assertEqual(calls, list(range(0, 60, 3)))
        self.assertEqual(detection_reason(1, 1/30, [], movement(cut=True), 0, 0., options), 'scene_cut')

    def test_flow_does_not_confirm_or_create_and_empty_detection_hides(self):
        tracker = ObjectTracker(min_hits=2)
        self.assertEqual(tracker.update(0., [detection()], movement()), [])
        self.assertEqual(tracker.update(.1, None, movement(local={1: [21, 20, 51, 50]})), [])
        self.assertEqual(tracker.update(.2, [detection((22, 20, 52, 50))], movement())[0]['track_id'], 1)
        self.assertEqual(tracker.update(.3, [], movement()), [])
        self.assertEqual(tracker.update(.4, None, movement()), [])
        self.assertEqual(tracker.update(.5, [detection((23, 20, 53, 50))], movement())[0]['track_id'], 1)

    def test_gmc_ids_class_gating_and_cut(self):
        tracker = ObjectTracker(min_hits=1)
        rows = tracker.update(0., [detection(), detection((80, 20, 110, 50), 2)], movement())
        self.assertEqual([r['track_id'] for r in rows], [1, 2])
        rows = tracker.update(.1, [detection((120, 20, 150, 50)), detection((180, 20, 210, 50), 2)], movement(100))
        self.assertEqual([r['track_id'] for r in rows], [1, 2])
        rows = tracker.update(.2, [detection()], movement(cut=True))
        self.assertEqual(rows[0]['track_id'], 3)

    def test_review_ignored_and_timestamp_expiry(self):
        tracker = ObjectTracker(min_hits=1, max_lost_seconds=1., max_prediction_seconds=.2)
        self.assertEqual(tracker.update(0., [detection(status='review')], movement()), [])
        tracker.update(.1, [detection()], movement())
        self.assertEqual(tracker.update(.4, None, movement()), [])
        self.assertEqual(tracker.update(2., [detection()], movement())[0]['track_id'], 2)

    def test_checkpoint_preserves_next_update(self):
        original = ObjectTracker(min_hits=1)
        original.update(0., [detection()], movement())
        original.update(.1, None, movement(local={1: [23, 20, 53, 50]}))
        restored = ObjectTracker(min_hits=1)
        restored.load_state_dict(json.loads(json.dumps(original.state_dict())))
        self.assertEqual(original.update(.2, [detection((25, 20, 55, 50))], movement()),
                         restored.update(.2, [detection((25, 20, 55, 50))], movement()))

    def test_camera_translation_and_independent_foreground_flow(self):
        rng = np.random.default_rng(18)
        first = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        second = cv2.warpAffine(first, np.array([[1., 0, 4], [0, 1., 2]]), (320, 240))
        # Object moves an additional 3 pixels horizontally over camera motion.
        second[82:122, 87:127] = first[80:120, 80:120]
        motion = CameraMotion(320)
        motion.update(first, {})
        original = cv2.calcOpticalFlowPyrLK
        with patch('cv2.calcOpticalFlowPyrLK', wraps=original) as optical_flow:
            result = motion.update(second, {1: [80, 80, 120, 120], 2: [180, 80, 220, 120]})
        self.assertEqual(optical_flow.call_count, 2)  # one forward/backward batch for all objects
        self.assertTrue(result['reliable'])
        np.testing.assert_allclose(result['matrix'][:, 2], [4, 2], atol=.6)
        self.assertIn(1, result['local_boxes'])
        self.assertGreater(result['local_boxes'][1][0], 85)


class FakeDetector:
    device = 'cpu'
    config = {'model': {'model_id': 'fake'}, 'inference': {}}

    def load_model(self):
        pass

    def provenance(self):
        return {'versions': {'transformers': 'test'}}

    def predict(self, image, **kwargs):
        return {'object_detections': [detection()], 'review_candidates': [detection(status='review')], 'model_calls': 6}


class VideoTests(unittest.TestCase):
    def test_video_resume_matches_uninterrupted_without_duplicate_frames(self):
        test_root = Path(__file__).resolve().parents[1] / 'outputs' / 'video_tests'
        test_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=test_root) as directory:
            root = Path(directory)
            source = root / 'source.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10., (160, 120))
            self.assertTrue(writer.isOpened())
            rng = np.random.default_rng(9)
            image = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
            for _ in range(8):
                writer.write(image)
            writer.release()
            options = VideoOptions(segment_frames=3, min_hits=1, keyframe_interval=2)
            detector = FakeDetector()
            # Exercise export without requiring ffmpeg installed on the test host.
            with patch('uav_pipeline.video.io.shutil.which', return_value=None):
                process_video(source, root / 'full', detector, options)
                process_video(source, root / 'resumed', detector, options, max_frames=4)
                process_video(source, root / 'resumed', detector, options, resume=True)
            def rows(folder):
                return [json.loads(s) for s in (root / folder / 'tracks.jsonl').read_text(encoding='utf-8').splitlines()]
            a, b = rows('full'), rows('resumed')
            self.assertEqual([r['frame_index'] for r in b], list(range(8)))
            for left, right in zip(a, b):
                for key in ('frame_index', 'timestamp_seconds', 'detector_ran', 'tracks', 'track_events'):
                    self.assertEqual(left[key], right[key], key)
            # Reproduce the detector storm: one object without local corners.
            # Previously every frame ran six model passes. Now fallback is bounded.
            logs = io.StringIO()
            with patch('uav_pipeline.video.io.shutil.which', return_value=None), \
                 patch.object(CameraMotion, 'update', return_value=movement()), redirect_stdout(logs):
                process_video(source, root / 'weak_flow', detector,
                              VideoOptions(keyframe_interval=6, min_detection_interval=3, min_hits=1))
            telemetry = rows('weak_flow')
            self.assertEqual([r['frame_index'] for r in telemetry if r['detector_ran']], [0, 3, 6])
            self.assertTrue((root / 'weak_flow' / 'latest_preview.jpg').is_file())
            status = json.loads((root / 'weak_flow' / 'status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['frame_index'], 7)
            self.assertEqual(status['model_calls'], 18)  # old schedule: 8 * 6 = 48
            self.assertIn('frame_fps=', logs.getvalue())
            self.assertIn('avg_fps=', logs.getvalue())
            self.assertGreater(status['average_processing_fps'], 0)
            state = json.loads((root / 'resumed' / 'checkpoint.json').read_text(encoding='utf-8'))
            self.assertTrue(state['completed'])
            self.assertEqual(state['counts']['discarded_review_candidates'], state['counts']['detector_frames'])
            with self.assertRaisesRegex(ValueError, 'not empty'):
                process_video(source, root / 'full', detector, options)
            with self.assertRaisesRegex(ValueError, 'changed'):
                process_video(source, root / 'resumed', detector, VideoOptions(min_hits=1), resume=True)
            # Simulate losing a server session: restore durable backup files into
            # a different local path, preserving IDs and source-frame indices.
            stepped = VideoOptions(frame_step=2, segment_frames=2, min_hits=1, keyframe_interval=2)
            with patch('uav_pipeline.video.io.shutil.which', return_value=None):
                process_video(source, root / 'local', detector, stepped, max_frames=2, backup_dir=root / 'backup')
                shutil.copytree(root / 'backup', root / 'restored')
                process_video(source, root / 'restored', detector, stepped, resume=True)
                # A completed resume regenerates exports without duplicate frames.
                process_video(source, root / 'restored', detector, stepped, resume=True)
            restored = rows('restored')
            self.assertEqual([r['frame_index'] for r in restored], [0, 2, 4, 6])
            self.assertEqual({t['track_id'] for r in restored for t in r['tracks']}, {1})
            for part in (root / 'restored' / 'segments').glob('part_*.mp4'):
                capture = cv2.VideoCapture(str(part))
                try:
                    self.assertTrue(capture.isOpened())
                    self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
                    self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 5., places=2)
                finally:
                    capture.release()
            # Fail halfway through an uncommitted segment; only committed frames
            # survive, then replay from the last checkpoint without duplicates.
            calls = [0]
            def fail(image, **kwargs):
                calls[0] += 1
                if calls[0] == 4:
                    raise RuntimeError('simulated interruption')
                return FakeDetector().predict(image, **kwargs)
            crash_options = VideoOptions(segment_frames=2, min_hits=1, keyframe_interval=1)
            with patch('uav_pipeline.video.io.shutil.which', return_value=None):
                with patch.object(detector, 'predict', side_effect=fail):
                    with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                        process_video(source, root / 'crash', detector, crash_options)
                process_video(source, root / 'crash', detector, crash_options, resume=True)
            self.assertEqual([r['frame_index'] for r in rows('crash')], list(range(8)))
            # Corruption cannot silently resume a different trajectory.
            state = json.loads((root / 'crash' / 'checkpoint.json').read_text(encoding='utf-8'))
            (root / 'crash' / state['segments'][0]['records']).write_text('corrupt', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                process_video(source, root / 'crash', detector, crash_options, resume=True)

    def test_invalid_options(self):
        for options in (VideoOptions(frame_step=0), VideoOptions(precision='bad'),
                        VideoOptions(max_prediction_seconds=2.), VideoOptions(log_interval=float('nan')),
                        VideoOptions(min_detection_interval=0), VideoOptions(weak_track_fraction=0.)):
            with self.assertRaises(ValueError):
                options.validate()
