"""Offline integration checks with real video I/O and deterministic model doubles."""
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

from tests.test_video import FakeDetector, detection
from uav_pipeline.video.pipeline import VideoOptions, process_video
from uav_pipeline.segmentation import PipelineConfig
from uav_pipeline.segmentation.media.video import VideoPreviewOptions, preview_video
from uav_pipeline.segmentation.postprocessing.visualization import create_overlay
from uav_pipeline.segmentation.session import SidewalkSession, keep_on_sidewalk


class FakeSidewalk:
    def __init__(self, regions=('left',)):
        self.config = PipelineConfig()
        self._predictor = None
        self.regions, self.calls = regions, []

    def process_frame(self, rgb, **kwargs):
        self.calls.append(kwargs)
        h, w = rgb.shape[:2]
        region = self.regions[min(len(self.calls)-1, len(self.regions)-1)]
        mask = np.zeros((h, w), np.uint8)
        if region == 'left':
            mask[:, :w//2] = 2
        elif region == 'right':
            mask[:, w//2:] = 2
        features = {'type': 'FeatureCollection', 'features': []}
        if region != 'empty':
            x1, x2 = (0, w//2) if region == 'left' else (w//2, w)
            features['features'].append({'type': 'Feature',
                'geometry': {'type': 'Polygon', 'coordinates': [[[x1, 0], [x2, 0], [x2, h], [x1, h], [x1, 0]]]},
                'properties': {'class_id': 2, 'area_pixels': (x2-x1)*h}})
        scores = np.stack([mask != 2, mask == 2, np.zeros_like(mask)]).astype(np.float32)
        return SimpleNamespace(mask=mask, pixel_features=features,
            overlay_rgb=create_overlay(rgb, mask, .45),
            completion=SimpleNamespace(completed_pixels=int(np.count_nonzero(mask == 2))),
            timings_seconds={}, inference=SimpleNamespace(scores=scores, runs=[{'tile_count': 1}]))


def make_video(path, frames=4):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10., (320, 240))
    if not writer.isOpened():
        raise RuntimeError('MJPG writer unavailable')
    image = np.random.default_rng(71).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    for _ in range(frames):
        writer.write(image)
    writer.release()


class CombinedVideoTests(unittest.TestCase):
    def test_corrupt_video_is_rejected_before_model_load_or_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'broken.mp4'
            source.write_bytes(b'incomplete video')
            output = Path(directory)/'output'
            detector = FakeDetector()
            with patch.object(detector,'load_model') as load:
                with self.assertRaisesRegex(ValueError,'Cannot decode input video'):
                    process_video(source,output,detector,VideoOptions())
                load.assert_not_called()
            self.assertFalse(output.exists())

    def test_ground_contact_unknown_holes_edges_and_invalid_boxes(self):
        mask = np.zeros((40, 60), np.uint8)
        mask[20:40, :30] = 2
        mask[30, 10] = 255
        rows = [detection((0, 0, 20, 30)), detection((30, 0, 50, 30)),
                detection((0, 0, 20, 31)), detection((-20, 0, -2, 30)),
                detection((0, 0, 20, 41)), detection((0, 0, float('nan'), 30)),
                detection((20, 20, 10, 30))]
        self.assertEqual([r['bbox_xyxy'] for r in keep_on_sidewalk(rows, mask)], [rows[0]['bbox_xyxy']])
        # Body over the road, contact on sidewalk: bottom-center accepts it.
        self.assertEqual(keep_on_sidewalk([rows[0]], mask, 'overlap', .5), [])
        self.assertEqual([r['bbox_xyxy'] for r in keep_on_sidewalk([rows[0]], mask, 'overlap', .3)], [rows[0]['bbox_xyxy']])
        mask[29, 10] = 0  # A hole must be respected.
        self.assertEqual(keep_on_sidewalk([rows[0]], mask), [])

    def test_filter_before_tracking_source_coordinates_and_empty_mask(self):
        class Detector(FakeDetector):
            def __init__(self):
                self.config = {'model': {'model_id': 'fake'}, 'inference': {}}
                self.calls = 0

            def predict(self, image, **kwargs):
                self.calls += 1
                return {'object_detections': [detection((5, 5, 15, 20)), detection((45, 5, 55, 20), 2)],
                        'review_candidates': [detection((45, 5, 55, 20), status='review')],
                        'conflict_reports': [{'outside_box': [45, 5, 55, 20]}], 'model_calls': 1}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.avi'
            make_video(source)
            pipeline = FakeSidewalk(('left', 'left', 'empty', 'right'))
            session = SidewalkSession(pipeline, VideoPreviewOptions(keyframe_interval=1, work_max_side=160), sample_step=1)
            detector = Detector()
            options = VideoOptions(work_max_side=80, min_hits=1, keyframe_interval=1, min_detection_interval=1, segment_frames=2)
            with redirect_stdout(io.StringIO()), patch('uav_pipeline.video.io.shutil.which', return_value=None):
                process_video(source, root/'out', detector, options, sidewalk=session)
            records = [json.loads(line) for line in (root/'out/tracks.jsonl').read_text().splitlines()]
            self.assertEqual([r['frame_index'] for r in records], list(range(4)))
            self.assertEqual(detector.calls, 4)  # Empty masks still allow finding signs/trees.
            self.assertEqual([t['class_id'] for t in records[0]['tracks']], [1])
            self.assertEqual(records[0]['tracks'][0]['bbox_xyxy'], [20., 20., 60., 80.])
            self.assertEqual(records[0]['detector_evidence']['object_detections'][0]['bbox_xyxy'], [20., 20., 60., 80.])
            self.assertEqual(records[0]['detector_evidence']['review_candidates'], [])
            self.assertNotIn('conflict_reports', records[0]['detector_evidence'])
            self.assertEqual(records[2]['tracks'], [])
            self.assertEqual(records[2]['detector_evidence']['object_detections'], [])
            self.assertTrue(records[2]['detector_ran'])
            self.assertEqual([t['class_id'] for t in records[3]['tracks']], [2])
            features = records[0]['sidewalk']['features']['features']
            self.assertEqual(features[0]['geometry']['coordinates'][0][2], [160., 240.])
            self.assertEqual(features[0]['properties']['area_pixels'], 160*240)
            mask = cv2.imread(str(root/'out'/records[0]['sidewalk']['mask_path']), cv2.IMREAD_UNCHANGED)
            self.assertEqual(mask.shape, (240, 320))
            self.assertEqual(set(np.unique(mask)), {0, 2})
            self.assertTrue((root/'out/images/00000000.jpg').is_file())
            summary = json.loads((root/'out/summary.json').read_text())
            self.assertTrue(summary['all_source_frames_processed'])
            self.assertEqual(summary['counts']['outside_sidewalk_detections'], 5)

    def test_signs_and_trunks_bypass_empty_unknown_and_outside_masks(self):
        rows = [detection((2, 2, 12, 18), cid) for cid in range(1, 11)]
        for mode in ('bottom-center', 'overlap'):
            for label in (0, 255):
                with self.subTest(mode=mode, label=label):
                    kept = keep_on_sidewalk(rows, np.full((40, 60), label, np.uint8), mode)
                    self.assertEqual([r['class_id'] for r in kept], [7, 8, 9])
            mask = np.zeros((40, 60), np.uint8)
            mask[:, 30:] = 2
            self.assertEqual([r['class_id'] for r in keep_on_sidewalk(rows, mask, mode)], [7, 8, 9])
        self.assertEqual(keep_on_sidewalk([detection((0, 0, float('nan'), 20), 9)], mask), [])

    def test_empty_video_mask_still_detects_tracks_and_exports_signs_and_trunks(self):
        class Detector(FakeDetector):
            def predict(self, image, **kwargs):
                return {'object_detections': [detection((cid*25, 30, cid*25+15, 70), cid)
                                               for cid in range(1, 11)],
                        'review_candidates': [detection((10, 10, 20, 20), 7, status='review'),
                                              detection((30, 10, 40, 20), 1, status='review')],
                        'model_calls': 1}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.avi'
            make_video(source, 2)
            session = SidewalkSession(FakeSidewalk(('empty',)), VideoPreviewOptions(keyframe_interval=1), sample_step=0)
            options = VideoOptions(min_hits=1, keyframe_interval=20, min_detection_interval=20)
            with redirect_stdout(io.StringIO()), patch('uav_pipeline.video.io.shutil.which', return_value=None):
                process_video(source, root/'out', Detector(), options, sidewalk=session)
            records = [json.loads(line) for line in (root/'out/tracks.jsonl').read_text().splitlines()]
            for record in records:
                self.assertEqual([r['class_id'] for r in record['tracks']], [7, 8, 9])
            self.assertTrue(records[0]['detector_ran'])
            self.assertFalse(records[1]['detector_ran'])
            evidence = records[0]['detector_evidence']
            self.assertEqual([r['class_id'] for r in evidence['object_detections']], [7, 8, 9])
            self.assertEqual([r['class_id'] for r in evidence['review_candidates']], [7])
            self.assertEqual(records[0]['sidewalk']['sidewalk_exempt_class_ids'], [7, 8, 9])
            summary = json.loads((root/'out/summary.json').read_text())
            self.assertTrue(summary['outside_sidewalk_objects_exported'])
            self.assertEqual(summary['sidewalk_filter']['sidewalk_exempt_class_ids'], [7, 8, 9])

    def test_rule_alerts_integrate_with_real_video_and_compact_outputs(self):
        from uav_pipeline.events import ViolationMonitor
        class ObjectDetector(FakeDetector):
            def predict(self,image,**kwargs):
                result=super().predict(image,**kwargs)
                for row in result['object_detections']:row['class_id']=4
                return result
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.avi'
            make_video(source,24)
            session=SidewalkSession(FakeSidewalk(),VideoPreviewOptions(keyframe_interval=1),sample_step=0,
                                    crisp_overlay=True)
            options=VideoOptions(min_hits=1,keyframe_interval=1,min_detection_interval=1)
            monitor=ViolationMonitor()
            with redirect_stdout(io.StringIO()):
                process_video(source,root/'out',ObjectDetector(),options,sidewalk=session,
                              output_mode='compact',events=monitor)
            report=json.loads((root/'out/events.json').read_text())
            self.assertEqual(report['alert_count'],1)
            event=report['events'][0]
            self.assertEqual((event['first_frame'],event['confirmation_frame']),(0,9))
            self.assertTrue(event['clip']['truncated_start'])
            self.assertTrue(event['clip']['truncated_end'])
            self.assertTrue((root/'out'/event['clip']['path']).is_file())
            records=[json.loads(line) for line in (root/'out/tracks.jsonl').read_text().splitlines()]
            self.assertEqual(records[9]['rule_alerts'][0]['event_id'],event['event_id'])
            self.assertEqual(set(p.name for p in (root/'out').iterdir()),
                             {'annotated.mp4','tracks.jsonl','summary.json','events.json','event_clips'})

    def test_repair_group_exports_source_coordinates_and_evidence_clip(self):
        from uav_pipeline.events import ViolationMonitor

        class PairDetector(FakeDetector):
            def predict(self,image,**kwargs):
                result = super().predict(image,**kwargs)
                result['object_detections'] = [detection((10,20,20,40),1),
                                               detection((23,25,43,40),2)]
                return result

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.avi'
            make_video(source,26)
            session = SidewalkSession(FakeSidewalk(),VideoPreviewOptions(keyframe_interval=1),
                                      sample_step=0,crisp_overlay=True)
            options = VideoOptions(work_max_side=160,min_hits=1,keyframe_interval=1,
                                   min_detection_interval=1,fast_render=True)
            monitor = ViolationMonitor()
            with redirect_stdout(io.StringIO()):
                process_video(source,root/'out',PairDetector(),options,sidewalk=session,
                              output_mode='compact',events=monitor)
            report = json.loads((root/'out/events.json').read_text())
            self.assertEqual(report['counts_by_kind']['motorcycle_repair_suspected'],1)
            event = report['events'][0]
            self.assertGreaterEqual(event['confirmation_timestamp_seconds']-event['first_timestamp_seconds'],2)
            self.assertTrue((root/'out'/event['clip']['path']).is_file())
            records = [json.loads(line) for line in (root/'out/tracks.jsonl').read_text().splitlines()]
            confirmed = [r for r in records if r['activity_regions']]
            self.assertTrue(confirmed)
            region = confirmed[0]['activity_regions'][0]
            self.assertEqual(region['kind'],'motorcycle_repair_suspected')
            self.assertEqual(region['label'],'Nghi sửa xe')
            np.testing.assert_allclose(region['bbox_xyxy'],[20,40,86,80],atol=1)

    def test_fast_render_uses_registry_colors_for_each_class(self):
        from uav_pipeline.video.pipeline import render
        from uav_pipeline.visualization.objects import get_class_color
        colors=[]
        for cid in range(1,11):
            row=dict(track_id=cid,class_id=cid,bbox_xyxy=[40,60,100,110])
            result=render(np.zeros((140,160,3),np.uint8),[row],0,True,fast=True)
            color=tuple(result[80,40].tolist())
            self.assertEqual(color,get_class_color(cid)[::-1])
            self.assertNotEqual(color,(0,0,255))
            colors.append(color)
        self.assertEqual(len(set(colors)),10)

    def test_objects_leave_sidewalk_between_detector_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.avi'
            make_video(source, 2)
            pipeline = FakeSidewalk(('left', 'right'))
            session = SidewalkSession(pipeline, VideoPreviewOptions(keyframe_interval=1), sample_step=0)
            options = VideoOptions(min_hits=1, keyframe_interval=20, min_detection_interval=20)
            with redirect_stdout(io.StringIO()), patch('uav_pipeline.video.io.shutil.which', return_value=None):
                process_video(source, root/'out', FakeDetector(), options, sidewalk=session)
            records = [json.loads(line) for line in (root/'out/tracks.jsonl').read_text().splitlines()]
            self.assertEqual(len(records[0]['tracks']), 1)
            self.assertFalse(records[1]['detector_ran'])
            self.assertEqual(records[1]['tracks'], [])
            self.assertEqual(records[1]['track_events'], [])

    def test_sidewalk_ai_warp_and_masks_match_standalone_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'source.avi'
            make_video(source, 8)
            options = VideoPreviewOptions(keyframe_interval=3, work_max_side=160, headless=True,
                                          save_video=True, mask_sample_step=1)
            standalone = FakeSidewalk()
            with redirect_stdout(io.StringIO()):
                preview_video(source, standalone, root/'baseline', options)
            combined = FakeSidewalk()
            session = SidewalkSession(combined, options, sample_step=0)
            cap = cv2.VideoCapture(str(source))
            modes = []
            try:
                for index in range(8):
                    success, frame = cap.read()
                    self.assertTrue(success)
                    result = session.process(frame, (120, 160, 3))
                    expected = cv2.imread(str(root/f'baseline/source/mask_samples/{index:08d}.png'), cv2.IMREAD_UNCHANGED)
                    np.testing.assert_array_equal(result['mask'], expected)
                    modes.append(result['metadata']['ai_ran'])
            finally:
                cap.release()
            self.assertIn(False, modes)
            self.assertEqual(len(combined.calls), len(standalone.calls))
            for a, b in zip(combined.calls, standalone.calls):
                self.assertEqual(a.keys(), b.keys())
                for key in a:
                    if isinstance(a[key], np.ndarray):
                        np.testing.assert_array_equal(a[key], b[key])
                    else:
                        self.assertEqual(a[key], b[key])

    def test_rejects_resume_instead_of_silently_losing_sidewalk_state(self):
        session = SidewalkSession(FakeSidewalk(), VideoPreviewOptions())
        with self.assertRaisesRegex(ValueError, 'does not yet support resume'):
            process_video('unused', 'unused', FakeDetector(), resume=True, sidewalk=session)
