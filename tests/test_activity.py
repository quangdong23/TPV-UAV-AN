"""Kiểm tra lọc nhiễu và phân loại hành động người–xe."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from tests import test_events
from uav_pipeline.events import EventOptions, ViolationMonitor


class ActivityTests(unittest.TestCase):
    def simulate(self, label='washing_vehicle', confidence=.9, moving=False, source='detection', car=False):
        helper = test_events.MotorcycleRepairTests()
        helper.setUp()
        classifier = Mock()
        classifier.predict.return_value = dict(label=label, confidence=confidence, margin=.5,
                                               scores={label: confidence}, crop_bbox_xyxy=[10,10,100,100])
        monitor = ViolationMonitor(EventOptions(repair_seconds=.2), activity_classifier=classifier)
        alerts = []
        for i in range(40):
            rows = helper.pair(shift=i*20 if moving else 0, source=source)
            if car:
                rows[1]['class_id'] = 3
            alerts.extend(helper.step(monitor, i, rows=rows, frame=np.zeros((240,320,3), np.uint8)))
        return monitor, classifier, [e for e in alerts if e['kind'].startswith('vehicle_')]

    def test_washing_and_repair_start_at_fifth_frame(self):
        for label, kind in [('washing_vehicle', 'vehicle_washing_suspected'),
                            ('repairing_vehicle', 'vehicle_repair_suspected')]:
            monitor, classifier, alerts = self.simulate(label)
            self.assertEqual(len(alerts), 1)
            self.assertEqual(alerts[0]['kind'], kind)
            self.assertEqual(alerts[0]['evidence']['stationary_frames'], 5)
            self.assertEqual(alerts[0]['confirmation_frame'], 4)
            self.assertTrue(monitor.repair_regions)

    def test_negative_low_confidence_motion_and_flow_do_not_alert(self):
        for kwargs in [dict(label='other_activity'), dict(label='street_vending'),
                       dict(confidence=.6), dict(moving=True), dict(source='prediction')]:
            self.assertEqual(self.simulate(**kwargs)[2], [])

    def test_car_supported(self):
        self.assertEqual(len(self.simulate(car=True)[2]), 1)

    def test_real_checkpoint(self):
        from uav_pipeline.detection.activity import ActivityClassifier
        from uav_pipeline.paths import MODELS_DIR
        classifier = ActivityClassifier(MODELS_DIR/'mobilenetv3_activity.pt')
        self.assertIsNone(classifier.predict(np.zeros((100,100,3), np.uint8), [0,0,4,4]))
        result = classifier.predict(np.zeros((100,100,3), np.uint8), [0,0,100,100])
        self.assertAlmostEqual(sum(result['scores'].values()), 1., places=5)

    def test_export_counts_new_activity_kind(self):
        import tempfile
        from pathlib import Path
        monitor, _, _ = self.simulate()
        with tempfile.TemporaryDirectory() as directory:
            summary = monitor.export(Path(directory), 10, 40, video_available=False)
            self.assertEqual(summary['counts_by_kind']['vehicle_washing_suspected'], 1)

    def test_checkpoint_input_size_is_used_after_letterbox(self):
        import torch
        from uav_pipeline.detection.activity import ActivityClassifier
        from uav_pipeline.paths import MODELS_DIR
        checkpoint = torch.load(MODELS_DIR/'mobilenetv3_activity.pt', weights_only=True, map_location='cpu')
        checkpoint['config'] = {'preprocessing': {'size': 160}}
        with patch('uav_pipeline.detection.activity.torch.load', return_value=checkpoint):
            classifier = ActivityClassifier('khong-can-file.pt')
        shapes = []
        hook = classifier.model.register_forward_pre_hook(lambda model, args: shapes.append(tuple(args[0].shape)))
        classifier.predict(np.zeros((100,160,3), np.uint8), [0,0,160,100])
        hook.remove()
        self.assertEqual(shapes, [(1,3,160,160)])

    def test_five_measured_frames_do_not_wait_for_track_confirmation(self):
        helper = test_events.MotorcycleRepairTests()
        helper.setUp()
        classifier = Mock()
        classifier.predict.return_value = dict(label='repairing_vehicle', confidence=.9, margin=.8,
                                               scores={'repairing_vehicle': .9}, crop_bbox_xyxy=[12,14,103,86])
        for fps in (10,30,60):
            monitor = ViolationMonitor(activity_classifier=classifier)
            classifier.reset_mock()
            for i in range(5):
                rows = helper.pair(source='detection' if i == 0 else 'optical_flow')
                for row in rows:
                    row['track_confirmed'] = False
                alerts = helper.step(monitor, i, fps=fps, rows=rows,
                                     frame=np.zeros((240,320,3), np.uint8))
                self.assertEqual(classifier.predict.call_count, int(i == 4))
                self.assertEqual(len(alerts), int(i == 4))
                self.assertEqual(monitor.repair_regions[0]['stationary_frames'], i+1)
            self.assertEqual(alerts[0]['independent_detector_observations'], 1)

    def test_crop_is_visible_before_classification_and_resets_when_pair_leaves(self):
        helper = test_events.MotorcycleRepairTests()
        helper.setUp()
        classifier = Mock()
        monitor = ViolationMonitor(activity_classifier=classifier)
        for i in range(4):
            helper.step(monitor, i, frame=np.zeros((240,320,3), np.uint8))
        region = monitor.repair_regions[0]
        self.assertFalse(region['ready'])
        self.assertEqual(region['bbox_xyxy'], [12,14,103,86])
        rendered = monitor.render(np.zeros((240,320,3), np.uint8), helper.pair())
        np.testing.assert_array_equal(rendered[86,70], [0,200,255])
        helper.step(monitor,4,rows=[])
        self.assertEqual(monitor.repair_regions, [])
        for i in range(5,9):
            helper.step(monitor,i,frame=np.zeros((240,320,3), np.uint8))
        classifier.predict.assert_not_called()

    def test_negative_label_remains_visible_without_action_alert(self):
        monitor, _, alerts = self.simulate(label='other_activity')
        self.assertEqual(alerts, [])
        region = monitor.repair_regions[0]
        self.assertIn('Hoạt động khác', region['label'])
        self.assertEqual(region['kind'], 'activity_candidate')

    def test_invalid_frame_count_is_rejected(self):
        for value in (0,-1,1.5):
            with self.assertRaises(ValueError):
                EventOptions(activity_frames=value).validate()

    def test_action_change_emits_each_kind_once_per_pair(self):
        helper = test_events.MotorcycleRepairTests()
        helper.setUp()
        classifier = Mock()
        classifier.predict.side_effect = [
            dict(label=label, confidence=.9, margin=.8, scores={label:.9}, crop_bbox_xyxy=[12,14,103,86])
            for label in ('washing_vehicle','repairing_vehicle','washing_vehicle')]
        monitor = ViolationMonitor(activity_classifier=classifier)
        alerts = []
        for i in range(15):
            alerts.extend(helper.step(monitor,i,frame=np.zeros((240,320,3),np.uint8)))
        self.assertEqual([e['kind'] for e in alerts if e['kind'].startswith('vehicle_')],
                         ['vehicle_washing_suspected','vehicle_repair_suspected'])
