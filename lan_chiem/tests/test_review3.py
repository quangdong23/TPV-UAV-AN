"""Regression cases from real sample review and sparse spatial coverage."""
import tempfile
from pathlib import Path
import numpy as np
from lan_chiem.configuration import load_config
from lan_chiem.roi import grid_regions, select_regions
from lan_chiem.postprocess import Detection, analyze_conflicts_and_partition
from lan_chiem.video_tracking import ObjectTracker


def test_configuration_inheritance_and_cycle():
    with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1] / 'outputs') as tmp:
        p = Path(tmp)
        (p/'base.yaml').write_text('model: {imgsz: 960, precision: fp16}\n', encoding='utf-8')
        (p/'child.yaml').write_text('extends: base.yaml\nmodel: {imgsz: 1280}\n', encoding='utf-8')
        assert load_config(p/'child.yaml')['model'] == {'imgsz': 1280, 'precision': 'fp16'}
        (p/'base.yaml').write_text('extends: child.yaml\n', encoding='utf-8')
        try:
            load_config(p/'child.yaml')
        except ValueError:
            pass
        else:
            raise AssertionError('Inheritance cycle accepted')


def test_grid_covers_unseeded_objects_and_rotates():
    regions = grid_regions(1920, 1040, 640)
    coverage = np.zeros((1040, 1920), bool)
    for x1, y1, x2, y2 in regions:
        coverage[y1:y2, x1:x2] = True
    assert coverage.all()
    chosen = [r for offset in range(0, len(regions), 2) for r in select_regions(regions, 2, offset)]
    assert set(chosen) == set(regions)
    assert len(select_regions(regions, 100)) == len(regions)


def test_weaker_shared_head_alternative_retains_audit_evidence():
    def cart():
        return Detection('cart', 6, 'cart', 'cart', 'street food cart', 'cart', 'full', .82, [0,0,100,100])
    negative = {'id': 'gate', 'score': .51, 'bbox_xyxy': [0,0,100,100], 'origin': 'shared_yoloe_head', 'source': 'full'}
    pred, review, reports = analyze_conflicts_and_partition([cart()], {'barrier_gate': [negative]}, {'same_head_counter_policy': 'stronger'})
    assert len(pred) == 1 and not review and reports[0]['requires_review'] is False
    assert pred[0].status == 'predicted'
    negative['score'] = .9
    assert analyze_conflicts_and_partition([cart()], {'barrier_gate': [negative]}, {'same_head_counter_policy': 'stronger'})[1]
    negative.update(score=.51, origin='same_caption')
    assert analyze_conflicts_and_partition([cart()], {'barrier_gate': [negative]}, {'same_head_counter_policy': 'stronger'})[1]


def test_roi_only_track_needs_two_observations_not_two_global_passes():
    tracker = ObjectTracker(min_hits=2, max_static_lost_seconds=2.5)
    motion = {'scene_cut': False, 'reliable': True, 'matrix': np.array([[1.,0,0],[0,1.,0]]), 'local_boxes': {}}
    d = {'class_id': 5, 'bbox_xyxy': [20,20,50,50], 'score': .8, 'source': 'roi_0_0'}
    assert not tracker.update(0, [d], motion, [[0,0,100,100]])
    for timestamp in (.2,.4,.6):
        assert not tracker.update(timestamp, [], motion, [[200,200,300,300]])
        assert tracker.tracks[0]['hits'] == 1
    assert tracker.update(.8, [d], motion, [[0,0,100,100]])[0]['track_id'] == 1
    assert not tracker.update(1., [], motion, [[0,0,100,100]])
    tracker.update(4., [], motion, [])
    assert not tracker.tracks


def test_feature_cache_tracks_known_camera_motion_and_resumes_exactly():
    import cv2
    from lan_chiem.video_motion import CameraMotion
    rng = np.random.default_rng(42)
    original = rng.integers(0,256,(240,320,3),dtype=np.uint8)
    motion = CameraMotion(320,6)
    restored = None
    for index in range(15):
        frame = cv2.warpAffine(original,np.array([[1.,0,index],[0,1.,0]]),(320,240))
        result = motion.update(frame,{1:[80+index-1,70,140+index-1,150]})
        if index:
            assert result['reliable']
            assert np.allclose(result['matrix'],[[1,0,1],[0,1,0]],atol=.1)
        if restored is not None:
            resumed = restored.update(frame,{1:[80+index-1,70,140+index-1,150]})
            assert np.array_equal(resumed['matrix'],result['matrix'])
            assert resumed['local_boxes'].keys() == result['local_boxes'].keys()
            for key in result['local_boxes']:
                assert np.array_equal(resumed['local_boxes'][key],result['local_boxes'][key])
        if index == 7:
            restored = CameraMotion(320,6)
            restored.previous = motion.previous.copy()
            restored.load_state_dict(motion.state_dict())


def test_all_ten_classes_and_unique_phrases_remain_required():
    from lan_chiem.yoloe_detector import YOLOEDetector, TARGET_PHRASES, COUNTER_PHRASES
    original = YOLOEDetector.build_vocabulary(TARGET_PHRASES,COUNTER_PHRASES)
    assert {row[1] for row in original if row[1] is not None} == set(range(1,11))
    for targets in ({1:['person']}, {**TARGET_PHRASES, 10:['person']}):
        try:
            YOLOEDetector.build_vocabulary(targets, COUNTER_PHRASES)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing class or duplicate phrase accepted')


def test_fused_head_does_not_leave_inference_parameters_for_backend_setup():
    import torch
    from unittest.mock import Mock, patch
    from types import SimpleNamespace
    from lan_chiem.yoloe_detector import YOLOEDetector
    class Head(torch.nn.Module):
        is_fused = False
        def fuse(self, pe):
            with torch.inference_mode():
                self.register_parameter('projection', torch.nn.Parameter(torch.ones(2), requires_grad=False))
                self.register_buffer('constant', torch.ones(2))
            self.is_fused = True
    head = Head()
    model = torch.nn.Module()
    model.model = torch.nn.ModuleList([head])
    engine = SimpleNamespace(model=model, to=Mock(), get_text_pe=Mock(return_value=torch.zeros(1,57,512)),
                             set_classes=Mock(), task='detect', ckpt_path=None)
    detector = YOLOEDetector(weights='fake-test.pt',device='cpu')
    detector.config['model']['reparameterize'] = True
    with patch.dict('sys.modules', {'ultralytics':SimpleNamespace(YOLOE=Mock(return_value=engine))}):
        detector.load_model()
    assert head.is_fused
    assert not torch.is_inference(head.projection) and not torch.is_inference(head.constant)
