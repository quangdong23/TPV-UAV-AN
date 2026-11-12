"""Production integrity, spatial contact, seating and finite persistence contracts."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from uav_pipeline.config import load_config
from uav_pipeline.detection.postprocess import Detection, analyze_conflicts_and_partition, annotate_encroachment
from uav_pipeline.tracking.motion import warp_label_mask
from uav_pipeline.tracking.objects import ObjectTracker
from uav_pipeline.detection.detector import YOLOEDetector
from tests.test_tracking_trunks import motion, detection
from uav_pipeline.events import ViolationMonitor

ROOT = Path(__file__).resolve().parents[1]


def test_standalone_production_configs_and_exact_vocabulary():
    obj = ROOT/'configs/detection.yaml'
    video = ROOT/'configs/video.yaml'
    for path in (obj, video):
        assert 'extends' not in load_config(path)
    d = YOLOEDetector(device='cpu')
    assert sorted(d.config['classes']) == list(range(1, 11))
    assert len(d.vocabulary) == 63
    assert d.vocabulary_sha256 == '85cec415ca14880e9ec4697192830d40fbd81895fc95349c261e60acef3b7f8d'
    assert d.vocabulary[61][2] == 'nonplastic_chair'
    before_path = ROOT/'reports/refactor_20261104/config_before.json'
    if before_path.exists():
        before = json.loads(before_path.read_text())
        assert d.config['classes'] == {int(k): v for k,v in before['object']['classes'].items()}
        cfg = load_config(video)
        for key, value in before['video'].items():
            assert cfg[key] == value


def test_production_checksum_and_master_are_identical():
    p = ROOT/'models/yoloe-v8s-single.pt'
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest == p.with_suffix('.sha256').read_text().split()[0]
    assert digest == json.loads(p.with_suffix('.json').read_text())['sha256']
    master = ROOT.parent/'yoloe_training_standalone/runs/candidate_balanced.pt'
    if master.is_file():
        assert digest == hashlib.sha256(master.read_bytes()).hexdigest()


def test_changed_canonical_checkpoint_fails_before_ultralytics_load():
    d = YOLOEDetector(device='cpu')
    with patch('uav_pipeline.detection.detector.file_hash', return_value='corrupt'):
        try:
            d.load_model()
        except ValueError as error:
            assert 'checksum' in str(error)
        else:
            raise AssertionError('Modified production checkpoint accepted')


def test_sign_facade_standee_tolerance_unknown_and_offscreen():
    mask = np.zeros((200, 300), np.uint8); mask[150:190] = 2
    sign = {'class_id': 7, 'bbox_xyxy': [50, 110, 80, 150]}
    assert annotate_encroachment(sign, mask)['subtype'] == 'sidewalk_standee'
    assert annotate_encroachment(sign, mask)['is_encroaching']
    assert not annotate_encroachment(sign, np.full_like(mask, 255))['is_encroaching']
    for bottom, expected in ((131, True), (130, False)):
        row = annotate_encroachment(dict(sign, bbox_xyxy=[50, 100, 70, bottom]), mask)
        assert row['is_encroaching'] is expected  # y_bottom-1: 20px inclusive.
    facade = dict(sign, bbox_xyxy=[30, 20, 200, 55])
    assert annotate_encroachment(facade, mask)['subtype'] == 'facade_attached'
    assert not annotate_encroachment(facade, mask)['is_encroaching']
    hanging = dict(sign, raw_label='hanging sign')
    assert annotate_encroachment(hanging, mask)['subtype'] == 'facade_attached'
    assert not annotate_encroachment(dict(sign, bbox_xyxy=[30, 190, 80, 220]), mask)['is_encroaching']
    assert 'is_encroaching' not in sign  # Caller data remains intact.


def test_only_commercial_objects_and_standees_raise_encroachment():
    mask = np.full((100, 100), 2, np.uint8)
    for cid in range(1, 11):
        monitor = ViolationMonitor()
        row = dict(track_id=cid, class_id=cid, bbox_xyxy=[20,20,40,60],
                   position_source='detection', detection_age_seconds=0., track_confirmed=True)
        for index in range(10):
            monitor.update(index, index/30, [row], mask, motion())
        assert bool(monitor.events) == (cid in (4,5,6,7,10))


def test_public_bench_needs_combined_evidence_but_metal_folding_chair_survives():
    def chair(box):
        return Detection('chair', 5, 'chair', 'Ghế nhựa', 'plastic chair', 'yoloe', 'full', .7, box)
    box = [10,10,110,50]
    counter = dict(id='metal', bbox_xyxy=box, score=.9, origin='shared_yoloe_head')
    counters = {'nonplastic_chair': [counter], 'construction_materials': [dict(counter, id='concrete')]}
    pred, review, _ = analyze_conflicts_and_partition([chair(box)], counters)
    assert not pred and review[0].subtype == 'public_bench_candidate'
    # A metal folding chair alone supplies no evidence of a public installation.
    pred, review, _ = analyze_conflicts_and_partition([chair([10,10,40,60])],
        {'nonplastic_chair':[dict(counter, bbox_xyxy=[10,10,40,60])]})
    assert len(pred) == 1 and not review
    assert pred[0].evidence['material_status'] == 'nonplastic_seating_candidate'
    # Grey paint alone does not exclude ordinary seating.
    pred, review, _ = analyze_conflicts_and_partition([chair(box)], {}, surface_image=np.full((100,150,3),128,np.uint8))
    assert pred and not review
    assert pred[0].evidence['chair_surface']['neutral_fraction'] == 1.


def test_poles_and_trees_survive_only_bounded_motion_supported_misses():
    for cid in (8,9):
        tracker = ObjectTracker(min_hits=2, max_static_prediction_seconds=.8, max_static_lost_seconds=1.)
        box = [20,20,30,90]
        for index,score in enumerate((.4,.3,.26)):
            row = dict(detection(box,cid), score=score)
            tracker.update(index*.1, [row], motion())
        assert tracker.tracks[0]['persistent_class']
        for index in (3,4):
            out = tracker.update(index*.1, [], motion())
            assert out[0]['track_id'] == 1 and out[0]['position_source'] == 'camera_motion'
            assert out[0]['detection_age_seconds'] > 0
            assert tracker.tracks[0]['hits'] == 3  # Predictions never confirm detections.
        assert not tracker.update(.5, [], motion())
        assert tracker.update(.6, [detection(box,cid)], motion())[0]['track_id'] == 1
        assert not tracker.update(1.7, None, motion()) and not tracker.tracks


def test_persistence_requires_three_consecutive_observed_keyframes_and_resets_on_cut():
    tracker = ObjectTracker(min_hits=2)
    row = detection([20,20,30,90], 9)
    tracker.update(0, [row], motion())
    tracker.update(.1, [row], motion())
    assert not tracker.tracks[0]['persistent_class']
    tracker.update(.2, [], motion())
    tracker.update(.3, [row], motion())
    assert not tracker.tracks[0]['persistent_class']
    tracker.update(.4, [row], motion())
    tracker.update(.5, [row], motion())
    assert tracker.tracks[0]['persistent_class']
    saved = tracker.state_dict()
    restored = ObjectTracker(); restored.load_state_dict(saved)
    assert restored.tracks[0]['persistent_class']
    assert not restored.update(.6, [], dict(motion(),scene_cut=True))
    assert not restored.tracks


def test_unreliable_motion_cannot_keep_missing_persistent_object_visible():
    tracker = ObjectTracker(min_hits=2)
    row = detection([20,20,30,90], 8)
    for i in range(3): tracker.update(i*.1, [row], motion())
    assert not tracker.update(.3, [], dict(motion(),reliable=False))


def test_affine_and_homography_masks_preserve_labels_and_anchor_alignment():
    mask = np.zeros((80,120),np.uint8); mask[20:50,30:70] = 2; mask[25,35] = 255
    matrix = np.array([[1.,0,7],[0,1.,-3]])
    expected = np.zeros_like(mask); expected[17:47,37:77] = 2; expected[22,42] = 255
    np.testing.assert_array_equal(warp_label_mask(mask,matrix,(120,80)), expected)
    np.testing.assert_array_equal(warp_label_mask(mask,np.vstack((matrix,[0,0,1])),(120,80)), expected)
    assert set(np.unique(warp_label_mask(mask,matrix,(120,80)))) == {0,2,255}
    for bad in (np.zeros((2,3)), np.full((3,3),np.nan)):
        try: warp_label_mask(mask,bad,(120,80))
        except ValueError: pass
        else: raise AssertionError('Invalid mask motion accepted')


def test_sidewalk_camera_shake_uses_anchor_and_refreshes_on_flow_failure():
    from uav_pipeline.segmentation.tracking.motion import VideoMotionPropagator
    rng = np.random.default_rng(123)
    frame = rng.integers(0,256,(200,300,3),dtype=np.uint8)
    mask = np.zeros((200,300),np.uint8); mask[70:150,80:220] = 2
    p = VideoMotionPropagator(keyframe_interval=5,tracking_max_side=300)
    p.update_keyframe(frame,mask)
    for dx,dy in ((2,1),(-1,2),(1,-1)):
        matrix = np.float32([[1,0,dx],[0,1,dy]])
        moved = cv2.warpAffine(frame,matrix,(300,200))
        warped,_ = p.propagate(moved)
        assert warped is not None, p.last_reason
        expected = warp_label_mask(mask,matrix,(300,200))
        union = np.count_nonzero((warped==2)|(expected==2))
        assert np.count_nonzero((warped==2)&(expected==2))/union > .98
    rejected,_ = p.propagate(np.zeros_like(frame))
    assert rejected is None


def test_processor_compatibility_removes_warning_without_changing_inputs():
    from PIL import Image
    from types import SimpleNamespace
    import warnings
    import torch
    from transformers import Mask2FormerImageProcessor
    from uav_pipeline.segmentation.models.backend import load_image_processor
    cfg = dict(do_resize=False, image_processor_type='Mask2FormerImageProcessor',
               _max_size=1333, reduce_labels=False, do_reduce_labels=False)
    with warnings.catch_warnings(record=True):
        old = Mask2FormerImageProcessor.from_dict(cfg)
    with patch.object(Mask2FormerImageProcessor, 'get_image_processor_dict', return_value=(cfg, {})), \
         warnings.catch_warnings(record=True) as seen:
        new = load_image_processor(SimpleNamespace(backend='mask2former',model_id='fixture'), '.', True)
    assert not seen
    image = Image.fromarray(np.random.default_rng(5).integers(0,256,(32,48,3),dtype=np.uint8))
    a,b = old(images=image,return_tensors='pt'), new(images=image,return_tensors='pt')
    assert a.keys() == b.keys()
    for key in a: assert torch.equal(a[key], b[key])


def test_weak_elongated_neutral_seating_goes_to_review_without_changing_chair_threshold():
    image = np.full((100,200,3),128,np.uint8)
    def seating(score,box):
        return Detection('chair',5,'chair','Ghế nhựa','low plastic stool','yoloe','full',score,box)
    pred,review,_ = analyze_conflicts_and_partition([seating(.35,[10,10,150,45])],surface_image=image)
    assert not pred and review[0].subtype == 'public_bench_candidate'
    assert 'scope_uncertain' in review[0].review_reason
    pred,review,_ = analyze_conflicts_and_partition([seating(.8,[10,10,150,45])],surface_image=image)
    assert pred and not review  # Strong positive preserved.
    pred,review,_ = analyze_conflicts_and_partition([seating(.35,[10,10,50,60])],surface_image=image)
    assert pred and not review  # Grey ordinary chair preserved.
