"""Check the pure detection YOLOE detector without downloading model weights."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

from uav_pipeline.detection.detector import YOLOEDetector


def test_yoloe_independence_from_grounding_dino():
    """Verify YOLOE detector initializes without importing transformers."""
    import subprocess
    cmd = [
        sys.executable,
        "-c",
        "import sys; from uav_pipeline.detection.detector import YOLOEDetector; d = YOLOEDetector(device='cpu'); assert 'transformers' not in sys.modules, 'transformers must not be loaded!'"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr or res.stdout


def test_yoloe_prompt_setup_and_image_output():
    detector = YOLOEDetector(device='cpu')
    engine = Mock()
    engine.ckpt_path = 'missing-test-weights.pt'
    engine.predict.return_value = [SimpleNamespace(boxes=SimpleNamespace(
        data=torch.tensor([[1., 2., 20., 30., .95, 0.]])))]
    with patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLOE=Mock(return_value=engine))}):
        result = detector.predict(Image.new('RGB', (64, 64)))
        detector.load_model()
    names = [phrase for phrase, _, _ in detector.vocabulary]
    engine.get_text_pe.assert_called_once_with(names)
    engine.set_classes.assert_called_once_with(names, engine.get_text_pe.return_value)
    assert result['backend'] == 'yoloe'
    assert result['object_detections'][0]['class_id'] == 1
    assert result['summary']['total_predicted_detections'] == 1
    assert result['_image_pil'].size == (64, 64)
    assert result['quality_verified'] is False


def test_video_frame_skips_image_hashing():
    detector = YOLOEDetector(device='cpu')
    engine = Mock()
    engine.ckpt_path = 'missing-test-weights.pt'
    engine.predict.return_value = [SimpleNamespace(boxes=SimpleNamespace(data=torch.empty((0, 6))))]
    with patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLOE=Mock(return_value=engine))}), \
         patch.object(detector, 'preprocess_image') as preprocess:
        result = detector.predict_video_frame(Image.new('RGB', (64, 64)))
    preprocess.assert_not_called()
    assert result['image_sha256'] is None
    assert result['model_calls'] == 1
    assert result['summary']['total_predicted_detections'] == 0


def test_target_classes_filtering():
    detector = YOLOEDetector(device='cpu')
    engine = Mock()
    engine.ckpt_path = 'missing-test-weights.pt'
    moto_idx = float(next(i for i, (_, cid, _) in enumerate(detector.vocabulary) if cid == 2))
    engine.predict.return_value = [SimpleNamespace(boxes=SimpleNamespace(
        data=torch.tensor([
            [1., 2., 20., 30., .95, 0.],
            [40., 50., 80., 90., .90, moto_idx],
        ])
    ))]
    with patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLOE=Mock(return_value=engine))}):
        result = detector.predict(Image.new('RGB', (100, 100)), target_classes=[2])
    assert len(result['object_detections']) == 1
    assert result['object_detections'][0]['class_id'] == 2


def test_extract_roi_clusters():
    """Verify small object seed grouping creates valid bounding boxes within image limits."""
    import numpy as np
    seeds = [
        np.array([100., 150., 140., 190., 0.6, 4]),  # table
        np.array([120., 160., 150., 200., 0.5, 5]),  # stool near table
        np.array([900., 800., 960., 860., 0.7, 6]),  # far cart
    ]
    boxes = YOLOEDetector.extract_roi_clusters(seeds, img_w=1920, img_h=1080, patch_size=640, max_patches=2)
    assert len(boxes) == 2
    for x1, y1, x2, y2 in boxes:
        assert 0 <= x1 < x2 <= 1920
        assert 0 <= y1 < y2 <= 1080
        assert (x2 - x1) >= int(640 * 0.8)
        assert (y2 - y1) >= int(640 * 0.8)


def test_static_anchor_retention():
    """Verify that confirmed static tracks (table/chair) have longer lifetime than dynamic tracks."""
    import numpy as np
    from uav_pipeline.tracking.objects import ObjectTracker
    tracker = ObjectTracker(min_hits=1, max_lost_seconds=0.5, max_static_lost_seconds=2.0)
    # Add a person (dynamic=1) and a chair (static=5)
    dets = [
        {'class_id': 1, 'bbox_xyxy': [10., 10., 30., 50.], 'score': 0.8, 'class_name_vi': 'Người'},
        {'class_id': 5, 'bbox_xyxy': [100., 100., 130., 130.], 'score': 0.8, 'class_name_vi': 'Ghế nhựa'},
    ]
    motion = {'reliable': True, 'matrix': np.eye(2, 3, dtype=np.float32), 'scene_cut': False, 'local_boxes': {}}
    tracker.update(timestamp=0.0, detections=dets, motion=motion)
    assert len(tracker.tracks) == 2

    # Step forward 1.0 second without detections (person exceeds 0.5s timeout, chair does not)
    tracker.update(timestamp=1.0, detections=None, motion=motion)
    track_cids = [t['class_id'] for t in tracker.tracks]
    assert 1 not in track_cids, "Dynamic track (person) should timeout after 0.5s"
    assert 5 in track_cids, "Static track (chair) should be retained under max_static_lost_seconds=2.0s"


def test_overlap_and_pixel_position_do_not_delete_or_relabel_objects():
    import numpy as np
    detector = YOLOEDetector(device='cpu')
    def index(cid):
        return next(i for i, (_, c, _) in enumerate(detector.vocabulary) if c == cid)
    rows = np.array([[350, 0, 400, 140, .9, index(8)],
                     [100, 100, 160, 220, .9, index(1)],
                     [95, 95, 170, 230, .8, index(2)],
                     [0, 0, 50, 50, .9, index(6)]])
    predicted, review, _ = detector.decode(rows, 1920, 1080)
    assert {d.class_id for d in predicted + review} == {1, 2, 6, 8}


def test_empty_roi_still_counts_model_call_and_restores_offsets():
    detector = YOLOEDetector(device='cpu')
    detector.config['model']['roi_strategy'] = 'seeded'
    chair = next(i for i, (_, cid, _) in enumerate(detector.vocabulary) if cid == 5)
    engine = Mock()
    engine.ckpt_path = 'missing-test-weights.pt'
    def output(data):
        return [SimpleNamespace(boxes=SimpleNamespace(data=torch.tensor(data).reshape(-1, 6)))]
    engine.predict.side_effect = [output([[200., 200., 220., 220., .8, chair]]), output([])]
    with patch.dict('sys.modules', {'ultralytics': SimpleNamespace(YOLOE=Mock(return_value=engine))}):
        result = detector.predict(Image.new('RGB', (1200, 900)))
    assert result['model_calls'] == 2
    assert result['roi_patches_evaluated'] == 1
    assert engine.predict.call_count == 2


def test_tracker_does_not_match_distinct_semantic_classes():
    import numpy as np
    from uav_pipeline.tracking.objects import association_costs, measurement
    track = {'class_id': 4, 'x': np.r_[measurement([10, 10, 40, 40]), np.zeros(4)]}
    detections = [{'class_id': 5, 'bbox_xyxy': [10, 10, 40, 40]}]
    assert association_costs([track], detections)[0, 0] == 1e6


def test_roi_can_be_disabled_per_frame():
    detector = YOLOEDetector(device='cpu')
    detector.engine = Mock()
    chair = next(i for i, (_, cid, _) in enumerate(detector.vocabulary) if cid == 5)
    detector.engine.predict.return_value = [SimpleNamespace(boxes=SimpleNamespace(
        data=torch.tensor([[200., 200., 220., 220., .8, chair]])))]
    result = detector.predict_video_frame(Image.new('RGB', (1200, 900)), roi_focus=False)
    assert result['model_calls'] == 1


def test_threshold_override_and_counter_evidence_are_honored():
    import numpy as np
    detector = YOLOEDetector(device='cpu')
    chair = next(i for i, (_, cid, _) in enumerate(detector.vocabulary) if cid == 5)
    foam = next(i for i, (_, _, group) in enumerate(detector.vocabulary) if group == 'foam_box')
    data = np.array([[10, 10, 40, 50, .3, chair], [10, 10, 40, 50, .8, foam]])
    predicted, review, _ = detector.decode(data, threshold_override=.4)
    assert not predicted and not review
    predicted, review, _ = detector.decode(data)
    assert not predicted and len(review) == 1
    predicted, review, _ = detector.decode(data, enable_counters=False)
    assert len(predicted) == 1 and not review


def test_cli_preserves_review_evidence_and_can_draw_yoloe_reviews():
    import json
    from pathlib import Path
    import tempfile
    from uav_pipeline.cli.detect import parse_args, process_single_image
    detector = YOLOEDetector(device='cpu')
    table = next(i for i, (_, cid, _) in enumerate(detector.vocabulary) if cid == 4)
    chair = next(i for i, (_, cid, _) in enumerate(detector.vocabulary) if cid == 5)
    detector.engine = Mock()
    detector.engine.predict.return_value = [SimpleNamespace(boxes=SimpleNamespace(
        data=torch.tensor([[10., 10., 40., 50., .8, table], [10., 10., 40., 50., .7, chair]])))]
    with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1] / 'outputs') as folder:
        folder = Path(folder)
        image = folder / 'input.png'
        Image.new('RGB', (64, 64)).save(image)
        with patch('sys.argv', ['predict', '--input', str(image), '--output', str(folder)]):
            args = parse_args()
        process_single_image(image, detector, folder, args, None, None, None)
        saved = json.loads((folder / 'input_result.json').read_text(encoding='utf-8'))
        assert len(saved['review_candidates']) == saved['summary']['total_review_candidates'] == 2
        args.keep_review = True
        with patch('uav_pipeline.cli.detect.draw_candidates_overlay', return_value=Image.new('RGB', (64, 64))) as draw:
            process_single_image(image, detector, folder, args, None, None, None)
        assert len(draw.call_args.kwargs['review_candidates']) == 2

