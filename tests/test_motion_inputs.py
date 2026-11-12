"""Regressions for video color, ROI batches, zoom and tentative identities."""
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from uav_pipeline.detection.detector import YOLOEDetector, resolve_yoloe_weights
from uav_pipeline.tracking.motion import local_box_from_flow, warp_box
from uav_pipeline.tracking.objects import ObjectTracker
from tests.test_video import movement, detection


def result(rows):
    return SimpleNamespace(boxes=SimpleNamespace(data=torch.tensor(rows, dtype=torch.float32).reshape(-1, 6)))


def test_video_rgb_and_bgr_arrays_have_same_engine_input():
    detector = YOLOEDetector(device='cpu', roi_focus=False)
    detector.engine = Mock()
    detector.engine.predict.return_value = [result([])]
    rgb = np.full((32, 48, 3), [220, 11, 3], dtype=np.uint8)
    detector.predict_video_frame(rgb)
    first = detector.engine.predict.call_args.kwargs['source'].copy()
    bgr = rgb[..., ::-1].copy()
    detector.predict_video_frame(bgr, color_order='BGR')
    assert detector.engine.predict.call_args.kwargs['source'] is bgr
    np.testing.assert_array_equal(first, bgr)


def test_roi_batch_keeps_offsets_counters_and_counts_for_two_images():
    detector = YOLOEDetector(device='cpu')
    detector.config['model']['roi_strategy'] = 'seeded'
    detector.roi_max_patches = 2
    chair = next(i for i, (_, cid, _) in enumerate(detector.vocabulary) if cid == 5)
    foam = next(i for i, (_, _, group) in enumerate(detector.vocabulary) if group == 'foam_box')
    detector.engine = Mock()
    detector.engine.predict.side_effect = [
        [result([])],
        [result([[10, 20, 40, 50, .8, chair], [10, 20, 40, 50, .9, foam]]),
         result([[10, 20, 40, 50, .8, chair]])],
    ]
    with patch.object(detector, 'extract_roi_clusters', return_value=[(100, 100, 612, 612), (700, 100, 1212, 612)]):
        output = detector.predict_video_frame(Image.new('RGB', (1800, 1000)))
    assert detector.engine.predict.call_count == output['model_calls'] == 2
    assert output['images_evaluated'] == 3 and output['roi_patches_evaluated'] == 2
    assert output['object_detections'][0]['bbox_xyxy'] == [710, 120, 740, 150]
    assert output['object_detections'][0]['source'] == 'roi_700_100'
    assert output['review_candidates'][0]['bbox_xyxy'] == [110, 120, 140, 150]
    assert output['review_candidates'][0]['source'] == 'roi_100_100'


def test_artificial_crop_edge_does_not_create_a_partial_object():
    detector = YOLOEDetector(device='cpu')
    detector.config['model']['roi_strategy'] = 'seeded'
    chair = next(i for i, (_, cid, _) in enumerate(detector.vocabulary) if cid == 5)
    detector.engine = Mock()
    detector.engine.predict.side_effect = [[result([])], [result([[0, 20, 40, 50, .8, chair]])]]
    with patch.object(detector, 'extract_roi_clusters', return_value=[(100, 100, 612, 612)]):
        output = detector.predict_video_frame(Image.new('RGB', (1800, 1000)))
    assert not output['object_detections'] and not output['review_candidates']
    assert output['model_calls'] == 2


def test_local_flow_preserves_camera_zoom_and_independent_object_motion():
    matrix = np.array([[1.1, 0., 10.], [0., 1.1, -5.]])
    before = np.array([[20., 20.], [25., 20.], [20., 25.], [25., 25.]])
    scale = np.array([.5, .4])
    after = (before / scale @ matrix[:, :2].T + matrix[:, 2] + [3., 2.]) * scale
    box = np.array([30., 30., 70., 80.])
    observed = local_box_from_flow(box, before, after, scale, matrix)
    np.testing.assert_allclose(observed, warp_box(box, matrix) + [3., 2., 3., 2.], atol=1e-5)
    np.testing.assert_allclose(observed[2:] - observed[:2], [44., 55.], atol=1e-5)


def test_camera_pan_does_not_lag_the_smoothed_box():
    tracker = ObjectTracker(min_hits=1)
    first = tracker.update(0., [detection()], movement())[0]
    following = tracker.update(.1, None, movement(100))[0]
    np.testing.assert_allclose(following['bbox_xyxy'], np.array(first['bbox_xyxy']) + [100, 0, 100, 0])


def test_tentative_tracks_need_consecutive_detector_observations():
    tracker = ObjectTracker(min_hits=2)
    tracker.update(0., [detection()], movement())
    tracker.update(.1, [detection((100, 100, 130, 130), cid=2)], movement())
    assert not tracker.update(.2, [detection()], movement())
    assert not tracker.confirmed_ids
    assert tracker.update(.3, [detection()], movement())[0]['track_id'] == 1


def test_lost_tracks_do_not_mask_background_or_consume_lk_work():
    tracker = ObjectTracker(min_hits=1, max_prediction_seconds=.2)
    tracker.update(0., [detection()], movement())
    assert 1 in tracker.boxes(.1)
    assert not tracker.boxes(.3)
    tracker.update(.1, [], movement())
    assert not tracker.boxes(.1)


def test_explicit_missing_checkpoint_cannot_select_stock_weights():
    try:
        resolve_yoloe_weights('outputs/nonexistent_experiment/missing_model.pt')
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('An explicit missing checkpoint silently selected another model')


def test_incomplete_roi_batch_is_rejected():
    detector = YOLOEDetector(device='cpu')
    detector.config['model']['roi_strategy'] = 'seeded'
    detector.engine = Mock()
    detector.engine.predict.side_effect = [[result([])], []]
    with patch.object(detector, 'extract_roi_clusters', return_value=[(100, 100, 612, 612)]):
        try:
            detector.predict_video_frame(Image.new('RGB', (1800, 1000)))
        except RuntimeError as exc:
            assert 'batch result count' in str(exc)
        else:
            raise AssertionError('A missing crop result was silently ignored')


def test_failed_encoder_cannot_commit_checkpoint():
    import cv2
    from pathlib import Path
    import tempfile
    from uav_pipeline.video.pipeline import VideoOptions, process_video
    from tests.test_video import FakeDetector
    with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1] / 'outputs') as folder:
        folder = Path(folder)
        source = folder / 'source.avi'
        source_writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10., (64, 48))
        assert source_writer.isOpened()
        for _ in range(3):
            source_writer.write(np.zeros((48, 64, 3), np.uint8))
        source_writer.release()
        encoder = Mock()
        encoder.release.side_effect = [RuntimeError('encode failed before flush'), None]
        with patch('uav_pipeline.video.pipeline.open_writer', return_value=encoder):
            try:
                process_video(source, folder / 'out', FakeDetector(),
                              VideoOptions(segment_frames=2, fast_render=True, preview_interval=0))
            except RuntimeError as error:
                assert 'encode failed' in str(error)
            else:
                raise AssertionError('An encoder failure was treated as a successful segment')
        assert not (folder / 'out/checkpoint.json').exists()


def test_nvenc_nonzero_exit_is_reported_and_resources_are_closed():
    import io
    from pathlib import Path
    import tempfile
    from uav_pipeline.video.io import NVENCWriter
    process = Mock()
    process.stdin = io.BytesIO()
    process.wait.return_value = 1
    with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1] / 'outputs') as folder:
        with patch('uav_pipeline.video.io.shutil.which', return_value='ffmpeg'), \
             patch('uav_pipeline.video.io.subprocess.Popen', return_value=process):
            encoder = NVENCWriter(Path(folder) / 'part.mp4', 15., (4, 2))
            encoder.write(np.zeros((2, 4, 3), np.uint8))
            try:
                encoder.release()
            except RuntimeError as error:
                assert 'NVENC encoder failed' in str(error)
            else:
                raise AssertionError('An unsuccessful encoder exit was ignored')
        assert encoder.closed and encoder.log.closed and process.stdin.closed
        encoder.release()  # Cleanup remains safe after the initial error.


def test_model_config_is_used_and_explicit_arguments_take_precedence():
    from pathlib import Path
    import tempfile
    import yaml
    root = Path(__file__).parents[1]
    config = yaml.safe_load((root / 'configs/detection.yaml').read_text(encoding='utf-8'))
    config['model'].update(weights='custom-config.pt', device='cpu', imgsz=640,
                           precision='fp32', roi_patch_size=320, roi_max_patches=0)
    with tempfile.TemporaryDirectory(dir=root / 'outputs') as folder:
        path = Path(folder) / 'config.yaml'
        path.write_text(yaml.safe_dump(config), encoding='utf-8')
        detector = YOLOEDetector(config_path=path)
        assert detector.weights == 'custom-config.pt' and detector.imgsz == 640
        assert detector.device == 'cpu' and detector.precision == 'fp32'
        assert detector.roi_patch_size == 320 and detector.roi_max_patches == 0
        with patch('torch.cuda.is_available', return_value=True):
            override = YOLOEDetector('explicit.pt', 'cuda', 960, path, 'fp16', roi_patch_size=640, roi_max_patches=2)
        assert override.weights == 'explicit.pt' and override.imgsz == 960
        assert override.device == 'cuda' and override.precision == 'fp16'
        assert override.roi_patch_size == 640 and override.roi_max_patches == 2


def test_configured_nms_and_explicit_override_have_observable_effects():
    detector = YOLOEDetector(device='cpu', roi_focus=False)
    chair = next(i for i, (_, cid, _) in enumerate(detector.vocabulary) if cid == 5)
    detector.engine = Mock()
    detector.engine.predict.return_value = [result([[10, 10, 30, 30, .8, chair], [17, 10, 37, 30, .7, chair]])]
    detector.config['inference']['nms_iou'] = .5
    configured = detector.predict_video_frame(Image.new('RGB', (64, 64)))
    override = detector.predict_video_frame(Image.new('RGB', (64, 64)), nms_iou=.45)
    assert len(configured['object_detections']) == 2
    assert len(override['object_detections']) == 1
    assert configured['inference_settings']['nms_iou'] == .5
    assert override['inference_settings']['nms_iou'] == .45


def test_precision_change_rebuilds_cached_backend():
    detector = YOLOEDetector(device='cpu', roi_focus=False)
    detector.engine = Mock()
    detector.engine.predictor = SimpleNamespace(model=SimpleNamespace(fp16=True))
    detector._predict_boxes(source=Image.new('RGB', (64, 64)), half=False)
    assert detector.engine.predictor is None


def test_async_encoder_copies_inputs_preserves_order_and_flushes():
    import threading
    from uav_pipeline.video.io import ThreadedVideoWriter
    ready = threading.Event()
    captured = []
    base = Mock()
    def consume(frame):
        assert ready.wait(2)
        captured.append(int(frame[0, 0, 0]))
    base.write.side_effect = consume
    writer = ThreadedVideoWriter(base)
    frame = np.full((2, 4, 3), 10, np.uint8)
    writer.write(frame)
    frame[:] = 20
    writer.write(frame)
    frame[:] = 30
    ready.set()
    writer.release()
    assert captured == [10, 20]
    assert not writer.worker.is_alive()
    writer.release()
    base.release.assert_called_once()


def test_async_encoder_propagates_worker_failure_without_hanging():
    from uav_pipeline.video.io import ThreadedVideoWriter
    base = Mock()
    base.write.side_effect = RuntimeError('encoder rejected frame')
    writer = ThreadedVideoWriter(base)
    writer.write(np.zeros((2, 4, 3), np.uint8))
    try:
        writer.release()
    except RuntimeError as error:
        assert 'encoder rejected frame' in str(error)
    else:
        raise AssertionError('A background encoder failure was hidden')
    assert not writer.worker.is_alive()
    base.release.assert_called_once()
