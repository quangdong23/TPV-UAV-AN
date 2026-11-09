"""Optional cached-weight GPU check; not part of the offline mock test suite."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from lan_chiem.video_io import atomic_json
from lan_chiem.yoloe_detector import YOLOEDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('lan_chiem/sample_data'))
    parser.add_argument('--output', type=Path, default=Path('lan_chiem/outputs/audit_mask_parity_final.json'))
    parser.add_argument('--weights', default='yoloe-v8s.pt')
    args = parser.parse_args()
    torch.set_num_threads(2)
    detector = YOLOEDetector(args.weights, device='cuda', roi_focus=False)
    detector.load_model()
    from ultralytics.engine.model import Model
    from ultralytics.models.yolo.segment.predict import SegmentationPredictor
    if not detector._predict_kwargs:
        raise ValueError('This parity check requires a segmentation checkpoint')
    rows = []
    mask_predictor = None
    for path in sorted(args.input.glob('*.png')):
        image = Image.open(path).convert('RGB')
        kwargs = dict(source=image, device='cuda', imgsz=960, half=True, conf=.15,
                      iou=.45, max_det=300, agnostic_nms=False, verbose=False, save=False)
        boxes = detector._predict_boxes(**kwargs)[0]
        box_predictor = detector.engine.predictor
        detector.engine.predictor = mask_predictor
        baseline = Model.predict(detector.engine, predictor=SegmentationPredictor, **kwargs)[0]
        mask_predictor = detector.engine.predictor
        detector.engine.predictor = box_predictor
        a, b = boxes.boxes.data.cpu().numpy(), baseline.boxes.data.cpu().numpy()
        equal = a.shape == b.shape and np.allclose(a, b, rtol=1e-5, atol=1e-3)
        # Verify the zero-copy OpenCV route matches the image API on real GPU
        # inference, including all class/conflict postprocessing.
        pil_output = detector.predict_video_frame(image)
        rgb = np.asarray(image)
        rgb_output = detector.predict_video_frame(rgb)
        bgr_output = detector.predict_video_frame(np.ascontiguousarray(rgb[..., ::-1]), color_order='BGR')
        color_equal = all(pil_output[key] == rgb_output[key] == bgr_output[key]
                          for key in ('object_detections', 'review_candidates', 'conflict_reports'))
        rows.append({'image': path.name, 'boxes_equal': bool(equal), 'boxes': len(a),
                     'color_paths_equal': color_equal,
                     'baseline_boxes': len(b), 'box_path_has_masks': boxes.masks is not None,
                     'baseline_has_masks': baseline.masks is not None})
    passed = bool(rows) and all(r['boxes_equal'] and r['color_paths_equal'] and not r['box_path_has_masks'] for r in rows)
    atomic_json(args.output, {'passed': passed, 'provenance': detector.provenance(), 'images': rows})
    if not passed:
        raise AssertionError(f'Box parity failed; see {args.output}')
    print(f'Box parity passed on {len(rows)} images', flush=True)


if __name__ == '__main__':
    main()
