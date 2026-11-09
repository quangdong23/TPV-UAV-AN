"""Measure warmed image inference and export reviewable predictions.

Optional labels JSON maps image filenames to lists of {class_id, bbox_xyxy}.
Coordinates are original-image pixels; include every evaluated image, even empty.
Reports precision/recall at IoU .5, not mAP or tracking accuracy.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lan_chiem.video_io import atomic_json, file_hash
from lan_chiem.video_audit import agreement
from lan_chiem.visualization import draw_candidates_overlay
from lan_chiem.yoloe_detector import YOLOEDetector


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('lan_chiem/sample_data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--sizes', type=int, nargs='+', default=[640, 960, 1280])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--labels', type=Path)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    torch.set_num_threads(2)
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if not paths:
        parser.error('No input images')
    labels = json.loads(args.labels.read_text(encoding='utf-8')) if args.labels else None
    if labels is not None and any(p.name not in labels for p in paths):
        parser.error('Labels must explicitly cover every evaluated image')
    detector = YOLOEDetector(args.weights, args.device, config_path=args.config)
    detector.load_model()
    rows = []
    for size in args.sizes:
        detector.imgsz = size
        for roi in (False, True):
            name = f'{size}_roi_{int(roi)}'
            output = args.output / name
            output.mkdir(parents=True, exist_ok=True)
            images = [(p, detector.preprocess_image(p)[0]) for p in paths]
            for _ in range(3):
                detector.predict_video_frame(images[0][1], roi_focus=roi)
            if detector.device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            times, calls, totals = [], 0, {}
            for path, image in images:
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    result = detector.predict_video_frame(image, roi_focus=roi)
                    times.append(time.perf_counter() - started)
                    calls += result['model_calls']
                clean = {k: v for k, v in result.items() if not k.startswith('_')}
                clean.update(image_name=path.name, image_sha256=file_hash(path))
                atomic_json(output / f'{path.stem}_result.json', clean)
                draw_candidates_overlay(image, result['object_detections'], result['review_candidates']).save(output / f'{path.stem}.jpg')
                if labels is not None:
                    for cid, row in agreement(labels[path.name], result['object_detections']).items():
                        total = totals.setdefault(cid, {'reference': 0, 'candidate': 0, 'matched': 0})
                        for key in total:
                            total[key] += row[key]
            metrics = None
            if labels is not None:
                reference = sum(v['reference'] for v in totals.values())
                candidate = sum(v['candidate'] for v in totals.values())
                matched = sum(v['matched'] for v in totals.values())
                metrics = {'precision_iou50': matched / candidate if candidate else None,
                           'recall_iou50': matched / reference if reference else None, 'per_class': totals}
            row = {'profile': name, 'imgsz': size, 'roi_focus': roi, 'images': len(paths),
                   'repeats': args.repeats, 'mean_seconds': float(np.mean(times)),
                   'median_seconds': float(np.median(times)),
                   'p95_seconds': float(np.percentile(times, 95)), 'fps': len(times) / sum(times),
                   'model_calls': calls, 'metrics': metrics, 'quality_verified': False}
            row['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated() if detector.device == 'cuda' else None
            rows.append(row)
            print(json.dumps(row), flush=True)
            atomic_json(args.output / 'benchmark.json', {'gpu': torch.cuda.get_device_name() if detector.device == 'cuda' else None,
                        'provenance': detector.provenance(), 'profiles': rows,
                        'note': 'Warmed detector + CPU postprocessing; excludes model load, hashing, drawing and exports. Counts are not accuracy.'})
            atomic_json(args.output / 'effective_config.json', detector.config)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
