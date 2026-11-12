"""Offline COCO box AP for labelled images or explicitly mapped detector frames."""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np

from .helpers import write_json
from .regression import records


def export_detector_frames(candidate, annotations, frame_map, category_map, output):
    """Export observed detections, never predicted tracker boxes, in source pixels."""
    folder = Path(candidate)
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    truth = json.loads(Path(annotations).read_text(encoding='utf-8'))
    mapping = json.loads(Path(frame_map).read_text(encoding='utf-8'))
    if not mapping.get('source_sha256') or mapping['source_sha256'] != summary['source_sha256']:
        raise ValueError('Frame-map video hash must match candidate')
    images = {row['id']: row for row in truth['images']}
    categories = {row['id'] for row in truth['categories']}
    ids = {int(k): v for k, v in mapping['image_to_frame'].items()}
    if (set(ids) != set(images) or any(type(v) is not int or v < 0 for v in ids.values())
            or len(set(ids.values())) != len(ids)):
        raise ValueError('Frame map must cover each labelled image exactly once with unique source frames')
    classes = {int(k): v for k, v in json.loads(Path(category_map).read_text(encoding='utf-8')).items()}
    if (not classes or not set(classes).issubset(range(1, 11))
            or any(type(v) is not int or v not in categories for v in classes.values())):
        raise ValueError('Explicit category map must map catalog IDs 1..10 to annotation categories')
    wanted = {frame: image_id for image_id, frame in ids.items()}
    seen, rows = set(), []
    for record in records(folder/'tracks.jsonl'):
        index = record['frame_index']
        if index not in wanted:
            continue
        if index in seen:
            raise ValueError('Duplicate detector frame')
        seen.add(index)
        image_id = wanted[index]
        image = images[image_id]
        if (record.get('coordinate_space') != 'source_pixels'
                or record.get('source_size') != [image['width'], image['height']]):
            raise ValueError('COCO image dimensions must match source-pixel detections')
        evidence = record.get('detector_evidence')
        if not record.get('detector_ran') or evidence is None or evidence.get('coordinate_space') != 'source_pixels':
            raise ValueError('Mapped frames must have actual source-pixel detector evidence')
        for detection in evidence['object_detections']:
            if detection['class_id'] not in classes:
                continue
            x1, y1, x2, y2 = detection['bbox_xyxy']
            rows.append({'image_id': image_id, 'category_id': classes[detection['class_id']],
                         'bbox': [x1, y1, x2-x1, y2-y1], 'score': detection['score']})
    if seen != set(wanted):
        raise ValueError('Candidate does not cover every requested labelled frame')
    write_json(output, rows)
    return sorted(set(classes.values()))


def evaluate_coco(annotations, predictions, output, category_ids=None):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise RuntimeError('Install requirements/evaluation.txt in the separate evaluation environment') from error
    captured = io.StringIO()
    with redirect_stdout(captured):
        truth = COCO(str(annotations))
        rows = json.loads(Path(predictions).read_text(encoding='utf-8'))
        if not isinstance(rows, list):
            raise ValueError('Predictions must be a COCO list')
        images, categories = set(truth.getImgIds()), set(truth.getCatIds())
        for row in rows:
            if row['image_id'] not in images or row['category_id'] not in categories:
                raise ValueError('Unknown COCO image/category ID')
            box = np.asarray(row['bbox'], float)
            if box.shape != (4,) or not np.isfinite(box).all() or min(box[2:]) <= 0:
                raise ValueError('COCO boxes must be finite xywh with positive area')
            if not isinstance(row['score'], (int, float)) or not math.isfinite(row['score']) or not 0 <= row['score'] <= 1:
                raise ValueError('COCO score must be finite in [0,1]')
        if rows:
            detected = truth.loadRes(rows)
        else:
            detected = COCO()
            detected.dataset = {'images': list(truth.imgs.values()), 'categories': list(truth.cats.values()),
                                'annotations': []}
            detected.createIndex()
        evaluator = COCOeval(truth, detected, 'bbox')
        if category_ids is not None:
            if not category_ids or not set(category_ids).issubset(categories):
                raise ValueError('Evaluation category IDs must be nonempty and present in annotations')
            evaluator.params.catIds = sorted(set(category_ids))
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    names = ['mAP_50_95', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
             'AR_1', 'AR_10', 'AR_100', 'AR_small', 'AR_medium', 'AR_large']
    report = {'accuracy_measured': True, 'metric': 'COCO bounding-box evaluation',
              'ground_truth': str(Path(annotations).resolve()), 'predictions': str(Path(predictions).resolve()),
              'annotations_sha256': hashlib.sha256(Path(annotations).read_bytes()).hexdigest(),
              'predictions_sha256': hashlib.sha256(Path(predictions).read_bytes()).hexdigest(),
              'images': len(images), 'category_ids': [int(v) for v in evaluator.params.catIds],
              'metrics': {k: float(v) if v >= 0 else None for k, v in zip(names, evaluator.stats)},
              'scope': 'Labelled COCO boxes only. Human label correctness must be verified independently. '
                       'Exports from UAV measure deployed, thresholded detections; never tracker predictions. '
                       'Null metrics have no applicable ground truth; not a live FPS or tracking measure.',
              'evaluation_log': captured.getvalue()}
    write_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--predictions', type=Path)
    source.add_argument('--candidate', type=Path)
    parser.add_argument('--frame-map', type=Path)
    parser.add_argument('--category-map', type=Path)
    parser.add_argument('--category-ids', type=int, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    categories = args.category_ids
    predictions = args.predictions
    if args.candidate:
        if not args.frame_map or not args.category_map:
            parser.error('--candidate requires --frame-map and --category-map')
        if args.category_ids:
            parser.error('--candidate selects categories using the explicit category map')
        predictions = args.output.with_name(args.output.stem+'_predictions.json')
        categories = export_detector_frames(args.candidate, args.annotations, args.frame_map,
                                             args.category_map, predictions)
    report = evaluate_coco(args.annotations, predictions, args.output, categories)
    print(json.dumps(report['metrics'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
