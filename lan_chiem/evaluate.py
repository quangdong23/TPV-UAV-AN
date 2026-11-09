"""Reproducible offline evaluation of the existing Grounding DINO Base weights.

python -m lan_chiem.evaluate --input lan_chiem/sample_data --output lan_chiem/outputs/evaluation
No annotations: counts and runtime are diagnostic, never accuracy metrics.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from lan_chiem.detector import GroundingDINODetector
from lan_chiem.object_classes import ClassId, CATALOG_VERSION
from lan_chiem.visualization import draw_detections, draw_candidates_overlay


def save(result, output, stem):
    output.mkdir(parents=True, exist_ok=True)
    image = result['_image_pil']
    draw_detections(image, result.get('object_detections', result.get('detections', []))).save(output / f'{stem}_annotated.jpg')
    if result.get('review_candidates'):
        draw_candidates_overlay(image, result['object_detections'], result['review_candidates']).save(output / f'{stem}_review.jpg')
    clean = {k: v for k, v in result.items() if not k.startswith('_')}
    (output / f'{stem}_result.json').write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', default='.hf_cache')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--images', default='1,5,7', help='Representative full-frame images')
    parser.add_argument('--tile-classes', default='7,9,10', help='Identical classes in targeted full/tiled comparison')
    parser.add_argument('--tile-images', default='3,7')
    parser.add_argument('--tile-size', type=int, default=1024)
    parser.add_argument('--tree-image', default='5')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'], help='Thiết bị thực thi (auto, cuda, cpu)')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    d = GroundingDINODetector(device=args.device, cache_dir=args.cache_dir)
    all_images = sorted(args.input.glob('*.png'))
    images = [p for p in all_images if p.stem in args.images.split(',')]
    if not images:
        parser.error('No PNG inputs')
    summary_path = args.output / 'summary.json'
    rows = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else []
    if any(r.get('catalog_version') != CATALOG_VERSION for r in rows):
        parser.error('Summary uses a different/legacy catalogue. Choose a new --output folder.')
    tile_paths = [p for p in all_images if p.stem in args.tile_images.split(',')]
    plans = [('class_thresholds_full', None, 'full', images),
             ('preset025_full', .25, 'full', images),
             ('targeted_full', None, 'full', tile_paths),
             ('targeted_full_tiled', None, 'full+tiled', tile_paths)]
    for name, preset, mode, paths in plans:
        for path in paths:
            # Resume only completed rows with the same config, input, and experiment.
            existing = args.output / name / f'{path.stem}_result.json'
            classes = [int(c) for c in args.tile_classes.split(',')] if name.startswith('targeted_') else None
            if existing.exists():
                prior = json.loads(existing.read_text(encoding='utf-8'))
                experiment = prior.get('provenance', {}).get('experiment', {})
                if (prior.get('image_sha256') == hashlib.sha256(path.read_bytes()).hexdigest()
                    and prior.get('provenance', {}).get('config_sha256') == d.provenance()['config_sha256']
                    and experiment.get('classes') == classes
                    and experiment.get('mode') == mode and experiment.get('preset') == preset
                    and experiment.get('tile_size', args.tile_size) == args.tile_size):
                    print(f'Resume {name}: {path.name}', flush=True)
                    continue
            print(f'{name}: {path.name}', flush=True)
            started = time.perf_counter()
            result = d.predict(path, preset=preset, mode=mode, tile_size=args.tile_size,
                               target_classes=classes, debug=True)
            result['provenance'].update({'cli_arguments': vars(args) | {'input': str(args.input), 'output': str(args.output)},
                'argv': sys.argv, 'threads': torch.get_num_threads(),
                'experiment': {'name': name, 'preset': preset, 'mode': mode, 'classes': classes,
                               'tile_size': args.tile_size, 'image_path': str(path.resolve())}})
            save(result, args.output / name, path.stem)
            rows = [r for r in rows if (r['experiment'], r['image']) != (name, path.name)]
            rows.append({'catalog_version': CATALOG_VERSION, 'experiment': name, 'image': path.name,
                         'seconds_with_load': round(time.perf_counter()-started, 3),
                         'inference_seconds': result['inference_time_seconds'], **result['summary']})
            (args.output / 'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    for path in images:
        if path.stem == args.tree_image:
            for prompt in (phrase.strip() for phrase in d.classes[ClassId.TREE_TRUNK]['prompt'].split('.') if phrase.strip()):
                print(f'isolated {prompt}: {path.name}', flush=True)
                result = d.predict_single_prompt(path, prompt, box_threshold=.20)
                result['provenance'].update({'argv': sys.argv, 'threads': args.threads})
                save(result, args.output / 'isolated_tree_prompts', f'{path.stem}_{prompt.replace(" ", "_")}')
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
