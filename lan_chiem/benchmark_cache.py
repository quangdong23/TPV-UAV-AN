"""Compare CPU cache speed and raw output equality with unchanged queries.

python -m lan_chiem.benchmark_cache --input lan_chiem/sample_data/7.png
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from lan_chiem.detector import GroundingDINODetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('lan_chiem/outputs/cache_benchmark.json'))
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    detector = GroundingDINODetector(device='cpu')
    detector.load_model()
    paths = [args.input] if args.input.is_file() else sorted(
        p for p in args.input.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    rows = []
    original = detector._run_model
    fingerprints = []

    def capture(inputs, prompt, source):
        output = original(inputs, prompt, source)
        fingerprints.append({key: hashlib.sha256(getattr(output, key).detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                             for key in ('logits', 'pred_boxes')})
        return output

    detector._run_model = capture
    # Warm the same model before timing either implementation.
    detector.predict_single_prompt(paths[0], 'person .')
    for path in paths:
        results = []
        for enabled in (False, True):
            fingerprints.clear()
            detector.config['inference']['enable_image_feature_cache'] = enabled
            started = time.perf_counter()
            result = detector.predict(path)
            wall = time.perf_counter() - started
            semantic = {k: result[k] for k in ('object_detections', 'review_candidates', 'counter_evidence', 'counter_plan', 'summary')}
            results.append((list(fingerprints), semantic, wall))
            print(f'{path.name}: cache={enabled}, seconds={wall:.3f}, calls={result["model_calls"]}', flush=True)
        row = {'image': str(path), 'baseline_seconds': results[0][2], 'cached_seconds': results[1][2],
               'speedup': results[0][2] / results[1][2],
               'raw_outputs_bitwise_equal': results[0][0] == results[1][0],
               'semantic_outputs_equal': results[0][1] == results[1][1],
               'cache_hits': detector._image_feature_cache.hits,
               'cache_misses': detector._image_feature_cache.misses}
        rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'threads': args.threads, 'results': rows}, indent=2), encoding='utf-8')
        if not row['raw_outputs_bitwise_equal'] or not row['semantic_outputs_equal']:
            raise RuntimeError(f'Cache parity failed: {path}')
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
