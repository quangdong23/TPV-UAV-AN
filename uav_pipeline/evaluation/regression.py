"""Streaming semantic regression check; ignores timings, never proves accuracy."""
import argparse
from itertools import zip_longest
import json
import math
from pathlib import Path

from uav_pipeline.evaluation.helpers import write_json


TIMING_FIELDS = frozenset({'frame_seconds', 'processing_fps', 'detector_seconds',
                           'model_stage_seconds', 'semantic_postprocess_seconds', 'stage_seconds'})


def difference(a, b, tolerance, path='$'):
    """First semantic difference, including missing keys and list ordering."""
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = set(a)-TIMING_FIELDS, set(b)-TIMING_FIELDS
        if ka != kb:
            return f'{path}: keys differ: {sorted(ka ^ kb)}'
        for key in sorted(ka):
            result = difference(a[key], b[key], tolerance, f'{path}.{key}')
            if result:
                return result
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f'{path}: lengths differ ({len(a)} vs {len(b)})'
        for i, (left, right) in enumerate(zip(a, b)):
            result = difference(left, right, tolerance, f'{path}[{i}]')
            if result:
                return result
        return None
    if type(a) is bool or type(b) is bool:
        equal = type(a) is type(b) and a == b
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        # Identity/count/index integers are exact. Float coordinates permit an
        # explicit absolute tolerance; NaN/Infinity never pass.
        equal = (a == b if isinstance(a, int) and isinstance(b, int) else
                 math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance)
    else:
        equal = type(a) is type(b) and a == b
    return None if equal else f'{path}: {a!r} != {b!r}'


def records(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def compare_runs(reference, candidate, tolerance=1e-4):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Tolerance must be finite and nonnegative')
    reference, candidate = Path(reference), Path(candidate)
    summaries = [json.loads((folder/'summary.json').read_text(encoding='utf-8'))
                 for folder in (reference, candidate)]
    if not summaries[0].get('source_sha256') or summaries[0]['source_sha256'] != summaries[1].get('source_sha256'):
        raise ValueError('Source hashes must match')
    examples, changed, count = [], 0, 0
    previous = None
    for count, (a, b) in enumerate(zip_longest(records(reference/'tracks.jsonl'),
                                             records(candidate/'tracks.jsonl')), 1):
        if a is None or b is None:
            result = 'Frame count differs'
        else:
            index = a.get('frame_index')
            if not isinstance(index, int) or (previous is not None and index <= previous):
                raise ValueError('Reference frame indices must be strictly increasing')
            previous = index
            result = difference(a, b, tolerance)
        if result:
            changed += 1
            if len(examples) < 20:
                examples.append({'record': count, 'difference': result})
    passed = count > 0 and changed == 0
    return {'semantic_regression_passed': passed, 'accuracy_measured': False,
            'records_compared': count, 'changed_records': changed,
            'float_absolute_tolerance': tolerance, 'ignored_fields': sorted(TIMING_FIELDS),
            'examples': examples, 'scope': 'tracks.jsonl including IDs, evidence, masks and per-frame alerts; excludes event clip files',
            'both_sources_complete': all(s.get('all_source_frames_processed', False) for s in summaries)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--tolerance', type=float, default=1e-4)
    args = parser.parse_args(argv)
    result = compare_runs(args.reference, args.candidate, args.tolerance)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0 if result['semantic_regression_passed'] else 1


if __name__ == "__main__":
    raise SystemExit(main())
