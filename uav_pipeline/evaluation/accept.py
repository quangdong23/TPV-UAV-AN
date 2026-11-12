"""Offline per-video speed and human-labelled quality acceptance; never deploys changes."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from .helpers import write_json
from .regression import compare_runs
from .tracking import evaluate_tracking


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def speed_check(reference, candidate, target, scope, max_slowdown):
    errors = []
    field = 'steady_pipeline_fps_this_run' if scope == 'steady' else 'processed_fps'
    for name, report in (('reference', reference), ('candidate', candidate)):
        if not report.get('completed') or not report.get('all_source_frames_processed'):
            errors.append(f'{name}: incomplete source video')
        if report.get('frame_skipping_enabled') is not False:
            errors.append(f'{name}: frame skipping must explicitly be disabled')
        if not finite_number(report.get(field)) or report[field] <= 0:
            errors.append(f'{name}: missing/invalid {field}')
    if reference.get('source_sha256') != candidate.get('source_sha256') or not candidate.get('source_sha256'):
        errors.append('Source video hashes differ or are missing')
    # CPU/GPU runs cannot substitute for one another. Exact environment is also
    # recorded in the report for the operator to verify load/driver conditions.
    if reference.get('gpu') != candidate.get('gpu'):
        errors.append('GPU identities differ')
    if reference.get('processed_frames') != candidate.get('processed_frames'):
        errors.append('Processed frame counts differ')
    for key in ('device', 'versions'):
        old = reference.get('runtime', {}).get(key)
        new = candidate.get('runtime', {}).get(key)
        if old != new:
            errors.append(f'Runtime {key} differs; rerun with the same environment')
    if finite_number(candidate.get(field)) and candidate[field] < target:
        errors.append(f'Candidate below {target:g} FPS ({scope})')
    if all(finite_number(s.get(field)) and s[field] > 0 for s in (reference, candidate)):
        if candidate[field] < reference[field] * (1 - max_slowdown):
            errors.append('Candidate slower than allowed relative to reference')
    return errors


def quality_check(reference, candidate, mode, minimum_gain):
    errors = []
    for report in (reference, candidate):
        if not report.get('accuracy_measured') or report.get('no_object_case'):
            errors.append('Quality acceptance requires labelled objects')
    a, b = reference['overall'], candidate['overall']
    for key in ('HOTA', 'IDF1', 'DetA', 'AssA'):
        if not all(finite_number(v.get(key)) for v in (a, b)):
            errors.append(f'Invalid {key}')
        elif b[key] + 1e-9 < a[key]:
            errors.append(f'{key} decreased')
    for key in ('ID_switches', 'false_positives', 'false_negatives'):
        if not all(finite_number(v.get(key)) for v in (a, b)) or b[key] > a[key]:
            errors.append(f'{key} increased or is invalid')
    # Aggregate gains must not hide a regression in a minority class.
    for cid, old in reference.get('by_class', {}).items():
        new = candidate.get('by_class', {}).get(cid, {})
        for key in ('HOTA', 'IDF1'):
            if (not finite_number(new.get(key)) or not finite_number(old.get(key))
                    or new[key] + 1e-9 < old[key]):
                errors.append(f'Class {cid} {key} degraded or missing')
    if mode == 'improve' and not any(
        finite_number(a.get(k)) and finite_number(b.get(k)) and b[k] - a[k] >= minimum_gain
        for k in ('HOTA', 'IDF1')
    ):
        errors.append('No sufficient HOTA/IDF1 improvement')
    for key in ('event_metrics', 'sidewalk_metrics'):
        old, new = reference.get(key, {}), candidate.get(key, {})
        if old.get('accuracy_measured'):
            metric = 'F1' if key == 'event_metrics' else 'IoU'
            if (not new.get('accuracy_measured') or not finite_number(new.get(metric))
                    or new[metric] + 1e-9 < old[metric]):
                errors.append(f'{key} degraded or missing')
    return errors


def accept_runs(references, candidates, annotations, *, target=15., scope='steady',
                max_slowdown=0., mode='improve', minimum_gain=.001):
    if (not references or len(references) != len(candidates)
            or (annotations is not None and len(annotations) != len(candidates))):
        raise ValueError('Provide equally sized nonempty reference/candidate/GT lists')
    if (not finite_number(target) or target <= 0 or scope not in ('steady', 'end-to-end')
            or not finite_number(max_slowdown) or not 0 <= max_slowdown < 1
            or mode not in ('preserve', 'improve')
            or not finite_number(minimum_gain) or minimum_gain <= 0):
        raise ValueError('Invalid acceptance thresholds')
    if mode == 'improve' and annotations is None:
        raise ValueError('Improvement requires human-verified ground truth')
    rows, seen = [], set()
    for i, (reference, candidate) in enumerate(zip(references, candidates)):
        reference, candidate = Path(reference), Path(candidate)
        a, b = [json.loads((p/'summary.json').read_text(encoding='utf-8')) for p in (reference, candidate)]
        identity = b.get('source_sha256')
        if identity in seen:
            raise ValueError('Duplicate source video in acceptance list')
        seen.add(identity)
        errors = speed_check(a, b, target, scope, max_slowdown)
        row = {'reference': str(reference.resolve()), 'candidate': str(candidate.resolve()),
               'source_sha256': identity, 'reference_summary': a, 'candidate_summary': b}
        if annotations is not None:
            old = evaluate_tracking(reference, annotations[i])
            new = evaluate_tracking(candidate, annotations[i])
            errors.extend(quality_check(old, new, mode, minimum_gain))
            if any(s.get('event_rules') is not None for s in (a, b)):
                if not all(q['event_metrics'].get('accuracy_measured') for q in (old, new)):
                    errors.append('Runs with events enabled require explicit event ground truth')
            row.update(reference_quality=old, candidate_quality=new)
        else:
            regression = compare_runs(reference, candidate)
            if not regression['semantic_regression_passed']:
                errors.append('Semantic regression failed')
            row['regression'] = regression
        row.update(passed=not errors, reasons=errors)
        rows.append(row)
    return {'passed': all(row['passed'] for row in rows), 'videos': rows,
            'mode': mode, 'target_fps': target, 'fps_scope': scope,
            'maximum_relative_slowdown': max_slowdown, 'minimum_quality_gain': minimum_gain,
            'accuracy_measured': annotations is not None,
            'gate_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope': 'Every supplied full video must pass; quality covers only annotated clips. '
                     'Steady throughput does not verify live latency or finalization speed. '
                     'Hardware load, environment and inference settings must be controlled by the operator.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, nargs='+', required=True)
    parser.add_argument('--candidate', type=Path, nargs='+', required=True)
    parser.add_argument('--ground-truth', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target-fps', type=float, default=15.)
    parser.add_argument('--fps-scope', choices=('steady', 'end-to-end'), default='steady')
    parser.add_argument('--max-slowdown', type=float, default=0.)
    parser.add_argument('--mode', choices=('preserve', 'improve'), default='improve')
    parser.add_argument('--minimum-gain', type=float, default=.001)
    args = parser.parse_args(argv)
    result = accept_runs(args.reference, args.candidate, args.ground_truth, target=args.target_fps,
                         scope=args.fps_scope, max_slowdown=args.max_slowdown,
                         mode=args.mode, minimum_gain=args.minimum_gain)
    write_json(args.output, result)
    print(json.dumps({'passed': result['passed'], 'videos': len(result['videos'])}))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
