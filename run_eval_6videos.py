"""Run the six-video production benchmark without overwriting the baseline.

Track counts are regression proxies, not accuracy; precision/recall need labelled
frame/box ground truth. Errors and incomplete runs make the command fail.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

BASE_DIR = Path(__file__).resolve().parent


def summarize(output):
    summary = json.loads((output/'summary.json').read_text())
    classes, raw, scores = {}, Counter(), {}
    frames = 0
    for line in (output/'tracks.jsonl').read_text().splitlines():
        row = json.loads(line)
        frames += 1
        for track in row.get('tracks', []):
            classes[track['track_id']] = track.get('class_name_vi', str(track['class_id']))
        for obj in (row.get('detector_evidence') or {}).get('object_detections', []):
            name = obj.get('class_name_vi', str(obj['class_id']))
            raw[name] += 1
            scores.setdefault(name, []).append(obj['score'])
    return {'frames': frames, 'confirmed_tracks_total': len(classes),
            'confirmed_tracks_by_class': dict(Counter(classes.values())),
            'raw_detection_counts': dict(raw),
            'score_stats': {k: {'avg': sum(v)/len(v), 'min': min(v), 'max': max(v)}
                            for k, v in scores.items()},
            'summary_counts': summary.get('counts', {}),
            'model': summary.get('runtime', {}),
            'completed': summary.get('completed', False)}


def compare(before, after):
    rows = {}
    for key, old in before.get('videos', {}).items():
        new = after.get('videos', {}).get(key, {})
        rows[key] = {'before_fps': old.get('fps'), 'after_fps': new.get('fps'),
                     'before_tracks': old.get('confirmed_tracks_by_class'),
                     'after_tracks': new.get('confirmed_tracks_by_class'),
                     'status': new.get('status', 'missing')}
    video2 = after.get('videos', {}).get('video_2', {})
    counts = video2.get('confirmed_tracks_by_class', {})
    checked = video2.get('status') == 'completed'
    checks = {'no_confirmed_tables': counts.get('Bàn nhựa', 0) == 0,
              'no_false_poles_on_trees': counts.get('Cột điện', 0) == 0}
    return {'videos': rows, 'video2_count_checks': checks if checked else None,
            'video2_count_checks_passed': all(checks.values()) if checked else None,
            'accuracy_status': 'unmeasured_requires_labelled_ground_truth',
            'watermark_status': 'requires_visual_or_labelled_check',
            'fps_comparable': 'Only compare full runs on the same hardware, device and precision.'}


def run_pipeline(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=None, help='Output directory; defaults to auto-generated path in outputs/')
    parser.add_argument('--baseline', type=Path, default=BASE_DIR/'outputs/benchmark_production/benchmark_report.json')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--weights', type=str, default='lan_chiem/models/yoloe-v8s-single.pt', help='Object detector weights (.pt)')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--videos', nargs='+', type=int, choices=range(1, 7), default=list(range(1, 7)))
    args = parser.parse_args(argv)
    if args.max_frames < 0:
        parser.error('--max-frames must be nonnegative')
    
    from lan_chiem.output_utils import generate_unique_output_dir
    args.output = generate_unique_output_dir(args.output, task_name='eval_6videos')
    before = json.loads(args.baseline.read_text()) if args.baseline.exists() else {}
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'device': args.device,
              'max_frames': args.max_frames, 'accuracy_measured': False,
              'weights': 'lan_chiem/models/yoloe-v8s-single.pt', 'videos': {}}
    started = time.perf_counter()
    for number in args.videos:
        name = f'video_{number}'
        output = args.output/name
        cmd = [sys.executable, '-m', 'uav_pipeline', '--input',
               str(BASE_DIR/f'lan_chiem/sample_data/{number}.mp4'), '--output', str(output),
               '--device', args.device, '--max-frames', str(args.max_frames),
               '--weights', str(args.weights)]
        tick = time.perf_counter()
        with (args.output/f'{name}.log').open('w') as log:
            result = subprocess.run(cmd, cwd=BASE_DIR, stdout=log, stderr=subprocess.STDOUT)
        seconds = time.perf_counter()-tick
        row = {'returncode': result.returncode, 'wall_seconds': seconds, 'status': 'error'}
        if result.returncode == 0:
            try:
                row.update(summarize(output))
                row['fps'] = row['frames']/seconds
                row['status'] = 'partial' if args.max_frames or not row['completed'] else 'completed'
            except (OSError, ValueError, KeyError) as error:
                row['error'] = str(error)
        else:
            row['error'] = (args.output/f'{name}.log').read_text()[-2000:]
        report['videos'][name] = row
        report['total_time_seconds'] = time.perf_counter()-started
        report['comparison'] = compare(before, report)
        (args.output/'benchmark_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(f"{name}: {row['status']} ({seconds:.1f}s)", flush=True)
    complete = all(v['status'] == 'completed' for v in report['videos'].values())
    checks = report['comparison']['video2_count_checks_passed']
    return 0 if complete and checks is not False else 1


if __name__ == '__main__':
    raise SystemExit(run_pipeline())
