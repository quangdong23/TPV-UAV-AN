"""Run the six-video production benchmark without overwriting the baseline.

Track counts are regression proxies, not accuracy; precision/recall need labelled
frame/box ground truth. Errors and incomplete runs make the command fail.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from uav_pipeline.paths import PROJECT_ROOT as BASE_DIR


def summarize(output):
    summary = json.loads((output/'summary.json').read_text())
    classes, raw, scores = {}, Counter(), {}
    frames = 0
    with (output/'tracks.jsonl').open(encoding='utf-8') as records:
        for line in records:
            row = json.loads(line)
            if row.get('frame_index') != frames:
                raise ValueError('Missing or reordered source frame in benchmark records')
            frames += 1
            for track in row.get('tracks', []):
                classes[track['track_id']] = track.get('class_name_vi', str(track['class_id']))
            for obj in (row.get('detector_evidence') or {}).get('object_detections', []):
                name = obj.get('class_name_vi', str(obj['class_id']))
                raw[name] += 1
                scores.setdefault(name, []).append(obj['score'])
    if not frames or frames != summary.get('processed_frames'):
        raise ValueError('Benchmark record count does not match summary')
    return {'frames': frames, 'confirmed_tracks_total': len(classes),
            'confirmed_tracks_by_class': dict(Counter(classes.values())),
            'raw_detection_counts': dict(raw),
            'score_stats': {k: {'avg': sum(v)/len(v), 'min': min(v), 'max': max(v)}
                            for k, v in scores.items()},
            'summary_counts': summary.get('counts', {}),
            'model': summary.get('runtime', {}),
            'completed': summary.get('completed', False),
            'source_sha256': summary.get('source_sha256'),
            'steady_pipeline_fps': summary.get('steady_pipeline_fps_this_run'),
            'processed_fps': summary.get('processed_fps'),
            'all_source_frames_processed': summary.get('all_source_frames_processed', False),
            'frame_skipping_enabled': summary.get('frame_skipping_enabled', True),
            'tracking_scope': summary.get('tracking_scope', 'sidewalk_filtered'),
            'latency_p95_seconds': summary.get('latency_p95_seconds_this_run'),
            'live_latency_verified': summary.get('live_latency_verified', False)}


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
    parser.add_argument('--weights', type=str, default='models/yoloe-v8s-single.pt', help='Object detector weights (.pt)')
    parser.add_argument('--video-config', type=Path, default=BASE_DIR/'configs/video.yaml')
    parser.add_argument('--sidewalk-config', type=Path, default=BASE_DIR/'configs/segmentation/gpu_t4.json')
    parser.add_argument('--object-config', type=Path, default=BASE_DIR/'configs/detection.yaml')
    parser.add_argument('--events', action='store_true')
    parser.add_argument('--require-target-fps', action='store_true')
    parser.add_argument('--target-fps', type=float, default=15.)
    parser.add_argument('--fps-scope', choices=['steady', 'end-to-end'], default='steady')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--videos', nargs='+', type=int, choices=range(1, 7), default=list(range(1, 7)))
    args = parser.parse_args(argv)
    if args.max_frames < 0:
        parser.error('--max-frames must be nonnegative')
    if not math.isfinite(args.target_fps) or args.target_fps <= 0:
        parser.error('--target-fps must be positive and finite')
    if args.require_target_fps and args.max_frames:
        parser.error('FPS acceptance requires full videos; omit --max-frames')
    
    from uav_pipeline.outputs import generate_unique_output_dir
    args.output = generate_unique_output_dir(args.output, task_name='eval_6videos')
    before = json.loads(args.baseline.read_text()) if args.baseline.exists() else {}
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'device': args.device,
              'max_frames': args.max_frames, 'accuracy_measured': False,
              'weights': str(args.weights), 'video_config': str(args.video_config.resolve()),
              'object_config': str(args.object_config.resolve()),
              'sidewalk_config': str(args.sidewalk_config.resolve()), 'events': args.events,
              'target_fps': args.target_fps, 'fps_scope': args.fps_scope, 'videos': {}}
    started = time.perf_counter()
    for number in args.videos:
        name = f'video_{number}'
        output = args.output/name
        cmd = [sys.executable, '-m', 'uav_pipeline', '--input',
               str(BASE_DIR/f'data/samples/{number}.mp4'), '--output', str(output),
               '--device', args.device, '--max-frames', str(args.max_frames),
               '--weights', str(args.weights), '--video-config', str(args.video_config.resolve()),
               '--object-config', str(args.object_config.resolve()),
               '--sidewalk-config', str(args.sidewalk_config.resolve())]
        if args.events:
            cmd.append('--events')
        tick = time.perf_counter()
        with (args.output/f'{name}.log').open('w') as log:
            result = subprocess.run(cmd, cwd=BASE_DIR, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL)
        seconds = time.perf_counter()-tick
        row = {'returncode': result.returncode, 'wall_seconds': seconds, 'status': 'error'}
        if result.returncode == 0:
            try:
                row.update(summarize(output))
                row['fps'] = row['frames']/seconds
                row['status'] = 'partial' if args.max_frames or not row['all_source_frames_processed'] else 'completed'
                measured = row['steady_pipeline_fps'] if args.fps_scope == 'steady' else row['processed_fps']
                row['target_met'] = (row['status'] == 'completed' and not row['frame_skipping_enabled']
                                     and isinstance(measured, (int, float)) and math.isfinite(measured)
                                     and measured >= args.target_fps)
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
    # Counts alone cannot establish false positives; labelled evaluation owns quality acceptance.
    speed_ok = all(v.get('target_met', False) for v in report['videos'].values())
    return 0 if complete and (not args.require_target_fps or speed_ok) else 1


if __name__ == '__main__':
    raise SystemExit(run_pipeline())
