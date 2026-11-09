"""Offline UAV object video inference, tracking and resumable segment exports.

python -m lan_chiem.video_predict --input video.mp4 --output outputs/video --device cuda
"""
import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import shutil
from pathlib import Path
from functools import lru_cache
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml

from lan_chiem.video_io import atomic_json, consolidate, file_hash, open_writer, write_frame, remove_committed_intermediates
from lan_chiem.video_motion import CameraMotion, gray_frame
from lan_chiem.video_tracking import ObjectTracker
from lan_chiem.video_scheduler import detection_reason
from lan_chiem.visualization import draw_detections, load_unicode_font, get_class_color
from lan_chiem.registry import ClassRegistry


@dataclass
class VideoOptions:
    keyframe_interval: int = 5
    min_detection_interval: int = 3
    weak_track_fraction: float = .5
    frame_step: int = 1
    tracking_max_side: int = 640
    work_max_side: int = 0
    min_hits: int = 2
    max_lost_seconds: float = 1.
    max_prediction_seconds: float = .5
    max_static_lost_seconds: float = 2.5
    max_static_prediction_seconds: float = 1.2
    segment_frames: int = 120
    precision: str = 'fp32'
    log_interval: float = 0.
    preview_interval: float = 5.
    target_fps: float = 9.
    cv_threads: int = 1
    warmup_runs: int = 0
    fast_render: bool = False
    prompt_batch_size: int = 1
    roi_detection_interval: int = 1
    video_encoder: str = 'opencv'
    async_encode: bool = False
    flow_refresh_interval: int = 1
    trunk_detection_interval_seconds: float = .8
    roi_patch_budget: int = 2
    class_persistence_boost: bool = True
    persistence_keyframes: int = 3
    persistence_missed_keyframes: int = 2

    def validate(self):
        if not isinstance(self.class_persistence_boost, bool):
            raise ValueError('class_persistence_boost must be boolean')
        if (not isinstance(self.persistence_keyframes, int) or self.persistence_keyframes < 3
                or not isinstance(self.persistence_missed_keyframes, int) or self.persistence_missed_keyframes < 0):
            raise ValueError('Invalid class persistence keyframe limits')
        for name in ('keyframe_interval', 'min_detection_interval', 'frame_step', 'min_hits', 'segment_frames', 'prompt_batch_size', 'roi_detection_interval', 'flow_refresh_interval'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('tracking_max_side', 'work_max_side', 'cv_threads', 'warmup_runs', 'roi_patch_budget'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        for name in ('max_lost_seconds', 'max_prediction_seconds', 'max_static_lost_seconds', 'max_static_prediction_seconds', 'log_interval', 'preview_interval', 'target_fps'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.max_prediction_seconds > self.max_lost_seconds:
            raise ValueError('max_prediction_seconds must be <= max_lost_seconds')
        if self.max_static_prediction_seconds > self.max_static_lost_seconds:
            raise ValueError('max_static_prediction_seconds must be <= max_static_lost_seconds')
        if self.precision not in ('fp32', 'fp16'):
            raise ValueError('precision must be fp32 or fp16')
        if not math.isfinite(self.trunk_detection_interval_seconds) or self.trunk_detection_interval_seconds < 0:
            raise ValueError('trunk_detection_interval_seconds must be finite and nonnegative')
        if self.video_encoder not in ('opencv', 'nvenc'):
            raise ValueError('video_encoder must be opencv or nvenc')
        if not isinstance(self.async_encode, bool):
            raise ValueError('async_encode must be a boolean')
        if not math.isfinite(self.weak_track_fraction) or not 0 < self.weak_track_fraction <= 1:
            raise ValueError('weak_track_fraction must be in (0, 1]')
        if self.prompt_batch_size > 8:
            raise ValueError('prompt_batch_size must be <=8; large caption batches can exceed GPU memory')


@lru_cache(maxsize=1024)
def _label_mask(text):
    # Cache small Unicode masks instead of converting the whole frame to PIL.
    font = _video_label_font()
    bounds = font.getbbox(text)
    mask = Image.new('L', (max(1, bounds[2]-bounds[0]), max(1, bounds[3]-bounds[1])))
    ImageDraw.Draw(mask).text((-bounds[0], -bounds[1]), text, font=font, fill=255)
    return np.asarray(mask)


@lru_cache(maxsize=1)
def _video_label_font():
    return load_unicode_font(16)


def _put_label(frame, text, x, y, color=(50,220,80)):
    mask = _label_mask(text)
    h, w = frame.shape[:2]
    x, y = max(0, min(x, w-1)), max(0, min(y, h-1))
    mask = mask[:h-y, :w-x]
    patch = frame[y:y+mask.shape[0], x:x+mask.shape[1]]
    alpha = mask[..., None].astype(np.float32)/255
    patch[:] = patch*(1-alpha) + np.asarray(color)*alpha


def render(frame, tracks, index, detector_ran, fps=0., fast=False):
    if fast:
        # OpenCV boxes with cached Unicode labels preserve Vietnamese accents.
        result = frame.copy()
        for track in tracks:
            x1, y1, x2, y2 = map(int, track['bbox_xyxy'])
            color=get_class_color(track['class_id'])[::-1]  # RGB registry to OpenCV BGR
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            meta = ClassRegistry.get(track['class_id'])
            name = meta.display_name if meta else track.get('class_name_vi', str(track['class_id']))
            label = f"#{track['track_id']} {name}"
            _put_label(result, label, x1, max(0, y1-20), color)
    else:
        rows = [{**t, 'score': t['detection_score'],
                 'class_name_vi': f"ID {t['track_id']} {t['class_name_vi']}",
                 'raw_label': t['position_source']} for t in tracks]
        result = np.asarray(draw_detections(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), rows))
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.putText(result, f"frame {index} | {'DETECT' if detector_ran else 'TRACK'} | processing FPS {fps:.2f}", (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
    return result


def fit_work(frame, max_side):
    h, w = frame.shape[:2]
    if max_side and max(h,w) > max_side:
        ratio = max_side / max(h,w)
        return cv2.resize(frame, (max(2, round(w*ratio)), max(2, round(h*ratio))), interpolation=cv2.INTER_AREA)
    return frame


def process_video(source, output, detector, options=None, resume=False, max_frames=0, backup_dir=None,
                  sidewalk=None, output_mode='full', events=None):
    options = options or VideoOptions()
    options.validate()
    if output_mode not in ('compact', 'full'):
        raise ValueError('output_mode must be compact or full')
    if events is not None and (sidewalk is None or options.frame_step != 1):
        raise ValueError('Event rules require combined video with every source frame processed')
    if sidewalk is not None and (resume or backup_dir):
        raise ValueError('Combined sidewalk video does not yet support resume/backup; use a fresh output directory')
    if max_frames < 0:
        raise ValueError('max_frames must be nonnegative')
    if options.precision == 'fp16' and not str(detector.device).startswith('cuda'):
        raise ValueError('fp16 requires CUDA; use fp32 for CPU')
    source, output = Path(source).resolve(), Path(output).resolve()
    backup_dir = Path(backup_dir).resolve() if backup_dir else None
    if backup_dir and (backup_dir == output or backup_dir.is_relative_to(output) or output.is_relative_to(backup_dir)):
        raise ValueError('backup_dir and output must be separate directories')
    if not source.is_file():
        raise FileNotFoundError(source)
    checkpoint_path = output / 'checkpoint.json'
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError('Output directory is not empty; choose a new output or --resume')
    if resume and not checkpoint_path.is_file():
        raise ValueError('No committed checkpoint to resume')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'segments').mkdir(exist_ok=True)
    detector.precision = options.precision
    detector.config.setdefault('model', {})['precision'] = options.precision
    detector.prompt_batch_size = options.prompt_batch_size
    print(f'[prepare] source={source.name} device={detector.device} precision={options.precision}', flush=True)
    source_hash = file_hash(source)
    code_hash = hashlib.sha256(b''.join((Path(__file__).parent / name).read_bytes() for name in
        ('video_predict.py', 'video_io.py', 'video_tracking.py', 'video_motion.py',
         'postprocess.py', 'object_classes.py',
         'registry.py', 'visualization.py', 'video_scheduler.py', 'yoloe_detector.py', 'yoloe_predictor.py', 'configuration.py', 'roi.py', 'embedding_cache.py', 'output_utils.py'))).hexdigest()
    print('[load] Loading model; first download may take time.', flush=True)
    load_started = time.perf_counter()
    detector.load_model()
    predict_frame = getattr(detector, 'predict_video_frame', detector.predict)
    bgr_input = detector.config.get('model', {}).get('backend') == 'yoloe'
    input_options = {'color_order': 'BGR'} if bgr_input else {}
    load_seconds = time.perf_counter() - load_started
    cuda = str(detector.device).startswith('cuda')
    gpu_name = torch.cuda.get_device_name() if cuda else 'CPU'
    model_device = str(next(detector.model.parameters()).device) if hasattr(getattr(detector, 'model', None), 'parameters') else str(detector.device)
    if cuda and not model_device.startswith('cuda'):
        raise RuntimeError(f'Model is on {model_device}, expected CUDA')
    print(f'[runtime] GPU={gpu_name} model_device={model_device} precision={options.precision} '
          f'model_load={load_seconds:.2f}s keyframe={options.keyframe_interval} '
          f'min_redetect={options.min_detection_interval}', flush=True)
    signature = {'schema': 1, 'source_sha256': source_hash, 'code_sha256': code_hash,
                 'video_options': asdict(options), 'detector_config': detector.config,
                 'device': str(detector.device),
                 'model_revision': detector.provenance().get('model_revision'),
                 'versions': {**detector.provenance().get('versions', {}),
                              'torch': torch.__version__, 'opencv': cv2.__version__}}
    if sidewalk is not None:
        signature['sidewalk'] = sidewalk.signature()
    signature['output_mode'] = output_mode
    if events is not None:
        signature['event_options'] = asdict(events.options)
    # JSON normalization handles YAML integer keys consistently on resume.
    signature = json.loads(json.dumps(signature))
    state = json.loads(checkpoint_path.read_text(encoding='utf-8')) if resume else None
    if state and state['signature'] != signature:
        raise ValueError('Resume rejected: source, code, environment or configuration changed')
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open video: {source}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError('Video FPS unavailable; normalize video before processing')
    tracker = ObjectTracker(
        min_hits=options.min_hits,
        max_lost_seconds=options.max_lost_seconds,
        max_prediction_seconds=options.max_prediction_seconds,
        max_static_lost_seconds=options.max_static_lost_seconds,
        max_static_prediction_seconds=options.max_static_prediction_seconds,
        class_persistence_boost=options.class_persistence_boost,
        persistence_keyframes=options.persistence_keyframes,
        persistence_missed_keyframes=options.persistence_missed_keyframes,
    )
    motion = CameraMotion(options.tracking_max_side, options.flow_refresh_interval)
    index = state['next_frame'] if state else 0
    total = state['processed_frames'] if state else 0
    last_detection_index = state['last_detection_index'] if state else -10**9
    last_detection_timestamp = state['last_detection_timestamp'] if state else -1e9
    last_trunk_timestamp = state.get('last_trunk_timestamp',-1e9) if state else -1e9
    last_timestamp = state['last_timestamp'] if state else -1.
    segments = state['segments'] if state else []
    counts = Counter(state['counts'] if state else {})
    timings = Counter(state['stage_seconds'] if state else {})
    initial_counts, initial_timings = counts.copy(), timings.copy()
    timings['model_load'] += load_seconds
    prior_wall = state['wall_seconds'] if state else 0.
    writer = records = None
    segment_count, current_count = 0, 0
    started = time.perf_counter() - load_seconds
    last_log = started
    last_preview = -1e9
    frame_latencies = []
    warmup_seconds = 0.
    ended = False
    frame = None
    frame_shape = None

    def backup_checkpoint():
        if backup_dir is None:
            return
        (backup_dir / 'segments').mkdir(parents=True, exist_ok=True)
        # Immutable segment files first, checkpoint last: interruption never
        # points a durable checkpoint at incomplete video/track data.
        for segment in segments:
            for key in ('video', 'records'):
                name = segment[key]
                destination = backup_dir / name
                if not destination.exists() or destination.stat().st_size != (output / name).stat().st_size:
                    temporary = destination.with_suffix(destination.suffix + '.tmp')
                    shutil.copyfile(output / name, temporary)
                    temporary.replace(destination)
        shutil.copyfile(output / state['previous_frame'], backup_dir / state['previous_frame'])
        atomic_json(backup_dir / 'checkpoint.json', state)

    def commit(completed):
        nonlocal writer, records, segment_count, state
        if writer is None:
            return
        writer.release()
        records.close()
        writer = records = None
        part = len(segments)
        image_name = f'segments/previous_{part:06d}.png'
        if not cv2.imwrite(str(output / image_name), frame):
            raise RuntimeError('Cannot save checkpoint frame')
        segments.append({'video': f'segments/part_{part:06d}.mp4',
                         'records': f'segments/part_{part:06d}.jsonl', 'frames': segment_count,
                         'video_sha256': file_hash(output / f'segments/part_{part:06d}.mp4'),
                         'records_sha256': file_hash(output / f'segments/part_{part:06d}.jsonl')})
        state = {'signature': signature, 'next_frame': index, 'processed_frames': total,
                 'last_detection_index': last_detection_index, 'last_timestamp': last_timestamp,
                 'last_detection_timestamp': last_detection_timestamp,
                 'last_trunk_timestamp': last_trunk_timestamp,
                 'source_shape': frame_shape,
                 'segments': segments, 'tracker': tracker.state_dict(), 'previous_frame': image_name,
                 'motion': motion.state_dict(),
                 'previous_frame_sha256': file_hash(output / image_name),
                 'counts': dict(counts), 'stage_seconds': dict(timings),
                 'wall_seconds': prior_wall + time.perf_counter() - started, 'completed': completed}
        atomic_json(checkpoint_path, state)
        backup_checkpoint()
        segment_count = 0

    try:
        if state:
            for part, segment in enumerate(segments):
                for key, suffix in (('video', 'mp4'), ('records', 'jsonl')):
                    expected = f'segments/part_{part:06d}.{suffix}'
                    if segment[key] != expected or file_hash(output / expected) != segment[key + '_sha256']:
                        raise ValueError('Checkpoint segment is missing, changed or corrupt')
            if state['previous_frame'] != f'segments/previous_{len(segments)-1:06d}.png':
                raise ValueError('Invalid checkpoint frame path')
            tracker.load_state_dict(state['tracker'])
            previous = cv2.imread(str(output / state['previous_frame']))
            if previous is None or file_hash(output / state['previous_frame']) != state['previous_frame_sha256']:
                raise ValueError('Checkpoint frame is missing or corrupt')
            frame_shape = tuple(state['source_shape'])
            motion.previous = gray_frame(previous, options.tracking_max_side)
            motion.load_state_dict(state.get('motion', {}))
            if state['completed']:
                ended = True
            elif not capture.set(cv2.CAP_PROP_POS_FRAMES, index) or abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - index) > .5:
                raise RuntimeError('Cannot seek to checkpoint frame')
        previous_cv_threads = cv2.getNumThreads()
        if options.cv_threads:
            cv2.setNumThreads(options.cv_threads)
        while not ended and (max_frames == 0 or current_count < max_frames):
            frame_started = time.perf_counter()
            t = time.perf_counter()
            success, decoded = capture.read()
            timings['decode'] += time.perf_counter() - t
            if not success:
                ended = True
                break
            frame = decoded
            if frame_shape is not None and frame.shape != frame_shape:
                raise ValueError('Video frame dimensions changed; split/normalize video first')
            frame_shape = frame.shape
            original_h, original_w = frame.shape[:2]
            frame = fit_work(frame, options.work_max_side)
            sidewalk_result = None
            has_sidewalk = True
            if sidewalk is not None:
                t = time.perf_counter()
                sidewalk_result = sidewalk.process(decoded, frame.shape)
                has_sidewalk = bool(np.any(sidewalk_result['mask'] == 2))
                counts['sidewalk_ai_frames' if sidewalk_result['metadata']['ai_ran'] else 'sidewalk_warp_frames'] += 1
                if not has_sidewalk:
                    counts['frames_without_sidewalk'] += 1
                timings['sidewalk'] += time.perf_counter() - t
            if current_count == 0 and options.warmup_runs:
                warm_started = time.perf_counter()
                warm_image = frame if bgr_input else Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                print(f'[warmup] runs={options.warmup_runs}; excluded from steady throughput, included in end-to-end', flush=True)
                for _ in range(options.warmup_runs):
                    predict_frame(warm_image, mode='full', **input_options)
                if cuda:
                    torch.cuda.synchronize()
                warmup_seconds = time.perf_counter()-warm_started
                frame_started = time.perf_counter()
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.
            if not math.isfinite(timestamp) or timestamp <= last_timestamp:
                timestamp = max(index / fps, last_timestamp + options.frame_step / fps)
                counts['timestamp_fallbacks'] += 1
            t = time.perf_counter()
            movement = motion.update(frame, tracker.boxes(timestamp))
            timings['motion'] += time.perf_counter() - t
            reason = detection_reason(index, timestamp, tracker.tracks, movement,
                                      last_detection_index, last_detection_timestamp, options)
            run_detector = reason is not None
            detections = None
            detector_evidence = None
            observed_regions = None
            observed_classes = None
            detector_seconds = 0.
            model_calls = 0
            if run_detector:
                if output_mode == 'full':
                    print(f'[detect] frame={index}/{source_frames or "?"} source_time={timestamp:.2f}s reason={reason}', flush=True)
                t = time.perf_counter()
                pass_count = [0]
                def forward_progress(stat):
                    pass_count[0] += 1
                    print(f'[forward] frame={index} pass={pass_count[0]} source={stat["source"]} '
                          f'forward_s={stat["forward_seconds"]:.3f} '
                          f'cache_hit={stat.get("image_feature_cache_hit", False)}', flush=True)
                # First detector update exposes cold-start/forward latency without
                # flooding the console with six messages for every subsequent update.
                detector._forward_progress_callback = forward_progress if current_count == 0 and output_mode == 'full' else None
                try:
                    predict_options = {'mode': 'full'}
                    if detector.config.get('model', {}).get('backend') == 'yoloe':
                        predict_options['roi_focus'] = (detector.roi_focus and
                            (reason == 'scene_cut' or counts['detector_frames'] % options.roi_detection_interval == 0))
                        patch_budget = min(options.roi_patch_budget,detector.roi_max_patches)
                        predict_options['roi_patch_budget'] = patch_budget
                        predict_options['roi_offset'] = (counts['detector_frames'] // options.roi_detection_interval) * patch_budget
                        predict_options['refine_trunks'] = (reason == 'scene_cut' or timestamp-last_trunk_timestamp >= options.trunk_detection_interval_seconds)
                    detector_image = frame if bgr_input else Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    result = predict_frame(detector_image, **predict_options, **input_options)
                    if cuda:
                        torch.cuda.synchronize()
                finally:
                    detector._forward_progress_callback = None
                detections = result['object_detections']
                if sidewalk is not None:
                    kept = sidewalk.filter(detections, sidewalk_result)
                    counts['outside_sidewalk_detections'] += len(detections) - len(kept)
                    detections = kept
                observed_regions = result.get('roi_regions')
                observed_classes = result.get('observed_class_ids')
                if result.get('trunk_refiner_ran'):
                    last_trunk_timestamp = timestamp
                    counts['trunk_detector_frames'] += 1
                detector_evidence = {'coordinate_space': 'working_pixels',
                                     'object_detections': detections,
                                     'review_candidates': result.get('review_candidates', []),
                                     'conflict_reports': result.get('conflict_reports', []),
                                     'model_stage_seconds': result.get('model_stage_seconds', {}),
                                     'semantic_postprocess_seconds': result.get('semantic_postprocess_seconds'),
                                     'roi_regions': observed_regions}
                detector_evidence['observed_class_ids'] = observed_classes
                detector_evidence['trunk_refiner_ran'] = result.get('trunk_refiner_ran',False)
                detector_evidence['trunk_refiner_evidence'] = result.get('trunk_refiner_evidence')
                if sidewalk is not None:
                    # Apply sidewalk rules, including classes allowed everywhere.
                    detector_evidence['review_candidates'] = sidewalk.filter(
                        result.get('review_candidates', []), sidewalk_result)
                # Review candidates intentionally never enter the tracker.
                detector_seconds = time.perf_counter() - t
                timings['detector'] += detector_seconds
                counts['detector_frames'] += 1
                counts['model_calls'] += result.get('model_calls', 0)
                counts['images_evaluated'] += result.get('images_evaluated', result.get('model_calls', 0))
                model_calls = result.get('model_calls', 0)
                counts[f'detect_reason_{reason}'] += 1
                counts['discarded_review_candidates'] += len(result.get('review_candidates', []))
                last_detection_index = index
                last_detection_timestamp = timestamp
            else:
                counts['tracking_only_frames'] += 1
            if movement['scene_cut']:
                counts['scene_cuts'] += 1
            t = time.perf_counter()
            tracks = tracker.update(timestamp, detections, movement, observed_regions=observed_regions, observed_classes=observed_classes)
            h, w = frame.shape[:2]
            for track in tracks:
                b = track['bbox_xyxy']
                track['bbox_xyxy'] = [float(np.clip(b[0], 0, w)), float(np.clip(b[1], 0, h)),
                                     float(np.clip(b[2], 0, w)), float(np.clip(b[3], 0, h))]
            tracks = [track for track in tracks if track['bbox_xyxy'][2] > track['bbox_xyxy'][0]
                      and track['bbox_xyxy'][3] > track['bbox_xyxy'][1]]
            if sidewalk is not None:
                tracks = sidewalk.filter(tracks, sidewalk_result)
            timings['tracking'] += time.perf_counter() - t
            new_events=[]
            event_observations=[]
            if events is not None:
                t=time.perf_counter()
                event_observations=sidewalk.filter(
                    tracker.event_observations(timestamp,frame.shape,movement), sidewalk_result)
                new_events=events.update(index,timestamp,event_observations,sidewalk_result['mask'],movement,
                                         source_size=[original_w,original_h])
                timings['event_rules']+=time.perf_counter()-t
            t = time.perf_counter()
            if writer is None:
                part = len(segments)
                writer = open_writer(output / f'segments/part_{part:06d}.mp4', fps / options.frame_step,
                                     frame.shape, encoder=options.video_encoder, async_encode=options.async_encode)
                records = (output / f'segments/part_{part:06d}.jsonl').open('w', encoding='utf-8')
            rendered = render(sidewalk_result['overlay'] if sidewalk_result is not None else frame,
                              tracks, index, run_detector,
                              (current_count + 1) / max(time.perf_counter()-started, 1e-6), options.fast_render)
            if events is not None:
                rendered=events.render(rendered,tracks)
            write_frame(writer, rendered)
            frame_seconds = time.perf_counter() - frame_started
            # Tracker/video use work pixels; public data always use original pixels.
            factor = [original_w/w, original_h/h, original_w/w, original_h/h]
            exported_tracks = [{**track, 'bbox_xyxy': (np.asarray(track['bbox_xyxy'])*factor).tolist()} for track in tracks]
            record = {'frame_index': index, 'timestamp_seconds': timestamp, 'detector_ran': run_detector,
                      'coordinate_space': 'source_pixels', 'source_size': [original_w, original_h],
                      'working_size': [w,h],
                      'detection_reason': reason, 'model_calls': model_calls,
                      'frame_seconds': frame_seconds, 'processing_fps': 1 / max(frame_seconds, 1e-6),
                      'detector_seconds': detector_seconds, 'camera_motion_reliable': movement['reliable'],
                      'motion_reason': movement['reason'], 'tracks': exported_tracks, 'track_events': tracker.events,
                      'detector_evidence': detector_evidence}
            if events is not None:
                record['rule_alerts']=[{'event_id':e['event_id'],'kind':e['kind'],
                                       'requires_review':True} for e in new_events]
            if sidewalk is not None:
                sidewalk.save_samples(sidewalk_result, rendered, index, output, [original_w, original_h])
                record['sidewalk'] = sidewalk_result['metadata']
                if detector_evidence is not None:
                    # Nested model evidence can reference rejected objects/ROI pixels.
                    # Publish a compact, source-coordinate view in combined mode.
                    public_evidence = {'coordinate_space': 'source_pixels'}
                    public_fields = ('class_id', 'class_name', 'class_name_vi', 'score',
                                     'status', 'review_reason', 'raw_label')
                    for key in ('object_detections', 'review_candidates'):
                        public_evidence[key] = [
                            {**{name: row[name] for name in public_fields if name in row},
                             'bbox_xyxy': (np.asarray(row['bbox_xyxy'])*factor).tolist()}
                            for row in detector_evidence[key]]
                    record['detector_evidence'] = public_evidence
                visible_ids = {track['track_id'] for track in tracks}
                record['track_events'] = [event for event in tracker.events if event['track_id'] in visible_ids]
            records.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            timings['render_encode_export'] += time.perf_counter() - t
            total += 1
            current_count += 1
            segment_count += 1
            index += options.frame_step
            last_timestamp = timestamp
            if segment_count >= options.segment_frames and output_mode == 'full':
                commit(False)
            now = time.perf_counter()
            frame_latencies.append(now-frame_started)
            if current_count == 1 or now - last_log >= options.log_interval:
                avg_fps = current_count / max(now-started, 1e-6)
                remaining_frames = max(0, source_frames-index) / options.frame_step
                if max_frames:
                    remaining_frames = min(remaining_frames, max(0, max_frames-current_count))
                eta = remaining_frames / max(avg_fps, 1e-6)
                status = {'frame_index': index-options.frame_step, 'source_frames': source_frames,
                          'source_time_seconds': timestamp, 'processed_frames': total,
                          'frame_seconds': frame_seconds, 'frame_processing_fps': 1/max(frame_seconds, 1e-6),
                          'average_processing_fps': avg_fps, 'eta_seconds': eta,
                          'detector_seconds': detector_seconds, 'detector_ran': run_detector,
                          'detection_reason': reason, 'detector_frames': counts['detector_frames'],
                          'model_calls': counts['model_calls'], 'device': str(detector.device),
                          'gpu': gpu_name, 'precision': options.precision}
                if output_mode == 'full':
                    atomic_json(output / 'status.json', status)
                print(f'[video] frame={index-options.frame_step}/{source_frames or "?"} '
                      f't={timestamp:.2f}s mode={"DETECT" if run_detector else "TRACK"} '
                      f'frame_fps={status["frame_processing_fps"]:.2f} avg_fps={avg_fps:.2f} '
                      f'detect_s={detector_seconds:.3f} AI={counts["detector_frames"]}/{total} '
                      f'calls={counts["model_calls"]} tracks={len(tracks)} ETA={eta:.0f}s', flush=True)
                last_log = now
            if output_mode == 'full' and options.preview_interval > 0 and now-last_preview >= options.preview_interval:
                scale = min(1., 960/max(rendered.shape[:2]))
                preview = cv2.resize(rendered, None, fx=scale, fy=scale)
                temporary = output / 'latest_preview.tmp.jpg'
                if not cv2.imwrite(str(temporary), preview, [cv2.IMWRITE_JPEG_QUALITY, 80]):
                    raise RuntimeError('Cannot write live preview')
                temporary.replace(output / 'latest_preview.jpg')
                print(f'[preview] {output / "latest_preview.jpg"}', flush=True)
                last_preview = now
            for _ in range(options.frame_step - 1):
                if not capture.grab():
                    ended = True
                    break
            if ended:
                break
        commit(ended)
        if state is None:
            raise ValueError('No decodable frames in video')
        # EOF may occur exactly at a segment boundary.
        state['completed'] = ended
        atomic_json(checkpoint_path, state)
        backup_checkpoint()
    finally:
        capture.release()
        if 'previous_cv_threads' in locals():
            cv2.setNumThreads(previous_cv_threads)
        if writer is not None:
            writer.release()
        if records is not None:
            records.close()
    analysis_finished = time.perf_counter()
    final_started = analysis_finished
    video_status = consolidate(output, segments, export_csv=output_mode == 'full',
                               consume_single=output_mode == 'compact' and (ended or sidewalk is not None))
    event_summary=None
    if events is not None:
        t=time.perf_counter()
        event_summary=events.export(output,fps/options.frame_step,total,video_status=='annotated.mp4')
        timings['event_clip_exports']=time.perf_counter()-t
    final_seconds = time.perf_counter() - final_started
    wall = prior_wall + time.perf_counter() - started
    steady_wall = max(analysis_finished-started-load_seconds-warmup_seconds, 1e-6)
    steady_fps = current_count / steady_wall if current_count else None
    run_detector_frames = counts['detector_frames'] - initial_counts['detector_frames']
    run_detector_seconds = timings['detector'] - initial_timings['detector']
    summary = {'source': str(source), 'source_fps': fps, 'source_sha256': source_hash,
               'processed_frames': total, 'completed': ended, 'wall_seconds': wall,
               'processed_fps': total / max(wall, 1e-6), 'finalize_seconds': final_seconds,
               'steady_pipeline_fps_this_run': steady_fps,
               'detector_fps_this_run': run_detector_frames / run_detector_seconds if run_detector_seconds else None,
               'detector_mean_seconds_this_run': run_detector_seconds / run_detector_frames if run_detector_frames else None,
               'ai_frame_fraction_this_run': run_detector_frames / current_count if current_count else None,
               'warmup_seconds_this_run': warmup_seconds,
               'target_fps': options.target_fps,
               'throughput_target_met': bool(steady_fps is not None and steady_fps >= options.target_fps and options.frame_step == 1),
               'end_to_end_target_met': total/max(wall,1e-6) >= options.target_fps and options.frame_step == 1,
               'frame_skipping_enabled': options.frame_step > 1,
               'speed_vs_source': (last_timestamp+1/fps)/max(wall,1e-6),
               'all_source_frames_processed': bool(ended and options.frame_step == 1 and total >= source_frames and source_frames > 0),
               'latency_p95_seconds_this_run': float(np.percentile(frame_latencies,95)) if frame_latencies else None,
               'latency_median_seconds_this_run': float(np.median(frame_latencies)) if frame_latencies else None,
               'live_latency_verified': False,
               'confirmed_track_count': len(tracker.confirmed_ids), 'unique_object_count_verified': False,
               'counts': dict(counts), 'stage_seconds': dict(timings), 'video': video_status,
               'tracker': 'class-aware Kalman/Hungarian + GMC + foreground LK (no ReID)',
               'runtime': signature, 'quality_verified': False,
               'gpu': torch.cuda.get_device_name() if str(detector.device).startswith('cuda') else None,
               'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if str(detector.device).startswith('cuda') else None,
               'timing_note': 'End-to-end excludes source hashing and resume seek; includes model load and final exports.',
               'media_note': 'Silent CFR output at source FPS/frame_step; JSONL timestamps are authoritative for VFR.'}
    if sidewalk is not None:
        summary['pipeline'] = 'sidewalk_then_objects'
        summary['sidewalk_filter'] = sidewalk.signature()
        summary['outside_sidewalk_objects_exported'] = bool(sidewalk.signature()['sidewalk_exempt_class_ids'])
        summary['resume_supported'] = False
    cleanup = output_mode == 'compact' and video_status == 'annotated.mp4' and (ended or sidewalk is not None)
    summary['output_mode'] = output_mode
    summary['intermediates_retained'] = not cleanup
    summary['resume_supported'] = sidewalk is None and not cleanup
    if event_summary is not None:
        summary['event_rules']=event_summary
    atomic_json(output / 'summary.json', summary)
    if backup_dir:
        atomic_json(backup_dir / 'summary.json', summary)
    if cleanup:
        remove_committed_intermediates(output, segments)
    elif output_mode == 'compact':
        print('[output] Intermediate segments preserved: run is incomplete or final video could not be assembled.', flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=None, help='Output directory; defaults to auto-generated path in outputs/')
    parser.add_argument('--video-config', type=Path, default=None)
    parser.add_argument('--config', type=Path, help='Detector YAML; defaults to existing image config')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='cuda')
    parser.add_argument('--backend', choices=['grounding-dino', 'world', 'yoloe', 'demo-hybrid'], default='yoloe')
    parser.add_argument('--weights', default=None)
    parser.add_argument('--imgsz', type=int, default=None, help='Override model.imgsz in detector config')
    parser.add_argument('--roi-focus', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--roi-detection-interval', type=int)
    parser.add_argument('--cache-dir', default='.hf_cache')
    parser.add_argument('--precision', choices=['fp32', 'fp16'])
    parser.add_argument('--keyframe-interval', type=int)
    parser.add_argument('--min-detection-interval', type=int)
    parser.add_argument('--log-interval', type=float)
    parser.add_argument('--preview-interval', type=float)
    parser.add_argument('--frame-step', type=int)
    parser.add_argument('--max-static-lost-seconds', type=float)
    parser.add_argument('--max-static-prediction-seconds', type=float)
    parser.add_argument('--max-frames', type=int, default=0, help='Limit frames this invocation; resume continues after them')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--backup-dir', type=Path, help='Mirror committed segments/checkpoint to a backup directory after each segment')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--cv-threads', type=int, help='OpenCV worker threads; 0 preserves the library default')
    parser.add_argument('--video-encoder', choices=['opencv', 'nvenc'])
    parser.add_argument('--async-encode', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--output-mode', choices=['compact', 'full'], default='compact',
                        help='compact: final video, JSONL and summary; full: also CSV, previews and resume files')
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable. Check the server GPU and CUDA-enabled PyTorch installation or explicitly use --device cpu.')
    torch.set_num_threads(args.threads)
    from lan_chiem.configuration import load_config
    config = load_config(args.video_config or Path(__file__).parent/'configs/video_pipeline.yaml')
    for name in ('precision', 'keyframe_interval', 'frame_step', 'min_detection_interval', 'log_interval', 'preview_interval', 'roi_detection_interval', 'max_static_lost_seconds', 'max_static_prediction_seconds', 'cv_threads', 'video_encoder', 'async_encode'):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    options = VideoOptions(**config)
    options.validate()
    if args.backend != 'yoloe':
        raise ValueError(f"Backend '{args.backend}' has been retired. The production pipeline exclusively uses YOLOE ('--backend yoloe').")
    from lan_chiem.yoloe_detector import YOLOEDetector
    from lan_chiem.output_utils import generate_unique_output_dir
    detector = YOLOEDetector(args.weights, args.device, args.imgsz, args.config or Path(__file__).parent/'configs/object_yoloe.yaml', roi_focus=args.roi_focus)
    output_dir = generate_unique_output_dir(args.output, task_name='video', source_name=args.input.stem, allow_existing=args.resume)
    path = process_video(args.input, output_dir, detector, options, args.resume, args.max_frames, args.backup_dir,
                         output_mode=args.output_mode)
    print(f'[saved] {path}')
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
