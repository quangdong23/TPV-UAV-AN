"""Offline UAV object video inference, tracking and resumable segment exports.

python -m uav_pipeline.video.pipeline --input video.mp4 --output outputs/video --device cuda
"""
import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
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

from uav_pipeline.video.io import atomic_json, consolidate, file_hash, open_writer, write_frame, remove_committed_intermediates
from uav_pipeline.tracking.motion import CameraMotion, gray_frame
from uav_pipeline.tracking.objects import ObjectTracker, SidewalkTrackVisibility
from uav_pipeline.video.capture import PrefetchCapture
from uav_pipeline.video.telemetry import LatencyHistogram, process_rss_bytes
from uav_pipeline.video.scheduler import detection_reason
from uav_pipeline.visualization.objects import draw_detections, load_unicode_font, get_class_color
from uav_pipeline.detection.registry import ClassRegistry
from uav_pipeline.paths import CONFIG_DIR, source_signature


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
    save_tracking: bool = True
    live_segment_seconds: float = 300.
    live_retention_seconds: float = 3600.
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
    upload_on_confirmation: bool = False
    recording_fps: float = 15.
    async_evidence_clips: bool = False
    upload_interval_seconds: float = 30.
    flow_refresh_interval: int = 1
    trunk_detection_interval_seconds: float = .8
    roi_patch_budget: int = 2
    class_persistence_boost: bool = True
    persistence_keyframes: int = 3
    persistence_missed_keyframes: int = 2
    track_full_frame: bool = True
    display_full_frame: bool = False
    adaptive_kalman_noise: bool = False
    kalman_process_noise: float = 4.0
    kalman_measurement_noise: float = 4.0
    kalman_size_ratio: float = .02
    kalman_class_scales: dict = field(default_factory=dict)
    dynamic_cart_tracking: bool = False
    motion_supported_persistence: bool = False
    sidewalk_display_hold_seconds: float = 0.
    decode_prefetch: int = 0
    roi_recovery_budget: int = 0

    def validate(self):
        if not math.isfinite(self.live_segment_seconds) or self.live_segment_seconds <= 0:
            raise ValueError('live_segment_seconds must be finite and positive')
        if not math.isfinite(self.live_retention_seconds) or self.live_retention_seconds <= 0:
            raise ValueError('live_retention_seconds must be finite and positive')
        if not isinstance(self.save_tracking,bool):
            raise ValueError('save_tracking must be boolean')
        SidewalkTrackVisibility(self.sidewalk_display_hold_seconds)
        if type(self.decode_prefetch) is not int or not 0 <= self.decode_prefetch <= 8:
            raise ValueError('decode_prefetch must be an integer in 0..8')
        if type(self.roi_recovery_budget) is not int or not 0 <= self.roi_recovery_budget <= 1:
            raise ValueError('roi_recovery_budget must be 0 or 1')
        for name in ('track_full_frame', 'adaptive_kalman_noise', 'dynamic_cart_tracking', 'motion_supported_persistence'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be boolean')
        for name in ('kalman_process_noise', 'kalman_measurement_noise', 'kalman_size_ratio'):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not isinstance(self.kalman_class_scales, dict):
            raise ValueError('kalman_class_scales must be a mapping')
        for key, value in self.kalman_class_scales.items():
            if (str(key) not in {str(i) for i in ClassRegistry.all_ids()} or not isinstance(value, (int, float))
                    or isinstance(value, bool) or not math.isfinite(value) or value <= 0):
                raise ValueError('kalman_class_scales needs registered class IDs and positive scales')
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
        if not math.isfinite(self.recording_fps) or self.recording_fps<=0:
            raise ValueError('recording_fps must be finite and positive')
        if not isinstance(self.upload_on_confirmation, bool):
            raise ValueError('upload_on_confirmation must be boolean')
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
            if track.get('sidewalk_visibility') == 'recent_confirmed_track':
                label += " (theo d\u00f5i)"
            _put_label(result, label, x1, max(0, y1-20), color)
    else:
        rows = [{**t, 'score': t['detection_score'],
                 'class_name_vi': f"ID {t['track_id']} {t['class_name_vi']}" + (" (theo d\u00f5i)" if t.get("sidewalk_visibility") == "recent_confirmed_track" else ""),
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
                  sidewalk=None, output_mode='full', events=None, activity=None, live_capture=None, live_output=None, event_uploader=None, recording_started_at=None):
    if event_uploader is not None and events is None:
        raise ValueError('Evidence upload requires event rules')
    recording_start = None
    if recording_started_at is not None:
        from datetime import datetime
        recording_start = datetime.fromisoformat(recording_started_at)
        if recording_start.tzinfo is None:
            raise ValueError('recording_started_at must include timezone')
    if event_uploader is not None and live_capture is None and recording_start is None:
        raise ValueError('Offline uploads require recording_started_at')
    options = options or VideoOptions()
    options.validate()
    if output_mode not in ('compact', 'full'):
        raise ValueError('output_mode must be compact or full')
    if activity is not None and (sidewalk is None or options.frame_step != 1):
        raise ValueError('Activity classification requires sidewalk masks and every source frame')
    if events is not None and (sidewalk is None or options.frame_step != 1):
        raise ValueError('Event rules require combined video with every source frame processed')
    if sidewalk is not None and (resume or backup_dir):
        raise ValueError('Combined sidewalk video does not yet support resume/backup; use a fresh output directory')
    if max_frames < 0:
        raise ValueError('max_frames must be nonnegative')
    if options.precision == 'fp16' and not str(detector.device).startswith('cuda'):
        raise ValueError('fp16 requires CUDA; use fp32 for CPU')
    live = live_capture is not None
    source = Path('rtmp_live') if live else Path(source).resolve()
    output = Path(output).resolve()
    if live and (resume or backup_dir):
        raise ValueError('Live input does not support resume/backup')
    backup_dir = Path(backup_dir).resolve() if backup_dir else None
    if backup_dir and (backup_dir == output or backup_dir.is_relative_to(output) or output.is_relative_to(backup_dir)):
        raise ValueError('backup_dir and output must be separate directories')
    if not live:
        if not source.is_file():
            raise FileNotFoundError(source)
        from uav_pipeline.video.io import validate_video_input
        validate_video_input(source)
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
    source_hash = None if live else file_hash(source)
    code_hash = source_signature()
    print('[load] Loading model; first download may take time.', flush=True)
    load_started = time.perf_counter()
    detector.load_model()
    if activity is not None:
        activity.load_model()
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
    if activity is not None:
        signature['activity_classifier'] = activity.provenance()
    signature['output_mode'] = output_mode
    if events is not None:
        signature['event_options'] = asdict(events.options)
    # JSON normalization handles YAML integer keys consistently on resume.
    signature = json.loads(json.dumps(signature))
    state = json.loads(checkpoint_path.read_text(encoding='utf-8')) if resume else None
    if state and state['signature'] != signature:
        raise ValueError('Resume rejected: source, code, environment or configuration changed')
    capture = live_capture if live else cv2.VideoCapture(str(source))
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
        adaptive_noise=options.adaptive_kalman_noise,
        process_noise=options.kalman_process_noise,
        measurement_noise=options.kalman_measurement_noise,
        noise_size_ratio=options.kalman_size_ratio,
        class_noise_scales=options.kalman_class_scales,
        dynamic_cart_tracking=options.dynamic_cart_tracking,
        motion_supported_persistence=options.motion_supported_persistence,
    )
    visibility = SidewalkTrackVisibility(options.sidewalk_display_hold_seconds)
    motion = CameraMotion(options.tracking_max_side, options.flow_refresh_interval)
    index = state['next_frame'] if state else 0
    total = state['processed_frames'] if state else 0
    last_detection_index = state['last_detection_index'] if state else -10**9
    last_detection_timestamp = state['last_detection_timestamp'] if state else -1e9
    last_trunk_timestamp = state.get('last_trunk_timestamp',-1e9) if state else -1e9
    roi_scan_cursor = state.get('roi_scan_cursor', 0) if state else 0
    last_timestamp = state['last_timestamp'] if state else -1.
    segments = state['segments'] if state else []
    counts = Counter(state['counts'] if state else {})
    timings = Counter(state['stage_seconds'] if state else {})
    sidewalk_stage_seconds = Counter()
    initial_counts, initial_timings = counts.copy(), timings.copy()
    timings['model_load'] += load_seconds
    prior_wall = state['wall_seconds'] if state else 0.
    publisher = None
    writer = records = None
    original_writer = None
    segment_count, current_count = 0, 0
    segment_start_timestamp = 0.
    recorded_frames=0
    last_recorded_timestamp=-1.
    last_recorded_source_index=None
    previous_recording_pair=None
    latest_source_fps=fps
    from uav_pipeline.video.cfr import TimestampCFRClock,open_cfr_writer
    recording_clock=TimestampCFRClock(options.recording_fps) if live else None
    started = time.perf_counter() - load_seconds
    last_log = started
    last_preview = -1e9
    frame_latencies = LatencyHistogram()
    rss_start = process_rss_bytes()
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
                name = segment.get(key)
                if not name:continue
                destination = backup_dir / name
                if not destination.exists() or destination.stat().st_size != (output / name).stat().st_size:
                    temporary = destination.with_suffix(destination.suffix + '.tmp')
                    shutil.copyfile(output / name, temporary)
                    temporary.replace(destination)
        shutil.copyfile(output / state['previous_frame'], backup_dir / state['previous_frame'])
        atomic_json(backup_dir / 'checkpoint.json', state)

    def commit(completed):
        nonlocal writer, records, original_writer, segment_count, state
        if writer is None:
            return
        writer.release()
        if original_writer is not None:
            original_writer.release()
            original_writer = None
        records.close()
        writer = records = None
        part = segments[-1].get('part',len(segments)-1)+1 if segments else 0
        old_image = state.get('previous_frame') if state else None
        image_name = f'segments/previous_{part:06d}.png'
        if not cv2.imwrite(str(output / image_name), frame):
            raise RuntimeError('Cannot save checkpoint frame')
        segments.append({'video': f'segments/part_{part:06d}.mp4',
                         'records': f'segments/part_{part:06d}.jsonl' if options.save_tracking else None, 'frames': segment_count,
                         'part': part, 'start_frame': (recorded_frames if live else total)-segment_count,
                         'video_sha256': file_hash(output / f'segments/part_{part:06d}.mp4'),
                         'records_sha256': file_hash(output / f'segments/part_{part:06d}.jsonl') if options.save_tracking else None})
        if live:
            segments[-1]['original_video'] = f'segments/original_{part:06d}.mp4'
            segments[-1].update(start_timestamp=segment_start_timestamp,
                                end_timestamp=last_recorded_timestamp+1/options.recording_fps)
        state = {'signature': signature, 'next_frame': index, 'processed_frames': total,
                 'last_detection_index': last_detection_index, 'last_timestamp': last_timestamp,
                 'last_detection_timestamp': last_detection_timestamp,
                 'last_trunk_timestamp': last_trunk_timestamp, 'roi_scan_cursor': roi_scan_cursor,
                 'source_shape': frame_shape,
                 'segments': segments, 'tracker': tracker.state_dict(), 'previous_frame': image_name,
                 'motion': motion.state_dict(),
                 'previous_frame_sha256': file_hash(output / image_name),
                 'counts': dict(counts), 'stage_seconds': dict(timings),
                 'wall_seconds': prior_wall + time.perf_counter() - started, 'completed': completed}
        atomic_json(checkpoint_path, state)
        backup_checkpoint()
        if old_image and old_image != image_name:
            (output/old_image).unlink(missing_ok=True)
            if backup_dir:(backup_dir/old_image).unlink(missing_ok=True)
        segment_count = 0

    live_evidence = None
    if live or event_uploader is not None:
        from uav_pipeline.video.evidence import LiveEvidence
        live_evidence = LiveEvidence(output, events, options.recording_fps if live else fps/options.frame_step, event_uploader,
                                     retention_seconds=30.,vfr=live,cfr=live,
                                     upload_on_confirmation=options.upload_on_confirmation,
                                     async_clips=options.async_evidence_clips,
                                     upload_interval_seconds=options.upload_interval_seconds,encoder=options.video_encoder)
    def emit_recording_pair(pair,tick):
        nonlocal writer,original_writer,records,segment_start_timestamp
        nonlocal segment_count,recorded_frames,last_recorded_timestamp,last_recorded_source_index
        if writer is None:
            part=segments[-1].get('part',len(segments)-1)+1 if segments else 0
            segment_start_timestamp=tick
            writer=open_cfr_writer(output/f'segments/part_{part:06d}.mp4',options.recording_fps,pair[1].shape,options.video_encoder)
            original_writer=open_cfr_writer(output/f'segments/original_{part:06d}.mp4',options.recording_fps,pair[0].shape,options.video_encoder)
            records=(output/f'segments/part_{part:06d}.jsonl').open('w',encoding='utf-8') if options.save_tracking else open(os.devnull,'w')
        write_frame(original_writer,pair[0])
        write_frame(writer,pair[1])
        if live_evidence is not None:live_evidence.write(pair[1],tick)
        if pair[2]==last_recorded_source_index:
            counts['recording_duplicate_frames']+=1
        else:
            counts['recording_unique_source_frames']+=1
        last_recorded_source_index=pair[2]
        recorded_frames+=1
        segment_count+=1
        last_recorded_timestamp=tick

    from uav_pipeline.evidence_groups import EvidenceGroups
    evidence_groups = EvidenceGroups()
    live_epoch = capture.epoch if live else 0
    try:
        if state:
            for part, segment in enumerate(segments):
                for key, suffix in (('video', 'mp4'), ('records', 'jsonl')):
                    if key=='records' and not segment.get(key):continue
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
        if options.decode_prefetch and not ended and not live:
            capture = PrefetchCapture(capture, options.decode_prefetch)
        previous_cv_threads = cv2.getNumThreads()
        if options.cv_threads:
            cv2.setNumThreads(options.cv_threads)
        if live_output is not None:
            from uav_pipeline.video.publish import RTMPPublisher
            publisher = RTMPPublisher(live_output, fps)
        while not ended and (max_frames == 0 or current_count < max_frames):
            frame_started = time.perf_counter()
            t = time.perf_counter()
            success, decoded = capture.read()
            timings['decode'] += time.perf_counter() - t
            if not success:
                ended = True
                break
            observed_at = None
            if live:
                from datetime import datetime
                from zoneinfo import ZoneInfo
                observed_at = datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat(timespec='milliseconds')
            stream_reconnected = live and capture.epoch != live_epoch
            if stream_reconnected:
                live_epoch = capture.epoch
                motion.previous = None
                motion.point_cache.clear()
                visibility.last_inside.clear()
                if sidewalk is not None:
                    # Recreate temporal mask state while retaining loaded models.
                    sidewalk.__init__(sidewalk.pipeline, sidewalk.options,
                        sidewalk.filter_mode, sidewalk.min_overlap, sidewalk.sample_step,
                        crisp_overlay=sidewalk.crisp_overlay)
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
                sidewalk_stage_seconds.update(sidewalk_result.get('stage_seconds', {}))
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
            if stream_reconnected:
                movement.update(scene_cut=True, reliable=False, reason='stream_reconnected')
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
                        predict_options['roi_offset'] = roi_scan_cursor
                        if options.roi_recovery_budget and predict_options['roi_focus']:
                            from uav_pipeline.detection.roi import track_recovery_regions
                            predict_options['roi_recovery_regions'] = track_recovery_regions(
                                tracker.tracks, timestamp, movement, frame.shape, detector.roi_patch_size,
                                min_hits=options.min_hits, dynamic_limit=options.max_prediction_seconds,
                                static_limit=options.max_static_prediction_seconds,
                                budget=min(options.roi_recovery_budget, max(0, patch_budget-1)))
                        predict_options['refine_trunks'] = (reason == 'scene_cut' or timestamp-last_trunk_timestamp >= options.trunk_detection_interval_seconds)
                    detector_image = frame if bgr_input else Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    result = predict_frame(detector_image, **predict_options, **input_options)
                    if detector.config.get('model', {}).get('backend') == 'yoloe' and predict_options['roi_focus']:
                        roi_scan_cursor += result.get('roi_grid_positions_consumed', patch_budget)
                        counts['roi_recovery_patches'] += result.get('roi_recovery_patches', 0)
                    if cuda:
                        torch.cuda.synchronize()
                finally:
                    detector._forward_progress_callback = None
                detections = result['object_detections']
                if sidewalk is not None:
                    from uav_pipeline.detection.postprocess import partition_road_surface_hypotheses
                    detections, surface_review = partition_road_surface_hypotheses(
                        detections, sidewalk_result.get('road_probability'),
                        detector.config.get('review_rules', {}),
                        sidewalk_probability=sidewalk_result.get('sidewalk_probability'))
                    result['review_candidates'] = [*result.get('review_candidates', []), *surface_review]
                eligible_detections = detections
                if sidewalk is not None:
                    kept = sidewalk.filter(detections, sidewalk_result)
                    eligible_detections = kept
                    counts['outside_sidewalk_detections'] += len(detections) - len(kept)
                    if not options.track_full_frame:
                        detections = kept
                observed_regions = result.get('roi_regions')
                observed_classes = result.get('observed_class_ids')
                if result.get('trunk_refiner_ran'):
                    last_trunk_timestamp = timestamp
                    counts['trunk_detector_frames'] += 1
                detector_evidence = {'coordinate_space': 'working_pixels',
                                     'object_detections': eligible_detections,
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
            sidewalk_tracks = tracks
            if sidewalk is not None:
                inside_tracks = sidewalk.filter(tracks, sidewalk_result)
                sidewalk_tracks = visibility.update(timestamp, tracks, inside_tracks, movement)
                if not options.display_full_frame:
                    tracks = sidewalk_tracks
            timings['tracking'] += time.perf_counter() - t
            new_events=[]
            event_observations=[]
            activity_predictions=[]
            activity_regions=None
            if events is not None or activity is not None:
                from uav_pipeline.violation_regions import violation_mask
                rule_mask=violation_mask(sidewalk_result['mask'],20.*frame.shape[1]/original_w,
                                         sidewalk_result.get('road_probability'))
                rule_result={**sidewalk_result,'mask':rule_mask}
                all_observations=tracker.event_observations(timestamp,frame.shape,movement)
                event_observations=sidewalk.filter(all_observations,rule_result)
                # Người cạnh vùng vẫn được ghép dù chân nằm ngoài mask vỉa hè.
                included={t['track_id'] for t in event_observations}
                event_observations.extend(t for t in all_observations
                    if t['class_id']==1 and t['track_id'] not in included)
            if activity is not None:
                t=time.perf_counter()
                previous_calls=activity.model_calls
                previous_images=activity.images_evaluated
                activity_predictions=activity.update(index,timestamp,frame,event_observations,
                    rule_mask,movement,run_detector)
                activity_regions=[row for row in activity_predictions if row['confirmed']]
                counts['activity_model_calls']+=activity.model_calls-previous_calls
                counts['activity_crops_evaluated']+=activity.images_evaluated-previous_images
                counts['activity_frames_with_predictions']+=bool(activity_predictions)
                timings['activity_classifier']+=time.perf_counter()-t
            if events is not None:
                t=time.perf_counter()
                event_kwargs={'activity_regions':activity_regions} if activity is not None else {}
                if live:
                    event_kwargs['detected_at'] = observed_at
                elif recording_start is not None:
                    from datetime import timedelta
                    event_kwargs['detected_at'] = (recording_start+timedelta(seconds=timestamp)).isoformat(timespec='milliseconds')
                evidence_groups.update(event_observations,timestamp,
                    movement.get('scene_cut',False) or stream_reconnected)
                new_events=events.update(index,timestamp,event_observations,rule_mask,movement,
                                         source_size=[original_w,original_h],**event_kwargs)
                for event in events.events:
                    if event.get('evidence_status')=='ready' or event.get('evidence_image_error'):
                        continue
                    try:
                        evidence_groups.capture(event,event_observations,decoded,output,frame.shape,
                            Path(str(getattr(detector,'weights','unknown'))).name, image_frame=index)
                    except (OSError,RuntimeError,ValueError,cv2.error) as error:
                        event['evidence_image_error']=type(error).__name__
                        event['images']=[]
                timings['event_rules']+=time.perf_counter()-t
            t = time.perf_counter()
            if writer is None and not live:
                part=segments[-1].get('part',len(segments)-1)+1 if segments else 0
                writer=open_writer(output/f'segments/part_{part:06d}.mp4',fps/options.frame_step,
                    frame.shape,encoder=options.video_encoder,async_encode=options.async_encode)
                records=(output/f'segments/part_{part:06d}.jsonl').open('w',encoding='utf-8') if options.save_tracking else open(os.devnull,'w')
            grouped_ids={t['track_id'] for r in events.violation_regions for t in r['members']} if events else set()
            display_tracks=[t for t in tracks if t['track_id'] not in grouped_ids]
            rendered = render(sidewalk_result['overlay'] if sidewalk_result is not None else frame,
                              display_tracks, index, run_detector,
                              (current_count + 1) / max(time.perf_counter()-started, 1e-6), options.fast_render)
            if events is not None:
                rendered=events.render(rendered,[t for t in sidewalk_tracks
                    if t.get('sidewalk_visibility') != 'recent_confirmed_track'])
            if activity is not None:
                rendered=activity.render(rendered,activity_predictions)
            if live:
                reported=capture.get(cv2.CAP_PROP_FPS)
                if math.isfinite(reported) and reported>0:latest_source_fps=reported
                pair=(decoded,rendered,index)
                for tick,use_current in recording_clock.push(timestamp):
                    emit_recording_pair(pair if use_current or previous_recording_pair is None else previous_recording_pair,tick)
                previous_recording_pair=(decoded.copy(),rendered.copy(),index)
            else:
                write_frame(writer,rendered)
                if live_evidence is not None:live_evidence.write(rendered,timestamp)
            if publisher is not None:
                publisher.write(rendered)
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
            if live:
                record['observed_at'] = observed_at
                record['source_fps_reported'] = capture.get(cv2.CAP_PROP_FPS)
                record['live_connection_epoch'] = live_epoch
                record['stream_reconnected'] = stream_reconnected
                if publisher is not None:
                    record['live_output'] = publisher.status()
            if events is not None:
                record['rule_alerts']=[{'event_id':e['event_id'],'kind':e['kind'],
                                       'requires_review':True} for e in new_events]
                record['activity_regions'] = [
                    {**region, 'bbox_xyxy': (np.asarray(region['bbox_xyxy'])*factor).tolist()}
                    for region in events.repair_regions]
            if activity is not None:
                record['activity_predictions']=[
                    {**row,'bbox_xyxy':(np.asarray(row['bbox_xyxy'])*factor).tolist(),
                     'coordinate_space':'source_pixels'} for row in activity_predictions]
                record['activity_regions']=[row for row in record['activity_predictions'] if row['confirmed']]
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
            if options.save_tracking and records is not None:
                records.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            if live and new_events and options.save_tracking:
                with (output/'event_journal.jsonl').open('a',encoding='utf-8') as journal:
                    for event in new_events:
                        journal.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
            timings['render_encode_export'] += time.perf_counter() - t
            total += 1
            current_count += 1
            if not live:segment_count += 1
            index += options.frame_step
            last_timestamp = timestamp
            segment_limit=max(1,math.ceil(options.live_segment_seconds*options.recording_fps)) if live else options.segment_frames
            if segment_count>=segment_limit and (output_mode=='full' or live):
                commit(False)
            now = time.perf_counter()
            frame_latencies.add(now-frame_started)
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
        if live and previous_recording_pair is not None:
            for tick in recording_clock.finish(last_timestamp+1/latest_source_fps):
                emit_recording_pair(previous_recording_pair,tick)
        commit(ended)
        if state is None:
            raise ValueError('No decodable frames in video')
        # EOF may occur exactly at a segment boundary.
        state['completed'] = ended
        atomic_json(checkpoint_path, state)
        backup_checkpoint()
    finally:
        if live_evidence is not None and sys.exc_info()[0] is not None:
            live_evidence.close()
        if publisher is not None:
            publisher.release()
        capture.release()
        if 'previous_cv_threads' in locals():
            cv2.setNumThreads(previous_cv_threads)
        if writer is not None:
            writer.release()
        if records is not None:
            records.close()
        if original_writer is not None:
            original_writer.release()
    analysis_finished = time.perf_counter()
    final_started = analysis_finished
    video_status = consolidate(output, segments, export_csv=output_mode == 'full',
                               consume_single=output_mode == 'compact' and (ended or sidewalk is not None))
    if live or event_uploader is not None:
        from uav_pipeline.video.io import consolidate_recording
        if video_status != 'annotated.mp4':
            video_status = consolidate_recording(output, [s['video'] for s in segments],
                                                 'annotated.mp4',sum(s['frames'] for s in segments),
                                                 start_timestamps=[s['start_timestamp'] for s in segments] if live else None)
    original_status = None
    if live:
        from uav_pipeline.video.io import consolidate_recording
        original_status = consolidate_recording(output,[s['original_video'] for s in segments],
            'original.mp4',sum(s['frames'] for s in segments),
            start_timestamps=[s['start_timestamp'] for s in segments])
    event_summary=None
    if live_evidence is not None:
        event_summary=live_evidence.finish()
        live_evidence.close()
    elif events is not None:
        t=time.perf_counter()
        event_summary=events.export(output,fps/options.frame_step,total,video_status=='annotated.mp4',merge_clips=True)
        timings['event_clip_exports']=time.perf_counter()-t
    upload_summary = dict(live_evidence.upload_counts) if live_evidence and event_uploader else None
    if event_uploader is not None and not live and live_evidence is None:
        upload_started = time.perf_counter()
        upload_summary = event_uploader.upload_events(output, events.events)
        timings['event_uploads'] = time.perf_counter()-upload_started
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
               'latency_p95_seconds_this_run': frame_latencies.quantile(.95),
               'latency_median_seconds_this_run': frame_latencies.quantile(.5),
               'latency_statistics_this_run': frame_latencies.summary(),
               'rss_start_bytes_this_run': rss_start, 'rss_end_bytes': process_rss_bytes(),
               'live_latency_verified': False,
               'confirmed_track_count': len(tracker.confirmed_ids), 'unique_object_count_verified': False,
               'counts': dict(counts), 'stage_seconds': dict(timings), 'video': video_status,
               'tracker': 'class-aware Kalman/Hungarian + GMC + foreground LK (no ReID)',
               'tracking_scope': 'full_frame' if options.track_full_frame else 'sidewalk_filtered',
               'display_scope': 'full_frame' if options.display_full_frame else 'sidewalk_filtered',
               'runtime': signature, 'quality_verified': False,
               'gpu': torch.cuda.get_device_name() if str(detector.device).startswith('cuda') else None,
               'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if str(detector.device).startswith('cuda') else None,
               'timing_note': 'End-to-end excludes source hashing and resume seek; includes model load and final exports.',
               'media_note': 'Silent CFR output at source FPS/frame_step; JSONL timestamps are authoritative for VFR.'}
    if upload_summary is not None:
        summary['event_uploads'] = upload_summary
    if live:
        from uav_pipeline.video.live import safe_source
        summary['source'] = getattr(live_capture, 'source_path', None) or safe_source(live_capture.url)
        summary['input_mode'] = getattr(live_capture, 'input_mode', None) or ('hls_live' if live_capture.url.startswith(('http://','https://')) else 'rtmp_live')
        if publisher is not None:
            summary['live_output'] = publisher.status()
        summary['original_video'] = original_status
        summary['original_recording_enabled'] = True
        summary['recording_segment_seconds'] = options.live_segment_seconds
        summary['recording_rotation_note'] = 'Main recordings finalize every configured interval; evidence uses an independent 30-second disk buffer. Main recordings are kept for the whole session.'
        summary['original_recording_note'] = 'Decoded live frames at source resolution, no AI overlay, no audio; CFR with the same rolling retention as AI video.'
        summary['source_sha256'] = None
        summary['live_reconnects'] = live_capture.reconnects
        summary['live_stopped_by_user'] = live_capture.interrupted
        summary['intentional_frame_dropping'] = False
        summary['live_latency_verified'] = False
        summary['source_fps_latest'] = latest_source_fps
        summary['recording_frame_rate_mode'] = 'CFR'
        summary['recording_fps'] = options.recording_fps
        summary['recorded_frames'] = recorded_frames
        summary['recording_dropped_source_frames'] = max(0,total-counts['recording_unique_source_frames'])
        summary['media_note'] = 'Fixed CFR output; original/annotated/evidence share timestamp slots. Duplication/dropping is recording-only; AI and event timing use source timestamps.'
    if sidewalk is not None:
        summary['pipeline'] = 'sidewalk_then_objects'
        summary['sidewalk_filter'] = sidewalk.signature()
        summary['sidewalk_stage_seconds'] = dict(sidewalk_stage_seconds)
        summary['sidewalk_timing_note'] = 'Nested wall timings inside sidewalk; do not add to stage_seconds totals.'
        summary['outside_sidewalk_objects_exported'] = bool(options.display_full_frame or sidewalk.signature()['sidewalk_exempt_class_ids'])
        summary['resume_supported'] = False
    cleanup = not live and output_mode == 'compact' and video_status == 'annotated.mp4' and (ended or sidewalk is not None)
    summary['output_mode'] = output_mode
    summary['intermediates_retained'] = not cleanup
    summary['resume_supported'] = sidewalk is None and not cleanup and not live
    if live:
        summary['retained_recording_frames'] = sum(s['frames'] for s in segments)
        summary['recording_start_frame'] = segments[0]['start_frame'] if segments else total
    if activity is not None:
        summary['activity_classifier']={**activity.provenance(),
            'model_calls':activity.model_calls,'crops_evaluated':activity.images_evaluated,
            'event_policy':'model_evidence_replaces_proximity_only_repair_rule',
            'events_enabled':events is not None}
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
    from uav_pipeline.cli.terminal import preserve_terminal
    with preserve_terminal():
        return run_cli(argv)


def run_cli(argv=None):
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
    parser.add_argument('--cache-dir', default='.cache/huggingface')
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
    from uav_pipeline.config import load_config
    config = load_config(args.video_config or CONFIG_DIR/'video.yaml')
    for name in ('precision', 'keyframe_interval', 'frame_step', 'min_detection_interval', 'log_interval', 'preview_interval', 'roi_detection_interval', 'max_static_lost_seconds', 'max_static_prediction_seconds', 'cv_threads', 'video_encoder', 'async_encode'):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    options = VideoOptions(**config)
    options.validate()
    if args.backend != 'yoloe':
        raise ValueError(f"Backend '{args.backend}' has been retired. The production pipeline exclusively uses YOLOE ('--backend yoloe').")
    from uav_pipeline.detection.detector import YOLOEDetector
    from uav_pipeline.outputs import generate_unique_output_dir
    detector = YOLOEDetector(args.weights, args.device, args.imgsz, args.config or CONFIG_DIR/'detection.yaml', roi_focus=args.roi_focus)
    output_dir = generate_unique_output_dir(args.output, task_name='video', source_name=args.input.stem, allow_existing=args.resume)
    path = process_video(args.input, output_dir, detector, options, args.resume, args.max_frames, args.backup_dir,
                         output_mode=args.output_mode)
    print(f'[saved] {path}')
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
