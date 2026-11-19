"""Video sidewalk/object analysis; signs, poles and tree trunks are kept everywhere."""
import argparse
import math
from pathlib import Path


def main(argv=None):
    from uav_pipeline.cli.terminal import preserve_terminal
    with preserve_terminal():
        return run(argv)


def run(argv=None):
    from uav_pipeline.paths import PROJECT_ROOT
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Video file, RTMP URL or HTTP(S) HLS playlist/player URL')
    parser.add_argument('--upload-events', action=argparse.BooleanOptionalAction, default=None,
                        help='Upload clips as soon as saved (default: enabled for live input with --events)')
    parser.add_argument('--recorded-at', help='Offline video start time, ISO 8601 with timezone; required for offline uploads')
    parser.add_argument('--upload-config', type=Path, default=root/'configs/upload_api.json')
    parser.add_argument('--live-output', help='Optional RTMP URL for annotated live output')
    parser.add_argument('--live-open-timeout', type=float, default=10.)
    parser.add_argument('--live-read-timeout', type=float, default=10.)
    parser.add_argument('--live-retry-seconds', type=float, default=3.)
    parser.add_argument('--live-fps', type=float, default=25., help='Fallback FPS if live source does not report FPS')
    parser.add_argument('--output', type=Path, default=None, help='Output directory; defaults to auto-generated path in outputs/')
    parser.add_argument('--sidewalk-config', type=Path, default=root/'configs/segmentation/gpu_t4.json')
    parser.add_argument('--object-config', type=Path, default=root/'configs/detection.yaml')
    parser.add_argument('--video-config', type=Path, default=root/'configs/video.yaml')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--tracking-scope', choices=['full-frame', 'sidewalk'],
                        help='Internal association scope; public output remains filtered by sidewalk')
    parser.add_argument('--decode-prefetch', type=int, choices=range(9),
                        help='Bounded file decode queue; 0 disables prefetch')
    parser.add_argument('--roi-recovery-budget', type=int, choices=[0, 1],
                        help='Recovery slots inside the existing ROI budget; always retains grid scanning')
    parser.add_argument('--weights', help='Override object model weights')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--cv-threads', type=int, default=4)
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--sample-step', type=int, default=None, help='Save image/mask every N frames; default compact=0, full=30')
    parser.add_argument('--output-mode', choices=['compact', 'full'], default='compact',
                        help='compact: final video, JSONL and summary; full: also CSV, samples and intermediate files')
    parser.add_argument('--filter-mode', choices=['bottom-center', 'overlap'], default='bottom-center')
    parser.add_argument('--min-overlap', type=float, default=0.1, help='Minimum sidewalk fraction of box with --filter-mode overlap')
    parser.add_argument('--sidewalk-keyframe-interval', type=int, default=5)
    parser.add_argument('--activity', action=argparse.BooleanOptionalAction, default=None,
                        help='Enable MobileNet activity classification (default: configs/activity.yaml)')
    parser.add_argument('--activity-config',type=Path,default=root/'configs/activity.yaml')
    parser.add_argument('--events', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable encroachment, sign attachment and suspected repair alerts with evidence clips')
    parser.add_argument('--violation-frames',type=int,default=10)
    parser.add_argument('--attachment-frames',type=int,default=10)
    parser.add_argument('--repair-seconds', type=float, default=None,
                        help='Override repair confirmation duration; applies to MobileNet or proximity-only mode')
    args = parser.parse_args(argv)
    if args.sample_step is None:
        args.sample_step = 30 if args.output_mode == 'full' else 0
    if args.threads < 1 or args.max_frames < 0 or args.sample_step < 0 or args.sidewalk_keyframe_interval < 1:
        parser.error('Invalid threads/frame/sampling options')
    if args.violation_frames < 1 or args.attachment_frames < 1:
        parser.error('Event frame thresholds must be positive')
    if args.repair_seconds is not None and (not math.isfinite(args.repair_seconds) or args.repair_seconds <= 0):
        parser.error('--repair-seconds must be positive and finite')
    if not 0 < args.min_overlap <= 1:
        parser.error('--min-overlap must be in (0, 1]')
    from uav_pipeline.video.live import is_live_url, RTMPCapture
    live = is_live_url(args.input)
    if args.upload_events is None:
        args.upload_events = live and args.events
    if args.upload_events and not live and not args.recorded_at:
        parser.error('Offline uploads require --recorded-at (video start time with timezone)')
    if args.upload_events and not args.events:
        parser.error('--upload-events requires --events')
    event_uploader = None
    if args.upload_events:
        from uav_pipeline.uploads import EvidenceUploader
        event_uploader = EvidenceUploader.from_config(args.upload_config)
        print('[upload] Enabled: saved evidence clips are queued on disk and uploaded during the live stream.',flush=True)
    if args.live_output:
        from uav_pipeline.video.live import safe_source
        from urllib.parse import urlsplit
        if not live or urlsplit(args.live_output).scheme.lower() not in ('rtmp','rtmps'):
            parser.error('--live-output requires live input and an RTMP output URL')
        if safe_source(args.input) == safe_source(args.live_output):
            parser.error('Input and output must use different stream paths')
        from uav_pipeline.video.publish import ffmpeg_path
        ffmpeg_path()
    if not live:
        args.input = Path(args.input)
        if not args.input.is_file():
            parser.error('Input video does not exist')
        from uav_pipeline.video.io import validate_video_input
        try:
            validate_video_input(args.input)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    for value in (args.live_open_timeout, args.live_read_timeout, args.live_retry_seconds, args.live_fps):
        if not math.isfinite(value) or value <= 0:
            parser.error('Live timeout/retry/FPS values must be finite and positive')

    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; check server PyTorch/CUDA or use --device cpu')
    torch.set_num_threads(args.threads)
    from uav_pipeline.config import load_config
    from uav_pipeline.detection.detector import YOLOEDetector
    from uav_pipeline.video.pipeline import VideoOptions, process_video
    from uav_pipeline.segmentation import PipelineConfig, SidewalkPipeline
    from uav_pipeline.segmentation.media.video import VideoPreviewOptions
    from uav_pipeline.segmentation.session import SidewalkSession
    from uav_pipeline.events import ViolationMonitor, EventOptions

    config = PipelineConfig.from_json(args.sidewalk_config)
    config.device = args.device
    options = VideoOptions(**load_config(args.video_config))
    if args.cv_threads < 1:
        parser.error('--cv-threads must be positive')
    options.cv_threads = args.cv_threads
    for name in ('decode_prefetch', 'roi_recovery_budget'):
        value = getattr(args, name)
        if value is not None:
            setattr(options, name, value)
    if args.tracking_scope is not None:
        options.track_full_frame = args.tracking_scope == 'full-frame'
    if args.device == 'cpu':
        config.precision = options.precision = 'fp32'
    config.validate()
    if not 1 <= options.keyframe_interval <= 5 or options.roi_detection_interval != 1:
        parser.error('Flight mode requires detector interval <=5 and ROI on every detector update')
    if not 1 <= args.sidewalk_keyframe_interval <= 5:
        parser.error('Flight mode requires sidewalk keyframe interval between 1 and 5')
    options.validate()
    if options.frame_step != 1:
        parser.error('Combined mode requires frame_step=1 to process every source frame')
    detector = YOLOEDetector(weights=args.weights, device=args.device, config_path=args.object_config)
    events=ViolationMonitor(EventOptions(encroachment_frames=args.violation_frames,
                           attachment_frames=args.attachment_frames,
                           repair_seconds=args.repair_seconds if args.repair_seconds is not None else 2.)) if args.events else None
    from uav_pipeline.activities import ActivityClassifier
    activity_config=load_config(args.activity_config) if args.activity is not False else {}
    use_activity=args.activity if args.activity is not None else activity_config.get('enabled',False)
    activity=ActivityClassifier(args.activity_config,device=args.device) if use_activity else None
    if activity is not None and args.repair_seconds is not None:
        activity.config['repair_confirmation_seconds']=args.repair_seconds
    with SidewalkPipeline(config) as pipeline:
        sidewalk_options = VideoPreviewOptions(keyframe_interval=args.sidewalk_keyframe_interval, cv_threads=options.cv_threads)
        session = SidewalkSession(pipeline, sidewalk_options, args.filter_mode, args.min_overlap, args.sample_step, crisp_overlay=True)
        from uav_pipeline.outputs import generate_unique_output_dir
        output_dir = generate_unique_output_dir(args.output, task_name='pipeline', source_name='rtmp_live' if live else args.input.stem)
        capture = None
        handlers = {}
        try:
            if live:
                import signal
                import threading
                stopped = threading.Event()
                for number in (signal.SIGINT, signal.SIGTERM):
                    handlers[number] = signal.getsignal(number)
                    signal.signal(number, lambda *_: stopped.set())
                capture = RTMPCapture(args.input, stop=stopped,
                    open_timeout=args.live_open_timeout, read_timeout=args.live_read_timeout,
                    retry_seconds=args.live_retry_seconds, fps_fallback=args.live_fps)
                if not capture.open():
                    print('[live] Stopped while waiting for source', flush=True)
                    return 0
                print('[live] All decoded frames are processed; upstream delay can grow if source FPS exceeds processing FPS.', flush=True)
                # Periodic preview/status are useful while waiting for a viewer.
                options.preview_interval = 2.
                options.warmup_runs = 0
            result = process_video(args.input, output_dir, detector, options, max_frames=args.max_frames,
                sidewalk=session, output_mode='full' if live else args.output_mode,
                events=events, activity=activity, live_capture=capture, live_output=args.live_output, event_uploader=event_uploader, recording_started_at=args.recorded_at)
        finally:
            if capture is not None:
                capture.release()
            if handlers:
                for number, previous in handlers.items():
                    signal.signal(number, previous)
    print(f'[saved] {result}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
