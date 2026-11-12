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
    parser.add_argument('--input', required=True, type=Path)
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
    parser.add_argument('--events', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable encroachment, sign attachment and suspected repair alerts with evidence clips')
    parser.add_argument('--violation-frames',type=int,default=10)
    parser.add_argument('--attachment-frames',type=int,default=10)
    parser.add_argument('--repair-seconds', type=float, default=2.,
                        help='Stationary person/motorcycle proximity duration on sidewalk; requires --events')
    args = parser.parse_args(argv)
    if args.sample_step is None:
        args.sample_step = 30 if args.output_mode == 'full' else 0
    if args.threads < 1 or args.max_frames < 0 or args.sample_step < 0 or args.sidewalk_keyframe_interval < 1:
        parser.error('Invalid threads/frame/sampling options')
    if args.violation_frames < 1 or args.attachment_frames < 1:
        parser.error('Event frame thresholds must be positive')
    if not math.isfinite(args.repair_seconds) or args.repair_seconds <= 0:
        parser.error('--repair-seconds must be positive and finite')
    if not 0 < args.min_overlap <= 1:
        parser.error('--min-overlap must be in (0, 1]')
    if not args.input.is_file():
        parser.error(f'Input video does not exist: {args.input}')
    from uav_pipeline.video.io import validate_video_input
    try:
        validate_video_input(args.input)
    except (ValueError, OSError) as error:
        parser.error(str(error))

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
    options.validate()
    if options.frame_step != 1:
        parser.error('Combined mode requires frame_step=1 to process every source frame')
    detector = YOLOEDetector(weights=args.weights, device=args.device, config_path=args.object_config)
    events=ViolationMonitor(EventOptions(encroachment_frames=args.violation_frames,
                           attachment_frames=args.attachment_frames,
                           repair_seconds=args.repair_seconds)) if args.events else None
    with SidewalkPipeline(config) as pipeline:
        sidewalk_options = VideoPreviewOptions(keyframe_interval=args.sidewalk_keyframe_interval, cv_threads=options.cv_threads)
        session = SidewalkSession(pipeline, sidewalk_options, args.filter_mode, args.min_overlap, args.sample_step, crisp_overlay=True)
        from uav_pipeline.outputs import generate_unique_output_dir
        output_dir = generate_unique_output_dir(args.output, task_name='pipeline', source_name=args.input.stem)
        result = process_video(args.input, output_dir, detector, options, max_frames=args.max_frames,
                               sidewalk=session, output_mode=args.output_mode, events=events)
    print(f'[saved] {result}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
