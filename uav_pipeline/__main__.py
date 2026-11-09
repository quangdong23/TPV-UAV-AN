"""Video sidewalk/object analysis; signs, poles and tree trunks are kept everywhere."""
import argparse
from pathlib import Path


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=None, help='Output directory; defaults to auto-generated path in outputs/')
    parser.add_argument('--sidewalk-config', type=Path, default=root/'configs/gpu_t4.json')
    parser.add_argument('--object-config', type=Path, default=root/'lan_chiem/configs/object_yoloe.yaml')
    parser.add_argument('--video-config', type=Path, default=root/'lan_chiem/configs/video_pipeline.yaml')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
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
                        help='Enable 10-frame rule alerts and 4-second-before/10-second-after evidence clips')
    parser.add_argument('--violation-frames',type=int,default=10)
    parser.add_argument('--attachment-frames',type=int,default=10)
    args = parser.parse_args(argv)
    if args.sample_step is None:
        args.sample_step = 30 if args.output_mode == 'full' else 0
    if args.threads < 1 or args.max_frames < 0 or args.sample_step < 0 or args.sidewalk_keyframe_interval < 1:
        parser.error('Invalid threads/frame/sampling options')
    if args.violation_frames < 1 or args.attachment_frames < 1:
        parser.error('Event frame thresholds must be positive')
    if not 0 < args.min_overlap <= 1:
        parser.error('--min-overlap must be in (0, 1]')
    if not args.input.is_file():
        parser.error(f'Input video does not exist: {args.input}')

    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; check server PyTorch/CUDA or use --device cpu')
    torch.set_num_threads(args.threads)
    from lan_chiem.configuration import load_config
    from lan_chiem.yoloe_detector import YOLOEDetector
    from lan_chiem.video_predict import VideoOptions, process_video
    from road_sidewalk_seg import PipelineConfig, SidewalkPipeline
    from road_sidewalk_seg.media.video import VideoPreviewOptions
    from uav_pipeline.sidewalk import SidewalkSession
    from uav_pipeline.events import ViolationMonitor, EventOptions

    config = PipelineConfig.from_json(args.sidewalk_config)
    config.device = args.device
    options = VideoOptions(**load_config(args.video_config))
    if args.cv_threads < 1:
        parser.error('--cv-threads must be positive')
    options.cv_threads = args.cv_threads
    if args.device == 'cpu':
        config.precision = options.precision = 'fp32'
    config.validate()
    options.validate()
    if options.frame_step != 1:
        parser.error('Combined mode requires frame_step=1 to process every source frame')
    detector = YOLOEDetector(weights=args.weights, device=args.device, config_path=args.object_config)
    events=ViolationMonitor(EventOptions(encroachment_frames=args.violation_frames,
                           attachment_frames=args.attachment_frames)) if args.events else None
    with SidewalkPipeline(config) as pipeline:
        sidewalk_options = VideoPreviewOptions(keyframe_interval=args.sidewalk_keyframe_interval, cv_threads=options.cv_threads)
        session = SidewalkSession(pipeline, sidewalk_options, args.filter_mode, args.min_overlap, args.sample_step, crisp_overlay=True)
        from lan_chiem.output_utils import generate_unique_output_dir
        output_dir = generate_unique_output_dir(args.output, task_name='pipeline', source_name=args.input.stem)
        result = process_video(args.input, output_dir, detector, options, max_frames=args.max_frames,
                               sidewalk=session, output_mode=args.output_mode, events=events)
    print(f'[saved] {result}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
