from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

from road_sidewalk_seg.core.config import PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segment sidewalk regions in UAV images and videos."
    )
    parser.add_argument(
        "--input",
        help="Image/video file or directory; omit to open the system file picker",
    )
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--output-mode", choices=["compact", "full"], default="compact",
                        help="compact: core results; full: also diagnostic rasters and sampled video masks")
    parser.add_argument(
        "--config",
        default="configs/gpu_t4.json",
        help="Pipeline JSON configuration",
    )
    parser.add_argument("--device", help="Override device: auto, cpu, cuda, cuda:0")
    parser.add_argument("--cpu-threads", type=int, help="CPU intra-op threads; 0 keeps PyTorch defaults")
    parser.add_argument("--cpu-attention", choices=["eager", "sdpa"])
    parser.add_argument("--inference-strategy", choices=["tiled", "compact", "whole", "auto"])
    parser.add_argument("--target-crs", help="Optional output CRS for georeferenced rasters")
    parser.add_argument(
        "--disable-low-light",
        action="store_true",
        help="Disable the CLAHE comparison branch",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop when the first input fails",
    )
    parser.add_argument(
        "--video-frame-step",
        type=int,
        default=1,
        help="Process every video frame by default; N > 1 explicitly skips frames",
    )
    parser.add_argument(
        "--video-max-frames",
        type=int,
        default=0,
        help="Stop after N analyzed frames; 0 means until video ends or Q is pressed",
    )
    parser.add_argument(
        "--video-max-side",
        type=int,
        default=768,
        help="Resize video frames for faster inference; 0 keeps the config value",
    )
    parser.add_argument(
        "--preview-width",
        type=int,
        default=1280,
        help="Maximum video preview width; 0 keeps native width",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save analyzed frames to preview.mp4",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI window; auto-enabled when no display is found (e.g. a GPU server)",
    )
    parser.add_argument(
        "--video-temporal-smoothing",
        type=float,
        default=0.3,
        help="Temporal smoothing weight for video (0.0=off, e.g. 0.3 for smooth moving video)",
    )
    parser.add_argument(
        "--keyframe-interval",
        type=int,
        default=8,
        help="Maximum AI refresh interval in processed frames; tracking failures refresh earlier; not a guaranteed FPS",
    )
    parser.add_argument("--video-work-max-side", type=int, default=960,
                        help="Cap all video processing/output dimensions; 0 preserves source size; validate boundary quality")
    parser.add_argument("--video-tracking-max-side", type=int, default=640,
                        help="Optical-flow working size only; 0 uses full frame")
    parser.add_argument("--video-log-interval", type=float, default=2.0,
                        help="Minimum seconds between headless progress messages; 0 logs every frame")
    parser.add_argument("--video-postprocess-max-side", type=int, default=768,
                        help="Cap mask/score processing separately from video output; 0 preserves reference path")
    parser.add_argument("--video-cv-threads", type=int, default=1,
                        help="OpenCV thread cap for video; 0 keeps current setting")
    parser.add_argument("--video-target-fps", type=float, default=15.0,
                        help="Report whether measured end-to-end throughput reaches this target; does not change output FPS")
    parser.add_argument("--video-mask-sample-step", type=int, default=None,
                        help="Export a mask every N frames; default compact=0, full=30; 0 disables")
    return parser


def _pick_media_files() -> list[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as error:
        raise RuntimeError(
            "The system file picker requires tkinter. "
            "Pass an image or video path with --input instead."
        ) from error

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askopenfilenames(
            parent=root,
            title="Chọn ảnh hoặc video UAV cần xử lý",
            filetypes=(
                (
                    "Ảnh và video hỗ trợ",
                    "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.mp4 *.avi *.mov *.mkv *.m4v *.webm",
                ),
                ("Video", "*.mp4 *.avi *.mov *.mkv *.m4v *.webm"),
                ("Ảnh", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("Tất cả tệp", "*.*"),
            ),
        )
    finally:
        root.destroy()
    return [Path(path) for path in selected]


def _discover_media(input_path: str | Path) -> list[Path]:
    from road_sidewalk_seg.media.image_io import SUPPORTED_SUFFIXES
    from road_sidewalk_seg.media.video import SUPPORTED_VIDEO_SUFFIXES

    path = Path(input_path)
    supported = SUPPORTED_SUFFIXES | SUPPORTED_VIDEO_SUFFIXES
    if path.is_file():
        if path.suffix.casefold() not in supported:
            raise ValueError(f"Unsupported input format: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.casefold() in supported
    )


def _configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _report_missing_dependency(error: ModuleNotFoundError) -> int:
    dependency = error.name or "an optional package"
    print(
        f"Missing Python dependency: {dependency}. "
        "Install this project's dependencies in the active environment with: "
        "python -m pip install -r requirements.txt",
        file=sys.stderr,
    )
    return 3


def _validate_video_args(args: argparse.Namespace) -> bool:
    if args.video_frame_step <= 0:
        print("--video-frame-step must be greater than zero", file=sys.stderr)
        return False
    if args.video_max_frames < 0 or args.video_max_side < 0 or args.preview_width < 0:
        print("Video numeric options cannot be negative", file=sys.stderr)
        return False
    if not (0.0 <= args.video_temporal_smoothing < 1.0):
        print("--video-temporal-smoothing must be in range [0.0, 1.0)", file=sys.stderr)
        return False
    if args.keyframe_interval < 1:
        print("--keyframe-interval must be an integer >= 1", file=sys.stderr)
        return False
    if min(args.video_work_max_side, args.video_tracking_max_side, args.video_log_interval,
           args.video_postprocess_max_side, args.video_cv_threads, args.video_target_fps,
           args.video_mask_sample_step) < 0:
        print("Video working/tracking/log options cannot be negative", file=sys.stderr)
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    _configure_console_encoding()
    args = build_parser().parse_args(argv)
    if args.video_mask_sample_step is None:
        args.video_mask_sample_step = 30 if args.output_mode == "full" else 0
    if not _validate_video_args(args):
        return 2

    config = PipelineConfig.from_json(args.config)
    if args.device:
        config.device = args.device
    if args.cpu_threads is not None:
        config.cpu_threads = args.cpu_threads
    if args.cpu_attention is not None:
        config.cpu_attention = args.cpu_attention
    if args.inference_strategy is not None:
        config.tiling.strategy = args.inference_strategy
    if args.target_crs:
        config.output.target_crs = args.target_crs
    if args.disable_low_light:
        config.preprocessing.low_light_mode = "disabled"
    config.validate()

    try:
        media_paths = _discover_media(args.input) if args.input else _pick_media_files()
    except ModuleNotFoundError as error:
        return _report_missing_dependency(error)
    if not media_paths:
        location = args.input or "system file picker"
        print(f"No supported image or video selected from: {location}", file=sys.stderr)
        return 2

    try:
        from road_sidewalk_seg.media.image_io import SUPPORTED_SUFFIXES
        from road_sidewalk_seg.models.backend import resolve_device
        from road_sidewalk_seg.core.pipeline import SidewalkPipeline
        from road_sidewalk_seg.media.video import SUPPORTED_VIDEO_SUFFIXES, VideoPreviewOptions, preview_video
    except ModuleNotFoundError as error:
        return _report_missing_dependency(error)

    image_paths = [path for path in media_paths if path.suffix.casefold() in SUPPORTED_SUFFIXES]
    video_paths = [
        path for path in media_paths if path.suffix.casefold() in SUPPORTED_VIDEO_SUFFIXES
    ]
    resolved_device = resolve_device(config.device)
    enabled_models = [model for model in config.models if model.enabled]
    model_names = ", ".join(model.name for model in enabled_models)
    print(f"[runtime] device={resolved_device}; models={model_names}")
    print(f"[runtime] strategy={config.tiling.strategy}; cpu_attention={config.cpu_attention}")
    if config.tiling.strategy != "tiled":
        print(
            "[notice] Experimental inference layout: compare boundaries on your images; "
            "same resolution does not guarantee the same predictions.",
            file=sys.stderr,
        )
    if resolved_device.type == "cpu" and any(
        model.backend == "mask2former" for model in enabled_models
    ):
        print(
            "[warning] Mask2Former on CPU is slow. "
            "Use --config configs/lightweight.json for faster tests.",
            file=sys.stderr,
        )

    failures = 0
    if image_paths:
        with SidewalkPipeline(config) as pipeline:
            for image_path in image_paths:
                print(f"[process] {image_path}")
                started_at = time.perf_counter()
                try:
                    output_dir = pipeline.process(image_path, args.output, diagnostics=args.output_mode == "full")
                    elapsed = time.perf_counter() - started_at
                    print(f"[saved]   {output_dir} ({elapsed:.1f}s)")
                    timings = pipeline.last_timings_seconds
                    print("[timing]  " + "; ".join(
                        f"{name}={seconds:.2f}s"
                        for name, seconds in timings.items()
                        if name != "total_before_metadata"
                    ))
                except Exception as error:
                    failures += 1
                    print(f"[error]   {image_path}: {error}", file=sys.stderr)
                    if args.fail_fast:
                        raise

    if video_paths:
        video_config = copy.deepcopy(config)
        if args.video_max_side > 0:
            video_config.tiling.max_image_side = args.video_max_side
        options = VideoPreviewOptions(
            frame_step=args.video_frame_step,
            max_frames=args.video_max_frames,
            preview_width=args.preview_width,
            save_video=args.save_video,
            headless=args.headless,
            temporal_smoothing=args.video_temporal_smoothing,
            keyframe_interval=args.keyframe_interval,
            work_max_side=args.video_work_max_side,
            tracking_max_side=args.video_tracking_max_side,
            log_interval=args.video_log_interval,
            postprocess_max_side=args.video_postprocess_max_side,
            cv_threads=args.video_cv_threads,
            target_fps=args.video_target_fps,
            mask_sample_step=args.video_mask_sample_step,
        )
        with SidewalkPipeline(video_config) as pipeline:
            for video_path in video_paths:
                print(f"[process-video] {video_path}")
                try:
                    saved_path = preview_video(
                        video_path,
                        pipeline,
                        args.output,
                        options,
                    )
                    if saved_path is not None:
                        print(f"[saved-video] {saved_path}")
                except Exception as error:
                    failures += 1
                    print(f"[error] {video_path}: {error}", file=sys.stderr)
                    if args.fail_fast:
                        raise

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
