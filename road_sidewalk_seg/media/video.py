from __future__ import annotations

import os
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import transformers

from road_sidewalk_seg.tracking.motion import VideoMotionPropagator
from road_sidewalk_seg.core.pipeline import SidewalkPipeline
from road_sidewalk_seg.postprocessing.visualization import create_overlay, draw_feature_boundaries


SUPPORTED_VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def is_gui_available() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


@dataclass(frozen=True, slots=True)
class VideoPreviewOptions:
    frame_step: int = 1
    max_frames: int = 0
    preview_width: int = 1280
    save_video: bool = False
    headless: bool = False
    temporal_smoothing: float = 0.3
    keyframe_interval: int = 8
    work_max_side: int = 960
    tracking_max_side: int = 640
    log_interval: float = 2.0
    postprocess_max_side: int = 768
    cv_threads: int = 1
    target_fps: float = 15.0
    mask_sample_step: int = 30
    start_frame: int = 0


def _fit_preview(frame, max_width: int):
    if max_width <= 0 or frame.shape[1] <= max_width:
        return frame
    scale = max_width / frame.shape[1]
    size = (max_width, max(1, round(frame.shape[0] * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _fit_work_frame(frame, max_side: int):
    height, width = frame.shape[:2]
    if max_side <= 0 or max(height, width) <= max_side:
        return frame
    scale = max_side / max(height, width)
    size = (max(2, round(width * scale)), max(2, round(height * scale)))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _open_writer(path: Path, fps: float, frame_size: tuple[int, int]):
    fourcc_candidates = ["mp4v", "avc1", "XVID"]
    for fourcc_str in fourcc_candidates:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(
            str(path),
            fourcc,
            max(fps, 0.1),
            frame_size,
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"Cannot create output video: {path}")


def preview_video(
    video_path: str | Path,
    pipeline: SidewalkPipeline,
    output_root: str | Path,
    options: VideoPreviewOptions,
) -> Path | None:
    if options.frame_step < 1 or options.keyframe_interval < 1 or min(options.max_frames, options.start_frame) < 0:
        raise ValueError("Invalid video frame sampling options")
    if min(options.work_max_side, options.tracking_max_side, options.log_interval,
           options.postprocess_max_side, options.cv_threads, options.target_fps,
           options.mask_sample_step) < 0:
        raise ValueError("Video size/log options cannot be negative")
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 25.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if options.start_frame:
        if (total_frames > 0 and options.start_frame >= total_frames) or not capture.set(cv2.CAP_PROP_POS_FRAMES, options.start_frame):
            capture.release()
            raise ValueError("Cannot seek to requested start frame")
    output_dir = Path(output_root) / path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "preview.mp4"
    writer = None
    window_name = f"Sidewalk preview - {path.name}"
    sampled_frames = 0
    frame_index = options.start_frame

    is_headless = options.headless or not is_gui_available()
    save_video = options.save_video or is_headless

    mode_label = "headless (no GUI)" if is_headless else "interactive GUI"
    tracking_info = (
        f"AI every {options.keyframe_interval} frame(s) + motion warp"
        if options.keyframe_interval > 1
        else "full AI every frame"
    )
    print(
        f"[video] {path.name} | frames={total_frames or '?'}; fps={source_fps:.2f}; "
        f"step={options.frame_step}; {tracking_info}; mode={mode_label}"
    )
    if options.work_max_side:
        print(f"[video] work/output cap={options.work_max_side}; model cap={pipeline.config.tiling.max_image_side}")
    if options.temporal_smoothing > 0 and options.keyframe_interval <= 1:
        print("[video] Temporal smoothing requires motion alignment; disabled for AI-every-frame mode.")
    if not is_headless:
        print("[keys]  Q/Esc: stop | Space: pause | S: save current frame")
    if save_video:
        print(f"[output] Saving result to: {output_path}")

    prev_scores = None
    stage_seconds = Counter()
    ai_stage_seconds = Counter()
    counts = Counter()
    fallback_reasons = Counter()
    frame_latencies = []
    ai_latencies = []
    last_log_at = 0.0
    work_shape = None
    original_cv_threads = cv2.getNumThreads()
    if options.cv_threads:
        cv2.setNumThreads(options.cv_threads)
    active_cv_threads = cv2.getNumThreads()
    loop_started_at = time.perf_counter()
    propagator = (
        VideoMotionPropagator(
            keyframe_interval=options.keyframe_interval,
            tracking_max_side=options.tracking_max_side,
        )
        if options.keyframe_interval > 1
        else None
    )

    try:
        while True:
            iteration_started_at = time.perf_counter()
            if options.frame_step > 1 and sampled_frames > 0:
                for _ in range(options.frame_step - 1):
                    if not capture.grab():
                        break
                    frame_index += 1

            success, frame_bgr = capture.read()
            stage_seconds["decode_and_skip"] += time.perf_counter() - iteration_started_at
            if not success:
                break

            started_at = time.perf_counter()
            frame_bgr = _fit_work_frame(frame_bgr, options.work_max_side)
            work_shape = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            stage_seconds["working_frame"] += time.perf_counter() - started_at

            run_ai = propagator is None or propagator.should_run_ai()
            warped_mask, warped_features = None, None
            aligned_scores = None
            if propagator is not None and propagator.last_mask is not None:
                if not run_ai or options.temporal_smoothing > 0:
                    motion_started_at = time.perf_counter()
                    warped_mask, warped_features = propagator.propagate(frame_rgb)
                    if warped_mask is not None and prev_scores is not None:
                        if run_ai and options.temporal_smoothing > 0:
                            aligned_scores = propagator.warp_keyframe_scores(prev_scores)
                    stage_seconds["motion_and_alignment"] += time.perf_counter() - motion_started_at
                    if warped_mask is None:
                        fallback_reasons[propagator.last_reason] += 1
                        if not run_ai:
                            counts["ai_fallbacks"] += 1
                        run_ai = True
            if run_ai:
                ai_started_at = time.perf_counter()
                process_options = {
                    "prev_scores": aligned_scores,
                    "temporal_smoothing": options.temporal_smoothing,
                }
                if options.postprocess_max_side:
                    process_options["postprocess_max_side"] = options.postprocess_max_side
                result = pipeline.process_frame(frame_rgb, **process_options)
                ai_elapsed = time.perf_counter() - ai_started_at
                stage_seconds["ai_pipeline"] += ai_elapsed
                ai_latencies.append(ai_elapsed)
                ai_stage_seconds.update(result.timings_seconds)
                counts["ai_frames"] += 1
                counts["model_forward_calls"] += sum(run["tile_count"] for run in result.inference.runs)
                prev_scores = result.inference.scores if options.temporal_smoothing > 0 and propagator is not None else None
                mask = result.mask
                overlay_rgb = result.overlay_rgb
                sidewalk_px = result.completion.completed_pixels
                features = result.pixel_features
                if propagator is not None:
                    motion_started_at = time.perf_counter()
                    propagator.update_keyframe(frame_rgb, mask, features)
                    stage_seconds["motion_and_alignment"] += time.perf_counter() - motion_started_at
                mode_tag = "AI"
            else:
                render_started_at = time.perf_counter()
                mask = warped_mask
                counts["warped_frames"] += 1
                sidewalk_px = int(np.count_nonzero(mask == 2))
                overlay_rgb = create_overlay(
                    frame_rgb, mask, alpha=pipeline.config.output.overlay_alpha,
                    show_unknown=pipeline.config.output.show_unknown,
                )
                if warped_features is not None and warped_features.get("features"):
                    overlay_rgb = draw_feature_boundaries(
                        overlay_rgb, warped_features, pipeline.config.polygon,
                    )
                mode_tag = "Warp"
                stage_seconds["render"] += time.perf_counter() - render_started_at

            render_started_at = time.perf_counter()
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            elapsed = time.perf_counter() - started_at

            status = (
                f"frame {frame_index}/{max(total_frames - 1, 0)} [{mode_tag}] | "
                f"inference {elapsed:.2f}s | sidewalk "
                f"{sidewalk_px:,} px"
            )
            cv2.rectangle(overlay_bgr, (0, 0), (overlay_bgr.shape[1], 42), (0, 0, 0), -1)
            cv2.putText(
                overlay_bgr,
                status,
                (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            stage_seconds["render"] += time.perf_counter() - render_started_at

            if save_video:
                encode_started_at = time.perf_counter()
                pad_bottom, pad_right = overlay_bgr.shape[0] % 2, overlay_bgr.shape[1] % 2
                encoded_frame = cv2.copyMakeBorder(
                    overlay_bgr, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE
                ) if pad_bottom or pad_right else overlay_bgr
                if writer is None:
                    height, width = encoded_frame.shape[:2]
                    writer = _open_writer(output_path, source_fps / options.frame_step, (width, height))
                writer.write(encoded_frame)
                stage_seconds["encode"] += time.perf_counter() - encode_started_at

            sampled_frames += 1
            counts[mode_tag] += 1
            if options.mask_sample_step > 0 and frame_index % options.mask_sample_step == 0:
                audit_started_at = time.perf_counter()
                audit_dir = output_dir / "mask_samples"
                audit_dir.mkdir(exist_ok=True)
                if not cv2.imwrite(str(audit_dir / f"{frame_index:08d}.png"), mask):
                    raise RuntimeError("Cannot write video mask sample")
                stage_seconds["mask_sample_export"] += time.perf_counter() - audit_started_at

            log_started_at = time.perf_counter()
            if is_headless and (
                sampled_frames == 1 or log_started_at - last_log_at >= options.log_interval
            ):
                total_elapsed = time.perf_counter() - loop_started_at
                avg_fps = sampled_frames / max(total_elapsed, 1e-4)
                if total_frames > 0:
                    pct = (frame_index / max(total_frames, 1)) * 100
                    remaining = (
                        (total_frames - frame_index) / (avg_fps * options.frame_step)
                        if avg_fps > 0
                        else 0
                    )
                    print(
                        f"[frame] {frame_index}/{total_frames} ({pct:.1f}%) | "
                        f"{elapsed:.2f}s | {avg_fps:.1f} fps | ETA {remaining:.0f}s"
                    )
                else:
                    print(f"[frame] {frame_index} | {elapsed:.2f}s | {avg_fps:.1f} fps")
                last_log_at = log_started_at
            elif not is_headless:
                try:
                    cv2.imshow(window_name, _fit_preview(overlay_bgr, options.preview_width))

                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        break
                    if key in (ord("s"), ord("S")):
                        screenshot_path = output_dir / f"frame_{frame_index:08d}.png"
                        cv2.imwrite(str(screenshot_path), overlay_bgr)
                        print(f"[saved] {screenshot_path}")
                    if key == ord(" "):
                        while True:
                            pause_key = cv2.waitKey(50) & 0xFF
                            if pause_key in (27, ord("q"), ord("Q"), ord(" ")):
                                key = pause_key
                                break
                        if key in (27, ord("q"), ord("Q")):
                            break
                except cv2.error:
                    is_headless = True
                    save_video = True
                    print("[video] Display unavailable, switched to headless mode.")

            stage_seconds["logging_and_display"] += time.perf_counter() - log_started_at
            frame_latencies.append(time.perf_counter() - iteration_started_at)

            if options.max_frames > 0 and sampled_frames >= options.max_frames:
                break
            frame_index += 1
    finally:
        capture.release()
        writer_error = None
        if writer is not None:
            finalize_started_at = time.perf_counter()
            try:
                writer.release()
            except Exception as error:
                writer_error = error
            stage_seconds["encode"] += time.perf_counter() - finalize_started_at
        if not is_headless:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass

        wall_seconds = time.perf_counter() - loop_started_at
        cv2.setNumThreads(original_cv_threads)
        source_seconds = total_frames / source_fps if total_frames > 0 else None
        processed_seconds = (
            max(0, min(total_frames, frame_index + 1) - options.start_frame) / source_fps if total_frames > 0
            else sampled_frames * options.frame_step / source_fps
        ) if sampled_frames else 0.0
        runtime = []
        if pipeline._predictor is not None:
            for model_config, backend in pipeline._predictor.backends:
                runtime.append({
                    "model": model_config.model_id, "device": str(backend.device),
                    "precision": backend.precision,
                    "gpu_name": torch.cuda.get_device_name(backend.device) if backend.device.type == "cuda" else None,
                })
        summary = {
            "source": str(path), "source_fps": source_fps,
            "source_frames": total_frames, "source_seconds": source_seconds,
            "processed_frames": sampled_frames, "processed_source_seconds": processed_seconds,
            "frame_step": options.frame_step, "keyframe_interval": options.keyframe_interval,
            "start_frame": options.start_frame,
            "work_max_side": options.work_max_side,
            "working_shape": list(work_shape) if work_shape else None,
            "tracking_max_side": options.tracking_max_side,
            "postprocess_max_side": options.postprocess_max_side,
            "cv_threads": active_cv_threads,
            "wall_seconds": wall_seconds,
            "processed_fps": sampled_frames / max(wall_seconds, 1e-6),
            "target_processed_fps": options.target_fps,
            "throughput_target_met": (
                sampled_frames / max(wall_seconds, 1e-6) >= options.target_fps
                if sampled_frames and options.target_fps > 0 else None
            ),
            "all_source_frames_processed": (
                total_frames > 0 and options.start_frame == 0
                and options.frame_step == 1 and sampled_frames >= total_frames
            ),
            "frame_skipping_enabled": options.frame_step > 1,
            "ai_updates_per_wall_second": counts["ai_frames"] / max(wall_seconds, 1e-6),
            "ai_results_applied": counts["AI"],
            "ai_applied_updates_per_wall_second": counts["AI"] / max(wall_seconds, 1e-6),
            "quality_verified": False,
            "writer_error": str(writer_error) if writer_error else None,
            "mask_sample_step": options.mask_sample_step,
            "warm_processed_fps_excluding_first_frame": (
                (sampled_frames - 1) / max(wall_seconds - frame_latencies[0], 1e-6)
                if sampled_frames > 1 and frame_latencies else None
            ),
            "speed_vs_source": processed_seconds / max(wall_seconds, 1e-6),
            "frame_latency_p50_seconds": float(np.percentile(frame_latencies, 50)) if frame_latencies else None,
            "frame_latency_p95_seconds": float(np.percentile(frame_latencies, 95)) if frame_latencies else None,
            "ai_latency_p50_seconds": float(np.percentile(ai_latencies, 50)) if ai_latencies else None,
            "counts": dict(counts), "fallback_reasons": dict(fallback_reasons),
            "stage_seconds": dict(stage_seconds), "ai_stage_seconds": dict(ai_stage_seconds),
            "runtime": runtime, "config": pipeline.config.to_dict(),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "opencv": cv2.__version__},
            "temporal_smoothing": options.temporal_smoothing,
            "timing_note": "Wall-clock includes model loading, decoding and complete encoding; excludes notebook setup and browser playback. Frame latency is loop service time, not live end-to-end latency.",
        }
        (output_dir / "video_profile.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[summary] {sampled_frames} frames in {wall_seconds:.2f}s; "
            f"{summary['processed_fps']:.2f} processed fps; "
            f"AI={counts['ai_frames']}, warp={counts['warped_frames']}, "
            f"fallback={counts['ai_fallbacks']}; profile={output_dir / 'video_profile.json'}"
        )
        if writer_error is not None:
            raise RuntimeError("Video encoding failed") from writer_error

    return output_path if writer is not None else None
