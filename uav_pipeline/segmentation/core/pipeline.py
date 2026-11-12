from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from uav_pipeline.segmentation.postprocessing.completion import CompletionResult, complete_sidewalk_mask
from uav_pipeline.segmentation.core.config import PipelineConfig
from uav_pipeline.segmentation.gis.georeference import georeference_feature_collection
from uav_pipeline.segmentation.media.image_io import load_image
from uav_pipeline.segmentation.inference.predictor import EnsemblePredictor, InferenceResult, decide_mask
from uav_pipeline.segmentation.postprocessing.polygonization import feature_collection_to_mask, mask_to_feature_collection
from uav_pipeline.segmentation.inference.preprocessing import build_variants
from uav_pipeline.segmentation.postprocessing.visualization import create_overlay, draw_feature_boundaries
from uav_pipeline.segmentation.tracking.motion import warp_feature_collection


@dataclass(frozen=True, slots=True)
class FrameResult:
    overlay_rgb: np.ndarray
    mask: np.ndarray
    confidence: np.ndarray
    pixel_features: dict
    inference: InferenceResult
    completion: CompletionResult
    luminance_score: float
    morphological_sidewalk_pixels: int
    timings_seconds: dict[str, float]


class SidewalkPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        config.validate()
        self.config = config
        self._predictor: EnsemblePredictor | None = None
        self.last_timings_seconds: dict[str, float] = {}

    def _get_predictor(self) -> EnsemblePredictor:
        if self._predictor is None:
            self._predictor = EnsemblePredictor(self.config)
        return self._predictor

    def close(self) -> None:
        if self._predictor is not None:
            self._predictor.close()
            self._predictor = None

    def __enter__(self) -> "SidewalkPipeline":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def process_frame(
        self,
        image_rgb: np.ndarray,
        prev_scores: np.ndarray | None = None,
        temporal_smoothing: float = 0.0,
        fast_video: bool = False,
        postprocess_max_side: int = 0,
        render_overlay: bool = True,
    ) -> FrameResult:
        if postprocess_max_side < 0:
            raise ValueError("postprocess_max_side cannot be negative")
        started_at = time.perf_counter()
        variants, light_score = build_variants(
            image_rgb,
            self.config.preprocessing,
        )
        preprocessed_at = time.perf_counter()
        predictor = self._get_predictor()
        loaded_at = time.perf_counter()
        defer_decision = prev_scores is not None and 0.0 < temporal_smoothing < 1.0
        predict_options = {'defer_decision': True} if defer_decision else {}
        if postprocess_max_side > 0:
            predict_options['output_max_side'] = postprocess_max_side
        result = predictor.predict(variants, **predict_options)
        inferred_at = time.perf_counter()

        if (
            prev_scores is not None
            and 0.0 < temporal_smoothing < 1.0
            and prev_scores.shape == result.scores.shape
        ):
            alpha = float(temporal_smoothing)
            smoothed_scores = alpha * prev_scores + (1.0 - alpha) * result.scores
            smoothed_scores /= np.maximum(smoothed_scores.sum(axis=0, keepdims=True), 1e-6)
            mask, confidence = decide_mask(smoothed_scores, self.config.decision)
            result = InferenceResult(
                mask=mask,
                confidence=confidence,
                scores=smoothed_scores,
                model_names=result.model_names,
                variant_names=result.variant_names,
                runs=result.runs,
            )

        if result.mask is None:
            # A changed score shape prevents smoothing, but still needs a mask.
            mask, confidence = decide_mask(result.scores, self.config.decision)
            result = replace(result, mask=mask, confidence=confidence)

        frame_height, frame_width = image_rgb.shape[:2]
        mask_height, mask_width = result.mask.shape
        compact = (mask_height, mask_width) != (frame_height, frame_width)
        area_scale = (mask_height * mask_width) / (frame_height * frame_width)
        completion_config = self.config.completion
        polygon_config = self.config.polygon
        if compact:
            completion_config = replace(
                completion_config,
                max_closing_kernel_size=(
                    max(3, round(completion_config.max_closing_kernel_size * min(mask_height / frame_height, mask_width / frame_width)))
                    if completion_config.max_closing_kernel_size else 0
                ),
            )
            polygon_config = replace(
                polygon_config,
                min_area_pixels=polygon_config.min_area_pixels * area_scale,
                min_hole_area_pixels=polygon_config.min_hole_area_pixels * area_scale,
                close_kernel_size=(
                    max(1, round(polygon_config.close_kernel_size * min(mask_height / frame_height, mask_width / frame_width)))
                    if polygon_config.close_kernel_size else 0
                ),
            )
        completion = complete_sidewalk_mask(result.mask, completion_config)
        morphological_pixels = completion.completed_pixels
        pixel_features = mask_to_feature_collection(
            completion.mask,
            polygon_config,
            completion_config,
        )
        base_mask = completion.mask
        if compact:
            matrix = np.array([
                [frame_width / mask_width, 0, 0], [0, frame_height / mask_height, 0],
            ], dtype=np.float32)
            pixel_features = warp_feature_collection(pixel_features, matrix)
            for feature in pixel_features["features"]:
                for key in ("area_pixels", "source_area_pixels"):
                    if key in feature["properties"]:
                        feature["properties"][key] /= area_scale
            base_mask = cv2.resize(base_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
            morphological_pixels = int(np.count_nonzero(base_mask == 2))
        completed_mask = feature_collection_to_mask(
            pixel_features,
            base_mask,
        )
        completion = CompletionResult(
            mask=completed_mask,
            visible_pixels=round(completion.visible_pixels / area_scale),
            retained_visible_pixels=round(completion.retained_visible_pixels / area_scale),
            completed_pixels=int(np.count_nonzero(completed_mask == 2)),
            closing_kernel_size=completion.closing_kernel_size,
        )
        postprocessed_at = time.perf_counter()
        overlay = create_overlay(
            image_rgb,
            completion.mask,
            alpha=self.config.output.overlay_alpha,
            show_unknown=self.config.output.show_unknown,
        ) if render_overlay else image_rgb
        overlay = draw_feature_boundaries(
            overlay,
            pixel_features,
            self.config.polygon,
        ) if render_overlay else overlay
        return FrameResult(
            overlay_rgb=overlay,
            mask=completion.mask,
            confidence=(
                cv2.resize(result.confidence, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
                if compact else result.confidence
            ),
            pixel_features=pixel_features,
            inference=result,
            completion=completion,
            luminance_score=light_score,
            morphological_sidewalk_pixels=morphological_pixels,
            timings_seconds={
                "preprocessing": preprocessed_at - started_at,
                "model_load": loaded_at - preprocessed_at,
                "inference": inferred_at - loaded_at,
                "postprocessing": postprocessed_at - inferred_at,
                "overlay": time.perf_counter() - postprocessed_at,
            },
        )

    def process(self, image_path: str | Path, output_root: str | Path, diagnostics: bool = True) -> Path:
        started_at = time.perf_counter()
        record = load_image(image_path)
        read_seconds = time.perf_counter() - started_at
        output_dir = Path(output_root) / record.path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        frame = self.process_frame(record.image_rgb)
        result = frame.inference
        completion = frame.completion
        pixel_features = frame.pixel_features
        morphological_pixels = frame.morphological_sidewalk_pixels

        save_started_at = time.perf_counter()
        self._save_raster_outputs(
            result,
            completion,
            frame.overlay_rgb,
            output_dir,
            diagnostics=diagnostics,
        )
        self._write_json(output_dir / "polygons_pixel.geojson", pixel_features)

        geospatial_path = None
        if record.georeference is not None:
            geospatial_features = georeference_feature_collection(
                pixel_features,
                record.georeference,
                target_crs=self.config.output.target_crs,
            )
            geospatial_path = output_dir / "polygons_geo.geojson"
            self._write_json(geospatial_path, geospatial_features)

        predictor = self._get_predictor()
        timings = dict(frame.timings_seconds)
        timings["read"] = read_seconds
        timings["save_and_georeference"] = time.perf_counter() - save_started_at
        timings["total_before_metadata"] = time.perf_counter() - started_at
        self.last_timings_seconds = timings
        metadata = {
            "source": str(record.path.resolve()),
            "image_width": int(record.image_rgb.shape[1]),
            "image_height": int(record.image_rgb.shape[0]),
            "luminance_score": frame.luminance_score,
            "variants": result.variant_names,
            "models": result.model_names,
            "device": self.config.device,
            "precision": self.config.precision,
            "runtime": [
                {
                    "model": model.name, "device": str(backend.device),
                    "precision": backend.precision, "cpu_attention": backend.cpu_attention,
                    "cpu_attention_modules": backend.cpu_attention_modules,
                }
                for model, backend in predictor.backends
            ],
            "timings_seconds": timings,
            "inference_runs": result.runs,
            "cpu_threads": torch.get_num_threads(),
            "target_classes": ["sidewalk"],
            "visible_sidewalk_pixels": completion.visible_pixels,
            "retained_visible_sidewalk_pixels": completion.retained_visible_pixels,
            "removed_component_pixels": (
                completion.visible_pixels - completion.retained_visible_pixels
            ),
            "morphological_sidewalk_pixels": morphological_pixels,
            "completed_sidewalk_pixels": completion.completed_pixels,
            "completion_growth_ratio": (
                completion.completed_pixels / max(completion.visible_pixels, 1)
            ),
            "completion_kernel_size": completion.closing_kernel_size,
            "envelope_growth_ratio": (
                completion.completed_pixels / max(morphological_pixels, 1)
            ),
            "georeferenced": record.georeference is not None,
            "geospatial_output": str(geospatial_path) if geospatial_path else None,
            "config": self.config.to_dict(),
        }
        self._write_json(output_dir / "metadata.json", metadata)
        return output_dir

    def _save_raster_outputs(
        self,
        result: InferenceResult,
        completion: CompletionResult,
        overlay_rgb: np.ndarray,
        output_dir: Path,
        diagnostics: bool = True,
    ) -> None:
        png_options = {"compress_level": self.config.output.png_compress_level}
        if diagnostics:
            Image.fromarray(result.mask, mode="L").save(output_dir / "visible_mask.png", **png_options)
        Image.fromarray(completion.mask, mode="L").save(output_dir / "mask.png", **png_options)
        if diagnostics:
            confidence_image = np.rint(result.confidence * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(confidence_image, mode="L").save(output_dir / "confidence.png", **png_options)

        Image.fromarray(overlay_rgb, mode="RGB").save(output_dir / "overlay.png", **png_options)

        if self.config.output.save_scores:
            np.savez_compressed(
                output_dir / "scores.npz",
                other=result.scores[0] + result.scores[1],
                sidewalk=result.scores[2],
            )

    @staticmethod
    def _write_json(path: Path, content: dict) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(content, file, ensure_ascii=False, indent=2)
