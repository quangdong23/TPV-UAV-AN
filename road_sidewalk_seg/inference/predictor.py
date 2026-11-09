from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional

from road_sidewalk_seg.core.config import DecisionConfig, ModelConfig, PipelineConfig
from road_sidewalk_seg.models.backend import SemanticBackend, create_backend, resolve_device
from road_sidewalk_seg.inference.preprocessing import ImageVariant
from road_sidewalk_seg.inference.tiling import blending_window, extract_padded_tile, generate_windows


@dataclass(frozen=True, slots=True)
class InferenceResult:
    mask: np.ndarray
    confidence: np.ndarray
    scores: np.ndarray
    model_names: list[str]
    variant_names: list[str]
    runs: list[dict]


def tiled_predict(
    image_rgb: np.ndarray,
    backend: SemanticBackend,
    config: PipelineConfig,
    run_stats: dict | None = None,
    output_max_side: int = 0,
) -> np.ndarray:
    if output_max_side < 0:
        raise ValueError("output_max_side cannot be negative")
    original_height, original_width = image_rgb.shape[:2]
    output_shape = (original_height, original_width)
    if output_max_side > 0 and max(output_shape) > output_max_side:
        output_scale = output_max_side / max(output_shape)
        output_shape = (
            max(1, round(original_height * output_scale)),
            max(1, round(original_width * output_scale)),
        )
    max_side = config.tiling.max_image_side
    resized = max_side > 0 and max(original_height, original_width) > max_side
    if resized:
        scale = max_side / max(original_height, original_width)
        resized_height = max(1, round(original_height * scale))
        resized_width = max(1, round(original_width * scale))
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float()
        image_tensor = functional.interpolate(
            image_tensor,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        image_rgb = image_tensor[0].permute(1, 2, 0).byte().numpy()

    image_height, image_width = image_rgb.shape[:2]
    strategy = config.tiling.strategy
    padded_height = math.ceil(image_height / 32) * 32
    padded_width = math.ceil(image_width / 32) * 32
    fits_whole = padded_height * padded_width <= config.tiling.whole_image_max_pixels
    use_whole = strategy == "whole" or (strategy == "auto" and fits_whole)
    if strategy == "whole" and not fits_whole:
        raise ValueError("Whole image exceeds tiling.whole_image_max_pixels; use auto or tiled")
    windows = [] if use_whole else generate_windows(image_height, image_width, config.tiling)
    if run_stats is not None:
        run_stats.update({
            "resized_width": image_width,
            "resized_height": image_height,
            "tile_size": config.tiling.tile_size,
            "output_height": output_shape[0],
            "output_width": output_shape[1],
            "tile_count": 1 if use_whole else len(windows),
            "strategy": "whole" if use_whole else strategy if strategy == "compact" else "tiled",
            "processed_pixels": (
                padded_height * padded_width if use_whole
                else sum(
                    math.ceil(window.height / 32) * 32 * math.ceil(window.width / 32) * 32
                    for window in windows
                ) if strategy == "compact" else len(windows) * config.tiling.tile_size ** 2
            ),
        })
    if use_whole:
        padded = np.pad(
            image_rgb,
            ((0, padded_height - image_height), (0, padded_width - image_width), (0, 0)),
            mode="edge",
        )
        scores = backend.predict_target_scores(padded)[:, :image_height, :image_width]
        scores /= np.maximum(scores.sum(axis=0, keepdims=True), 1e-6)
        if scores.shape[1:] != output_shape:
            scores = functional.interpolate(
                torch.from_numpy(scores).unsqueeze(0),
                size=output_shape,
                mode="bilinear",
                align_corners=False,
            )[0].numpy()
            scores /= np.maximum(scores.sum(axis=0, keepdims=True), 1e-6)
        return scores
    tile_weights = blending_window(
        config.tiling.tile_size,
        config.tiling.weight_floor,
    )
    score_sum = np.zeros((3, image_height, image_width), dtype=np.float32)
    weight_sum = np.zeros((image_height, image_width), dtype=np.float32)

    for window in windows:
        tile = extract_padded_tile(
            image_rgb, window, config.tiling.tile_size, compact=strategy == "compact"
        )
        tile_scores = backend.predict_target_scores(tile)
        valid_scores = tile_scores[:, : window.height, : window.width]
        valid_weights = tile_weights[: window.height, : window.width]
        y_slice = slice(window.y, window.y + window.height)
        x_slice = slice(window.x, window.x + window.width)
        score_sum[:, y_slice, x_slice] += valid_scores * valid_weights[None]
        weight_sum[y_slice, x_slice] += valid_weights

    scores = score_sum / np.maximum(weight_sum[None], 1e-6)
    scores /= np.maximum(scores.sum(axis=0, keepdims=True), 1e-6)
    if scores.shape[1:] != output_shape:
        score_tensor = torch.from_numpy(scores).unsqueeze(0)
        score_tensor = functional.interpolate(
            score_tensor,
            size=output_shape,
            mode="bilinear",
            align_corners=False,
        )
        scores = score_tensor[0].numpy()
        scores /= np.maximum(scores.sum(axis=0, keepdims=True), 1e-6)
    return scores


def decide_mask(scores: np.ndarray, config: DecisionConfig) -> tuple[np.ndarray, np.ndarray]:
    if scores.shape[0] == 3:
        a,b,c=scores
        high=np.maximum(a,b)
        confidence=np.maximum(high,c)
        second=np.maximum(np.minimum(a,b),np.minimum(high,c))
        margin=confidence-second
    else:
        sorted_scores = np.sort(scores, axis=0)
        confidence = sorted_scores[-1]
        margin = sorted_scores[-1] - sorted_scores[-2]
    entropy = -(scores * np.log(np.maximum(scores, 1e-8))).sum(axis=0)
    normalized_entropy = entropy / math.log(scores.shape[0])

    winning_group = scores.argmax(axis=0)
    mask = np.where(winning_group == 2, 2, 0).astype(np.uint8)
    uncertain = (
        (confidence < config.min_confidence)
        | (margin < config.min_margin)
        | (normalized_entropy > config.max_normalized_entropy)
    )
    mask[uncertain] = np.uint8(config.unknown_id)
    return mask, confidence.astype(np.float32, copy=False)


class EnsemblePredictor:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        if resolve_device(config.device).type == "cpu" and config.cpu_threads:
            torch.set_num_threads(config.cpu_threads)
        self.backends: list[tuple[ModelConfig, SemanticBackend]] = []
        for model_config in config.models:
            if not model_config.enabled:
                continue
            backend = create_backend(
                model_config,
                device=config.device,
                precision=config.precision,
                cache_dir=config.cache_dir,
                cpu_attention=config.cpu_attention,
            )
            self.backends.append((model_config, backend))

    def predict(self, variants: list[ImageVariant], output_max_side: int = 0) -> InferenceResult:
        aggregate: np.ndarray | None = None
        total_weight = 0.0
        model_names: list[str] = []
        variant_names = [variant.name for variant in variants]
        runs = []

        for model_config, backend in self.backends:
            model_names.append(model_config.name)
            for variant in variants:
                run = {"model": model_config.name, "variant": variant.name}
                started_at = time.perf_counter()
                scores = tiled_predict(
                    variant.image, backend, self.config, run_stats=run,
                    output_max_side=output_max_side,
                )
                run["seconds"] = time.perf_counter() - started_at
                runs.append(run)
                run_weight = model_config.weight / len(variants)
                if aggregate is None:
                    aggregate = np.zeros_like(scores)
                aggregate += scores * run_weight
                total_weight += run_weight

        if aggregate is None or total_weight <= 0:
            raise RuntimeError("No enabled model produced a prediction")

        aggregate /= total_weight
        aggregate /= np.maximum(aggregate.sum(axis=0, keepdims=True), 1e-6)
        mask, confidence = decide_mask(aggregate, self.config.decision)
        return InferenceResult(
            mask=mask,
            confidence=confidence,
            scores=aggregate,
            model_names=model_names,
            variant_names=variant_names,
            runs=runs,
        )

    def close(self) -> None:
        self.backends.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "EnsemblePredictor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def run_ensemble(
    variants: list[ImageVariant],
    config: PipelineConfig,
) -> InferenceResult:
    with EnsemblePredictor(config) as predictor:
        return predictor.predict(variants)

