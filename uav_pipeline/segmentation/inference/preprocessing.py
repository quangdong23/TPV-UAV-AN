from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from uav_pipeline.segmentation.core.config import PreprocessingConfig


@dataclass(frozen=True, slots=True)
class ImageVariant:
    name: str
    image: np.ndarray


def luminance_score(image_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    if gray.dtype == np.uint8 and gray.size <= 2**24:
        # Exact uint8 median without partitioning/copying the image. Float32
        # histogram counters are exact up to 2**24 source pixels.
        cumulative = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().cumsum()
        lower = np.searchsorted(cumulative, (gray.size-1)//2 + 1)
        upper = np.searchsorted(cumulative, gray.size//2 + 1)
        return float((int(lower)+int(upper))/2.0/255.0)
    return float(np.median(gray) / 255.0)


def apply_luminance_clahe(
    image_rgb: np.ndarray,
    clip_limit: float,
    grid_size: int,
) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_size, grid_size),
    )
    enhanced = cv2.merge((clahe.apply(lightness), channel_a, channel_b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def build_variants(
    image_rgb: np.ndarray,
    config: PreprocessingConfig,
) -> tuple[list[ImageVariant], float]:
    score = luminance_score(image_rgb)
    variants = [ImageVariant(name="raw", image=image_rgb)]

    use_low_light = config.low_light_mode == "always" or (
        config.low_light_mode == "auto" and score < config.low_light_threshold
    )
    if use_low_light:
        variants.append(
            ImageVariant(
                name="clahe",
                image=apply_luminance_clahe(
                    image_rgb,
                    clip_limit=config.clahe_clip_limit,
                    grid_size=config.clahe_grid_size,
                ),
            )
        )
    return variants, score

