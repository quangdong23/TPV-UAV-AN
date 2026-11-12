from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_pipeline.segmentation.core.config import TilingConfig


@dataclass(frozen=True, slots=True)
class TileWindow:
    x: int
    y: int
    width: int
    height: int


def _axis_starts(length: int, tile_size: int, step: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, step))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def generate_windows(
    image_height: int,
    image_width: int,
    config: TilingConfig,
) -> list[TileWindow]:
    step = max(1, int(round(config.tile_size * (1.0 - config.overlap))))
    x_starts = _axis_starts(image_width, config.tile_size, step)
    y_starts = _axis_starts(image_height, config.tile_size, step)
    return [
        TileWindow(
            x=x,
            y=y,
            width=min(config.tile_size, image_width - x),
            height=min(config.tile_size, image_height - y),
        )
        for y in y_starts
        for x in x_starts
    ]


def extract_padded_tile(
    image_rgb: np.ndarray,
    window: TileWindow,
    tile_size: int,
    compact: bool = False,
) -> np.ndarray:
    tile = image_rgb[
        window.y : window.y + window.height,
        window.x : window.x + window.width,
    ]
    padded_height = ((window.height + 31) // 32) * 32 if compact else tile_size
    padded_width = ((window.width + 31) // 32) * 32 if compact else tile_size
    pad_bottom = padded_height - window.height
    pad_right = padded_width - window.width
    if pad_bottom == 0 and pad_right == 0:
        return tile
    return np.pad(
        tile,
        ((0, pad_bottom), (0, pad_right), (0, 0)),
        mode="edge",
    )


def blending_window(tile_size: int, weight_floor: float) -> np.ndarray:
    axis = np.hanning(tile_size).astype(np.float32)
    weights = np.outer(axis, axis)
    if float(weights.max()) > 0:
        weights /= float(weights.max())
    return np.maximum(weights, weight_floor)

