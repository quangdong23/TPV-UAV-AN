from __future__ import annotations

import cv2
import numpy as np

from road_sidewalk_seg.core.config import PolygonConfig


CLASS_COLORS_RGB = {
    2: np.array([235, 45, 45], dtype=np.uint8),
    255: np.array([255, 190, 30], dtype=np.uint8),
}


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.uint8:
        # Three tiny lookup tables avoid boolean indexing over every RGB pixel.
        palette = np.zeros((256, 3), dtype=np.uint8)
        for class_id, color in CLASS_COLORS_RGB.items():
            palette[class_id] = color
        return cv2.merge([cv2.LUT(mask, palette[:, channel]) for channel in range(3)])
    colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS_RGB.items():
        colored[mask == class_id] = color
    return colored


def create_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    show_unknown: bool = False,
) -> np.ndarray:
    colors = colorize_mask(mask)
    active = mask != 0 if show_unknown else mask == 2
    overlay = image_rgb.copy()
    blended = cv2.addWeighted(image_rgb, 1.0 - alpha, colors, alpha, 0.0)
    cv2.copyTo(blended, active.astype(np.uint8), overlay)
    return overlay


def _draw_polygon(
    image_rgb: np.ndarray,
    rings,
    color: tuple[int, int, int],
    config: PolygonConfig,
) -> None:
    for ring_index, ring in enumerate(rings):
        points = np.rint(np.asarray(ring, dtype=np.float32)).astype(np.int32)
        if len(points) < 3:
            continue
        cv2.polylines(image_rgb, [points], True, color, 2, cv2.LINE_AA)
        draw_vertices = config.draw_vertices and (
            ring_index == 0 or config.draw_hole_vertices
        )
        if draw_vertices:
            for x, y in points:
                cv2.circle(image_rgb, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
                cv2.circle(image_rgb, (int(x), int(y)), 2, (255, 255, 255), -1, cv2.LINE_AA)


def draw_feature_boundaries(
    image_rgb: np.ndarray,
    feature_collection: dict,
    config: PolygonConfig,
) -> np.ndarray:
    output = image_rgb.copy()
    colors = {2: (235, 45, 45)}
    for feature in feature_collection["features"]:
        class_id = int(feature["properties"]["class_id"])
        geometry = feature["geometry"]
        if geometry["type"] == "Polygon":
            polygons = [geometry["coordinates"]]
        elif geometry["type"] == "MultiPolygon":
            polygons = geometry["coordinates"]
        else:
            continue
        for rings in polygons:
            _draw_polygon(output, rings, colors[class_id], config)
    return output
