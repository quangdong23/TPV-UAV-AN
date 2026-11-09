from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from road_sidewalk_seg.core.config import CompletionConfig


@dataclass(frozen=True, slots=True)
class CompletionResult:
    mask: np.ndarray
    visible_pixels: int
    retained_visible_pixels: int
    completed_pixels: int
    closing_kernel_size: int


def _odd_kernel_size(image_shape: tuple[int, int], config: CompletionConfig) -> int:
    if config.closing_kernel_ratio <= 0:
        return 0
    size = max(3, round(min(image_shape) * config.closing_kernel_ratio))
    if config.max_closing_kernel_size > 0:
        size = min(size, config.max_closing_kernel_size)
    if size % 2 == 0:
        size += 1
    return size


def _remove_small_components(
    binary_mask: np.ndarray,
    config: CompletionConfig,
) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )
    if component_count <= 1:
        return binary_mask.copy()

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = int(areas.max())
    image_area = binary_mask.shape[0] * binary_mask.shape[1]
    minimum_area = image_area * config.min_component_area_ratio
    retained = np.zeros(component_count, dtype=np.uint8)
    for component_id in range(1, component_count):
        component_area = int(stats[component_id, cv2.CC_STAT_AREA])
        if component_area < minimum_area:
            continue
        relative_area = component_area / max(largest_area, 1)
        if component_area == largest_area or relative_area >= config.min_component_relative_area:
            retained[component_id] = 1
            continue
        left = stats[component_id, cv2.CC_STAT_LEFT]
        top = stats[component_id, cv2.CC_STAT_TOP]
        width = stats[component_id, cv2.CC_STAT_WIDTH]
        height = stats[component_id, cv2.CC_STAT_HEIGHT]
        component = (labels[top : top + height, left : left + width] == component_id).astype(np.uint8)
        points = cv2.findNonZero(component)
        if points is None:
            continue
        _, (rect_width, rect_height), _ = cv2.minAreaRect(points)
        aspect_ratio = max(rect_width, rect_height) / max(
            min(rect_width, rect_height),
            1.0,
        )
        if (
            component_area == largest_area
            or relative_area >= config.min_component_relative_area
            or aspect_ratio >= config.min_component_aspect_ratio
        ):
            retained[component_id] = 1
    return retained[labels]


def _fill_small_holes(
    binary_mask: np.ndarray,
    max_hole_area: float,
) -> np.ndarray:
    output = binary_mask.copy()
    contours, hierarchy = cv2.findContours(
        output,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if hierarchy is None:
        return output
    for contour_index, contour in enumerate(contours):
        if (
            hierarchy[0][contour_index][3] != -1
            and cv2.contourArea(contour) <= max_hole_area
        ):
            cv2.drawContours(output, [contour], -1, 1, cv2.FILLED)
    return output


def complete_sidewalk_mask(
    visible_mask: np.ndarray,
    config: CompletionConfig,
) -> CompletionResult:
    sidewalk = (visible_mask == 2).astype(np.uint8)
    visible_pixels = int(sidewalk.sum())
    if not config.enabled or visible_pixels == 0:
        return CompletionResult(
            visible_mask.copy(),
            visible_pixels,
            visible_pixels,
            visible_pixels,
            0,
        )

    candidate = _remove_small_components(sidewalk, config)
    retained_visible_pixels = int(candidate.sum())
    if config.unknown_expansion_ratio > 0:
        radius = max(1, round(min(sidewalk.shape) * config.unknown_expansion_ratio))
        support_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1),
        )
        nearby = cv2.dilate(sidewalk, support_kernel)
        candidate[(visible_mask == 255) & (nearby > 0)] = 1

    kernel_size = _odd_kernel_size(sidewalk.shape, config)
    completed = candidate
    if kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        closed = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
        maximum_pixels = int(candidate.sum() * config.max_closing_growth_ratio)
        if int(closed.sum()) <= maximum_pixels:
            completed = closed

    if config.fill_holes:
        max_hole_area = completed.size * config.max_hole_area_ratio
        completed = _fill_small_holes(completed, max_hole_area)

    output = visible_mask.copy()
    output[output == 2] = 0
    output[completed > 0] = 2
    return CompletionResult(
        mask=output,
        visible_pixels=visible_pixels,
        retained_visible_pixels=retained_visible_pixels,
        completed_pixels=int(completed.sum()),
        closing_kernel_size=kernel_size,
    )
