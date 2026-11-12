from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union

from uav_pipeline.segmentation.core.config import CompletionConfig, PolygonConfig
from uav_pipeline.segmentation.models.taxonomy import TARGET_LABELS

try:
    from shapely import concave_hull, make_valid
except ImportError:
    concave_hull = None
    from shapely.validation import make_valid


def _remove_small_components(binary_mask: np.ndarray, min_area: float) -> np.ndarray:
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )
    retained = (stats[:, cv2.CC_STAT_AREA] >= min_area).astype(np.uint8)
    retained[0] = 0
    return retained[labels]


def _polygon_parts(geometry) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    elif isinstance(geometry, GeometryCollection):
        for part in geometry.geoms:
            yield from _polygon_parts(part)


def _contour_points(contour: np.ndarray) -> list[tuple[float, float]]:
    return [(float(point[0][0]), float(point[0][1])) for point in contour]


def binary_mask_to_geometry(
    binary_mask: np.ndarray,
    config: PolygonConfig,
):
    mask = (binary_mask > 0).astype(np.uint8)
    image_area = float(mask.shape[0] * mask.shape[1])
    min_area = max(config.min_area_pixels, image_area * config.min_area_ratio)
    min_hole_area = max(
        config.min_hole_area_pixels,
        image_area * config.min_hole_area_ratio,
    )
    if config.close_kernel_size > 0:
        size = config.close_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = _remove_small_components(mask, min_area)

    contours, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_NONE,
    )
    if hierarchy is None:
        return None

    polygons: list[Polygon] = []
    hierarchy_rows = hierarchy[0]
    for contour_index, contour in enumerate(contours):
        if hierarchy_rows[contour_index][3] != -1:
            continue
        if cv2.contourArea(contour) < min_area or len(contour) < 3:
            continue

        holes: list[list[tuple[float, float]]] = []
        child_index = int(hierarchy_rows[contour_index][2])
        while child_index != -1:
            child = contours[child_index]
            if cv2.contourArea(child) >= min_hole_area and len(child) >= 3:
                holes.append(_contour_points(child))
            child_index = int(hierarchy_rows[child_index][0])

        polygon = Polygon(_contour_points(contour), holes)
        if not polygon.is_valid:
            polygon = make_valid(polygon)
        for part in _polygon_parts(polygon):
            tolerance = max(0.0, part.length * config.simplify_epsilon_ratio)
            simplified = part.simplify(tolerance, preserve_topology=True)
            if not simplified.is_empty and simplified.area >= min_area:
                polygons.extend(_polygon_parts(simplified))

    if not polygons:
        return None
    merged = make_valid(unary_union(polygons))
    parts = list(_polygon_parts(merged))
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def _complete_geometry_envelope(
    geometry,
    config: CompletionConfig | None,
):
    if (
        config is None
        or not config.enabled
        or not config.envelope_enabled
        or concave_hull is None
    ):
        return geometry

    completed_parts: list[Polygon] = []
    for part in _polygon_parts(geometry):
        candidate = concave_hull(
            part,
            ratio=config.concave_hull_ratio,
            allow_holes=False,
        )
        candidate = make_valid(unary_union([part, candidate]))
        candidate_parts = list(_polygon_parts(candidate))
        candidate_area = sum(candidate_part.area for candidate_part in candidate_parts)
        maximum_area = part.area * config.max_envelope_growth_ratio
        if candidate_parts and candidate_area <= maximum_area:
            completed_parts.extend(candidate_parts)
        else:
            completed_parts.append(part)

    if not completed_parts:
        return geometry
    merged = make_valid(unary_union(completed_parts))
    parts = list(_polygon_parts(merged))
    if not parts:
        return geometry
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def mask_to_feature_collection(
    mask: np.ndarray,
    config: PolygonConfig,
    completion_config: CompletionConfig | None = None,
) -> dict:
    features = []
    class_id = 2
    geometry = binary_mask_to_geometry(mask == class_id, config)
    if geometry is not None and not geometry.is_empty:
        source_area = float(geometry.area)
        geometry = _complete_geometry_envelope(geometry, completion_config)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "class_id": class_id,
                    "class_name": TARGET_LABELS[class_id],
                    "area_pixels": float(geometry.area),
                    "source_area_pixels": source_area,
                    "envelope_growth_ratio": float(
                        geometry.area / max(source_area, 1.0)
                    ),
                },
                "geometry": mapping(geometry),
            }
        )
    return {"type": "FeatureCollection", "features": features}


def feature_collection_to_mask(
    feature_collection: dict,
    base_mask: np.ndarray,
) -> np.ndarray:
    output = base_mask.copy()
    output[output == 2] = 0

    for feature in feature_collection["features"]:
        class_id = int(feature["properties"]["class_id"])
        geometry = shape(feature["geometry"])
        for part in _polygon_parts(geometry):
            exterior = np.rint(np.asarray(part.exterior.coords)).astype(np.int32)
            if len(exterior) >= 3:
                cv2.fillPoly(output, [exterior], class_id)
            for interior in part.interiors:
                hole = np.rint(np.asarray(interior.coords)).astype(np.int32)
                if len(hole) >= 3:
                    cv2.fillPoly(output, [hole], 0)
    return output
