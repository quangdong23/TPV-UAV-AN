from __future__ import annotations

import copy

import cv2
import numpy as np
from uav_pipeline.tracking.motion import warp_label_mask


def _warp_ring(ring: list, matrix: np.ndarray) -> list:
    pts = np.asarray(ring, dtype=np.float32)
    if len(pts) == 0:
        return ring
    warped = cv2.transform(pts.reshape(1, -1, 2), matrix).reshape(-1, 2)
    return warped.tolist()


def _warp_geometry(geometry: dict, matrix: np.ndarray) -> dict:
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if geom_type == "Polygon":
        new_coords = [_warp_ring(ring, matrix) for ring in coords]
        return {"type": "Polygon", "coordinates": new_coords}
    elif geom_type == "MultiPolygon":
        new_coords = [[_warp_ring(ring, matrix) for ring in poly] for poly in coords]
        return {"type": "MultiPolygon", "coordinates": new_coords}
    return geometry


def warp_feature_collection(feature_collection: dict, matrix: np.ndarray) -> dict:
    new_features = []
    for feat in feature_collection.get("features", []):
        new_feat = {
            "type": "Feature",
            "properties": feat.get("properties", {}),
            "geometry": _warp_geometry(feat.get("geometry", {}), matrix),
        }
        new_features.append(new_feat)
    return {"type": "FeatureCollection", "features": new_features}


class VideoMotionPropagator:
    """
    Camera-motion propagation with quality checks and periodic AI refresh.
    """

    def __init__(
        self, keyframe_interval: int = 1, max_features: int = 300,
        tracking_max_side: int = 640,
    ) -> None:
        if keyframe_interval < 1 or tracking_max_side < 0 or max_features < 12:
            raise ValueError("Invalid motion tracking options")
        self.keyframe_interval = max(1, keyframe_interval)
        self.max_features = max_features
        self.tracking_max_side = tracking_max_side
        self.prev_gray: np.ndarray | None = None
        self.last_mask: np.ndarray | None = None
        self.last_features: dict | None = None
        self.frames_since_keyframe: int = 0
        self.anchor_mask: np.ndarray | None = None
        self.anchor_features: dict | None = None
        self.cumulative_matrix = np.eye(3, dtype=np.float64)
        self.points: np.ndarray | None = None
        self.frame_shape: tuple[int, int] | None = None
        self.last_reason = "initial"
        self.last_metrics: dict[str, float] = {}

    def _gray(self, frame_rgb: np.ndarray) -> np.ndarray:
        height, width = frame_rgb.shape[:2]
        if self.tracking_max_side and max(height, width) > self.tracking_max_side:
            scale = self.tracking_max_side / max(height, width)
            frame_rgb = cv2.resize(
                frame_rgb, (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    def _reject(self, reason: str):
        self.last_reason = reason
        return None, None

    def should_run_ai(self) -> bool:
        if self.keyframe_interval <= 1:
            return True
        if self.last_mask is None or self.prev_gray is None:
            return True
        return self.frames_since_keyframe >= self.keyframe_interval - 1

    def update_keyframe(
        self,
        frame_rgb: np.ndarray,
        mask: np.ndarray,
        features: dict | None = None,
    ) -> None:
        if mask.shape != frame_rgb.shape[:2]:
            raise ValueError("Motion mask and frame dimensions must match")
        self.prev_gray = self._gray(frame_rgb)
        self.last_mask = mask.copy()
        self.last_features = copy.deepcopy(features) if features is not None else None
        self.anchor_mask = self.last_mask
        self.anchor_features = self.last_features
        self.frame_shape = frame_rgb.shape[:2]
        self.cumulative_matrix = np.eye(3, dtype=np.float64)
        self.points = None
        self.frames_since_keyframe = 0
        self.last_reason = "keyframe"
        self.last_metrics = {}

    def warp_keyframe_scores(self, scores: np.ndarray) -> np.ndarray | None:
        if self.frame_shape is None or scores.ndim != 3:
            return None
        height, width = scores.shape[1:]
        frame_height, frame_width = self.frame_shape
        frame_to_scores = np.diag([width / frame_width, height / frame_height, 1.0])
        matrix = (frame_to_scores @ self.cumulative_matrix @ np.linalg.inv(frame_to_scores))[:2]
        return np.stack([
            cv2.warpAffine(
                channel, matrix, (width, height),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            for channel in scores
        ])

    def propagate(self, frame_rgb: np.ndarray) -> tuple[np.ndarray | None, dict | None]:
        if self.prev_gray is None or self.last_mask is None:
            return None, None

        self.last_metrics = {}
        if frame_rgb.shape[:2] != self.frame_shape:
            return self._reject("frame_size_changed")
        curr_gray = self._gray(frame_rgb)
        difference = float(np.mean(cv2.absdiff(self.prev_gray, curr_gray))) / 255.0
        if difference > 0.25:
            return self._reject("scene_change")
        p0 = self.points
        if p0 is None or len(p0) < max(24, self.max_features // 3):
            p0 = cv2.goodFeaturesToTrack(
                self.prev_gray, maxCorners=self.max_features, qualityLevel=0.01,
                minDistance=10, blockSize=7,
            )
        if p0 is None or len(p0) < 12:
            return self._reject("too_few_corners")

        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            curr_gray,
            p0,
            None,
            winSize=(21, 21),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if p1 is None or status is None:
            return self._reject("flow_failed")

        backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, self.prev_gray, p1, None, winSize=(21, 21), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if backward is None or backward_status is None:
            return self._reject("backward_flow_failed")

        valid = (
            (status.reshape(-1) == 1) & (backward_status.reshape(-1) == 1)
            & (np.linalg.norm((p0 - backward).reshape(-1, 2), axis=1) <= 1.0)
            & np.isfinite(p1.reshape(-1, 2)).all(axis=1)
        )
        good_prev = p0[valid]
        good_curr = p1[valid]
        if len(good_prev) < 12:
            return self._reject("too_few_consistent_tracks")

        matrix, inliers = cv2.estimateAffinePartial2D(
            good_prev,
            good_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
        )
        if matrix is None or inliers is None or not np.isfinite(matrix).all():
            return self._reject("affine_failed")

        inlier_mask = inliers.reshape(-1).astype(bool)
        inlier_ratio = float(inlier_mask.mean())
        if inlier_mask.sum() < 12 or inlier_ratio < 0.65:
            return self._reject("low_inlier_ratio")
        projected = cv2.transform(good_prev, matrix)
        residual = np.linalg.norm((projected - good_curr).reshape(-1, 2), axis=1)
        median_error = float(np.median(residual[inlier_mask]))
        coverage = float(cv2.contourArea(cv2.convexHull(good_prev[inlier_mask]))) / curr_gray.size
        scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
        rotation = float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
        translation = float(np.linalg.norm(matrix[:, 2])) / max(curr_gray.shape)
        self.last_metrics = {
            "inlier_ratio": inlier_ratio, "median_error": median_error,
            "coverage": coverage, "scale": scale, "rotation_degrees": rotation,
        }
        if median_error > 1.5 or coverage < 0.02:
            return self._reject("poor_geometric_support")
        if not 0.95 <= scale <= 1.05 or abs(rotation) > 8 or translation > 0.1:
            return self._reject("large_camera_motion")

        height, width = frame_rgb.shape[:2]
        tracking_to_frame = np.diag([width / curr_gray.shape[1], height / curr_gray.shape[0], 1.0])
        incremental = np.eye(3, dtype=np.float64)
        incremental[:2] = matrix
        cumulative = tracking_to_frame @ incremental @ np.linalg.inv(tracking_to_frame) @ self.cumulative_matrix
        anchor_points = np.array([[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]], dtype=np.float32)
        transformed = cv2.transform(anchor_points, cumulative[:2]).reshape(-1, 2)
        frame_polygon = anchor_points.reshape(-1, 2)
        overlap, _ = cv2.intersectConvexConvex(frame_polygon, transformed.astype(np.float32))
        if overlap / max((width - 1) * (height - 1), 1) < 0.95:
            return self._reject("new_scene_area")
        matrix = cumulative[:2]
        warped_mask = warp_label_mask(
            self.anchor_mask,
            matrix,
            (width, height),
        )

        warped_features = (
            warp_feature_collection(self.anchor_features, matrix)
            if self.anchor_features is not None
            else None
        )

        self.prev_gray = curr_gray
        self.cumulative_matrix = cumulative
        self.points = good_curr[inlier_mask].reshape(-1, 1, 2)
        self.last_mask = warped_mask
        self.last_features = warped_features
        self.frames_since_keyframe += 1
        self.last_reason = "tracked"
        return warped_mask, warped_features
