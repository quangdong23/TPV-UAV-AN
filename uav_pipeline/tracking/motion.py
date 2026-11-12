"""Sparse optical flow and camera compensation, following uav_pipeline.segmentation's
forward/backward consistency + RANSAC approach. No segmentation dependency.
"""
import cv2
import numpy as np


def warp_label_mask(mask, matrix, size, unknown_id=0):
    """Warp categorical labels from an anchor with affine or homography motion.

    Nearest-neighbour sampling preserves class IDs; newly exposed pixels never
    inherit sidewalk. The caller must reject unreliable motion before warping.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if mask.ndim != 2 or matrix.shape not in ((2, 3), (3, 3)) or not np.isfinite(matrix).all():
        raise ValueError('Invalid label mask or motion matrix')
    full = np.vstack((matrix, [0, 0, 1])) if matrix.shape == (2, 3) else matrix
    if abs(np.linalg.det(full)) < 1e-12 or min(size) <= 0:
        raise ValueError('Singular motion matrix or invalid mask size')
    warp = cv2.warpAffine if matrix.shape == (2, 3) else cv2.warpPerspective
    return warp(mask, matrix, tuple(size), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=unknown_id)


def gray_frame(frame, max_side=640):
    h, w = frame.shape[:2]
    scale = min(1., max_side / max(h, w)) if max_side else 1.
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    gray = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2GRAY)
    return gray


def warp_box(box, matrix):
    x1, y1, x2, y2 = box
    corners = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], np.float32)
    points = cv2.transform(corners, np.asarray(matrix, dtype=np.float32))[0]
    return np.concatenate((points.min(0), points.max(0)))


def warp_boxes(boxes, matrix):
    """Same float32 corner transform as warp_box, in one OpenCV call."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if not len(boxes):
        return np.empty((0, 4), dtype=np.float32)
    corners = boxes[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 4, 2)
    points = cv2.transform(corners.reshape(1, -1, 2), np.asarray(matrix, np.float32)).reshape(-1, 4, 2)
    return np.concatenate((points.min(1), points.max(1)), axis=1)


def local_box_from_flow(box, previous_points, current_points, scale_xy, camera_matrix=None):
    """Combine camera scale/rotation with the object's median residual motion.

    LK points are in tracking pixels; boxes/matrix are in working-image pixels.
    A translated old box alone would undo GMC scale changes during Kalman update.
    """
    scale = np.asarray(scale_xy, float)
    before = np.asarray(previous_points, float) / scale
    after = np.asarray(current_points, float) / scale
    if camera_matrix is None:
        reference = before
        moved_box = np.asarray(box, float)
    else:
        matrix = np.asarray(camera_matrix, float)
        reference = before @ matrix[:, :2].T + matrix[:, 2]
        moved_box = warp_box(box, matrix)
    residual = np.median(after - reference, axis=0)
    return moved_box + np.tile(residual, 2)


class CameraMotion:
    def __init__(self, max_side=640, refresh_interval=1):
        self.max_side = max_side
        self.previous = None
        if not isinstance(refresh_interval, int) or refresh_interval < 1:
            raise ValueError('refresh_interval must be a positive integer')
        self.refresh_interval = refresh_interval
        self.frame_index = 0
        self.point_cache = {}

    def state_dict(self):
        return {'frame_index': self.frame_index,
                'point_cache': {str(k): v.tolist() for k, v in self.point_cache.items()}}

    def load_state_dict(self, state):
        self.frame_index = state.get('frame_index', 0)
        self.point_cache = {int(k): np.asarray(v, np.float32)
                            for k, v in state.get('point_cache', {}).items()}

    def update(self, frame, boxes):
        current = gray_frame(frame, self.max_side)
        identity = np.eye(2, 3, dtype=np.float64)
        result = {'matrix': identity, 'local_boxes': {}, 'reliable': False,
                  'scene_cut': False, 'reason': 'initial'}
        previous, self.previous = self.previous, current
        self.frame_index += 1
        if previous is None or previous.shape != current.shape:
            self.point_cache = {}
            return result
        difference = np.mean(cv2.absdiff(previous, current)) / 255.
        hist_a = cv2.calcHist([previous], [0], None, [32], [0, 256])
        hist_b = cv2.calcHist([current], [0], None, [32], [0, 256])
        hist_distance = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA)
        if difference > .30 and hist_distance > .5:
            self.point_cache = {}
            result.update(scene_cut=True, reason='scene_cut')
            return result
        h, w = frame.shape[:2]
        sx, sy = current.shape[1] / w, current.shape[0] / h

        def corners(mask, count):
            return cv2.goodFeaturesToTrack(previous, maxCorners=count, qualityLevel=.01,
                                          minDistance=5, mask=mask, blockSize=5)

        background = np.full_like(previous, 255)
        regions = {}
        refresh = self.frame_index % self.refresh_interval == 0
        def cached_points(key, x1, y1, x2, y2, minimum):
            points = self.point_cache.get(key)
            if refresh or points is None:
                return None
            xy = points.reshape(-1, 2)
            valid = (xy[:,0] >= x1) & (xy[:,0] < x2) & (xy[:,1] >= y1) & (xy[:,1] < y2)
            if key == -1:
                pixel = np.clip(xy.astype(int), [0,0], [current.shape[1]-1,current.shape[0]-1])
                valid &= background[pixel[:,1], pixel[:,0]] != 0
            points = points[valid]
            return points if len(points) >= minimum else None
        for track_id, box in boxes.items():
            b = np.asarray(box) * [sx, sy, sx, sy]
            x1, y1 = np.maximum(0, np.floor(b[:2]).astype(int))
            x2, y2 = np.minimum([current.shape[1], current.shape[0]], np.ceil(b[2:]).astype(int))
            if x2 <= x1 or y2 <= y1:
                continue
            background[y1:y2, x1:x2] = 0
            # Detect ROI corners on the small patch, not the entire frame N times.
            points = cached_points(track_id, x1,y1,x2,y2,8)
            if points is not None:
                regions[track_id] = points
                continue
            points = cv2.goodFeaturesToTrack(previous[y1:y2, x1:x2], maxCorners=40,
                                             qualityLevel=.01, minDistance=5, blockSize=5)
            if points is not None and len(points) >= 4:
                regions[track_id] = points + np.array([x1, y1], np.float32)
        background_points = cached_points(-1,0,0,current.shape[1],current.shape[0],100)
        if background_points is None:
            background_points = corners(background, 400)
        groups = ([(-1, background_points)] if background_points is not None else []) + list(regions.items())
        pair = None
        per_track = {}
        self.point_cache = {}
        if groups:
            points = np.concatenate([p for _, p in groups]).astype(np.float32)
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None,
                                                            winSize=(21, 21), maxLevel=3)
            if next_points is not None and status is not None:
                back, back_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, next_points, None,
                                                              winSize=(21, 21), maxLevel=3)
                if back is not None and back_status is not None:
                    valid = ((status.ravel() == 1) & (back_status.ravel() == 1)
                             & (np.linalg.norm((back - points).reshape(-1, 2), axis=1) < 1.)
                             & np.isfinite(next_points.reshape(-1, 2)).all(1))
                    offset = 0
                    for key, group in groups:
                        span = slice(offset, offset + len(group))
                        good = valid[span]
                        item = (points[span].reshape(-1, 2)[good],
                                next_points[span].reshape(-1, 2)[good])
                        self.point_cache[key] = item[1].reshape(-1,1,2).copy()
                        if key == -1:
                            pair = item
                        else:
                            per_track[key] = item
                        offset += len(group)
        if pair is not None and len(pair[0]) >= 12:
            a, b = pair
            cv2.setRNGSeed(0)  # Reproducible after restoring a checkpoint.
            matrix, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                       ransacReprojThreshold=2.)
            if matrix is not None and inliers is not None and np.isfinite(matrix).all():
                good = inliers.ravel().astype(bool)
                coverage = cv2.contourArea(cv2.convexHull(a[good])) / current.size if good.sum() >= 3 else 0
                scale = np.hypot(matrix[0, 0], matrix[1, 0])
                if good.sum() >= 10 and good.mean() >= .55 and coverage > .015 and .80 < scale < 1.25:
                    scaling = np.diag([sx, sy, 1.])
                    affine = np.eye(3)
                    affine[:2] = matrix
                    result.update(matrix=(np.linalg.inv(scaling) @ affine @ scaling)[:2],
                                  reliable=True, reason='tracked')
        if not result['reliable']:
            result['reason'] = 'camera_flow_unreliable'
        # Independent foreground flow is essential for moving people/vehicles.
        for track_id, box in boxes.items():
            pair = per_track.get(track_id)
            if pair is None or len(pair[0]) < 4:
                continue
            a, b = pair
            delta = b - a
            median = np.median(delta, axis=0)
            good = np.linalg.norm(delta - median, axis=1) < 2.
            if good.sum() < 4 or good.mean() < .6:
                continue
            local = local_box_from_flow(box, a[good], b[good], [sx, sy],
                                        result['matrix'] if result['reliable'] else None)
            result['local_boxes'][track_id] = local
        return result
