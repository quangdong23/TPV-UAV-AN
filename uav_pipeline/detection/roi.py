"""Deterministic overlapping image coverage, independent of model detections."""
import math


def select_recovery_regions(regions, limit, offset=0, recovery=()):
    """Spend existing crop budget on recovery while retaining a fair grid scan.

    Return the number of grid positions consumed, including duplicates skipped.
    The caller persists this cursor so focused crops never permanently starve
    one parity of grid cells. A single-crop budget always stays with discovery.
    """
    if limit <= 0:
        return [], 0
    chosen = []
    for region in recovery:
        region = tuple(region)
        if region not in chosen:
            chosen.append(region)
        if len(chosen) >= min(1, limit-1):
            break
    if limit == 1:
        chosen = []
    consumed = 0
    while regions and consumed < len(regions) and len(chosen) < limit:
        region = tuple(regions[(offset+consumed) % len(regions)])
        consumed += 1
        if region not in chosen:
            chosen.append(region)
    return chosen, consumed


def track_recovery_regions(tracks, timestamp, motion, shape, patch_size, *,
                           min_hits, dynamic_limit, static_limit, budget=1):
    """Find recent confirmed weak tracks using measured flow/camera geometry.

    These are crop requests, never synthetic detections or new identities.
    Expired tracks and dynamic tracks with no measured local flow are excluded.
    """
    if budget <= 0 or motion.get('scene_cut'):
        return []
    import numpy as np
    from uav_pipeline.tracking.motion import warp_box
    height, width = shape[:2]
    candidates = []
    for track in tracks:
        if track['hits'] < min_hits:
            continue
        age = timestamp-track['last_detection']
        static = track['class_id'] in (4, 5, 6, 7, 8, 9, 10) and not track.get('moving_cart', False)
        limit = static_limit if static else dynamic_limit
        if not 0 <= age <= limit or limit <= 0:
            continue
        if not (track.get('roi_dependent') or track.get('miss_count', 0)):
            continue
        if age < .5*limit and not track.get('miss_count', 0):
            continue
        box = motion.get('local_boxes', {}).get(track['id'])
        if box is None:
            if not static or not motion.get('reliable'):
                continue
            state = track['x']
            size = np.maximum(1., state[2:4])
            box = warp_box(np.r_[state[:2]-size/2, state[:2]+size/2], motion['matrix'])
        box = np.asarray(box, float)
        if (box.shape != (4,) or not np.isfinite(box).all()
                or np.any(box[2:] <= box[:2]) or max(box[2:]-box[:2]) >= patch_size*.8):
            continue
        x = max(0, min(max(0, width-patch_size), round((box[0]+box[2]-patch_size)/2)))
        y = max(0, min(max(0, height-patch_size), round((box[1]+box[3]-patch_size)/2)))
        region = (x, y, min(width, x+patch_size), min(height, y+patch_size))
        # Keep full-object context and the same artificial-edge margin used by decode.
        if (box[0] <= x+2 or box[1] <= y+2 or box[2] >= region[2]-2 or box[3] >= region[3]-2):
            continue
        candidates.append((age/limit, track['id'], region))
    chosen = []
    for _, _, region in sorted(candidates, key=lambda item: (-item[0], item[1])):
        if region not in chosen:
            chosen.append(region)
        if len(chosen) >= budget:
            break
    return chosen


def grid_regions(width, height, patch_size, overlap=.2):
    if not 0 <= overlap < 1:
        raise ValueError('ROI overlap must be in [0, 1)')
    def positions(length):
        end = max(0, length - patch_size)
        count = max(1, math.ceil(end / max(1, patch_size * (1-overlap))))
        return sorted({round(i * end / count) for i in range(count + 1)})
    return [(x, y, min(x+patch_size, width), min(y+patch_size, height))
            for y in positions(height) for x in positions(width)]


def select_regions(regions, limit, offset=0):
    if not regions or limit <= 0:
        return []
    return [regions[(offset+i) % len(regions)] for i in range(min(limit, len(regions)))]
