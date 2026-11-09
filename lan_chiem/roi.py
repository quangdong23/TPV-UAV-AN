"""Deterministic overlapping image coverage, independent of model detections."""
import math


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
