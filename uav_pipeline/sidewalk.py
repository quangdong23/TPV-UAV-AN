"""Stateful adapter using the existing sidewalk video's AI/warp schedule."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np

from lan_chiem.object_classes import ClassId
from lan_chiem.postprocess import annotate_encroachment


SIDEWALK_EXEMPT_CLASSES = frozenset({ClassId.ADVERTISING_SIGN, ClassId.UTILITY_POLE, ClassId.TREE_TRUNK})


def keep_on_sidewalk(rows, mask, mode='bottom-center', min_overlap=0.1, sign_contact_tolerance=20.):
    """Boxes and mask share pixels; signs/poles/trunks bypass the sidewalk condition.

    Only label 2 counts as sidewalk for the remaining classes.
    Bottom-center is a ground-contact approximation, not a legal encroachment
    decision. Overlap uses the entire box as denominator, including offscreen area.
    """
    if mode not in ('bottom-center', 'overlap'):
        raise ValueError('Unknown sidewalk filter mode')
    if not math.isfinite(min_overlap) or not 0 < min_overlap <= 1:
        raise ValueError('min_overlap must be in (0, 1]')
    h, w = mask.shape
    kept = []
    for row in rows:
        row = annotate_encroachment(row, mask, sign_contact_tolerance)
        box = np.asarray(row['bbox_xyxy'], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            continue
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        if row.get('class_id') in SIDEWALK_EXEMPT_CLASSES:
            kept.append(row)
            continue
        if mode == 'bottom-center':
            x, y = math.floor((x1+x2)/2), math.ceil(y2)-1
            inside = 0 <= x < w and 0 <= y < h and mask[y, x] == 2
        else:
            left, top = max(0, math.floor(x1)), max(0, math.floor(y1))
            right, bottom = min(w, math.ceil(x2)), min(h, math.ceil(y2))
            area = max(0, right-left)*max(0, bottom-top)
            sidewalk = np.count_nonzero(mask[top:bottom, left:right] == 2) if area else 0
            denominator = (math.ceil(x2)-math.floor(x1))*(math.ceil(y2)-math.floor(y1))
            inside = sidewalk / denominator >= min_overlap
        if inside:
            kept.append(row)
    return kept


class SidewalkSession:
    def __init__(self, pipeline, options, filter_mode='bottom-center', min_overlap=0.1, sample_step=30,
                 crisp_overlay=False):
        from road_sidewalk_seg.tracking.motion import VideoMotionPropagator

        if sample_step < 0 or options.keyframe_interval < 1:
            raise ValueError('Invalid sampling/keyframe interval')
        if min(options.work_max_side, options.tracking_max_side, options.postprocess_max_side) < 0:
            raise ValueError('Sidewalk sizes cannot be negative')
        if not 0 <= options.temporal_smoothing < 1:
            raise ValueError('Temporal smoothing must be in [0, 1)')
        keep_on_sidewalk([], np.zeros((1, 1), np.uint8), filter_mode, min_overlap)
        self.pipeline, self.options = pipeline, options
        self.filter_mode, self.min_overlap, self.sample_step = filter_mode, min_overlap, sample_step
        self.prev_scores = None
        self.crisp_overlay = crisp_overlay
        self.propagator = (VideoMotionPropagator(keyframe_interval=options.keyframe_interval,
                           tracking_max_side=options.tracking_max_side)
                           if options.keyframe_interval > 1 else None)

    def signature(self):
        root = Path(__file__).resolve().parents[1]
        paths = sorted((root/'road_sidewalk_seg').rglob('*.py')) + sorted((root/'uav_pipeline').glob('*.py'))
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
        return {'config': self.pipeline.config.to_dict(), 'video_options': asdict(self.options),
                'filter_mode': self.filter_mode, 'min_overlap': self.min_overlap,
                'sidewalk_exempt_class_ids': sorted(SIDEWALK_EXEMPT_CLASSES),
                'sample_step': self.sample_step, 'crisp_overlay': self.crisp_overlay, 'code_sha256': digest.hexdigest()}

    def process(self, source_bgr, working_shape):
        previous_threads = cv2.getNumThreads()
        try:
            if self.options.cv_threads and self.options.cv_threads != previous_threads:
                cv2.setNumThreads(self.options.cv_threads)
            return self._process(source_bgr, working_shape)
        finally:
            if cv2.getNumThreads() != previous_threads:
                cv2.setNumThreads(previous_threads)

    def _process(self, source_bgr, working_shape):
        from road_sidewalk_seg.media.video import _fit_work_frame
        from road_sidewalk_seg.tracking.motion import warp_feature_collection
        from road_sidewalk_seg.postprocessing.visualization import create_overlay, draw_feature_boundaries

        o, p = self.options, self.propagator
        rgb = cv2.cvtColor(_fit_work_frame(source_bgr, o.work_max_side), cv2.COLOR_BGR2RGB)
        run_ai = p is None or p.should_run_ai()
        mask = features = aligned_scores = None
        if p is not None and p.last_mask is not None and (not run_ai or o.temporal_smoothing > 0):
            mask, features = p.propagate(rgb)
            if mask is not None and self.prev_scores is not None and run_ai and o.temporal_smoothing > 0:
                aligned_scores = p.warp_keyframe_scores(self.prev_scores)
            if mask is None:
                run_ai = True
        if run_ai:
            kwargs = {'prev_scores': aligned_scores, 'temporal_smoothing': o.temporal_smoothing}
            if o.postprocess_max_side:
                kwargs['postprocess_max_side'] = o.postprocess_max_side
            if self.crisp_overlay:
                kwargs['render_overlay'] = False
            result = self.pipeline.process_frame(rgb, **kwargs)
            mask, features, overlay = result.mask, result.pixel_features, result.overlay_rgb
            self.prev_scores = result.inference.scores if o.temporal_smoothing > 0 and p is not None else None
            if p is not None:
                p.update_keyframe(rgb, mask, features)
        else:
            overlay = rgb if self.crisp_overlay else create_overlay(rgb, mask, self.pipeline.config.output.overlay_alpha,
                                                                    self.pipeline.config.output.show_unknown)
            if not self.crisp_overlay and features and features.get('features'):
                overlay = draw_feature_boundaries(overlay, features, self.pipeline.config.polygon)
        mh, mw = mask.shape
        h, w = working_shape[:2]
        source_h, source_w = source_bgr.shape[:2]
        matrix = np.array([[source_w/mw, 0, 0], [0, source_h/mh, 0]], dtype=np.float32)
        exported = deepcopy(warp_feature_collection(features or {'features': []}, matrix))
        for feature in exported['features']:
            for key in ('area_pixels', 'source_area_pixels'):
                if key in feature['properties']:
                    feature['properties'][key] *= source_w*source_h/(mw*mh)
        working_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if self.crisp_overlay:
            working_bgr = source_bgr if source_bgr.shape[:2] == (h, w) else cv2.resize(source_bgr, (w, h), interpolation=cv2.INTER_AREA)
            overlay = create_overlay(cv2.cvtColor(working_bgr, cv2.COLOR_BGR2RGB), working_mask,
                                     self.pipeline.config.output.overlay_alpha, self.pipeline.config.output.show_unknown)
            work_features = warp_feature_collection(features or {'features': []},
                np.array([[w/mw,0,0],[0,h/mh,0]], dtype=np.float32))
            overlay = draw_feature_boundaries(overlay, work_features, self.pipeline.config.polygon)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        else:
            overlay_bgr = cv2.resize(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), (w, h), interpolation=cv2.INTER_LINEAR)
        return {'mask': working_mask,
                'segmentation_mask': mask,
                'overlay': overlay_bgr,
                'metadata': {'coordinate_space': 'source_pixels', 'features': exported,
                             'segmentation_size': [mw, mh], 'ai_ran': run_ai,
                             'sign_contact_tolerance_pixels': 20.*h/source_h,
                             'motion_reason': p.last_reason if p else 'every_frame_ai',
                             'motion_metrics': dict(p.last_metrics) if p else {},
                             'mode': 'AI' if run_ai else 'Warp',
                             'sidewalk_pixels': int(np.count_nonzero(mask == 2)),
                             'pixel_count_space': 'segmentation_pixels',
                             'sidewalk_exempt_class_ids': sorted(SIDEWALK_EXEMPT_CLASSES),
                             'filter_mode': self.filter_mode, 'min_overlap': self.min_overlap}}

    def filter(self, rows, frame_result):
        return keep_on_sidewalk(rows, frame_result['mask'], self.filter_mode, self.min_overlap,
            frame_result.get('metadata', {}).get('sign_contact_tolerance_pixels', 20.))

    def save_samples(self, result, rendered, index, output, source_size):
        if not self.sample_step or index % self.sample_step:
            return
        output = Path(output)
        for folder in ('sidewalk_masks', 'images'):
            (output/folder).mkdir(exist_ok=True)
        mask_path, image_path = f'sidewalk_masks/{index:08d}.png', f'images/{index:08d}.jpg'
        # Samples and exported polygons use source pixels; video/images use working pixels.
        mask = cv2.resize(result['segmentation_mask'], tuple(source_size), interpolation=cv2.INTER_NEAREST)
        if not cv2.imwrite(str(output/mask_path), mask) or not cv2.imwrite(str(output/image_path), rendered):
            raise RuntimeError('Cannot write combined video samples')
        result['metadata'].update(mask_path=mask_path, image_path=image_path)
