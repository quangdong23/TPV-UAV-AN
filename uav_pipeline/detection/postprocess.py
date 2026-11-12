"""Postprocessing, geometry, deduplication, and spatial conflict analysis.

Completely independent of model architectures (no HuggingFace / Grounding DINO dependency).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from uav_pipeline.detection.classes import ClassId


ENCROACHMENT_CLASSES = frozenset({ClassId.PLASTIC_TABLE, ClassId.PLASTIC_CHAIR,
                                ClassId.VENDING_CART, ClassId.GOODS})


def annotate_encroachment(row, mask, contact_tolerance=20.):
    """Return current-frame contact evidence in the mask's pixel coordinates.

    A 2D foot/mask intersection is a screening heuristic, not proof of a base
    or attachment. Missing sidewalk and offscreen feet stay unclassified.
    """
    result = {**row, 'is_encroaching': False}
    box = np.asarray(row['bbox_xyxy'], float)
    h, w = mask.shape
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        return result
    x, y = math.floor((box[0]+box[2])/2), math.ceil(box[3])-1
    if not (0 <= x < w and 0 <= y < h):
        if row.get('class_id') == ClassId.ADVERTISING_SIGN:
            result.update(subtype='sign_unresolved', contact_evidence={'reason': 'offscreen_foot'})
        return result
    contact = mask[y, x] == 2
    cid = row.get('class_id')
    if cid in ENCROACHMENT_CLASSES:
        result['is_encroaching'] = bool(contact)
    if cid != ClassId.ADVERTISING_SIGN:
        return result
    radius = max(0, math.ceil(contact_tolerance))
    left, top = max(0, x-radius), max(0, y-radius)
    right, bottom = min(w, x+radius+1), min(h, y+radius+1)
    ys, xs = np.nonzero(mask[top:bottom, left:right] == 2)
    distance = float(np.min(np.hypot(xs+left-x, ys+top-y))) if len(xs) else None
    contact = distance is not None and distance <= contact_tolerance
    label = row.get('raw_label', '').lower()
    attachment = bool(re.search(r'\b(hanging|wall mounted|wall-mounted|facade)\b', label))
    below = np.flatnonzero(mask[y+1:, x] == 2)
    horizontal = box[2]-box[0] >= 1.6*(box[3]-box[1])
    elevated = bool(len(below) and below[0]+1 > contact_tolerance and horizontal)
    subtype = ('facade_attached' if attachment or elevated else
               'sidewalk_standee' if contact else 'sign_unresolved')
    result.update(subtype=subtype, is_encroaching=subtype == 'sidewalk_standee',
                  contact_evidence={'foot_distance_pixels': distance,
                      'tolerance_pixels': float(contact_tolerance),
                      'attachment_phrase': attachment, 'elevated_horizontal': elevated,
                      'interpretation': 'image_space_contact_requires_review'})
    return result


@dataclass
class Detection:
    """Đại diện cho một bounding box đối tượng được nhận diện."""
    id: str
    class_id: Optional[int]
    class_name: str
    class_name_vi: str
    raw_label: str
    prompt_group: str
    source: str  # "full" | "tile_x_y" | "fused"
    score: float
    bbox_xyxy: List[float]  # [x_min, y_min, x_max, y_max]
    subtype: Optional[str] = None  # Cho ID 7 (biển quảng cáo) hoặc ID 10 (hàng hóa)
    status: str = "predicted"  # "predicted" | "review" (không tự gán "confirmed")
    review_reason: Optional[str] = None
    conflict_info: Optional[Dict[str, Any]] = None
    related_detection_ids: List[str] = None
    queried_class_id: Optional[int] = None
    query_prompt: Optional[str] = None
    candidate_class_ids: List[int] = field(default_factory=list)
    status_history: List[Dict[str, Any]] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    is_encroaching: bool = False

    def __post_init__(self):
        if self.related_detection_ids is None:
            self.related_detection_ids = []
        if not self.status_history:
            self.status_history.append({"from": None, "to": self.status,
                                        "reason": self.review_reason or "raw_model_hypothesis"})

    def require_review(self, reason: str) -> None:
        """Record every review decision; never implicitly promote a hypothesis."""
        self.status_history.append({"from": self.status, "to": "review", "reason": reason})
        self.status = "review"
        self.review_reason = "; ".join(filter(None, [self.review_reason, reason]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "class_name_vi": self.class_name_vi,
            "raw_label": self.raw_label,
            "prompt_group": self.prompt_group,
            "source": self.source,
            "score": round(float(self.score), 4),
            "bbox_xyxy": [round(float(c), 2) for c in self.bbox_xyxy],
            "subtype": self.subtype,
            "is_encroaching": self.is_encroaching,
            "status": self.status,
            "review_reason": self.review_reason,
            "conflict_info": self.conflict_info,
            "related_detection_ids": self.related_detection_ids,
            "queried_class_id": self.queried_class_id,
            "query_prompt": self.query_prompt,
            "candidate_class_ids": self.candidate_class_ids,
            "status_history": self.status_history,
            "evidence": self.evidence,
        }


@dataclass
class RejectionRecord:
    """Ghi vết một kết quả bị loại bỏ ở từng giai đoạn suy luận (phục vụ debug trace)."""
    stage: str  # "threshold" | "mapping" | "nms" | "geometry" | "conflict"
    bbox_xyxy: List[float]
    score: float
    raw_label: str
    prompt_group: str
    source: str
    reason: str
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "bbox_xyxy": [round(float(c), 2) for c in self.bbox_xyxy],
            "score": round(float(self.score), 4),
            "raw_label": self.raw_label,
            "prompt_group": self.prompt_group,
            "source": self.source,
            "reason": self.reason,
            "details": self.details,
        }


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Tính Intersection over Union (IoU) giữa 2 box [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def calculate_ios(box_small: List[float], box_large: List[float]) -> float:
    """Tính Intersection over Smaller Box (IoS) - độ chứa của box nhỏ trong box lớn."""
    x1 = max(box_small[0], box_large[0])
    y1 = max(box_small[1], box_large[1])
    x2 = min(box_small[2], box_large[2])
    y2 = min(box_small[3], box_large[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_small = max(0.0, box_small[2] - box_small[0]) * max(0.0, box_small[3] - box_small[1])
    if area_small <= 0.0:
        return 0.0
    return inter_area / area_small


def calculate_center_distance(box1: List[float], box2: List[float]) -> float:
    """Tính khoảng cách Euclidean giữa tâm của 2 box."""
    c1_x = (box1[0] + box1[2]) / 2.0
    c1_y = (box1[1] + box1[3]) / 2.0
    c2_x = (box2[0] + box2[2]) / 2.0
    c2_y = (box2[1] + box2[3]) / 2.0
    return float(np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2))


def apply_intra_class_nms(
    detections: List[Detection],
    iou_threshold: float = 0.50,
    rejections: Optional[List[RejectionRecord]] = None,
) -> List[Detection]:
    """
    Áp dụng NMS nội bộ nhóm (Intra-class NMS):
    Chỉ triệt tiêu các box thuộc CÙNG một class_id nếu IoU > iou_threshold.
    Không bao giờ triệt tiêu box giữa các lớp khác nhau.
    """
    if len(detections) <= 1:
        return detections

    by_class: Dict[int, List[Detection]] = {}
    for d in detections:
        by_class.setdefault(d.class_id, []).append(d)

    kept: List[Detection] = []
    for cid, c_dets in by_class.items():
        if len(c_dets) <= 1:
            kept.extend(c_dets)
            continue

        c_dets.sort(key=lambda x: x.score, reverse=True)
        selected: List[Detection] = []
        for candidate in c_dets:
            suppress = False
            winner = None
            for chosen in selected:
                iou = calculate_iou(candidate.bbox_xyxy, chosen.bbox_xyxy)
                if iou > iou_threshold:
                    suppress = True
                    winner = chosen
                    break
            if not suppress:
                selected.append(candidate)
            else:
                if rejections is not None and winner is not None:
                    rejections.append(
                        RejectionRecord(
                            stage="nms",
                            bbox_xyxy=candidate.bbox_xyxy,
                            score=candidate.score,
                            raw_label=candidate.raw_label,
                            prompt_group=candidate.prompt_group,
                            source=candidate.source,
                            reason="suppressed_duplicate",
                            details=f"Bị loại bởi {winner.id} với IoU={calculate_iou(candidate.bbox_xyxy, winner.bbox_xyxy):.2f} > {iou_threshold}",
                        )
                    )
        kept.extend(selected)

    return kept


def reconcile_roi_fragments(detections, regions, image_size):
    """Prefer a complete view over a corroborated crop-boundary fragment.

    Only same-class, different-view hypotheses participate. Never expand a box,
    boost its confidence, or infer a missing class from geometry.
    """
    width, height = image_size
    removed = set()
    resolved = []
    for fragment in detections:
        if fragment.class_id == 9:
            continue  # Trunk versus whole-tree scope needs semantic evidence.
        region = regions.get(fragment.source)
        if region is None:
            continue
        x1,y1,x2,y2 = region
        margin = min(32., .05 * min(x2-x1,y2-y1))
        a = fragment.bbox_xyxy
        area = (a[2]-a[0])*(a[3]-a[1])
        candidates = []
        for complete in detections:
            if (complete is fragment or complete.class_id != fragment.class_id
                    or complete.source == fragment.source):
                continue
            b = complete.bbox_xyxy
            other_area = (b[2]-b[0])*(b[3]-b[1])
            if not 0 < area <= .75*other_area or calculate_ios(a,b) < .95:
                continue
            clipped = ((x1>0 and a[0]-x1<=margin and b[0]<x1-2) or
                       (y1>0 and a[1]-y1<=margin and b[1]<y1-2) or
                       (x2<width and x2-a[2]<=margin and b[2]>x2+2) or
                       (y2<height and y2-a[3]<=margin and b[3]>y2+2))
            if not clipped:
                continue
            # A large box spanning separate objects is not a complete view.
            aggregate = any(d is not fragment and d is not complete and
                d.class_id == fragment.class_id and calculate_ios(d.bbox_xyxy,b)>=.95
                and calculate_iou(d.bbox_xyxy,a)<.15 for d in detections)
            if not aggregate:
                candidates.append(complete)
        if candidates:
            winner = min(candidates,key=lambda d:((d.bbox_xyxy[2]-d.bbox_xyxy[0])*
                         (d.bbox_xyxy[3]-d.bbox_xyxy[1]),-d.score))
            winner.evidence.setdefault('roi_fragments',[]).append({
                'id':fragment.id,'source':fragment.source,'score':fragment.score,
                'bbox_xyxy':list(a),'reason':'corroborated_artificial_crop_boundary'})
            removed.add(id(fragment))
            resolved.append((fragment,winner))
    # The global pass may repeat exactly the same partial scope. Reconcile
    # that clone only after another view supplied crop-boundary evidence.
    for fragment,winner in resolved:
        b=winner.bbox_xyxy
        area=(b[2]-b[0])*(b[3]-b[1])
        for clone in detections:
            a=clone.bbox_xyxy
            if (id(clone) in removed or clone is winner or clone.class_id!=fragment.class_id
                    or (a[2]-a[0])*(a[3]-a[1])>.75*area
                    or calculate_iou(a,fragment.bbox_xyxy)<.90 or calculate_ios(a,b)<.95):
                continue
            winner.evidence.setdefault('roi_fragments',[]).append({
                'id':clone.id,'source':clone.source,'score':clone.score,
                'bbox_xyxy':list(a),'reason':'agrees_with_corroborated_crop_fragment'})
            removed.add(id(clone))
    return [d for d in detections if id(d) not in removed]


def consolidate_nested_boxes(detections: List[Detection]) -> Tuple[List[Detection], List[Detection]]:
    kept, removed = [], []
    by_class: Dict[int, List[Detection]] = {}
    for det in detections:
        if det.class_id is None or det.candidate_class_ids:
            kept.append(det)
        else:
            by_class.setdefault(det.class_id, []).append(det)
    for cid, items in by_class.items():
        aggregates = []
        for outer in items:
            contained = [d for d in items if d is not outer and
                         calculate_ios(d.bbox_xyxy, outer.bbox_xyxy) >= .95 and
                         calculate_iou(d.bbox_xyxy, outer.bbox_xyxy) < .5]
            if any(calculate_iou(a.bbox_xyxy, b.bbox_xyxy) < .15
                   for i, a in enumerate(contained) for b in contained[i+1:]):
                outer.evidence['contained_detection_ids'] = [d.id for d in contained]
                outer.require_review('aggregate_scope_uncertain')
                aggregates.append(outer)
        selected = []
        for det in sorted([d for d in items if d not in aggregates], key=lambda d: -d.score):
            winner = None
            if cid != 9:  # Trunk scope cannot be resolved by preferring a nested box.
                for other in selected:
                    a, b = det.bbox_xyxy, other.bbox_xyxy
                    aa, ba = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
                    ios = max(calculate_ios(a, b), calculate_ios(b, a))
                    distance = math.hypot((a[0]+a[2]-b[0]-b[2])/2, (a[1]+a[3]-b[1]-b[3])/2)
                    diagonal = min(math.hypot(a[2]-a[0], a[3]-a[1]), math.hypot(b[2]-b[0], b[3]-b[1]))
                    if ios >= .95 and min(aa, ba)/max(aa, ba, 1) >= .30 and distance/max(diagonal, 1) <= .35:
                        winner = other
                        break
            if winner:
                winner.evidence.setdefault('duplicate_hypotheses', []).append(det.to_dict())
                removed.append(det)
            else:
                selected.append(det)
        kept.extend(aggregates + selected)
    return kept, removed


def mark_exclusive_overlaps(detections: List[Detection]) -> List[Dict[str, Any]]:
    """Keep both interpretations; no comparison of scores across class queries."""
    exclusive = {(2, 6), (3, 5), (3, 6), (3, 7), (4, 5), (5, 6), (6, 7)}
    reports = []
    for i, a in enumerate(detections):
        for b in detections[i+1:]:
            if a.class_id is None or b.class_id is None:
                continue
            if tuple(sorted((a.class_id, b.class_id))) in exclusive and calculate_iou(a.bbox_xyxy, b.bbox_xyxy) >= .80:
                reports.append({'conflict_type': 'exclusive_class_overlap_unresolved',
                                'det_id': a.id, 'related_id': b.id,
                                'bbox_xyxy': a.bbox_xyxy, 'related_bbox_xyxy': b.bbox_xyxy,
                                'class_ids': [a.class_id, b.class_id],
                                'interpretation': 'alternative_hypotheses_not_verified'})
                for det, other in [(a, b), (b, a)]:
                    det.require_review('exclusive_class_overlap_unresolved')
                    det.evidence.setdefault('alternative_hypotheses', []).append(
                        {'id': other.id, 'class_id': other.class_id, 'bbox_xyxy': other.bbox_xyxy})
                    if other.id not in det.related_detection_ids:
                        det.related_detection_ids.append(other.id)
    return reports


def analyze_conflicts_and_partition(
    detections: List[Detection],
    counter_objects: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    review_rules: Optional[Dict[str, Any]] = None,
    surface_image=None,
    image_size=None,
) -> Tuple[List[Detection], List[Detection], List[Dict[str, Any]]]:
    """Phân tách kết quả thành predicted detections và review candidates dựa trên luật xung đột không gian."""
    rules = review_rules or {}
    reports = mark_exclusive_overlaps(detections) if rules.get('exclusive_class_overlap', True) else []
    predicted, review = [], []
    counters = counter_objects or {}
    context = rules.get('target_context', {})
    context_enabled = context.get('enabled', False)

    def same_object(first, second):
        return (calculate_iou(first, second) >= context.get('competing_iou', .60)
                and min(calculate_ios(first, second), calculate_ios(second, first))
                >= context.get('competing_mutual_coverage', .70))

    def enabled(name: str) -> bool:
        return rules.get(name, True)

    def has_trunk_extent(box):
        dimensions = image_size
        if dimensions is None and surface_image is not None:
            dimensions = (np.asarray(surface_image).shape[1], np.asarray(surface_image).shape[0])
        x1,y1,x2,y2 = box
        return (dimensions is not None and (y2-y1) >= .20*dimensions[1]
                and (y2-y1) >= 2.5*(x2-x1))

    def relate(det: Detection, other: Any, name: str, iou_limit: float = .35, containment: float = .65):
        ob = other.bbox_xyxy if isinstance(other, Detection) else other["bbox_xyxy"]
        oid = other.id if isinstance(other, Detection) else other.get("id")
        iou = calculate_iou(det.bbox_xyxy, ob)
        ios = calculate_ios(det.bbox_xyxy, ob)
        reverse_ios = calculate_ios(ob, det.bbox_xyxy)
        if (context_enabled and context.get('pole_tree_same_object_only')
                and isinstance(other, Detection) and {det.class_id, other.class_id} == {8, 9}):
            if not same_object(det.bbox_xyxy, ob):
                return
            # A weak utility-pole alternative cannot overrule a corroborated
            # trunk hypothesis merely by a small score difference. Keep the
            # trunk label provisional; video confirmation needs more detections.
            if rules.get('weak_pole_tree_policy', False):
                pole, trunk = (det, other) if det.class_id == 8 else (other, det)
                if pole.score < .45 and -.05 <= pole.score-trunk.score < .15:
                    trunk.evidence['pole_tree_uncertain'] = True
                    trunk.evidence['confirmation_hits'] = 4
                    if det.class_id == 9:
                        return
                    name = 'weak_pole_competing_trunk'
                    # Ensure the weak pole is reviewed independent of which
                    # hypothesis has the slightly greater raw head score.
                    gap = other.score-det.score
                    margin = context.get('competing_score_margin', .05)
                else:
                    gap = other.score-det.score
                    margin = context.get('competing_score_margin', .05)
            else:
                gap = other.score-det.score
                margin = context.get('competing_score_margin', .05)
            if gap < -margin and name != 'weak_pole_competing_trunk':
                det.evidence.setdefault('weaker_pole_tree_alternatives', []).append(other.id)
                return
            # A clear winner is retained; close alternatives remain unresolved.
            name = ('pole_tree_class_unresolved' if abs(gap) < margin
                    and name != 'weak_pole_competing_trunk' else name)
        # Focus-class counter hypotheses must describe the same physical
        # extent. A plant inside a goods display or canopy behind a sign is
        # contextual overlap, not evidence that the target class is wrong.
        if (context_enabled and det.class_id in (7, 8, 9, 10)
                and isinstance(other, dict)
                and other.get('origin') == 'shared_yoloe_head'):
            if not same_object(det.bbox_xyxy, ob):
                return
            gap = other.get('score', 0)-det.score
            margin = context.get('competing_score_margin', .05)
            if gap < margin:
                det.evidence.setdefault('weaker_counter_alternatives', []).append({
                    'id': oid, 'counter_group': other.get('counter_group'),
                    'score': other.get('score'), 'score_gap': gap})
                return
        if isinstance(other, dict) and other.get('origin') in ('same_caption', 'shared_yoloe_head'):
            if other.get('score', 0) < .30 or iou < .60 or min(ios, reverse_ios) < .70:
                return
        coverage = ios if name == "contained_in_cart_candidate" else max(ios, reverse_ios)
        if iou < iou_limit and coverage < containment:
            return
        report = {"conflict_type": name, "det_id": det.id, "related_id": oid,
                  "bbox_xyxy": det.bbox_xyxy, "related_bbox_xyxy": ob,
                  "iou": iou, "ios": ios, "reverse_ios": reverse_ios,
                  "interpretation": "spatial_overlap_only_not_verified"}
        if isinstance(other, Detection):
            report["related_status_at_observation"] = other.status
            report["related_class_id"] = other.class_id
            if det.id not in other.related_detection_ids:
                other.related_detection_ids.append(det.id)
        else:
            report["counter_evidence"] = other
        if oid and oid not in det.related_detection_ids:
            det.related_detection_ids.append(oid)
        det.evidence.setdefault("spatial_relations", []).append(report)
        det.conflict_info = det.conflict_info or report
        weaker_same_head = (isinstance(other, dict) and other.get('origin') == 'shared_yoloe_head'
                            and rules.get('same_head_counter_policy', 'strict') == 'stronger'
                            and other.get('score', 0) < det.score)
        contextual = (isinstance(other, Detection) and
                      name in ('contained_in_cart_candidate', 'overlap_with_table', 'goods_chair_overlap')
                      and not rules.get('spatial_context_review', True))
        report['requires_review'] = not (weaker_same_head or contextual)
        if report['requires_review']:
            det.require_review(name)
        reports.append(report)

    for det in detections:
        cid = det.class_id
        if det.candidate_class_ids or cid is None:
            det.require_review("ambiguous_class_unresolved")
            review.append(det)
            continue
        if (cid == ClassId.TREE_TRUNK and rules.get('weak_trunk_candidates', False)
                and det.score < .18):
            sufficient_extent = has_trunk_extent(det.bbox_xyxy)
            if sufficient_extent and surface_image is not None:
                # Recovery below the former threshold is deliberately limited
                # to a visible dark central body, rather than nearby foliage.
                # This is supporting contrast, never proof of tree material.
                image = np.asarray(surface_image)
                ih, iw = image.shape[:2]
                x1,y1,x2,y2 = det.bbox_xyxy
                width,height = x2-x1,y2-y1
                top,bottom = max(0,int(y1+.4*height)),min(ih,math.ceil(y1+.9*height))
                core = image[top:bottom,max(0,int(x1+.3*width)):min(iw,math.ceil(x1+.7*width)),:3]
                left = image[top:bottom,max(0,int(x1)):min(iw,int(x1+.2*width)),:3]
                right = image[top:bottom,max(0,int(x1+.8*width)):min(iw,math.ceil(x2)),:3]
                reference = [part.mean(2).ravel() for part in (left,right) if part.size]
                if core.size and reference:
                    body = core.mean(2)
                    background = float(np.median(np.concatenate(reference)))
                    core_median = float(np.median(body))
                    contrast_support = (background > 0 and core_median < .8*background
                                        and float(np.mean(body < .8*background)) >= .4)
                    det.evidence['weak_trunk_body_contrast'] = {
                        'body_median':core_median,'surround_median':background,
                        'supported':bool(contrast_support)}
                    sufficient_extent = bool(contrast_support)
                else:
                    sufficient_extent = False
            if sufficient_extent:
                det.evidence.update(weak_trunk_candidate=True, confirmation_hits=4)
            else:
                det.require_review('weak_trunk_extent_uncertain')
        if enabled('upright_object_geometry') and cid in (ClassId.UTILITY_POLE, ClassId.TREE_TRUNK):
            x1, y1, x2, y2 = det.bbox_xyxy
            w = max(x2 - x1, 1.0)
            h = max(y2 - y1, 1.0)
            if (x2 - x1) > 1.6 * max(y2 - y1, 1):
                det.evidence['width_height_ratio'] = (x2 - x1) / max(y2 - y1, 1)
                if not (context_enabled and context.get('geometry_requires_competing_evidence')):
                    det.require_review('upright_object_scope_uncertain')
            elif cid == ClassId.UTILITY_POLE:
                aspect = h / w
                max_w = float(rules.get('pole_max_width_non_slender', 55.0))
                min_aspect = float(rules.get('pole_max_aspect_ratio_non_slender', 2.8))
                if w > max_w and aspect < min_aspect and det.score < 0.45:
                    det.evidence['pole_width'] = round(w, 1)
                    det.evidence['aspect_ratio'] = round(aspect, 2)
                    if not (context_enabled and context.get('geometry_requires_competing_evidence')):
                        det.require_review('pole_width_or_aspect_non_slender')

                if surface_image is not None:
                    img_arr = np.asarray(surface_image)
                    ih, iw = img_arr.shape[:2]
                    top_h = max(20, min(int(h * 0.8), 120))
                    cy1 = max(0, int(y1 - top_h))
                    cy2 = max(0, int(y1))
                    cx1 = max(0, int(x1 - w * 0.3))
                    cx2 = min(iw, int(x2 + w * 0.3))
                    if cy2 > cy1 and cx2 > cx1:
                        canopy_patch = img_arr[cy1:cy2, cx1:cx2, :3].astype(float)
                        if canopy_patch.size > 0:
                            greenness = float(np.mean(canopy_patch[..., 1] - (canopy_patch[..., 0] + canopy_patch[..., 2]) / 2.0))
                            det.evidence['canopy_greenness'] = round(greenness, 2)
                            min_green = float(rules.get('pole_tree_canopy_min_greenness', 12.0))
                            if greenness >= min_green and not (
                                    context_enabled and context.get('green_background_is_not_negative_evidence')):
                                det.require_review('pole_has_tree_canopy_misclassified_tree')
        if context_enabled and cid in (ClassId.UTILITY_POLE, ClassId.TREE_TRUNK):
            # Competing model hypotheses matter; width/foliage alone do not.
            for group in ('column', 'wall'):
                for counter in counters.get(group, []):
                    if (counter.get('score', 0) >= max(.30, det.score+context.get('competing_score_margin', .05))
                            and same_object(det.bbox_xyxy, counter['bbox_xyxy'])):
                        relate(det, counter, 'upright_structure_competing_'+group)
        if cid in (ClassId.PLASTIC_TABLE, ClassId.PLASTIC_CHAIR):
            det.evidence['material_status'] = 'unverified_prompt_hypothesis'
            if cid == ClassId.PLASTIC_TABLE:
                x1, y1, x2, y2 = det.bbox_xyxy
                if (y2 - y1) > 1.4 * max(x2 - x1, 1):
                    det.require_review('table_aspect_ratio_tall_post')
            else:
                x1, y1, x2, y2 = det.bbox_xyxy
                aspect = (x2-x1)/max(y2-y1, 1)
                def strong(group):
                    return any(c.get('score', 0) >= det.score
                               and calculate_iou(det.bbox_xyxy, c['bbox_xyxy']) >= .6
                               for c in counters.get(group, []))
                neutral = False
                surface = {'width_height_ratio': aspect}
                if surface_image is not None:
                    image = np.asarray(surface_image)
                    ih, iw = image.shape[:2]
                    crop = image[max(0, int(y1)):min(ih, math.ceil(y2)),
                                 max(0, int(x1)):min(iw, math.ceil(x2)), :3].astype(float)
                    if crop.size:
                        # Channel-order independent; colour alone never excludes a chair.
                        chroma = (crop.max(2)-crop.min(2))/np.maximum(crop.max(2), 1)
                        surface['neutral_fraction'] = float(np.mean(chroma < .15))
                        surface['intensity_std'] = float(crop.mean(2).std()/255.)
                        neutral = surface['neutral_fraction'] >= .8
                counter_bench = (aspect >= rules.get('public_bench_min_aspect', 1.65)
                                 and strong('nonplastic_chair')
                                 and (strong('construction_materials') or neutral))
                # Weak, elongated neutral seating is ambiguous even when the
                # counter head emits no box. Preserve it for review, never
                # assert concrete material from colour or shape alone.
                elongated_seating = (aspect >= rules.get('elongated_seating_min_aspect', 2.8)
                                     and neutral
                                     and det.score <= rules.get('elongated_seating_max_score', .55))
                bench = counter_bench or elongated_seating
                det.evidence['chair_surface'] = surface
                if bench:
                    det.subtype = 'public_bench_candidate'
                    det.require_review('public_bench_geometry_and_counter_evidence' if counter_bench
                                       else 'elongated_neutral_seating_scope_uncertain')
                elif strong('nonplastic_chair'):
                    det.evidence['material_status'] = 'nonplastic_seating_candidate'
        if cid == ClassId.ADVERTISING_SIGN:
            x1, y1, x2, y2 = det.bbox_xyxy
            if context_enabled:
                dimensions = image_size
                if dimensions is None and surface_image is not None:
                    dimensions = (np.asarray(surface_image).shape[1], np.asarray(surface_image).shape[0])
                for region in context.get('known_watermark_regions', []):
                    if dimensions is not None:
                        width, height = dimensions
                        rectangle = [region[0]*width, region[1]*height, region[2]*width, region[3]*height]
                        if calculate_ios(det.bbox_xyxy, rectangle) >= context.get('watermark_min_box_coverage', .9):
                            det.require_review('configured_watermark_region')
            elif y2 >= 1020 and y1 >= 900 and (x2 - x1) > 200:
                det.require_review('video_watermark_text_ignored')
        if cid == ClassId.VENDING_CART:
            det.evidence["sale_context_status"] = "sale_context_unknown"
            if re.search(r"\b(bicycle|motorcycle)\b", (det.raw_label or "").lower()):
                det.require_review("vendor_vehicle_sale_context_unknown")
        if cid == ClassId.GOODS:
            phrase_threshold = rules.get('phrase_review_thresholds', {}).get(det.raw_label)
            if phrase_threshold is not None and det.score < phrase_threshold:
                det.require_review('weak_packaged_goods_content_unverified')
            words = (det.raw_label or "").lower()
            content = bool(re.search(r"\b(fruit|fruits|vegetable|vegetables|products|packaged)\b", words))
            container = bool(re.search(r"\b(basket|crate|box|container)\b", words))
            det.evidence.update({"observed_subtype": det.subtype,
                "content_status": "content_candidate" if content else
                                  ("container_only" if container else "content_unknown"),
                    "sale_context_status": "sale_context_unknown",
                    "verification_status": "unverified",
                "source": "raw_model_phrase_only"})
            if not content and not container and det.score < 0.22:
                det.require_review(det.evidence["content_status"] + "; sale_context_unknown")

        counter_pairs = {
            2: [("bicycle", "motorcycle_bicycle_overlap")],
            ClassId.PLASTIC_TABLE: [("car", "table_car_overlap"),
                ("construction_materials", "table_materials_overlap"), ("foam_box", "table_foam_box_overlap")],
            ClassId.PLASTIC_CHAIR: [("foam_box", "chair_foam_box_overlap"),
                ("construction_materials", "chair_materials_overlap"),
                ("nonplastic_chair", "chair_nonplastic_overlap")],
            6: [("barrier_gate", "cart_barrier_overlap"), ("car", "cart_car_overlap"),
                ("delivery_vehicle", "cart_delivery_vehicle_overlap"),
                ("stroller", "cart_stroller_overlap"), ("column", "cart_column_overlap"),
                ("wall", "cart_wall_overlap"), ("fence", "cart_fence_overlap")],
            ClassId.UTILITY_POLE: [("tree_trunk", "pole_tree_counter_overlap")],
            ClassId.ADVERTISING_SIGN: [("traffic_sign", "advertising_traffic_overlap"),
                                     ("canopy", "sign_canopy_overlap")],
            ClassId.GOODS: [("potted_plant", "goods_plant_overlap"),
                 ("construction_materials", "goods_materials_overlap")],
        }
        for group, name in rules.get('counter_pairs', counter_pairs).get(cid, []):
            # Prompt 61 is a metal folding chair, also used by street vendors.
            # It cannot establish a fixed public bench without additional evidence.
            if cid == ClassId.PLASTIC_CHAIR and group == 'nonplastic_chair':
                continue
            if enabled(name):
                for counter in counters.get(group, []):
                    relate(det, counter, name)
        if enabled("ambiguous_counter_overlap"):
            applicable = {g for g, _ in counter_pairs.get(cid, [])}
            for counter in counters.get("_ambiguous", []):
                if applicable.intersection(counter.get("candidate_counter_groups", [])):
                    relate(det, counter, "ambiguous_counter_overlap")

        object_pairs = {
            ClassId.PLASTIC_TABLE: [(ClassId.VENDING_CART, "contained_in_cart_candidate"), (ClassId.CAR, "table_car_overlap")],
            ClassId.VENDING_CART: [(ClassId.MOTORCYCLE, "cart_motorcycle_overlap"), (ClassId.CAR, "cart_car_overlap")],
            ClassId.UTILITY_POLE: [(ClassId.TREE_TRUNK, "pole_trunk_overlap")],
            ClassId.GOODS: [(ClassId.PLASTIC_TABLE, "overlap_with_table"), (ClassId.PLASTIC_CHAIR, "goods_chair_overlap")],
        }
        if context_enabled and context.get('pole_tree_same_object_only'):
            object_pairs[ClassId.TREE_TRUNK] = [(ClassId.UTILITY_POLE, 'trunk_pole_overlap')]
        for other_cid, name in object_pairs.get(cid, []):
            if enabled(name):
                for other in detections:
                    if other.class_id == other_cid and not other.candidate_class_ids:
                        relate(det, other, name,
                               iou_limit=1.01 if name in ("overlap_with_table", "contained_in_cart_candidate") else (.25 if name == "pole_trunk_overlap" else .35),
                               containment=.70 if name in ("overlap_with_table", "contained_in_cart_candidate") else (.45 if name == "pole_trunk_overlap" else .65))
        if cid == ClassId.TREE_TRUNK and enabled("tree_scope_uncertain"):
            for other in detections:
                if other is not det and other.class_id == ClassId.TREE_TRUNK:
                    if (rules.get('weak_trunk_candidates', False) and other.score < .18
                            and not has_trunk_extent(other.bbox_xyxy)):
                        continue  # A rejected tiny fragment cannot veto a strong trunk.
                    if (calculate_ios(det.bbox_xyxy, other.bbox_xyxy) >= .85 or
                        calculate_ios(other.bbox_xyxy, det.bbox_xyxy) >= .85) and calculate_iou(det.bbox_xyxy, other.bbox_xyxy) < .35:
                        relate(det, other, "tree_scope_uncertain", containment=.85)
        (review if det.status == "review" else predicted).append(det)

    return predicted, review, reports


def deduplicate_all_detections(detections: list, img_w: int, img_h: int) -> tuple:
    """Compatibility helper: suppress same-class duplicates only.

    Position, size and overlap with another class cannot establish a label.
    Spatial ambiguity is handled by analyze_conflicts_and_partition.
    """
    kept = apply_intra_class_nms(detections, iou_threshold=.5)
    kept_ids = {id(d) for d in kept}
    return kept, [d for d in detections if id(d) not in kept_ids]


def partition_road_surface_hypotheses(rows, road_probability, rules=None, sidewalk_probability=None):
    """Screen weak signs/poles against independent drivable-surface evidence.

    Road probability is the explicit surface channel, never the binary
    non-sidewalk mask. Missing/invalid evidence leaves detections unchanged.
    No shape, colour, or source-coordinate exception is used.
    """
    rules = rules or {}
    if not rules.get('road_surface_consistency', False) or road_probability is None:
        return rows, []
    probability = np.asarray(road_probability)
    if (probability.ndim != 2 or not np.isfinite(probability).all()
            or np.any(probability < 0) or np.any(probability > 1)):
        return rows, []
    h, w = probability.shape
    sidewalk = np.asarray(sidewalk_probability) if sidewalk_probability is not None else None
    if (sidewalk is not None and (sidewalk.shape != probability.shape
            or not np.isfinite(sidewalk).all() or np.any(sidewalk < 0) or np.any(sidewalk > 1))):
        sidewalk = None
    kept, review = [], []
    for row in rows:
        if row.get('class_id') not in (7, 8, 9) or row.get('score', 1.) >= .5:
            kept.append(row)
            continue
        x1,y1,x2,y2 = row['bbox_xyxy']
        left,top,right,bottom = max(0,math.floor(x1)),max(0,math.floor(y1)),min(w,math.ceil(x2)),min(h,math.ceil(y2))
        if right <= left or bottom <= top:
            kept.append(row)
            continue
        patch = probability[top:bottom,left:right]
        road_fraction = float(np.mean(patch >= .8))
        row = {**row, 'evidence': {**row.get('evidence', {}),
               'road_surface_fraction': road_fraction, 'road_mean_probability': float(patch.mean())}}
        boundary = False
        if (sidewalk is not None and row.get('class_id') == 8
                and (y2-y1) >= 8.*max(x2-x1, 1.)):
            sidewalk_patch = sidewalk[top:bottom,left:right]
            ground_fraction = float(np.mean(patch+sidewalk_patch >= .8))
            boundary = (ground_fraction >= .90 and float(patch.mean()) >= .35
                        and float(sidewalk_patch.mean()) >= .15
                        and float((patch+sidewalk_patch).mean()) >= .85)
            row['evidence'].update(pavement_fraction=ground_fraction,
                sidewalk_mean_probability=float(sidewalk_patch.mean()))
        if boundary or (road_fraction >= .85 and float(patch.mean()) >= .85):
            reason = ('weak_pole_on_surface_boundary' if boundary
                      else 'weak_object_on_drivable_surface')
            row['status_history'] = [*row.get('status_history', []),
                {'from':row.get('status','predicted'),'to':'review','reason':reason}]
            row.update(status='review',review_reason='; '.join(filter(None,[row.get('review_reason'),reason])))
            review.append(row)
        else:
            kept.append(row)
    return kept, review
