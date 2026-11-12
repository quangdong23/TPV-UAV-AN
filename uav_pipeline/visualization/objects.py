"""
Module visualization: Vẽ bounding box và nhãn tiếng Việt trên ảnh (Phiên bản 2.1).
Cải tiến:
- Tách bạch xuất ảnh kết quả:
  + draw_detections: Vẽ các giả thuyết predicted và vùng vỉa hè; chưa xác minh đúng.
  + draw_candidates: Vẽ cả đối tượng xác nhận và ứng viên cần kiểm tra (review_candidates) với viền nổi bật (Amber/Gold), ký hiệu [CAN: ID], giúp người dùng đối chiếu mà không gây nhiễu ảnh chính.
- 4 chế độ hiển thị nhãn: "short", "full", "id_only", "none".
- Giảm chồng lấn nhãn: Thuật toán so le và tự động đổi vị trí nhãn (trên/dưới/trong box).
- Font Unicode chuẩn hệ thống (Arial, Segoe UI, Tahoma,...) hiển thị tiếng Việt có dấu sắc nét.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

from uav_pipeline.detection.registry import ClassRegistry

CANDIDATE_FONTS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/Library/Fonts/Arial.ttf",
]


def load_unicode_font(font_size: int = 16) -> ImageFont.ImageFont:
    """Nạp font TrueType hỗ trợ Unicode tiếng Việt."""
    for font_path in CANDIDATE_FONTS:
        if os.path.isfile(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:
        return ImageFont.load_default()


def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """Lấy màu RGB của class từ registry."""
    meta = ClassRegistry.get(class_id)
    if meta:
        return meta.default_color
    return (128, 128, 128)


def _draw_single_box_and_badge(
    draw: ImageDraw.ImageDraw,
    bbox: List[float],
    cid: int,
    cname_vi: str,
    raw_label: str,
    score: float,
    subtype: Optional[str],
    is_candidate: bool,
    review_reason: Optional[str],
    conflict_info: Optional[Dict[str, Any]],
    label_mode: str,
    color: Tuple[int, int, int],
    line_width: int,
    font: ImageFont.ImageFont,
    img_w: int,
    img_h: int,
    occupied_badge_boxes: List[Tuple[float, float, float, float]],
) -> None:
    x1, y1, x2, y2 = bbox

    # 1. Vẽ khung chữ nhật
    if not is_candidate:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
    else:
        # Ứng viên vẽ màu Amber / Gold viền đôi
        cand_color = (255, 179, 0)
        draw.rectangle([x1, y1, x2, y2], outline=cand_color, width=line_width)
        offset = line_width + 1
        if x2 - x1 > 2 * offset and y2 - y1 > 2 * offset:
            draw.rectangle([x1 + offset, y1 + offset, x2 - offset, y2 - offset], outline=(255, 235, 59), width=1)

    if label_mode == "none":
        return

    # 2. Xây dựng nội dung nhãn
    sub_str = f" ({subtype})" if subtype else ""
    if cid is None:
        label_text = f"[RAW] {raw_label or cname_vi} {score:.2f}" if not is_candidate else f"[REVIEW] {cname_vi} {score:.2f} [{raw_label}]"
    elif is_candidate:
        c_tag = conflict_info.get("conflict_type", "review") if conflict_info else "review"
        if label_mode == "full":
            label_text = f"[CAN: {cid}] {cname_vi} {score:.2f}{sub_str} [{c_tag}] [{raw_label}]"
        elif label_mode == "id_only":
            label_text = f"[CAN: {cid}] {score:.2f}"
        else:  # short
            label_text = f"[CAN: {cid}] {cname_vi} {score:.2f} ({c_tag})"
    else:
        if label_mode == "full":
            label_text = f"[{cid}] {cname_vi} {score:.2f}{sub_str} [{raw_label}]"
        elif label_mode == "id_only":
            label_text = f"[{cid}] {score:.2f}"
        else:  # short
            label_text = f"[{cid}] {cname_vi} {score:.2f}"

    # 3. Kích thước badge
    try:
        bbox_t = draw.textbbox((0, 0), label_text, font=font)
        text_w = bbox_t[2] - bbox_t[0]
        text_h = bbox_t[3] - bbox_t[1]
    except AttributeError:
        text_w, text_h = draw.textsize(label_text, font=font)

    padding = 3
    badge_w = text_w + padding * 2
    badge_h = text_h + padding * 2

    # 4. Tìm vị trí đặt badge tránh đè nhau
    pos_candidates = [
        (max(0.0, x1), y1 - badge_h),
        (max(0.0, x1), y1 + line_width),
        (max(0.0, x1), y2),
        (max(0.0, x1 + badge_w + 5), y1 - badge_h),
    ]

    chosen_x = max(0.0, x1)
    chosen_y = y1 - badge_h if (y1 - badge_h >= 0) else y1 + line_width

    for cx, cy in pos_candidates:
        if cy < 0 or cy + badge_h > img_h or cx + badge_w > img_w:
            continue
        candidate_box = (cx, cy, cx + badge_w, cy + badge_h)
        has_overlap = False
        for ob in occupied_badge_boxes:
            if not (candidate_box[2] < ob[0] or candidate_box[0] > ob[2] or candidate_box[3] < ob[1] or candidate_box[1] > ob[3]):
                has_overlap = True
                break
        if not has_overlap:
            chosen_x, chosen_y = cx, cy
            break

    badge_x1 = max(0.0, min(float(img_w - badge_w), chosen_x))
    badge_y1 = max(0.0, min(float(img_h - badge_h), chosen_y))
    badge_x2 = badge_x1 + badge_w
    badge_y2 = badge_y1 + badge_h

    occupied_badge_boxes.append((badge_x1, badge_y1, badge_x2, badge_y2))

    # Màu nền badge
    badge_fill = (255, 179, 0) if is_candidate else color
    draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=badge_fill)

    # Màu chữ tương phản
    luminance = 0.299 * badge_fill[0] + 0.587 * badge_fill[1] + 0.114 * badge_fill[2]
    text_color = (0, 0, 0) if luminance > 155 else (255, 255, 255)

    draw.text(
        (badge_x1 + padding, badge_y1 + padding),
        label_text,
        fill=text_color,
        font=font,
    )


def draw_detections(
    image: Image.Image,
    detections: List[Any],
    label_mode: str = "short",
    filter_class_ids: Optional[Set[int]] = None,
    sidewalk_regions: Optional[List[Any]] = None,
    font_size: Optional[int] = None,
    line_width: Optional[int] = None,
) -> Image.Image:
    """Vẽ predicted + vỉa hè, hoặc nhãn RAW cho truy vấn không có class ID."""
    annotated = image.copy().convert("RGBA")
    img_w, img_h = annotated.size

    # 1. Vẽ vỉa hè (ID 0)
    if sidewalk_regions:
        sidewalk_overlay = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
        sw_draw = ImageDraw.Draw(sidewalk_overlay)
        for r in sidewalk_regions:
            poly = r.get("polygon", []) if isinstance(r, dict) else getattr(r, "polygon", [])
            if len(poly) >= 3:
                flat_poly = [(float(pt[0]), float(pt[1])) for pt in poly]
                sw_draw.polygon(flat_poly, fill=(0, 191, 255, 55))
                sw_draw.line(flat_poly + [flat_poly[0]], fill=(0, 140, 240, 200), width=max(2, int(img_w / 400)))
        annotated = Image.alpha_composite(annotated, sidewalk_overlay)

    draw = ImageDraw.Draw(annotated)
    min_dim = min(img_w, img_h)
    lw = line_width or max(2, int(min_dim / 350))
    fs = font_size or max(12, int(min_dim / 55))
    font = load_unicode_font(fs)

    occupied_badge_boxes: List[Tuple[float, float, float, float]] = []

    for det in detections:
        cid = getattr(det, "class_id", None) if hasattr(det, "class_id") else det.get("class_id")
        if filter_class_ids is not None and cid not in filter_class_ids:
            continue

        cname_vi = getattr(det, "class_name_vi", "") if hasattr(det, "class_name_vi") else det.get("class_name_vi", det.get("raw_label", det.get("prompt", "Raw prediction")))
        raw_label = getattr(det, "raw_label", "") if hasattr(det, "raw_label") else det.get("raw_label", "")
        score = getattr(det, "score", 0.0) if hasattr(det, "score") else det.get("score", 0.0)
        bbox = getattr(det, "bbox_xyxy", [0, 0, 0, 0]) if hasattr(det, "bbox_xyxy") else det.get("bbox_xyxy", [0, 0, 0, 0])
        subtype = getattr(det, "subtype", None) if hasattr(det, "subtype") else det.get("subtype")
        color = get_class_color(cid)

        _draw_single_box_and_badge(
            draw=draw,
            bbox=bbox,
            cid=cid,
            cname_vi=cname_vi,
            raw_label=raw_label,
            score=score,
            subtype=subtype,
            is_candidate=False,
            review_reason=None,
            conflict_info=None,
            label_mode=label_mode,
            color=color,
            line_width=lw,
            font=font,
            img_w=img_w,
            img_h=img_h,
            occupied_badge_boxes=occupied_badge_boxes,
        )

    return annotated.convert("RGB")


def draw_candidates_overlay(
    image: Image.Image,
    confirmed_detections: List[Any],
    review_candidates: List[Any],
    label_mode: str = "short",
    sidewalk_regions: Optional[List[Any]] = None,
    font_size: Optional[int] = None,
    line_width: Optional[int] = None,
) -> Image.Image:
    """Vẽ predicted và review. Tên tham số confirmed_detections giữ để tương thích."""
    annotated = image.copy().convert("RGBA")
    img_w, img_h = annotated.size

    # 1. Vẽ vỉa hè
    if sidewalk_regions:
        sidewalk_overlay = Image.new("RGBA", annotated.size, (255, 255, 255, 0))
        sw_draw = ImageDraw.Draw(sidewalk_overlay)
        for r in sidewalk_regions:
            poly = r.get("polygon", []) if isinstance(r, dict) else getattr(r, "polygon", [])
            if len(poly) >= 3:
                flat_poly = [(float(pt[0]), float(pt[1])) for pt in poly]
                sw_draw.polygon(flat_poly, fill=(0, 191, 255, 45))
                sw_draw.line(flat_poly + [flat_poly[0]], fill=(0, 140, 240, 180), width=max(2, int(img_w / 400)))
        annotated = Image.alpha_composite(annotated, sidewalk_overlay)

    draw = ImageDraw.Draw(annotated)
    min_dim = min(img_w, img_h)
    lw = line_width or max(2, int(min_dim / 350))
    fs = font_size or max(12, int(min_dim / 55))
    font = load_unicode_font(fs)

    occupied_badge_boxes: List[Tuple[float, float, float, float]] = []

    # Vẽ confirmed trước
    for det in confirmed_detections:
        cid = getattr(det, "class_id", None) if hasattr(det, "class_id") else det.get("class_id", -1)
        cname_vi = getattr(det, "class_name_vi", "") if hasattr(det, "class_name_vi") else det.get("class_name_vi", str(cid))
        raw_label = getattr(det, "raw_label", "") if hasattr(det, "raw_label") else det.get("raw_label", "")
        score = getattr(det, "score", 0.0) if hasattr(det, "score") else det.get("score", 0.0)
        bbox = getattr(det, "bbox_xyxy", [0, 0, 0, 0]) if hasattr(det, "bbox_xyxy") else det.get("bbox_xyxy", [0, 0, 0, 0])
        subtype = getattr(det, "subtype", None) if hasattr(det, "subtype") else det.get("subtype")
        color = get_class_color(cid)

        _draw_single_box_and_badge(
            draw=draw,
            bbox=bbox,
            cid=cid,
            cname_vi=cname_vi,
            raw_label=raw_label,
            score=score,
            subtype=subtype,
            is_candidate=False,
            review_reason=None,
            conflict_info=None,
            label_mode=label_mode,
            color=color,
            line_width=lw,
            font=font,
            img_w=img_w,
            img_h=img_h,
            occupied_badge_boxes=occupied_badge_boxes,
        )

    # Vẽ review_candidates với viền nổi bật
    for det in review_candidates:
        cid = getattr(det, "class_id", None) if hasattr(det, "class_id") else det.get("class_id", -1)
        cname_vi = getattr(det, "class_name_vi", "") if hasattr(det, "class_name_vi") else det.get("class_name_vi", str(cid))
        candidate_ids = getattr(det, "candidate_class_ids", []) if hasattr(det, "candidate_class_ids") else det.get("candidate_class_ids", [])
        if candidate_ids:
            cname_vi += " " + "/".join(map(str, candidate_ids))
        raw_label = getattr(det, "raw_label", "") if hasattr(det, "raw_label") else det.get("raw_label", "")
        score = getattr(det, "score", 0.0) if hasattr(det, "score") else det.get("score", 0.0)
        bbox = getattr(det, "bbox_xyxy", [0, 0, 0, 0]) if hasattr(det, "bbox_xyxy") else det.get("bbox_xyxy", [0, 0, 0, 0])
        subtype = getattr(det, "subtype", None) if hasattr(det, "subtype") else det.get("subtype")
        rev_reason = getattr(det, "review_reason", None) if hasattr(det, "review_reason") else det.get("review_reason")
        conf_info = getattr(det, "conflict_info", None) if hasattr(det, "conflict_info") else det.get("conflict_info")

        _draw_single_box_and_badge(
            draw=draw,
            bbox=bbox,
            cid=cid,
            cname_vi=cname_vi,
            raw_label=raw_label,
            score=score,
            subtype=subtype,
            is_candidate=True,
            review_reason=rev_reason,
            conflict_info=conf_info,
            label_mode=label_mode,
            color=(255, 179, 0),
            line_width=lw + 1,
            font=font,
            img_w=img_w,
            img_h=img_h,
            occupied_badge_boxes=occupied_badge_boxes,
        )

    return annotated.convert("RGB")
