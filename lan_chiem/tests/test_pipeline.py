"""
Kiểm thử tích hợp pipeline hoàn chỉnh phiên bản 2.1:
- Chạy thực tế Grounding DINO từ .hf_cache.
- Xác thực đầy đủ cấu trúc JSON phân vùng:
  + object_detections (xác nhận)
  + review_candidates (ứng viên cần kiểm tra)
  + conflict_reports (báo cáo xung đột)
  + sidewalk_regions (vùng vỉa hè ID 0)
  + action_candidates
  + class_support_status
  + debug_trace (rejections_by_stage, rejections_by_reason, counter_objects_detected)
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw

from lan_chiem.detector import GroundingDINODetector
from lan_chiem.sidewalk import SidewalkRegion
from lan_chiem.visualization import draw_detections


def test_full_pipeline_2_1():
    img = Image.new("RGB", (400, 400), color=(235, 235, 235))
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 60, 160, 250], fill=(40, 140, 40))   # Cây / cột
    draw.rectangle([250, 200, 380, 320], fill=(210, 60, 60)) # Bàn / ghế

    detector = GroundingDINODetector(device="cpu", cache_dir=".hf_cache")

    # Tạo giả lập 1 vùng vỉa hè
    sw_sample = [
        SidewalkRegion(
            polygon=[[0.0, 300.0], [400.0, 300.0], [400.0, 400.0], [0.0, 400.0]],
            source="manual_test",
            confidence=1.0,
        )
    ]

    result = detector.predict(
        image_input=img,
        mode="full",
        box_threshold_override=0.20,
        text_threshold_override=0.20,
        nms_iou=0.50,
        debug=True,
        sidewalk_regions=sw_sample,
    )

    # 1. Kiểm tra các trường cấu trúc bắt buộc
    assert "object_detections" in result
    assert "review_candidates" in result
    assert "conflict_reports" in result
    assert "sidewalk_regions" in result
    assert len(result["sidewalk_regions"]) == 1
    assert "action_candidates" in result
    assert "class_support_status" in result
    assert "summary" in result
    assert "debug_trace" in result

    # 2. Kiểm tra summary
    summary = result["summary"]
    assert "total_confirmed_detections" in summary
    assert "total_review_candidates" in summary
    assert "total_conflicts_detected" in summary
    assert "confirmed_class_counts" in summary
    assert "review_class_counts" in summary

    # 3. Kiểm tra debug_trace
    debug_trace = result["debug_trace"]
    assert "rejections_by_stage" in debug_trace
    assert "rejections_by_reason" in debug_trace
    assert "rejected_records" in debug_trace
    assert "counter_objects_detected" in debug_trace

    # 4. Kiểm tra serialize JSON sạch
    clean_json = {k: v for k, v in result.items() if not k.startswith("_")}
    json_str = json.dumps(clean_json, ensure_ascii=False, indent=2)
    assert len(json_str) > 0

    # 5. Kiểm tra vẽ chú thích ảnh kết quả kiểm tra chính
    annotated_clean = draw_detections(
        img,
        result["object_detections"],
        label_mode="short",
        sidewalk_regions=result["sidewalk_regions"],
    )
    assert isinstance(annotated_clean, Image.Image)
