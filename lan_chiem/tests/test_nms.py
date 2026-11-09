"""
Kiểm thử NMS nội bộ nhóm, hợp nhất box chân (leg consolidation), lọc hình học thân cây, và bảo tồn ghế bị che khuất.
"""

from lan_chiem.detector import (
    Detection,
    apply_intra_class_nms,
    calculate_ios,
    calculate_iou,
    consolidate_person_boxes,
    filter_tree_trunk_geometry,
)


def test_intra_class_nms():
    """Hai box cùng lớp 5 (ghế) giao nhau mạnh -> chỉ giữ box điểm cao hơn."""
    d1 = Detection(
        id="det_1",
        class_id=5,
        class_name="ghế",
        class_name_vi="Ghế",
        raw_label="chair",
        prompt_group="tables_chairs",
        source="full",
        score=0.85,
        bbox_xyxy=[10.0, 10.0, 100.0, 100.0],
    )
    d2 = Detection(
        id="det_2",
        class_id=5,
        class_name="ghế",
        class_name_vi="Ghế",
        raw_label="plastic chair",
        prompt_group="tables_chairs",
        source="tile_0_0",
        score=0.72,
        bbox_xyxy=[12.0, 11.0, 101.0, 99.0],
    )

    kept = apply_intra_class_nms([d1, d2], iou_threshold=0.50)
    assert len(kept) == 1
    assert kept[0].id == "det_1"
    assert kept[0].score == 0.85


def test_inter_class_preservation():
    """Hai box thuộc hai lớp khác nhau (Người ID 1 và Ghế ID 5) chồng lấn -> cả 2 đều được giữ lại (không NMS chéo)."""
    d1 = Detection(
        id="det_1",
        class_id=1,
        class_name="người",
        class_name_vi="Người",
        raw_label="person",
        prompt_group="people_vehicles",
        source="full",
        score=0.90,
        bbox_xyxy=[50.0, 20.0, 120.0, 150.0],
    )
    d2 = Detection(
        id="det_2",
        class_id=5,
        class_name="ghế",
        class_name_vi="Ghế",
        raw_label="chair",
        prompt_group="tables_chairs",
        source="full",
        score=0.80,
        bbox_xyxy=[45.0, 60.0, 125.0, 160.0],
    )

    kept = apply_intra_class_nms([d1, d2], iou_threshold=0.50)
    assert len(kept) == 2


def test_person_leg_consolidation():
    """Box người chỉ bao phần chân (subpart) được phát hiện và hợp nhất vào box toàn thân."""
    d_full = Detection(
        id="det_person_full",
        class_id=1,
        class_name="người",
        class_name_vi="Người",
        raw_label="person",
        prompt_group="people_vehicles",
        source="full",
        score=0.85,
        bbox_xyxy=[100.0, 50.0, 180.0, 300.0],  # Full body H=250, W=80
    )
    d_legs = Detection(
        id="det_person_legs",
        class_id=1,
        class_name="người",
        class_name_vi="Người",
        raw_label="person",
        prompt_group="people_vehicles",
        source="full",
        score=0.60,
        bbox_xyxy=[105.0, 180.0, 175.0, 298.0],  # Legs only: starts at Y=180 (> 35% H)
    )

    consolidated = consolidate_person_boxes([d_full, d_legs])
    assert len(consolidated) == 1
    assert consolidated[0].id == "det_person_full"


def test_tree_trunk_geometry_filter():
    """Giữ cả box cây rộng với review; hình học không tự chứng minh phạm vi thân."""
    d_trunk = Detection(
        id="det_trunk",
        class_id=9,
        class_name="cây",
        class_name_vi="Thân cây",
        raw_label="tree trunk",
        prompt_group="trunks_poles",
        source="full",
        score=0.75,
        bbox_xyxy=[500.0, 200.0, 560.0, 700.0],  # W=60, H=500 (thân đứng)
    )
    d_canopy = Detection(
        id="det_canopy",
        class_id=9,
        class_name="cây",
        class_name_vi="Thân cây",
        raw_label="tree",
        prompt_group="trunks_poles",
        source="full",
        score=0.65,
        bbox_xyxy=[10.0, 10.0, 1800.0, 950.0],  # Chiếm gần trọn ảnh
    )

    filtered = filter_tree_trunk_geometry([d_trunk, d_canopy], img_width=1920, img_height=1080)
    assert len(filtered) == 2
    assert filtered[0].id == "det_trunk"
    assert filtered[1].status == "review"
    assert filtered[1].bbox_xyxy == d_canopy.bbox_xyxy
