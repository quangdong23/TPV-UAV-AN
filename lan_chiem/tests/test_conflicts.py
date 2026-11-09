"""
Kiểm thử phân tích xung đột phân lớp (Conflict Detection) và phân tách Review Candidates.
"""

from lan_chiem.detector import Detection, GroundingDINODetector


def test_conflict_cart_barrier_overlap():
    """Trụ barie / cổng an ninh bị gán xe đẩy bán hàng -> chuyển sang review_candidates kèm báo cáo xung đột."""
    detector = GroundingDINODetector()
    cart_det = Detection(
        id="det_cart_err",
        class_id=6,
        class_name="xe đẩy bán hàng",
        class_name_vi="Xe đẩy bán hàng",
        raw_label="vendor pushcart",
        prompt_group="vendor_vehicles",
        source="full",
        score=0.35,
        bbox_xyxy=[20.0, 15.0, 125.0, 210.0],  # Trụ barie
    )
    counter_obs = {
        "barrier_gate": [
            {"score": 0.55, "bbox_xyxy": [21.0, 16.0, 124.0, 209.0], "label": "security gate post"}
        ],
        "car": [],
        "potted_plant": [],
        "construction_materials": [],
        "bicycle": [],
    }

    confirmed, review, conflicts = detector.analyze_conflicts_and_partition(
        detections=[cart_det],
        counter_objects=counter_obs,
    )

    assert len(confirmed) == 0
    assert len(review) == 1
    assert review[0].id == "det_cart_err"
    assert review[0].status == "review"
    assert len(conflicts) == 1
    assert conflicts[0]["conflict_type"] == "cart_barrier_overlap"


def test_conflict_cart_car_overlap():
    """Xe ô tô trong bãi xe bị gán xe đẩy bán hàng -> chuyển sang review_candidates."""
    detector = GroundingDINODetector()
    cart_det = Detection(
        id="det_car_err",
        class_id=6,
        class_name="xe đẩy bán hàng",
        class_name_vi="Xe đẩy bán hàng",
        raw_label="street vending cart",
        prompt_group="vendor_vehicles",
        source="full",
        score=0.29,
        bbox_xyxy=[1040.0, 580.0, 1660.0, 1020.0],  # Chiếc ô tô lớn
    )
    counter_obs = {
        "car": [
            {"score": 0.45, "bbox_xyxy": [1045.0, 585.0, 1662.0, 1022.0], "label": "car"}
        ],
        "barrier_gate": [],
        "potted_plant": [],
        "construction_materials": [],
        "bicycle": [],
    }

    confirmed, review, conflicts = detector.analyze_conflicts_and_partition(
        detections=[cart_det],
        counter_objects=counter_obs,
    )

    assert len(confirmed) == 0
    assert len(review) == 1
    assert conflicts[0]["conflict_type"] == "cart_car_overlap"


def test_conflict_goods_plant_overlap():
    """Chậu cây cảnh bị gán thành hàng hóa -> chuyển sang review_candidates."""
    detector = GroundingDINODetector()
    goods_det = Detection(
        id="det_plant_err",
        class_id=10,
        class_name="hàng hóa",
        class_name_vi="Hàng hóa",
        raw_label="basket of fruit",
        prompt_group="goods",
        source="full",
        score=0.32,
        bbox_xyxy=[60.0, 395.0, 180.0, 515.0],
    )
    counter_obs = {
        "potted_plant": [
            {"score": 0.44, "bbox_xyxy": [61.0, 398.0, 178.0, 514.0], "label": "potted plant"}
        ],
        "car": [],
        "barrier_gate": [],
        "construction_materials": [],
        "bicycle": [],
    }

    confirmed, review, conflicts = detector.analyze_conflicts_and_partition(
        detections=[goods_det],
        counter_objects=counter_obs,
    )

    assert len(confirmed) == 0
    assert len(review) == 1
    assert conflicts[0]["conflict_type"] == "goods_plant_overlap"


def test_conflict_contained_in_cart_candidate():
    """Mặt trên của xe bán hàng bị nhận nhầm thành bàn -> ghi vết xung đột."""
    detector = GroundingDINODetector()
    cart = Detection(
        id="det_cart_real",
        class_id=6,
        class_name="xe đẩy bán hàng",
        class_name_vi="Xe đẩy bán hàng",
        raw_label="food vending cart",
        prompt_group="vendor_vehicles",
        source="full",
        score=0.50,
        bbox_xyxy=[540.0, 430.0, 750.0, 670.0],
    )
    table_on_cart = Detection(
        id="det_table_err",
        class_id=4,
        class_name="bàn",
        class_name_vi="Bàn",
        raw_label="table",
        prompt_group="tables_chairs",
        source="full",
        score=0.38,
        bbox_xyxy=[550.0, 435.0, 745.0, 570.0],  # Nằm trọn trên mặt xe
    )

    confirmed, review, conflicts = detector.analyze_conflicts_and_partition(
        detections=[cart, table_on_cart],
        counter_objects={},
    )

    assert any(c.get("conflict_type") == "contained_in_cart_candidate" for c in conflicts)
    assert any(r.id == "det_table_err" for r in review)
    assert any(c.id == "det_cart_real" for c in confirmed)
