"""
Kiểm thử logic ánh xạ nhãn theo nhóm Prompt và xác định Subtype.
"""

from lan_chiem.detector import GroundingDINODetector, determine_sign_subtype


def test_group_mapping():
    detector = GroundingDINODetector()

    # Nhóm people_vehicles (ID 1, ID 2)
    pv_aliases = [
        ("pedestrian", 1, "người", "Người"),
        ("person", 1, "người", "Người"),
        ("motorcycle", 2, "xe máy", "Xe máy"),
        ("motorbike", 2, "xe máy", "Xe máy"),
    ]

    cid, cname, _, is_amb, m_ids = detector._map_grouped_label("person", pv_aliases)
    assert cid == 1
    assert not is_amb
    assert m_ids == [1]

    cid, cname, _, is_amb, m_ids = detector._map_grouped_label("motorcycle", pv_aliases)
    assert cid == 2
    assert not is_amb
    assert m_ids == [2]

    # Nhãn lạ trong nhóm -> không khớp
    cid, _, _, is_amb, m_ids = detector._map_grouped_label("table", pv_aliases)
    assert cid is None
    assert not is_amb
    assert m_ids == []

    # Nhãn ghép chứa dấu hiệu của cả ID 5 và ID 6 -> phải đánh dấu ambiguous
    vendor_aliases = [
        ("car", 3, "xe ô tô", "Xe ô tô"),
        ("vendor pushcart", 6, "xe đẩy bán hàng", "Xe đẩy bán hàng"),
        ("street food cart", 6, "xe đẩy bán hàng", "Xe đẩy bán hàng"),
    ]
    cid, _, _, is_amb, m_ids = detector._map_grouped_label("car vendor pushcart", vendor_aliases)
    assert cid is None
    assert is_amb is True
    assert 3 in m_ids and 6 in m_ids


def test_sign_subtypes():
    """Kiểm tra xác định subtype cho ID 7 (biển quảng cáo)."""
    assert determine_sign_subtype("traffic sign") == "traffic_sign"
    assert determine_sign_subtype("advertising board") == "advertising_board"
    assert determine_sign_subtype("billboard") == "advertising_board"
    assert determine_sign_subtype("shop sign") == "shop_sign"
    assert determine_sign_subtype("sandwich board") == "standing_sign"
    assert determine_sign_subtype("standing sign") == "standing_sign"
    assert determine_sign_subtype("unknown sign") == "unknown"
