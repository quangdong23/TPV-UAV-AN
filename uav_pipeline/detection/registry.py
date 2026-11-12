"""Registry for catalogue objects-v3-10, IDs 1–10. Sidewalk masks are separate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ClassMeta:
    id: int
    name: str  # Tên định danh dữ liệu chuẩn (vd: 'cây', 'vỉa hè')
    display_name: str  # Tên hiển thị giao diện (vd: 'Thân cây', 'Vỉa hè')
    category_type: str  # 'object' | 'region' | 'action'
    support_status: str  # 'supported' | 'candidate_verification' | 'polygon_mask' | 'unsupported'
    convention_note: str  # Ghi chú quy ước tạm thời
    default_color: Tuple[int, int, int]  # Màu RGB mặc định


# Danh mục mười lớp mới, sinh từ object_classes.py
from uav_pipeline.detection.classes import ADVANCED_OBJECT_CLASSES, CATALOG_VERSION

_COLORS = [(42,157,143), (30,144,255), (100,149,237), (255,215,0),
           (244,162,97), (0,230,230), (142,68,173), (233,196,106),
           (34,139,34), (218,112,214)]
CLASS_DEFINITIONS: Dict[int, ClassMeta] = {
    item['id']: ClassMeta(item['id'], item['name_vi'].lower(), item['name_vi'],
                          'object', 'supported', item['detail'], _COLORS[item['id']-1])
    for item in ADVANCED_OBJECT_CLASSES
}


class ClassRegistry:
    """Registry tra cứu thông tin lớp đối tượng duy nhất của toàn hệ thống."""

    @classmethod
    def get(cls, class_id: int) -> Optional[ClassMeta]:
        return CLASS_DEFINITIONS.get(class_id)

    @classmethod
    def get_by_name(cls, name: str) -> Optional[ClassMeta]:
        name_clean = name.strip().lower()
        for meta in CLASS_DEFINITIONS.values():
            if meta.name.lower() == name_clean or meta.display_name.lower() == name_clean:
                return meta
        return None

    @classmethod
    def all_classes(cls) -> List[ClassMeta]:
        return list(CLASS_DEFINITIONS.values())

    @classmethod
    def all_ids(cls) -> List[int]:
        return sorted(list(CLASS_DEFINITIONS.keys()))

    @classmethod
    def object_classes(cls) -> List[ClassMeta]:
        """Mười lớp vật thể ID 1–10."""
        return [c for c in CLASS_DEFINITIONS.values() if c.category_type == "object"]

    @classmethod
    def action_classes(cls) -> List[ClassMeta]:
        """Danh mục mới không chứa lớp hành vi."""
        return [c for c in CLASS_DEFINITIONS.values() if c.category_type == "action"]

    @classmethod
    def region_classes(cls) -> List[ClassMeta]:
        """Polygon vỉa hè lưu riêng, không có ID trong danh mục vật thể."""
        return [c for c in CLASS_DEFINITIONS.values() if c.category_type == "region"]

    @classmethod
    def get_support_status_dict(cls) -> Dict[int, Dict[str, Any]]:
        """Trả về từ điển trạng thái hỗ trợ chuẩn hóa để lưu vào JSON kết quả."""
        status_dict = {}
        for cid, meta in CLASS_DEFINITIONS.items():
            status_dict[cid] = {
                "name": meta.name,
                "display_name": meta.display_name,
                "type": meta.category_type,
                "support_status": meta.support_status,
                "convention_note": meta.convention_note,
            }
        return status_dict
