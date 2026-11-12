"""
Module sidewalk: Quản lý vùng không gian vỉa hè (mask/polygon riêng, ngoài danh mục vật thể).
Hỗ trợ:
- Tích hợp với phân hệ uav_pipeline.segmentation sẵn có nếu khả dụng.
- Nạp polygon từ file GeoJSON hoặc nhập thủ công (manual).
- Ghi rõ nguồn gốc: "uav_pipeline.segmentation", "manual", hoặc "geojson_file".
- Vẽ overlay vùng vỉa hè trong suốt lên ảnh.
- Tuyệt đối không giả lập polygon bằng cách lấy bounding box 'sidewalk'.
- Không chặn quy trình nhận diện vật thể khi chưa có dữ liệu vỉa hè.
- Chưa đưa ra bất kỳ kết luận pháp lý nào về lấn chiếm vỉa hè.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw


@dataclass
class SidewalkRegion:
    """Đại diện cho một vùng vỉa hè dạng đa giác (polygon) theo tọa độ pixel."""
    polygon: List[List[float]]  # Danh sách đỉnh [[x1, y1], [x2, y2], ...]
    source: str  # "uav_pipeline.segmentation" | "manual" | "geojson_file"
    confidence: Optional[float] = None
    area_pixels: float = 0.0

    def __post_init__(self):
        if self.area_pixels == 0.0 and len(self.polygon) >= 3:
            # Tính diện tích bằng công thức Shoelace
            x = [p[0] for p in self.polygon]
            y = [p[1] for p in self.polygon]
            self.area_pixels = 0.5 * abs(
                sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(len(self.polygon) - 1))
                + (x[-1] * y[0] - x[0] * y[-1])
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "confidence": round(float(self.confidence), 4) if self.confidence is not None else None,
            "area_pixels": round(float(self.area_pixels), 1),
            "polygon": [[round(float(c), 2) for c in pt] for pt in self.polygon],
        }


def extract_polygons_from_geojson(geojson_data: Dict[str, Any], source: str = "geojson_file") -> List[SidewalkRegion]:
    """Trích xuất danh sách SidewalkRegion từ GeoJSON FeatureCollection."""
    regions: List[SidewalkRegion] = []
    features = geojson_data.get("features", [])
    for feat in features:
        geom = feat.get("geometry", {})
        gtype = geom.get("type", "")
        coords = geom.get("coordinates", [])
        props = feat.get("properties", {})
        conf = props.get("confidence")

        if gtype == "Polygon" and coords:
            # coords[0] là vòng ngoài (exterior ring)
            exterior = coords[0]
            if len(exterior) >= 3:
                regions.append(SidewalkRegion(polygon=exterior, source=source, confidence=conf))
        elif gtype == "MultiPolygon" and coords:
            for poly_coords in coords:
                if poly_coords and len(poly_coords[0]) >= 3:
                    regions.append(SidewalkRegion(polygon=poly_coords[0], source=source, confidence=conf))

    return regions


def load_sidewalk_from_pipeline(
    image_path: Union[str, Path],
    config_dict: Optional[Dict[str, Any]] = None,
) -> Tuple[List[SidewalkRegion], str]:
    """
    Thử nạp kết quả phân vùng vỉa hè từ uav_pipeline.segmentation qua interface SidewalkPipeline.
    Nếu uav_pipeline.segmentation chưa có weights hoặc không khả dụng, trả về rỗng kèm thông điệp.
    """
    try:
        from uav_pipeline.segmentation import PipelineConfig, SidewalkPipeline
    except ImportError as e:
        return [], f"Không thể import uav_pipeline.segmentation: {e}"

    try:
        cfg = PipelineConfig()
        if config_dict:
            for k, v in config_dict.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

        with SidewalkPipeline(cfg) as pipeline:
            result = pipeline.process_image(image_path)
            # Trích xuất pixel_features
            features = getattr(result, "pixel_features", None)
            if features and isinstance(features, dict):
                regions = extract_polygons_from_geojson(features, source="model:uav_pipeline.segmentation")
                return regions, f"Thành công: tìm thấy {len(regions)} vùng vỉa hè từ uav_pipeline.segmentation."
            return [], "uav_pipeline.segmentation không tìm thấy vùng vỉa hè nào."
    except Exception as e:
        return [], f"Không thể chạy uav_pipeline.segmentation trên ảnh: {e}"


def load_sidewalk_from_file(file_path: Union[str, Path]) -> Tuple[List[SidewalkRegion], str]:
    """Nạp vùng vỉa hè từ file GeoJSON hoặc JSON đã lưu trước đó."""
    fp = Path(file_path)
    if not fp.is_file():
        return [], f"Không tìm thấy file vỉa hè tại: {fp}"

    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "features" in data:
            regions = extract_polygons_from_geojson(data, source=f"file:{fp.name}")
        elif "sidewalk_regions" in data:
            regions = [
                SidewalkRegion(
                    polygon=r.get("polygon", []),
                    source=r.get("source", f"file:{fp.name}"),
                    confidence=r.get("confidence"),
                )
                for r in data["sidewalk_regions"]
                if r.get("polygon")
            ]
        else:
            return [], "File JSON không chứa cấu trúc 'features' (GeoJSON) hoặc 'sidewalk_regions'."

        return regions, f"Đã nạp {len(regions)} vùng vỉa hè từ {fp.name}."
    except Exception as e:
        return [], f"Lỗi đọc file vỉa hè: {e}"


def create_manual_sidewalk_region(points: List[List[float]]) -> Optional[SidewalkRegion]:
    """Tạo một vùng vỉa hè thủ công từ danh sách các điểm click/nhập."""
    if len(points) < 3:
        return None
    return SidewalkRegion(polygon=points, source="manual")


def draw_sidewalk_overlay(
    image: Image.Image,
    regions: List[SidewalkRegion],
    fill_rgba: Tuple[int, int, int, int] = (0, 191, 255, 65),
    line_rgba: Tuple[int, int, int, int] = (0, 150, 255, 230),
    line_width: int = 3,
) -> Image.Image:
    """Vẽ vùng đa giác vỉa hè phủ mờ (transparent overlay) lên ảnh."""
    if not regions:
        return image.copy()

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for r in regions:
        flat_poly = [(float(pt[0]), float(pt[1])) for pt in r.polygon]
        if len(flat_poly) >= 3:
            # Vẽ phần mặt phẳng vỉa hè mờ
            draw.polygon(flat_poly, fill=fill_rgba)
            # Vẽ đường viền
            draw.line(flat_poly + [flat_poly[0]], fill=line_rgba, width=line_width)

    merged = Image.alpha_composite(base, overlay)
    return merged.convert("RGB")
