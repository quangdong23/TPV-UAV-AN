"""
Kiểm thử phân hệ vỉa hè (ID 0): đa giác, tính diện tích Shoelace, nạp GeoJSON.
"""

from uav_pipeline.segmentation.regions import (
    SidewalkRegion,
    create_manual_sidewalk_region,
    extract_polygons_from_geojson,
)


def test_sidewalk_region_area_calculation():
    """Kiểm tra tính diện tích đa giác Shoelace."""
    # Hình vuông 100x100 -> diện tích = 10000
    pts = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
    region = SidewalkRegion(polygon=pts, source="test", confidence=0.95)
    assert abs(region.area_pixels - 10000.0) < 1e-4
    assert region.source == "test"
    assert region.confidence == 0.95


def test_extract_from_geojson():
    """Kiểm tra trích xuất polygon từ dữ liệu GeoJSON chuẩn."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[10.0, 10.0], [200.0, 10.0], [200.0, 80.0], [10.0, 80.0], [10.0, 10.0]]
                    ],
                },
                "properties": {"confidence": 0.88},
            }
        ],
    }
    regions = extract_polygons_from_geojson(geojson, source="test_geojson")
    assert len(regions) == 1
    assert regions[0].source == "test_geojson"
    assert regions[0].confidence == 0.88
    assert len(regions[0].polygon) == 5


def test_manual_sidewalk_region():
    pts = [[0.0, 0.0], [50.0, 0.0], [50.0, 50.0]]
    r = create_manual_sidewalk_region(pts)
    assert r is not None
    assert r.source == "manual"
    assert r.area_pixels > 0
