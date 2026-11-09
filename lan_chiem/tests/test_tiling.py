"""
Kiểm thử chế độ chia mảnh (tiling) cho ảnh drone độ phân giải cao:
- Đảm bảo các mảnh phủ kín toàn bộ diện tích ảnh mà không bỏ sót góc hoặc rìa.
- Đảm bảo biến đổi tọa độ từ tile về ảnh toàn cục là chính xác.
"""

from lan_chiem.detector import generate_tiles, crop_box_to_global


def test_tiles_smaller_than_tile_size():
    """Nếu ảnh nhỏ hơn kích thước tile, chỉ trả về 1 tile bao trọn ảnh."""
    tiles = generate_tiles(image_width=800, image_height=600, tile_size=1024)
    assert len(tiles) == 1
    assert tiles[0] == (0, 0, 800, 600)


def test_tiles_coverage_for_large_image():
    """Với ảnh lớn (ví dụ 3000x2000), đảm bảo các tile phủ kín từ (0,0) tới (3000, 2000)."""
    img_w, img_h = 3000, 2000
    tile_size = 1024
    overlap = 0.25

    tiles = generate_tiles(img_w, img_h, tile_size=tile_size, tile_overlap=overlap)
    assert len(tiles) > 1

    # Kiểm tra góc trên bên trái được phủ
    assert any(x1 == 0 and y1 == 0 for x1, y1, _, _ in tiles)

    # Kiểm tra góc dưới bên phải được phủ
    assert any(x2 == img_w and y2 == img_h for _, _, x2, y2 in tiles)

    # Kiểm tra tất cả các tile có kích thước hợp lệ
    for x1, y1, x2, y2 in tiles:
        assert 0 <= x1 < x2 <= img_w
        assert 0 <= y1 < y2 <= img_h
        assert (x2 - x1) <= tile_size
        assert (y2 - y1) <= tile_size


def test_coordinate_transform():
    """Kiểm tra phép tịnh tiến tọa độ từ tile nội bộ sang ảnh toàn cục."""
    tile_x1, tile_y1 = 500, 300
    local_box = [50.0, 40.0, 150.0, 120.0]

    # Tọa độ toàn cục = tọa độ nội bộ + gốc của tile
    global_x1, global_y1, global_x2, global_y2 = crop_box_to_global(
        local_box, (tile_x1, tile_y1), (1024, 1024), (1920, 1080))

    assert global_x1 == 550.0
    assert global_y1 == 340.0
    assert global_x2 == 650.0
    assert global_y2 == 420.0
