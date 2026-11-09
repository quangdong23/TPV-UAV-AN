# Nhận diện UAV bằng YOLOE-v8s fused

Production dùng một checkpoint, 63 prompt cố định, catalog 10 lớp. Hai cấu hình duy nhất: `configs/object_yoloe.yaml`, `configs/video_pipeline.yaml`. Detector mặc định cách tối đa 5 frame, ROI budget video 2; tracking cập nhật mỗi frame.

Chạy từ `UAV`:

```bash
# Toàn bộ kết quả tự động gom về UAV/outputs/ với tên duy nhất (hoặc tùy chọn truyền --output):
python -m lan_chiem.predict --input lan_chiem/sample_data --device cuda
python -m lan_chiem.video_predict --input lan_chiem/sample_data/2.mp4 --device cuda
python -m lan_chiem.tests.run_tests --video
```

Video mặc định compact, gồm video, JSONL và summary. CLI vật thể riêng không có mask nên `is_encroaching=false` chưa đánh giá; dùng `python -m uav_pipeline` cho phân tích tiếp xúc vỉa hè. `full` giữ CSV, checkpoint và segment để resume.

Checksum canonical được kiểm tra khi nạp. Đổi vocabulary cần checkpoint tương thích, không thay prompt trực tiếp trên trọng số fused. Xuất từ model nguồn tương thích bằng `python -m lan_chiem.export_single_yoloe --source-weights /path/to/source.pt`; sau khi xuất cần duyệt và cập nhật checksum cho artifact mới.

DINO, YOLO-World, hybrid và YAML thử nghiệm nằm trong `../archive/legacy_dino/`. Các backend legacy chỉ nạp khi chọn rõ bằng tham số backend; không thuộc pipeline production. Số đo cũ trong archive chỉ có giá trị lịch sử.

Xem [README hệ thống](../README.md), [schema đầu ra](../OUTPUTS.md) và [báo cáo refactor](../reports/refactor_20261104/REPORT.md). Số lượng track và confidence chưa thay thế đánh giá nhãn chuẩn.
