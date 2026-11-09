# UAV phát hiện vật thể và sàng lọc lấn chiếm vỉa hè

Pipeline production: **Mask2Former Swin-Large Mapillary → YOLOE-v8s fused 63 prompt → Kalman/Hungarian + Lucas–Kanade → kiểm tra tiếp xúc vỉa hè → video và JSONL**. Catalog gồm 10 lớp; segmentation quản lý vỉa hè riêng.

## Chạy hệ thống

Chạy từ thư mục `UAV`, dùng output mới:

```bash
python -m pip install -r requirements.txt -r lan_chiem/requirements-yoloe.txt -r lan_chiem/requirements-video.txt
# Chạy trực tiếp (tự động lưu vào outputs/ với tên duy nhất không bao giờ trùng lặp):
python -m uav_pipeline --input lan_chiem/sample_data/2.mp4 --device cuda
```

`--device cpu` dùng FP32 cho cả hai model. `--max-frames 30` giới hạn lượt thử. Mặc định `compact` giữ `annotated.mp4`, `tracks.jsonl`, `summary.json`; xem [OUTPUTS.md](OUTPUTS.md). Dùng `--events` để bật cảnh báo theo chuỗi frame và xuất clip bằng chứng.

## Cấu hình và tính toàn vẹn

Chỉ có hai YAML production, độc lập, không dùng `extends`:

| File | Nội dung |
| --- | --- |
| `lan_chiem/configs/object_yoloe.yaml` | 10 lớp, 38 target + 25 counter prompt, ngưỡng từng lớp, counter-pairs, ROI, hậu xử lý |
| `lan_chiem/configs/video_pipeline.yaml` | Keyframe 5, frame_step 1, ROI budget 2, tracking buffer, persistence |

Mask2Former dùng `configs/gpu_t4.json`. Có thể ghi đè bằng `--object-config`, `--video-config`, `--sidewalk-config`. CLI ảnh và video vật thể dùng chung YAML production.

Checkpoint: `lan_chiem/models/yoloe-v8s-single.pt`, giống `../yoloe_training_standalone/runs/candidate_balanced.pt`.

- SHA256 trọng số: `80ccadad6beb805acb93c78db97ec1b0d764e717dea38d8a5c314c2699915dd9`.
- SHA256 vocabulary: `85cec415ca14880e9ec4697192830d40fbd81895fc95349c261e60acef3b7f8d`.
- Checksum kiểm tra khi nạp checkpoint canonical; file `.sha256` và metadata `.json` nằm cạnh trọng số. `--weights` hỗ trợ checkpoint thử nghiệm riêng.

```bash
cd lan_chiem/models
sha256sum -c yoloe-v8s-single.sha256
```

## Lọc và cảnh báo

Biển quảng cáo (7), cột điện (8), thân cây (9) vẫn được nhận diện trên toàn khung hình. Các lớp còn lại phải thuộc vỉa hè: mặc định điểm giữa đáy box trên nhãn `2`; tùy chọn `--filter-mode overlap --min-overlap 0.1` kiểm tra diện tích giao. Unknown `255` không được tính là vỉa hè.

Mỗi track xuất `is_encroaching`. Cờ chỉ bật cho bàn (4), ghế (5), xe đẩy (6), hàng hóa (10) có tiếp xúc vỉa hè, hoặc biển `sidewalk_standee`. Người, xe cộ, cột và cây không tự phát cảnh báo lấn chiếm.

Biển được phân loại theo từng frame:

- `sidewalk_standee`: điểm đáy nằm trên hoặc cách vỉa hè tối đa 20 pixel nguồn, không có bằng chứng biển treo.
- `facade_attached`: prompt biển treo hoặc box ngang nằm cao hơn vỉa hè phía dưới.
- `sign_unresolved`: thiếu mặt vỉa hè, chân ngoài khung, hoặc chưa đủ bằng chứng.

Đây là sàng lọc hình học 2D, chưa xác minh chân đế hoặc mặt tiền vật lý. Các biển sát cột/cây còn được kiểm tra quan hệ theo thời gian; sự kiện attachment chỉ là đề nghị kiểm tra, không phải kết luận vi phạm. `--events` yêu cầu mặc định 10 frame hợp lệ và ít nhất hai quan sát detector độc lập.

Prompt `nonplastic_chair` là “metal folding chair”. Ghế kim loại không tự bị loại. Box rộng cùng counter nonplastic mạnh và bằng chứng vật liệu xây dựng hoặc bề mặt ít sắc độ được chuyển thành `public_bench_candidate` để review. Ghế rất dài, ít sắc độ và confidence thấp cũng được chuyển sang review dù counter không phát box. Màu xám hoặc tỷ lệ box đơn lẻ không chứng minh ghế đá. Ngưỡng confidence production được giữ nguyên.

Cột/cây có ba keyframe liên tiếp được giữ qua tối đa hai keyframe hụt khi camera/LK còn đáng tin; vẫn hết hạn theo tracking buffer và reset khi đổi cảnh. Dự đoán không tăng số quan sát detector. Mask vỉa hè được warp từ keyframe gốc bằng affine RANSAC, tránh resample nối tiếp; flow không đạt kiểm tra chất lượng sẽ chạy segmentation lại. Helper warp hỗ trợ cả homography, nhưng bộ ước lượng production vẫn dùng affine.

## Kiểm thử và benchmark

```bash
python -m lan_chiem.tests.run_tests
python -m lan_chiem.tests.run_tests --video
python run_eval_6videos.py
```

Benchmark mới lưu `runs/eval_refactor_6videos`, giữ nguyên baseline `runs/eval_full_6videos/benchmark_report.json`, ghi log từng video và trả mã lỗi khi pipeline lỗi hoặc kiểm tra số track Video 2 thất bại. Dùng `--output` mới cho mỗi lượt; có `--device cpu`, `--videos 2`, `--max-frames N`.

Số track/detection là chỉ báo hồi quy, không phải Accuracy/precision/recall/mAP. Chỉ so FPS khi cùng phần cứng, precision và đủ video. Báo cáo refactor, manifest cleanup, baseline và log: [reports/refactor_20261104](reports/refactor_20261104/REPORT.md). Môi trường refactor không có CUDA nên chưa nghiệm thu zero regression sáu video trên GPU.

## Cấu trúc và huấn luyện

```text
uav_pipeline/                 CLI kết hợp, adapter vỉa hè, events
lan_chiem/configs/            Hai YAML production
lan_chiem/models/             Checkpoint, checksum, metadata
lan_chiem/postprocess.py      Counter, hình học và tiếp xúc vỉa hè
lan_chiem/video_tracking.py   Tracking và class persistence
lan_chiem/video_motion.py     Camera/LK và warp label mask
road_sidewalk_seg/            Mask2Former, mask, polygon, propagation
archive/legacy_dino/          DINO, YOLO-World, hybrid, benchmark và YAML cũ
reports/refactor_20261104/    Bằng chứng kiểm thử và thay đổi
../yoloe_training_standalone/ Trainer fused, dataset v4, checkpoint giữ lại
```

Legacy được archive để tra cứu và giữ test hồi quy; production YOLOE không import các backend này. Hướng dẫn huấn luyện v5 và staging ảnh: [training README](../yoloe_training_standalone/README.md).

Phân đoạn độc lập vẫn chạy bằng:

```bash
python -m road_sidewalk_seg --config configs/gpu_t4.json --input video.mp4 --headless --save-video
```

Đầu ra phân đoạn riêng: `preview.mp4`, `video_profile.json`; mask mẫu chỉ bật ở full hoặc tùy chọn sampling. GIS là tùy chọn qua `requirements-gis.txt`, cần raster có georeference. Luồng kết hợp chưa hỗ trợ resume; CLI vật thể riêng hỗ trợ `--resume` với `--output-mode full`.
