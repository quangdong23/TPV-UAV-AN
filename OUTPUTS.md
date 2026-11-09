# Đầu ra hệ thống

## Thư mục lưu tập trung duy nhất (Unified Outputs)

Toàn bộ kết quả phân tích, video kết hợp, benchmark và tracking đều được gom về lưu tại một nơi duy nhất: **`UAV/outputs/`**.
Thư mục `runs/` cũ đã được gỡ bỏ hoàn toàn; mọi lệnh nếu có truyền đường dẫn `runs/...` cũng sẽ được tự động chuyển hướng về lưu trong `outputs/`.

### Cơ chế tự động sinh tên không trùng lặp (Auto-naming & Non-collision)
- **Không bắt buộc truyền `--output`**: Khi chạy bất kỳ lệnh nào (`uav_pipeline`, `video_predict`, `predict`, `run_eval_6videos.py`), nếu không truyền `--output`, hệ thống sẽ tự động tạo thư mục đầu ra mới có dạng:
  ```text
  outputs/<tiền_tố_tác_vụ>_<tên_nguồn>_<YYYYMMDD_HHMMSS>
  ```
  *Ví dụ*: `outputs/pipeline_2_20261104_144512`, `outputs/video_6_20261104_144600`
- **Tự động tránh trùng tên khi chỉ định `--output`**: Nếu người dùng chỉ định một thư mục đã tồn tại và có chứa tệp tin (và không dùng cờ `--resume`), hệ thống tự động tăng chỉ số đuôi (`outputs/my_run_1`, `outputs/my_run_2`,...) để không ghi đè, không mất dữ liệu cũ và loại bỏ hoàn toàn lỗi dừng ngang `Output directory is not empty`.

Các CLI mặc định dùng `--output-mode compact`. Chỉ giảm file xuất và I/O;
cấu hình production tại `lan_chiem/configs/object_yoloe.yaml` và
`lan_chiem/configs/video_pipeline.yaml`.

## Video kết hợp và nhận diện vật thể

Khi một lượt compact hoàn tất thành công, thư mục output chỉ còn:

```text
annotated.mp4
tracks.jsonl
summary.json
```

| Thành phần trước đây | Mặc định mới | Khi cần thêm |
| --- | --- | --- |
| Video chú thích | Giữ `annotated.mp4` | Luôn giữ |
| JSONL từng frame, box/track/polygon và evidence | Giữ `tracks.jsonl`, không cắt nội dung phân tích | Luôn giữ |
| Thống kê, cấu hình và provenance | Giữ `summary.json` | Luôn giữ |
| `tracks.csv` (bản bảng của dữ liệu đã có trong JSONL) | Không tạo | `--output-mode full` |
| `status.json`, `latest_preview.jpg` | Không tạo; vẫn có tiến độ Terminal | Full; preview còn phụ thuộc `preview_interval` |
| `images/`, `sidewalk_masks/` của luồng kết hợp | Không tạo | Full mặc định mỗi 30 frame, hoặc `--sample-step N` |
| `segments/part_*.mp4`, `part_*.jsonl` | Tạm thời; dọn sau khi kết quả cuối được ghi thành công | Full giữ lại |
| `previous_*.png`, `checkpoint.json`, `concat.txt`, log encoder thành công | Dọn cùng trung gian sau thành công | Full giữ lại |

Compact ghi một đoạn liên tục mỗi lượt, không tạo checkpoint định kỳ. Nếu chỉ
có một đoạn, video được chuyển sang tên cuối bằng đổi tên, không encode lại và
không giữ hai bản video. Ghép nhiều đoạn (ví dụ sau một lần dừng có kiểm soát rồi
resume ở CLI vật thể) dùng FFmpeg copy stream, không encode lại.

Nếu thiếu FFmpeg khi có nhiều đoạn, giữ segment và ghi trạng thái trong summary;
không xóa bản video duy nhất. Một đoạn không cần FFmpeg để hoàn tất. Nếu encoder,
render hoặc export lỗi, không tự dọn các file đang có. Lượt CLI vật thể bị dừng
bởi `--max-frames` giữ checkpoint đã commit để có thể tiếp tục; chế độ kết hợp
vẫn không hỗ trợ resume và có thể hoàn tất gọn ngay số frame đã yêu cầu.

Cleanup chỉ xóa các tên file do lượt chạy sinh ra, không xóa đệ quy file không
biết trong thư mục segment, không đụng dữ liệu đầu vào hay output lượt khác.
Muốn checkpoint định kỳ và khả năng tiếp tục sau gián đoạn, dùng full ngay từ
đầu; khi resume phải dùng cùng output mode và cấu hình. Backup chỉ tạo khi yêu
cầu `--backup-dir`; không tự xóa bản backup.

```bash
# Cách chạy thuận tiện nhất: không cần chỉ định --output, hệ thống tự sinh tên trong outputs/
python -m uav_pipeline --input lan_chiem/sample_data/6.mp4 --device cuda

# Hoặc chỉ định tên tùy ý (tự động gom về outputs/ và tự động tăng số nếu đã tồn tại)
python -m uav_pipeline --input lan_chiem/sample_data/6.mp4 --output combined_run --device cuda

# Khi cần chẩn đoán chi tiết / xuất frame mẫu
python -m uav_pipeline --input lan_chiem/sample_data/6.mp4 --output combined_full --device cuda --output-mode full
```

## Phân đoạn vỉa hè chạy riêng

- Video CLI: `preview.mp4` và `video_profile.json`; không tự tạo `mask_samples/`.
  Full hoặc `--video-mask-sample-step N` bật lại mask mẫu. Chế độ GUI không lưu
  video trừ khi có `--save-video`; headless luôn lưu video như trước.
- Ảnh CLI compact: `overlay.png`, `mask.png`, `polygons_pixel.geojson`,
  `metadata.json`. Mask cuối, polygon và metadata giữ nguyên nội dung phân tích.
- Ảnh full: thêm `visible_mask.png`, `confidence.png` phục vụ chẩn đoán.
- Raster có georeference vẫn xuất `polygons_geo.geojson`. `scores.npz` chỉ có khi
  chủ động bật `output.save_scores` trong cấu hình; mặc định đã tắt.

## Ảnh nhận diện vật thể và công cụ khác

CLI ảnh đã xuất hai kết quả chính: `<tên>_annotated.jpg` và `<tên>_result.json`;
giữ nguyên. Debug trace và vẽ ứng viên review chỉ bật khi người dùng yêu cầu.
Benchmark production chạy bằng `python run_eval_6videos.py`; các công cụ
audit/compare/benchmark legacy nằm tại `archive/legacy_dino/`. Export model chạy
bằng `python -m lan_chiem.export_single_yoloe` và cần duyệt checksum sau khi xuất.
Cache model/embedding phục vụ nạp/suy luận không phải đầu ra phân tích video.

## Tương thích và kiểm tra

API Python `process_video()` và `SidewalkPipeline.process()` giữ mặc định đầy đủ
để không làm hỏng công cụ/test đang gọi trực tiếp. Các entry point CLI chọn gọn.
Muốn API video gọn, truyền `output_mode='compact'`; ảnh dùng `diagnostics=False`.

Đã đối chiếu full/compact trên video tổng hợp với model giả lập: JSONL giống
nhau khi bỏ trường thời gian, box/ID/evidence/polygon không đổi; mask và overlay
ảnh cuối giống từng byte. Có kiểm tra ba file cuối, trường hợp thiếu FFmpeg,
lượt partial/resume, lỗi render, yêu cầu ảnh mẫu và không xóa file lạ. Các kiểm
tra này không thay thế đánh giá độ chính xác model thật trên dữ liệu mới.

## Cảnh báo theo thời gian (bật riêng)

`--events` thêm `events.json` và clip bằng chứng trong `event_clips/` khi có cảnh báo.
Clip mặc định lấy 4 giây trước + 10 giây sau lần đầu vật thể nằm trên vỉa hè.
Chi tiết quy tắc 10 frame, giới hạn góc nhìn drone và lệnh chạy: [QUALITY_AND_EVENTS.md](QUALITY_AND_EVENTS.md).

## Thuộc tính tiếp xúc và phân loại biển

Mỗi dòng JSONL dùng `coordinate_space=source_pixels`, box nguồn và polygon vỉa hè.
Track thêm các trường:

| Trường | Ý nghĩa |
| --- | --- |
| `is_encroaching` | Boolean sàng lọc theo mask hiện tại; false khi chưa có mask hoặc lớp không thuộc diện cảnh báo |
| `subtype` (biển) | `sidewalk_standee`, `facade_attached`, `sign_unresolved` |
| `contact_evidence` (biển) | Khoảng cách điểm đáy đến vỉa hè, tolerance, dấu hiệu treo/box ngang trên cao; khoảng cách đang tính trong pixel working, tolerance đã quy đổi từ 20 pixel nguồn |
| `class_persistence_active` | Cột/cây đã đạt ba keyframe detector liên tiếp; không có nghĩa frame hiện tại có detection |
| `position_source`, `detection_age_seconds` | Phân biệt detection với optical flow/camera motion/prediction và tuổi detection thật |

Ghế nghi ghế đá được giữ ở `review_candidates` với subtype
`public_bench_candidate` và `evidence.chair_surface`, không dùng để tạo track mới.
Combined compact vẫn bỏ review theo chính sách sẵn có; muốn xem đầy đủ bằng chứng
counter/chair, chạy CLI ảnh với `--debug` hoặc gọi API detector trực tiếp.

`--events` thêm `events.json` và `event_clips/`. Chỉ lớp 4/5/6/10 hoặc biển
`sidewalk_standee` hợp lệ được xét `sidewalk_encroachment`; cột/cây vẫn có thể làm
điểm tựa trong `sign_attachment_suspected`. Cả hai sự kiện đều cần người xem lại.
Giữ detection của biển mặt tiền không đồng nghĩa cảnh báo vi phạm.

## Báo cáo refactor

`reports/refactor_20261104/REPORT.md` chứa bảng FPS/số track trước–sau và giới hạn
đo lường; `cleanup_manifest.json` liệt kê archive, checkpoint và dataset đã xóa.
`runs/eval_full_6videos/benchmark_report.json` là baseline, không bị ghi đè.
Benchmark mới ở `runs/eval_refactor_6videos/benchmark_report.json`, gồm lỗi từng
video và trạng thái chưa đo Accuracy khi thiếu ground truth hoặc GPU.
