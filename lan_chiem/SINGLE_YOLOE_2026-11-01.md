# Kiểm tra một model YOLOE — 01/11/2026

> Báo cáo lịch sử của lần xuất checkpoint. Xem [kết quả và thay đổi mới nhất](MAINTENANCE_AND_ACCURACY.md). Các thử nghiệm/adapter không được chọn đã được dọn; baseline ảnh, video cuối, số đo và bản sao mã trước sửa vẫn được giữ.

## Kết luận

Đã chuyển mặc định sang một YOLOE-v8s gộp vocabulary cố định; không gọi DINO hoặc text encoder trong suy luận thường ngày. Hai video mẫu vượt mục tiêu 15 FPS. **Chưa ngang hybrid về thân cây**: hai cây chính trong video 2 vẫn thiếu. Chưa đủ nhãn độc lập để kết luận độ chính xác tổng thể hoặc cả 10 lớp đều đạt yêu cầu.

Chỉ chỉnh mã, cấu hình, tài liệu và tạo kết quả trong `lan_chiem`. Checkpoint nguồn ngoài thư mục được đọc để xuất; hash không đổi.

## Vì sao trước đây có hai model?

YOLOE nhận diện thông thường; DINO tiny bổ sung thân cây mỗi 0,8 giây nguồn vì YOLOE bỏ sót. Cách này có ích cho cây nhưng thêm thời gian nạp/suy luận và bộ nhớ. Đây là đánh đổi chất lượng lớp khó với tốc độ, không phải yêu cầu của YOLOE.

Text encoder tạo embedding khi thiết lập prompt cần phân biệt với detector xử lý từng ảnh. [Tài liệu YOLOE chính thức](https://docs.ultralytics.com/models/yoloe/) mô tả các chế độ prompt. Bản mới gộp embedding cố định vào head và lưu checkpoint, nên không nạp encoder khi dùng bản đã xuất.

## Thay đổi

- `yoloe_detector.py`: bỏ đường gọi TrunkRefiner/DINO; profile cũ bật refiner được cảnh báo và vô hiệu hóa. Provenance ghi `detector_count=1`. Giữ đủ 10 lớp, kể cả thân cây.
- `export_single_yoloe.py`: xuất `models/yoloe-v8s-single.pt`, gộp head vocabulary và Conv/BN, không fine-tune, không sửa checkpoint nguồn. Nạp checkpoint kiểm tra tên lớp/hash vocabulary để tránh lệch mapping.
- `embedding_cache.py`: cache cho checkpoint chưa gộp theo weights/vocabulary/versions; kiểm tra kích thước, số hữu hạn, ghi atomic và xử lý cache hỏng. Checkpoint mặc định bỏ qua cache và encoder.
- `yoloe_single.yaml`: đầu vào 1280, giữ mô tả lớp/counter-prompt hiện có, ROI cho ảnh tĩnh. Thử 1536 không giúp cây ổn định nên chọn 1280.
- `video_yoloe_single.yaml`, `video_predict.py`: 2 ROI/lượt, xoay vòng theo ngân sách video thực thay vì cap ảnh; toàn cảnh/ROI dùng chung engine. Giữ phát hiện mỗi 6 frame và phát hiện sớm khi track yếu, tracking mọi frame.
- Giữ sửa tracking lần trước: detection neo hộp vào đo mới, bỏ EMA vị trí gây trễ; cập nhật LK không làm co covariance Kalman qua phép đo giả.
- `predict.py`: không tự chuyển sang DINO vì `--prompt`; backend khác phải chọn rõ ràng. CLI ảnh đặt CPU threads mặc định 2 phù hợp benchmark.

## Tốc độ đo được

Máy RTX 3050 Ti Laptop 4 GB; PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, Ultralytics 8.4.45. Các job GPU chạy riêng.

| Cấu hình | Video 1: FPS liên tục | Video 1: FPS gồm nạp/xuất | Video 2: FPS liên tục | Video 2: FPS gồm nạp/xuất |
|---|---:|---:|---:|---:|
| Hybrid trước, tham chiếu review4 | 22,41 | 15,07 | 18,01 | 13,02 |
| Một YOLOE hiện tại | 28,01 | 23,20 | 29,73 | 24,22 |

Hybrid là tham chiếu lịch sử cùng nguồn, khác lần chạy và kích thước (960 so với 1280 mới); không coi chênh lệch là thử nghiệm chỉ đổi một biến. GPU cấp phát đỉnh khoảng 1.947 MiB ở hybrid, 180 MiB ở bản mới. Đây là `max_memory_allocated`, không phải tổng VRAM tiến trình.

- Video 1: xử lý/giải mã kiểm tra đủ 602 frame xuất; detector 101 lượt, 202 calls, 303 ảnh tính ROI; 501 frame chỉ tracking.
- Video 2: đủ 584 frame; detector 106 lượt, 212 calls, 318 ảnh; 478 frame chỉ tracking.
- Hai calls/lượt là toàn cảnh và batch 2 ROI trên **cùng model**.
- Nạp model khoảng 3,05/3,09 giây. FPS gồm nạp/xuất không gồm khởi động interpreter, import, hash nguồn và seek khôi phục. FPS liên tục bỏ nạp/warmup. Chưa đo live latency; không tuyên bố AI chạy 28–30 lượt trên từng giây nguồn.
- 9 ảnh × 3 lần đo ấm, 1280: không ROI trung bình 44,36 ms (22,54 ảnh/s), p95 47,76 ms; đầy đủ ROI trung bình 157,21 ms (6,36 ảnh/s), p95 165,67 ms. Ngân sách ảnh lớn hơn video.

Nguồn: [benchmark ảnh](outputs/review5/final_images/benchmark.json), [video 1](outputs/review5/final_video_1/summary.json), [video 2](outputs/review5/final_video_2/summary.json), [kiểm chứng](outputs/review5/final_verification.json).

## Chất lượng và phương án đã thử

Đối chiếu checkpoint nguồn/bản gộp trên 9 ảnh + frame 217: cùng số detection/candidate theo lớp, ghép một-một IoU ≥ 0,95. Chặn `get_text_pe` xác nhận không gọi encoder ở bản gộp; không import Transformers. Đây là bảo toàn kết quả khi xuất, **không phải kiểm tra đúng với ground truth**. Xem [export_parity.json](outputs/review5/export_parity.json).

Đã thử mô tả text mở rộng, embedding visual từ mask/crop thân cây, secondary class trước NMS, toàn cảnh/ROI 1280 và 1536, mask lớp cây chung. Không phương án nào bắt ổn định hai cây chính; không gán tán cây thành thân cây. Head đặt hộp còn có khả năng nhưng điểm phân loại trunk thường khoảng 0,001–0,015 tại các cây đã kiểm tra. Hạ ngưỡng dễ báo sai. Không sửa NMS diện rộng khi nút thắt là phân loại; [NMS chính thức](https://docs.ultralytics.com/reference/utils/nms/).

Đã thử thích nghi riêng kênh trunk ở 3 scale, đóng băng phần còn lại, bằng nhãn nháp vài ảnh và augmentation. Xuất hiện nhiều false positive nên **không chọn `trunk_adapter_*.pt` làm model chạy**. Nhãn nháp chỉ phục vụ chẩn đoán, không phải ground truth đầy đủ. Đóng băng kênh khác không đảm bảo chúng không bị ảnh hưởng vì lớp mới có thể giành argmax.

Ảnh 5 nhận ra xe đẩy/bảng quảng cáo/nhiều xe nhưng vẫn có hộp người bộ phận/ROI trùng, xe che khuất chưa đầy đủ. Frame 217 giữ hộp xe máy gần vị trí hiện tại; hai cây hybrid thấy được thì YOLOE đơn chưa thấy. [So sánh trực tiếp](outputs/review5/comparison_frame217.jpg), [ảnh 5](outputs/review5/final_images/1280_roi_1/5.jpg), [video 2](outputs/review5/final_video_2/annotated.mp4). Không công bố precision/recall/mAP/IDF1 hoặc số vật thể duy nhất từ số track khi chưa có nhãn chuẩn.

Để giữ một model và khắc phục cây đáng tin, cần fine-tune YOLOE với bbox/mask đúng định nghĩa thân cây, hard negative cột điện/cột nhà/cờ, nhiều cảnh/kích thước/che khuất và dữ liệu toàn bộ 10 lớp. Chia train/validation/test theo cảnh/video, không chia ngẫu nhiên frame gần nhau. Đánh giá precision/recall từng lớp, hộp trùng và tracking, cùng FPS trên video độc lập. Không đưa bản thích nghi vài ảnh vào mặc định để làm đẹp hình mẫu.

## Kiểm chứng và sử dụng

`python -m lan_chiem.tests.run_tests --video`: 124 test, 121 đạt, 3 bỏ qua; [log](outputs/review5/tests_final.log). Có cache hỏng/khác weights, không gọi encoder, không bật hybrid, ROI xoay vòng, tracking và chặn chuyển backend ngầm. Test prompt cũ nay chọn DINO rõ ràng và giữ kiểm tra summary.

Xem [README](README.md). Khởi động lại tiến trình cũ để nạp bản mới. Đổi vocabulary cần xuất lại checkpoint.

SHA-256 nguồn `yoloe-v8s-seg.pt`: `ac2b90ed23011495a3e86d89caeb3432a15129cac8d849ba121293c8fc1e0536` (không đổi). Bản thực tế `lan_chiem/models/yoloe-v8s-single.pt`: `06bb25a39531f41f7b1e2969b027e1b089d0b01420b5f943f6e30401d1288535`.

`outputs/review5/source_before.zip` giữ mã/cấu hình/tài liệu trước đổi. Probe/adapter là hồ sơ thử nghiệm. `final_verification.json` kiểm tra hash mã runtime khớp hai video, checkpoint khớp, đủ 9 ảnh, đủ frame video xuất và log test đạt.
