# Dọn dẹp và kiểm tra độ chính xác YOLOE

## Đã dọn

Xóa 1.440 file thử nghiệm, khoảng **1.881 MiB (1,84 GiB)**: benchmark cũ, video xuất lặp, checkpoint adapter bị loại, probe/crop và log cũ. Sau đó bỏ thêm 11 profile không được mã hiện tại tham chiếu và module `trunk_refiner.py` đã ngừng sử dụng. Các script thử riêng ảnh 1/4/5 và tài liệu lịch sử trùng cũng nằm trong danh sách dọn. [Danh sách lần dọn chính](outputs/maintenance/cleanup_plan.json).

Giữ nguyên `sample_data`, checkpoint đang dùng trong `models`, mã ứng dụng và bộ test. `outputs/current` chứa kết quả mới; `outputs/review5` chỉ giữ baseline ảnh/video cuối, số đo, kiểm chứng và bản sao mã trước thay đổi. Cache embedding nhỏ được giữ để xuất lại checkpoint. Không xóa backend so sánh độc lập còn được công cụ đánh giá sử dụng; chúng không chạy trong luồng YOLOE. Bổ sung `.gitignore` cho output/cache, không tự động xóa dữ liệu khi chạy ứng dụng.

## Lỗi đã sửa

Luồng đang dùng: CLI → YOLOE cố định vocabulary → toàn cảnh/ROI → ánh xạ ID chuẩn → hợp nhất fragment và NMS cùng lớp → kiểm tra xung đột. Video bổ sung lịch gọi detector, Kalman và Lucas–Kanade trên các frame ở giữa. Không có model nhận diện thứ hai.

Ở ảnh 5, ROI cắt gần chân người nhưng vẫn trả hộp phần thân với confidence cao. Toàn cảnh cũng lặp lại hộp phần thân đó. NMS theo IoU không loại được vì hộp phần thân và hộp đầy đủ chênh kích thước; ưu tiên score lại giữ hộp thiếu.

Đã thêm `reconcile_roi_fragments` trong `postprocess.py`, nối vào `YOLOEDetector.decode` **trước NMS**:

- Chỉ xét cùng lớp, khác lượt nhìn, có bằng chứng gần biên ROI nhân tạo và hộp đầy đủ từ lượt khác bao chứa ≥95% hộp nhỏ, vượt ra ngoài biên crop.
- Giữ hộp đầy đủ đã được model dự đoán; không tự nới tọa độ, tăng confidence hoặc đổi lớp.
- Sau khi xác nhận fragment, hợp nhất bản toàn cảnh gần như trùng fragment đó (IoU ≥0,90).
- Không gộp rider với xe, không coi biên ảnh thật là biên crop, không chọn hộp lớn bao nhiều vật thể rời, không giải quyết phạm vi thân cây/tán cây bằng quy tắc này. Giữ evidence hộp bị hợp nhất.

Kết quả ảnh 5: từ **hai hộp người thành một hộp đầy đủ**, giữ confidence gốc 0,6929. Tổng detection từ 10 còn 9. **Tám ảnh khác giữ nguyên danh sách dự đoán**, gồm lớp, nguồn, tọa độ và điểm đã xuất. Đây là sửa một lỗi cụ thể, không phải bằng chứng tăng mAP toàn bộ hệ thống. [So sánh trước/sau](outputs/current/roi_before_after.jpg).

## Kiểm tra sau khi dọn mã

| Video mẫu | Frame xử lý/xuất đã giải mã lại | FPS liên tục | FPS gồm nạp model và hoàn tất xuất |
|---|---:|---:|---:|
| 09-10-14 | 602/602 | 30,08 | 24,63 |
| 09-12-21 | 584/584 | 29,42 | 24,08 |

RTX 3050 Ti Laptop 4 GB, cùng checkpoint YOLOE đơn. Chạy từng job riêng. Pipeline dùng detector mỗi 6 frame nguồn, có phát hiện sớm khi track yếu và 2 ROI xoay vòng; tracking xử lý mọi frame. Số trên là thông lượng offline, không phải mỗi frame đều chạy AI hoặc phép đo live latency. Phần gồm nạp/xuất không tính khởi động Python, import và hash nguồn. FPS thay đổi giữa các lần chạy.

129 test: **126 đạt, 3 bỏ qua**. Đã bỏ test riêng của refiner đã xóa; giữ kiểm tra tracking và chặn profile cũ tự bật hybrid. Có sáu test hồi quy mới cho fragment/clone, hộp người thực tế, phân biệt lớp, hộp nhóm và biên ảnh. [Log](outputs/maintenance/tests.log), [kiểm chứng ảnh/video/hash](outputs/current/verification.json), [frame 217](outputs/current/video_2/frame217.jpg), [video mới](outputs/current/video_2/annotated.mp4).

Checkpoint nguồn và bản dùng thực tế có SHA-256 không đổi so với lần trước. Kiểm chứng đảm bảo hash mã runtime khớp cả hai video, đủ 9 ảnh, source hash khớp dữ liệu mẫu và checkpoint vẫn là một detector không nạp text encoder.

## Vấn đề còn lại và hướng nâng độ chính xác

**Thân cây vẫn là lỗi bỏ sót của model.** Lần kiểm tra trước cho thấy điểm phân loại trunk rất thấp dù head có thể đặt hộp tại cây. Dọn file, tăng thời gian giữ track hoặc hạ ngưỡng cực thấp không giải quyết khả năng phân loại; hạ ngưỡng còn tăng báo sai. Xe che khuất cũng có thể nhận hộp thiếu một phần; tracker không thể tự sửa định nghĩa hộp mà detector chưa học tốt. Không có nhãn đầy đủ nên chưa công bố precision/recall/mAP/IDF1.

Hướng ưu tiên là **fine-tune một YOLOE-v8s cho 10 lớp thực tế**, thay vì thêm model khi chạy:

1. Gán nhãn toàn cảnh, kể cả vật thể bị bỏ sót, không chỉ crop từ detection. Thống nhất bbox thân cây khác tán cây/cột nhà; bbox người/xe khi che khuất; vật liệu xây dựng và hàng hóa đúng định nghĩa catalog. Thêm hard negative cho các nhầm lẫn này. DINO có thể hỗ trợ gợi ý nhãn offline nếu cần, nhưng phải duyệt và không coi dự đoán là ground truth.
2. Chia train/validation/test theo camera/cảnh/thời điểm, không chia ngẫu nhiên frame gần nhau. Chín ảnh và hai đoạn cùng bối cảnh chưa đủ để chứng minh tính tổng quát. Không dùng tập test để chỉnh ngưỡng lặp đi lặp lại.
3. Dùng trainer YOLOE chính thức, với detection YAML cùng scale rồi nạp trọng số nguồn YOLOE tương thích do người dùng cung cấp, chọn `YOLOEPETrainer` cho nhãn bbox. Không dùng trainer mặc định như thể nó tự học đúng vocabulary tùy chỉnh. [Hướng dẫn chính thức](https://docs.ultralytics.com/models/yoloe/#train-usage). Đã đối chiếu class trainer trong Ultralytics đang cài; chưa chạy huấn luyện mới khi chưa có bộ nhãn được duyệt.
4. Khi huấn luyện thành 10 class, triển khai mapping rõ ràng YOLO 0–9 → catalog 1–10. Checkpoint 10 class cần config **đã resolve đầy đủ**, mỗi class một tên đúng thứ tự, `counter_phrases: {}`; không nạp nó với vocabulary 63 phrase hiện tại. Kế thừa config bằng deep merge không xóa các counter cũ chỉ bằng một dict rỗng ở file con. Loader hiện kiểm tra tên/hash nên sẽ từ chối mapping sai.
5. Đo precision/recall từng lớp, lỗi hộp bộ phận/trùng, ID switch và FPS trên dữ liệu độc lập. Chỉ thay model khi lớp khó cải thiện và lớp khác không suy giảm; sau đó mới cân nhắc TensorRT để tối ưu tốc độ. TensorRT không sửa được lỗi học nhãn.

Lưu ý tích hợp: CLI hiện chọn `yoloe_single.yaml` (1280, grid ROI). API khởi tạo không truyền config vẫn dùng `config.yaml` với cấu hình nhẹ hơn; không được lấy số đo CLI để khẳng định chất lượng API đó. Để dùng đúng profile đã kiểm tra:

```python
from lan_chiem.yoloe_detector import YOLOEDetector
detector = YOLOEDetector(config_path="lan_chiem/yoloe_single.yaml", device="cuda")
result = detector.predict("lan_chiem/sample_data/5.png")
```

Video nên dùng pipeline `video_predict` như README để có lịch phát hiện, ROI xoay vòng và tracking; gọi trực tiếp `predict` liên tục không tự có những cơ chế này. Khởi động lại tiến trình đang chạy sau khi cập nhật mã.


## Dọn bổ sung ngày 02/11/2026

Đã xóa 209 file, giải phóng 1.116,8 MiB (khoảng 1,09 GiB): cache Python/test, các segment trung gian của video đã hoàn tất, checkpoint resume tương ứng và video so sánh model lịch sử. Giữ video xuất cuối trong current/demo/review5 và video demo tiếng Việt, toàn bộ mã nguồn, cấu hình kế thừa, model, dữ liệu mẫu, nhãn và số đo. Các lượt đã dọn segment không còn resume; muốn phân tích lại hãy chọn output mới. Video so sánh model lịch sử không còn trên đĩa, số đo trong báo cáo là kết quả lần chạy trước. [Danh sách đã xóa](outputs/maintenance/cleanup_2026-11-02.json).
