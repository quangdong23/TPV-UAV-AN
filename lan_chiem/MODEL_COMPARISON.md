# So sánh bốn model YOLOE trên dữ liệu lan_chiem

**Khuyến nghị hiện tại: giữ YOLOE-v8S làm mặc định.** Ba model bổ sung chưa cải thiện độ chính xác chung trên phần dữ liệu đã kiểm tra. Không thay checkpoint mặc định hoặc chạy ensemble.

Đã thử thêm **YOLOE-11S, YOLOE-11M, YOLOE-v8M**. Chọn các model cùng hỗ trợ vocabulary mở để đối chiếu đủ bộ lớp đang cấu hình; YOLO pretrained COCO thông thường không có sẵn đầy đủ thân cây, xe hàng rong và các lớp đặc thù. Các họ model này được mô tả trong [tài liệu Ultralytics YOLOE](https://docs.ultralytics.com/models/yoloe/). Kết luận dưới đây chỉ áp dụng các cấu hình đã thử, không phải mọi model YOLO.

## Điều kiện đo

- RTX 3050 Ti Laptop 4 GB, cùng môi trường thư viện hiện tại; không nâng/cài lại môi trường.
- Cùng 9 ảnh PNG, hai video 602 và 584 frame; mỗi job GPU chạy riêng, mỗi lượt một detector. Không DINO bổ trợ. Các checkpoint được gộp vocabulary/Conv-BN để không nạp text encoder khi chạy.
- Cùng 63 phrase, ánh xạ 10 lớp, ngưỡng từng lớp, kích thước 1280, ROI 640 và bộ xử lý hậu kỳ. Ảnh tĩnh dùng grid ROI đầy đủ; video dùng tối đa 2 ROI xoay vòng, lịch AI mỗi 6 frame và phát hiện sớm khi track yếu, tracking mọi frame. Không bỏ frame nguồn.
- Người dùng xác nhận đang dùng pin/tiết kiệm điện, sau đó xác nhận đã cắm sạc. Số đo trước khi ổn định nguồn được lưu riêng, không dùng xếp hạng tốc độ. Đã chạy lại trên sạc, đổi thứ tự và kiểm tra độ dao động. Một lượt v8M trùng giai đoạn kiểm tra xuất video bằng CPU đã bị loại, thay bằng lượt riêng.

## Kết quả cuối trên sạc

| Model | FPS liên tục: video 1 / 2 | FPS gồm nạp/xuất: video 1 / 2 | Hộp khớp / 34 nhãn nháp | Hộp dư trong vùng audit |
|---|---:|---:|---:|---:|
| **YOLOE-v8S hiện tại** | **24,07 / 23,67** | **20,06 / 20,08** | **19** | **3** |
| YOLOE-11S | 23,35 / 23,93 | 16,98 / 16,98 | 14 | 12 |
| YOLOE-11M | 22,75 / 22,85 | 16,82 / 16,95 | 14 | 13 |
| YOLOE-v8M | 14,14 / 22,31 | 12,09 / 16,90 | 13 | 10 |

v8S, 11S và 11M đạt 15 FPS ở cả hai video trong lượt cuối. v8M không đạt ở một video và tốc độ dao động giữa các lượt, nên không xem nó là lựa chọn đáp ứng ổn định mục tiêu trên máy này. Không dùng lại kết luận sơ bộ 11M chậm trên pin sau khi đã có số đo trên sạc.

FPS liên tục loại thời gian nạp/warmup. FPS gồm nạp/xuất vẫn không gồm khởi động interpreter, import module, hash nguồn và seek khôi phục. Đây là thông lượng offline trên các đoạn video khoảng 18–19 giây, chưa phải live latency, tốc độ AI trên mọi frame hoặc cam kết cho mọi camera. Các lượt đầu chuyển nguồn và lịch sử đo cho thấy hiệu năng máy biến động; không diễn giải vài FPS chênh lệch thành ưu thế kiến trúc chắc chắn. [Lịch sử số đo](outputs/model_comparison/timing_history.json).

## Phạm vi đánh giá chất lượng

34 hộp tham chiếu **dạng nháp do trợ lý quan sát ảnh gốc**, không phải bộ ground truth được người gán nhãn độc lập duyệt. Bao gồm bàn/ghế ảnh 1, người/ô tô/xe đẩy/biển quảng cáo và 10 xe máy ảnh 5, biển quảng cáo ảnh 8 và bốn thân cây cụ thể ở ảnh 3/8. Một số vùng cột cổng được kiểm tra báo nhầm cột điện/xe đẩy. Chưa có positive cho lớp 8 và 10. Tập này nhấn vào một số lớp khó, không đại diện mọi tình huống triển khai.

Ghép một-một theo cùng lớp và IoU ≥0,5, chỉ tính dự đoán có tâm nằm trong vùng đã audit. Bbox gồm phần thân nhìn thấy; nhãn material và hộp xe che khuất còn cần duyệt. Các nhãn được áp dụng giống nhau cho mọi model, không chỉnh ngưỡng theo kết quả này. Có lưu lưới tọa độ ảnh gốc và ghi chú chỉnh lại phạm vi cây để người khác kiểm tra. **Không gọi các số trên là accuracy/mAP/precision/recall của toàn hệ thống.**

Quan sát đáng chú ý:

- Ở cảnh bàn/ghế, các model bổ sung còn nhầm ghế thành bàn, nhận vật thể không đúng material và tạo nhiều hộp dư. Model lớn không tự giải quyết semantic của bộ lớp này.
- 11M đặt thêm nhiều hộp xe ở bãi xe, nhưng một số hộp là bộ phận/trùng hoặc không đủ khớp IoU với nhãn nháp. Số box tăng không chứng minh số xe đúng tăng. Kết quả chỉ khớp 5/10 xe ở phần chẩn đoán này, ngang v8S; cần duyệt nhãn che khuất trước khi đưa ra kết luận rộng hơn.
- 11M nhận được thêm một số biển quảng cáo. Tuy nhiên xe đẩy thật ở ảnh 5 có điểm 0,90 lại bị đưa vào review do xung đột biển gắn trên xe. Đây là ví dụ lỗi chính sách hậu xử lý ngoài khả năng neural detector; không nên quy mọi lần thiếu output cho model.
- Tính cả candidate review, v8S và 11M cùng khớp 22/34 hộp nháp, nhưng hộp dư trong vùng audit lần lượt 8 và 22. Vì vậy không tự tắt bộ lọc review để tăng số vật thể hiển thị.
- Ba model mới chưa giải quyết các thân cây lớn đã bỏ sót. Trên bốn cây có nhãn nháp, v8S/v8M chỉ khớp một cây nhỏ; 11S/11M không khớp cây nào ở output predicted.

Ảnh đối chiếu: [bàn/ghế](outputs/model_comparison/comparison_image_1.jpg), [thân cây lớn](outputs/model_comparison/comparison_image_3.jpg), [xe máy/xe đẩy](outputs/model_comparison/comparison_image_5.jpg), [biển/cổng/cây](outputs/model_comparison/comparison_image_8.jpg). Xem [nhãn và phạm vi audit](outputs/model_comparison/diagnostic_labels.json), [kết quả chi tiết theo lớp](outputs/model_comparison/comparison.json).

## Quyết định và bước tiếp theo

Giữ v8S: đạt tốc độ, khớp nhiều hộp tham chiếu hơn và ít hộp dư hơn ở phần kiểm tra hiện tại. Đây là lựa chọn tạm thời có bằng chứng tốt hơn cho cấu hình đang dùng, chưa phải khẳng định v8S tốt nhất trên mọi cảnh. Vocabulary/ngưỡng hiện có được xây từ v8S; chưa tối ưu và xác nhận ngưỡng riêng từng model trên tập validation độc lập.

Muốn nâng chất lượng đáng kể, ưu tiên nhãn được duyệt cho cả 10 lớp, đặc biệt thân cây/cột, bàn/ghế, xe bị che và xe đẩy có biển gắn kèm. Đánh giá riêng lỗi model và lỗi chuyển sang review. Fine-tune một YOLOE theo hướng trong [báo cáo độ chính xác](MAINTENANCE_AND_ACCURACY.md), rồi so lại trên camera/cảnh độc lập. Không chuyển mặc định sang 11M chỉ vì nhiều tham số hoặc box hơn.

Có thể thử riêng một candidate mà vẫn chạy một model:

```powershell
python -m lan_chiem.predict --input lan_chiem/sample_data/5.png --output lan_chiem/outputs/try_11m --weights lan_chiem/models/comparison/yoloe-11m-seg-fixed.pt --device cuda
```

Chạy lại đối chiếu trong thư mục output mới, khi nguồn điện ổn định:

```powershell
python -m lan_chiem.compare_models --include-baseline --output lan_chiem/outputs/new_comparison
```

Đã xóa các bản source weights tải về sau khi xuất checkpoint gộp và dọn các lượt video/ảnh bị thay thế; giữ ba candidate gộp để chạy lại, kết quả cuối và lịch sử số đo gọn. `--download-only` có thể tải lại source weights chính thức nếu cần fine-tune. Không chỉnh phần ngoài `lan_chiem`, checkpoint nguồn và mặc định cũ không đổi.

[Kiểm chứng](outputs/model_comparison/verification.json): 4 checkpoint khác nhau, cùng vocabulary/mã runtime; 36 kết quả ảnh; giải mã lại đủ **4.744 frame của 8 video xuất cuối**. Hash checkpoint thực nạp được đối chiếu với file, không suy từ tên profile. Đây là kiểm tra pipeline và nguồn dữ liệu, không thay thế nhãn chuẩn.


Ghi chú dọn ngày 02/11/2026: đã bỏ các video so sánh lịch sử và segment trung gian để giảm dung lượng. Số đo, nhãn, ảnh đối chiếu và bản xác minh của lần chạy được giữ; không thể chạy lại kiểm chứng giải mã video cũ khi chưa tái xuất.
