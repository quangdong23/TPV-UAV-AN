# Demo YOLOE-v8S kết hợp Grounding DINO

Có thể trình diễn có kiểm soát trên video mẫu. Chưa đủ bằng chứng để công bố nhận diện chính xác cả 10 lớp hoặc tracking không đổi ID. Chế độ này được bật riêng; lệnh mặc định vẫn dùng một YOLOE.

## Cách kết hợp

- YOLOE-v8S xử lý toàn cảnh 1280, mỗi 6 frame nguồn. Tracking cập nhật trên mọi frame; không bỏ frame nguồn.
- DINO tiny chỉ bổ sung **thân cây**, khoảng mỗi 1,2 giây theo thời gian video. Không chạy DINO cho mọi frame, không dùng kết quả cũ như một phát hiện mới.
- Nhãn đối chứng `flag`, `utility pole`, `building column` giúp đưa kết quả mơ hồ sang review. Không xem việc DINO im lặng giữa hai lượt là bằng chứng cây đã biến mất.
- Video làm việc cạnh dài 1280, tracking 480, không ROI, xuất H.264 NVENC. Một lượt ROI có cùng độ phân giải thực trong cấu hình này làm tăng chi phí mà không thêm chi tiết.
- Nạp và warmup hai model một lần trước khi phát. Lần gọi DINO vẫn có thể gây một nhịp dừng; đây là thông lượng xử lý video, chưa xác minh độ trễ camera trực tiếp.

Helper dùng model/cache DINO tiny có sẵn trên máy, không tải mạng. Xử lý nhãn grounded theo [tài liệu Transformers](https://huggingface.co/docs/transformers/model_doc/grounding-dino); không tự gán mọi box thành cây khi thiếu nhãn. Phạm vi hỗ trợ hiện tại là thân cây, không phải DINO xác minh toàn bộ lớp của YOLOE.

## Chạy tại thư mục UAV

Chuẩn bị sẵn model trước phần trình chiếu, cắm sạc và giữ tải máy ổn định. Dùng thư mục output mới cho mỗi lượt:

```powershell
python -m lan_chiem.demo_session --input "lan_chiem/sample_data/Media Player 2026-10-28 09-10-14.mp4" "lan_chiem/sample_data/Media Player 2026-10-28 09-12-21.mp4" --output lan_chiem/outputs/my_demo --evaluate-images
```

Lệnh in `READY` sau nạp/warmup, tái sử dụng model cho hai video, xuất `session.json`, `summary.json`, `tracks.jsonl` và `annotated.mp4`. Thời gian chuẩn bị được ghi riêng và cộng lại trong `fps_including_preparation`; chưa tính thời gian import Python. Đây là xử lý/xuất video, không tự mở trình phát.

Chạy một video qua CLI hiện có:

```powershell
python -m lan_chiem.video_predict --backend demo-hybrid --input "lan_chiem/sample_data/Media Player 2026-10-28 09-12-21.mp4" --output lan_chiem/outputs/my_demo_video --device cuda
```

Ảnh tĩnh ưu tiên độ chi tiết hơn tốc độ video: giữ profile ROI của `yoloe_single.yaml`, DINO ở độ phân giải mặc định cao hơn:

```powershell
python -m lan_chiem.predict --backend demo-hybrid --config lan_chiem/yoloe_single.yaml --input lan_chiem/sample_data --output lan_chiem/outputs/my_demo_images --device cuda
```

## Kết quả và giới hạn

Kết quả cuối nằm trong [bản xác minh](outputs/demo/verification.json), [session hybrid](outputs/demo/hybrid/session.json), [đối chứng một model cùng profile](outputs/demo/primary/session.json). Hai đầu vào có 602 và 584 frame; kiểm tra giải mã toàn bộ video xuất, không chỉ đọc số FPS trong log. Cấu hình video cuối gọi YOLOE mỗi 6 frame, tăng tracking từ 320 lên 480 để giảm hộp trễ trên người đi xe; đã kiểm tra thêm frame 217, 227 và 350.

RTX 3050 Ti Laptop 4 GB, FP16, cắm sạc: hybrid đạt **31,90 / 30,36 FPS** trên hai video. Chuẩn bị model/warmup **8,23 giây** cho phiên này; thông lượng gồm chuẩn bị được ghi trong session. Trung bình hai video là **31,12 FPS**, tính cả chuẩn bị là **25,59 FPS**. Đối chứng YOLOE đơn cùng profile đạt 29,02 / 28,04 FPS ở lượt chạy riêng, gồm chuẩn bị đạt 26,17 FPS. Không suy ra hybrid nhanh hơn YOLOE: hai lượt không chạy đồng thời, xung/tải máy biến động; DINO vẫn thêm chi phí tính toán. Các lượt thử trước biến động lớn theo trạng thái máy, không bảo đảm con số này cho mọi máy hoặc tải nền.

Xem [video 1](outputs/demo/hybrid/video_1/annotated.mp4), [video 2](outputs/demo/hybrid/video_2/annotated.mp4), [đối chiếu cùng frame](outputs/demo/comparison_frame217.jpg) và [ảnh tĩnh chất lượng cao hơn](outputs/demo/images_quality).

DINO giúp bổ sung cây mà YOLOE đơn bỏ sót trong mẫu đối chiếu. Tuy nhiên vẫn quan sát được hộp trùng, nhận diện sai và hộp người/xe chưa ôm đủ vật thể. Không cộng số track ID thành số vật thể thật. Chưa đo IDF1/HOTA, chưa có nhãn video đầy đủ để đo mAP/precision/recall. Ở profile ảnh nhanh đi kèm phiên video, hybrid đạt 11 TP / 6 FP / 23 FN so với YOLOE đơn 9 TP / 6 FP / 25 FN trên phạm vi nhãn nháp; cải thiện đến từ cây, vẫn còn nhiều bỏ sót. Profile ảnh tĩnh có ROI và DINO 640 đạt 20 TP / 4 FP / 14 FN trên cùng phạm vi nháp; vì vậy giữ profile riêng cho ảnh, không dùng kết quả ảnh nhanh để đại diện ảnh tĩnh chất lượng cao. Bộ nhãn ảnh chẩn đoán hiện có chỉ là bản nháp 34 box theo vùng/lớp được chọn; số TP/FP trong bản xác minh không đại diện cả tập hay cả 10 lớp.

Khuyến nghị demo bằng video đã kiểm tra này, trình bày nhận diện và tracking tương đối. Nếu cần chứng minh độ chính xác toàn bộ lớp ở nhiều bối cảnh, cần bổ sung nhãn kiểm thử và dữ liệu thực tế; thay đổi nhịp chạy hai model không giải quyết hết sai số ngữ nghĩa của model.

## Kiểm tra mã

`python -m lan_chiem.tests.run_tests --video`: 133 test, 130 đạt, 3 bỏ qua theo điều kiện môi trường; log tại [tests.log](outputs/demo/tests.log). Có test nhịp DINO, không tái dùng phát hiện cũ, lọc nhãn sai/thiếu và input đường dẫn ảnh. Checkpoint YOLOE không fine-tune, không thay trọng số gốc. Provenance, hash mã và dữ liệu nằm trong kết quả từng video. Các thư mục lần chạy stable/primary_stable đã được chuyển thành hybrid/primary sau khi chốt; đường dẫn output ghi trong metadata gốc có thể giữ tên lúc chạy, không dùng chúng để resume. Các lượt thử loại bỏ được lưu gọn trong 	rials.json.
