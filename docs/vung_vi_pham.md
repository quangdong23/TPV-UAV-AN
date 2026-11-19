# Gom vùng và bằng chứng vi phạm

- Bàn/ghế gần nhau thành một cụm, một cảnh báo. Nếu có người ngồi/cúi sát cụm thì dùng tên “kinh doanh dịch vụ ăn uống lấn chiếm vỉa hè/lòng lề đường”.
- Hàng hoá ghép với người gần bên; vật liệu không ghép người.
- Máy móc chỉ ghép người gần có bbox tư thế ngồi/cúi. Đây là heuristic, chưa có model xác minh động tác sửa máy hoặc kinh doanh ăn uống; sự kiện luôn cần kiểm duyệt.
- Biển đứng trên vùng xét có tên “đặt biển quảng cáo sai quy định”; nhánh biển gắn cây/cột giữ rule riêng, dùng chung mô tả cảnh báo này.
- Sửa/rửa xe giữ MobileNetV3 và crop người–xe, đổi mô tả thêm “lấn chiếm vỉa hè/lòng lề đường”.

## Ngưỡng

`violation_regions.py`: gom bàn ghế với khoảng cách bbox ≤ 0,6 đường chéo bbox nhỏ hơn; nhóm khác dùng 0,3. Người cần ở gần ≤ 0,2 và độ lệch chân ≤ 0,6 chiều cao người. Tư thế ngồi/cúi dựa vào tỉ lệ rộng/cao ≥ 0,65 hoặc nhãn `seated` của YOLOE.

Giữ ngưỡng thời gian của `EventOptions`: 10 frame hợp lệ và 2 quan sát detector; nối sự kiện qua mất dấu tối đa 2 giây. Mặt đường sát vỉa hè được xét trong dải 20 pixel ảnh gốc khi có bằng chứng mặt đường; không mở rộng ra toàn bộ lòng đường. Đây là khoảng cách ảnh, không phải ranh giới pháp lý hoặc khoảng cách mét.

## Đầu ra

- `event_images/<event_id>_crop.jpg`: ảnh gốc, bbox nhóm, padding 15% và ít nhất 8 pixel mỗi bên, giới hạn trong ảnh.
- `event_images/snapshot_<frame>.jpg`: ảnh bối cảnh dùng chung cho các sự kiện chụp cùng frame.
- `event_images/snapshot_<frame>.json`: danh sách cảnh báo, bbox và mô tả tổng hợp của snapshot.
- `events.json` (video file) hoặc `closed_events.jsonl` (live): từng sự kiện và thông tin clip chung.
- Clip có cửa sổ chồng nhau được gộp nếu tổng độ dài không quá 15 giây, gồm đủ `alerts`, `description`; upload một lần với ảnh của các sự kiện trong clip. Hết giới hạn thì tách clip.

Chạy: `python -m uav_pipeline run --input <video> --events --activity --no-upload-events`.

## Cụm người

Từ 5 người đứng gần nhau trên toàn ảnh: một bbox, một cảnh báo `crowd_gathering_suspected`, tên hiển thị “tụ tập trái phép”. Chỉ dùng track đã xác nhận, còn mới ≤ 0,5 giây; người ngồi/cúi không được tính. Khoảng cách chân bbox chuẩn hoá theo chiều cao người nhỏ hơn ≤ `EventOptions.crowd_gap_ratio` (hiện là 0,5); giới hạn độ trải cụm bằng 2 lần ngưỡng này để tránh nối một hàng người dài.

Giữ rule 10 frame hợp lệ, ít nhất 2 quan sát detector, mất dấu tối đa 2 giây. Crop padding và clip chung như các vùng khác. Tên hiển thị do cấu hình nghiệp vụ; số người, tư thế và khoảng cách không xác minh tính hợp pháp. Sự kiện luôn có `requires_review: true`.

## Hồ sơ giả cho demo

Danh sách 10 hồ sơ nằm trong `configs/demo_people.json`. Chỉ sự kiện có người mới được gắn `demo_people` trong log/snapshot; mỗi hồ sơ có `is_demo: true`, `identity_verified: false`, `identity_source: synthetic`. Đây không phải dữ liệu nhận diện từ video. Họ tên luôn có hậu tố `(DEMO)`; CCCD giả giữ dạng chuỗi để không mất số 0 đầu.

API `/api/uploads/initiate` nhận một hồ sơ đại diện qua `people` (họ tên) và `cccd` (CCCD giả), ví dụ:

```json
{"people":"Nguyễn Văn An (DEMO)","cccd":"000000000001"}
```

Clip gộp giữ hồ sơ của tất cả track trong log, gửi một hồ sơ đại diện cho clip. Retry dùng hồ sơ đã lưu, không random lại. Sự kiện không có người không gửi hai trường này. Chưa thay đổi giao diện/backend API và không gửi dữ liệu giả lên API thật trong test.

## Confidence demo

Mỗi cảnh báo và clip có `confidence` giả trong khoảng `0.85–0.98`, luôn kèm `confidence_source: "demo"`. Đây không phải xác suất model hay kết luận vi phạm. Điểm giả được tạo ổn định theo mã sự kiện/clip và thời gian, lưu trong log/manifest và sổ upload; retry không đổi điểm và không tạo upload mới chỉ vì bổ sung confidence. API `/api/uploads/initiate` nhận cả hai trường này. Các điểm model gốc (`score`, `probabilities`) được giữ nguyên. Giao diện/backend cần hiển thị nhãn demo tương ứng; test không gửi lên API thật.
