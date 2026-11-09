# Chất lượng, tốc độ và cảnh báo theo dõi

## Chạy hệ thống

Từ `~/dong_workspace/UAV`, trong môi trường conda đã cài dependencies:

```bash
python -m uav_pipeline \
  --input lan_chiem/sample_data/6.mp4 \
  --output outputs/events_6_run_01 \
  --device cuda --events
```

Chọn một thư mục output mới cho mỗi lần chạy. Mặc định phân đoạn AI mỗi 5 frame;
những frame ở giữa dùng truyền mask theo chuyển động có kiểm tra chất lượng.
Không bỏ frame nguồn. Chỉ dùng một model nhận diện vật thể YOLOE; model phân đoạn
vỉa hè riêng. `--cv-threads 4` thống nhất số luồng OpenCV giữa hai nhánh, tránh
liên tục đổi thread pool. Lớp phủ được vẽ trực tiếp lên ảnh gốc để tránh làm mờ
nền do thu nhỏ rồi phóng lớn. Các phép LUT màu và lấy hai điểm số lớn nhất được
kiểm thử bằng kết quả chính xác từng phần tử so với phép tính trước.

Không có `--events`: vẫn chỉ có `annotated.mp4`, `tracks.jsonl`, `summary.json`.
Có `--events`: thêm `events.json` và `event_clips/` nếu có cảnh báo. Không tự xuất
ảnh mẫu/CSV/log chẩn đoán. Những clip có cùng cửa sổ frame dùng chung một file.

## Quy tắc đã triển khai

- Xét lấn chiếm cho bàn (4), ghế (5), xe đẩy (6), hàng hóa (10) và biển
  `sidewalk_standee` (7). Người, xe cộ, cột và cây không phát cảnh báo lấn chiếm.
  Cột/cây vẫn được giữ trên toàn khung hình để theo dõi và kiểm tra quan hệ biển.
  Điểm đáy bàn/ghế/xe đẩy/hàng hóa phải nằm trên nhãn vỉa hè 2; biển có tolerance
  20 pixel nguồn. Biển treo hoặc box ngang trên cao là `facade_attached`;
  thiếu bằng chứng là `sign_unresolved`. Các nhãn này là heuristic 2D.
- Biển có điểm tiếp xúc vỉa hè và không có cây/cột sát box được xét lấn chiếm
  sau 10 frame như các vật thể khác, không cần xác nhận gắn biển. Nếu box biển
  sát cây/cột, tạm chuyển sang nhánh theo dõi quan hệ biển–cây/cột; không đồng
  thời kết luận lấn chiếm dựa vào điểm đáy chiếu lên mask. Khi có nhiều điểm tựa
  mơ hồ hoặc chưa đủ bằng chứng thời gian, chưa phát cảnh báo cho biển đó.
  Đây là phân loại theo hình học 2D: biển đứng dưới đất sát cây có thể bị trì
  hoãn và cần xem lại; chưa có model phân loại đứng/treo độc lập.
- Mặc định 10 frame nguồn liên tiếp có vị trí hợp lệ. AI detection hoặc optical
  flow hợp lệ được tính. Vật thể tĩnh có thể dùng chuyển động camera đáng tin
  cậy giữa các lần AI, khi lần detection gần nhất chưa quá 0,5 giây. Box chỉ
  ngoại suy bằng Kalman, không có đo chuyển động đáng tin, không được tính. Cần ít
  nhất 2 lần detector quan sát độc lập và ID đã xác nhận. Do đó 10 frame theo
  dõi không được diễn giải thành 10 lần AI xác nhận độc lập.
- Mất dấu, ra khỏi vỉa hè, frame bị đứt hoặc cắt cảnh sẽ ngắt chuỗi. Box đã quá
  0,5 giây từ lần detector gần nhất không được dùng xác nhận. Một đợt liên tục
  chỉ phát một cảnh báo; cùng ID có thời gian chống lặp 14 giây.
- Clip lấy mốc **frame đầu tiên của chuỗi nằm trên vỉa hè**: 4 giây trước và 10
  giây sau, kể cả khi việc xác nhận xảy ra muộn hơn. Video đầu/cuối thiếu dữ
  liệu thì clip ngắn hơn, ghi rõ `truncated_start` / `truncated_end`.
- Clip cắt từ video chú thích sau khi phân tích xong, không giữ 14 giây ảnh trong
  RAM. Ưu tiên FFmpeg/H.264, nếu không có encoder phù hợp thì dùng OpenCV. Dùng thời gian CFR của video xuất; timestamp nguồn nằm trong metadata.
  Với nguồn VFR, cửa sổ 14 giây theo CFR có thể khác 14 giây theo PTS nguồn.
- Nghi vấn gắn biển: box biển gần cây/cột, chồng lấn theo chiều cao, quan hệ vị
  trí tương đối ổn định 10 frame, có ít nhất 2 lần cả hai được detector quan sát
  và có chuyển động camera đáng tin cậy. Cặp tách ra, thay đổi vị trí tương đối,
  mất dấu hoặc có hai điểm tựa mơ hồ sẽ ngắt chuỗi. Camera đứng yên không đủ
  chứng cứ để phát cảnh báo loại này.

Các tên sự kiện là `sidewalk_encroachment` và `sign_attachment_suspected`.
Cả hai có `requires_review: true`. Cây hoặc cột chỉ làm điểm tựa trong nghi vấn gắn biển;
không tự thỏa quy tắc lấn chiếm. Cần người xem xét trước khi kết luận vi phạm. Đặc biệt, quan hệ gần nhau trong ảnh 2D, kể cả
kéo dài nhiều frame, không chứng minh được liên kết vật lý trong không gian 3D.

Có thể điều chỉnh riêng bằng `--violation-frames 10 --attachment-frames 10`.
Không dùng frame skipping với cảnh báo. Luồng kết hợp hiện không hỗ trợ resume.

## Màu hiển thị và dấu vi phạm

Mỗi lớp có một màu riêng, dùng chung bảng màu cho box và nhãn ở chế độ vẽ
nhanh/chế độ đầy đủ. Hai màu gần đỏ trước đây của bàn và xe hàng rong đã đổi
sang vàng và cyan; đỏ thuần dành cho viền cảnh báo. Người giữ màu lớp riêng.

Điểm tiếp xúc để tính cảnh báo mặc định dùng đúng điểm giữa đáy box như chế độ
lọc `bottom-center`, bỏ yêu cầu toàn bộ vùng 5×5 pixel phải là vỉa hè. Khi chọn
`--filter-mode overlap`, việc hiển thị vẫn dựa vào tỷ lệ overlap, còn cảnh báo
vẫn dựa vào điểm tiếp xúc; hai khái niệm này có thể cho kết quả khác nhau.

Một ID đã xác nhận vi phạm giữ viền đỏ khi tiếp tục được hiển thị và
`is_encroaching` của frame hiện tại còn đúng; không tô đỏ biển mặt tiền hoặc
vật thể đã mất tiếp xúc. Chu kỳ chờ detector vẫn dùng kiểm tra mask hiện tại. Chuỗi đếm mới vẫn
được kiểm tra đầy đủ; không tạo cảnh báo/clip trùng chỉ để duy trì màu đỏ. Cắt
cảnh hoặc frame nguồn bị đứt sẽ xóa trạng thái tô đỏ của cảnh trước. ID mới sau
mất dấu phải xác nhận lại. Luôn bật `--events` để có dấu đỏ và clip bằng chứng.

## Kết quả đo và giới hạn chất lượng

Trên GPU công ty, video 6 có 712 frame, bản tối ưu không bật cảnh báo chạy hết
57,89 giây: **12,30 FPS toàn lượt**, 12,65 FPS phần xử lý sau tải model/warmup.
Mục tiêu 15 FPS **chưa đạt**; không dùng FPS riêng của detector để thay cho FPS
cả pipeline. Trong lượt này phân đoạn chiếm 34,15 giây, detector 11,14 giây.
Xem `summary.json` của từng lượt để lấy số đo đúng cấu hình và video đó.

Không thay trọng số/ngưỡng detector để giả định rằng chất lượng đã tăng. Thử
Grounding DINO tiny làm detector duy nhất trên ảnh mẫu mất khoảng 2–3,9 giây/ảnh
với cấu hình đối chiếu, nên không đưa vào mặc định. Thử đổi cách dùng FP16 của
phân đoạn không nhanh hơn; một cách còn thay đổi mask, nên cũng không áp dụng.

Cải thiện recall của cây/cột/biển **chưa được xác nhận**. Bộ dữ liệu đánh giá
độc lập hiện không có sau lần xóa datasets; ảnh/video chưa gán nhãn chỉ dùng
kiểm tra trực quan, không cho phép kết luận precision/recall đã tốt hơn. Không
nên đưa cảnh báo vào vận hành tự động trước khi đánh giá detector, ID tracking,
mask vỉa hè và cảnh báo trên các cảnh độc lập có nhãn. Tối ưu hiển thị và tốc độ
không thay thế được cải thiện trọng số cho những lớp model đang nhận diện kém.

## Cơ sở kỹ thuật

- [OpenCV: homography và giới hạn hình học](https://docs.opencv.org/4.12.0/d9/dab/tutorial_homography.html)
- [MIT Vision Book: quan hệ giữa mặt phẳng, chuyển động camera và homography](https://visionbook.mit.edu/homography.html)
- [Ultralytics: YOLOE và huấn luyện với prompt](https://docs.ultralytics.com/models/yoloe/)
- [Grounding DINO: bài báo gốc](https://arxiv.org/abs/2303.05499)

Các ngưỡng hình học là heuristic kỹ thuật cần kiểm định trên dữ liệu drone của
hệ thống; không phải tiêu chuẩn pháp lý hoặc xác suất đã được hiệu chuẩn.

Kiểm thử: `python -m lan_chiem.tests.run_tests --video` — 158 bài, không lỗi,
3 bài bỏ qua vì thiếu artifact. Có kiểm tra cửa sổ clip 140 frame ở 10 FPS,
clip chồng lấn, thời điểm xác nhận frame thứ 10 và tích hợp video thực với model giả lập.

Lượt video 6 với quy tắc cũ (cả 10 lớp, trước khi loại người và tách loại biển) bật cảnh báo: 16 cảnh báo lấn chiếm, gộp cửa sổ giống nhau còn
13 clip; không có cảnh báo nghi vấn gắn biển trong lượt này. Phân tích đạt
13,09 FPS sau tải/warmup. Xuất riêng 13 clip bằng FFmpeg mất 36,14 giây
(so với 78,01 giây bằng OpenCV). Tất cả clip đọc được và đúng số frame.
Đây là kiểm thử hoạt động, chưa phải số đo độ chính xác cảnh báo. Thời gian
xuất clip làm FPS toàn lượt thấp hơn FPS phân tích; mục tiêu 15 FPS toàn lượt
bao gồm xuất clip chưa đạt.
