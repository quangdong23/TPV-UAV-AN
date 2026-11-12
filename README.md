# Hệ Thống UAV Giám Sát và Nhận Diện Vi Phạm Lấn Chiếm Vỉa Hè

Hệ thống thị giác máy tính tự động tích hợp xử lý video từ thiết bị bay không người lái (UAV) để **phát hiện, theo vết và ghi nhận các hành vi lấn chiếm vỉa hè đô thị**, phân biệt đối tượng vi phạm với các cấu trúc hạ tầng hợp lệ theo ngữ cảnh đường phố Việt Nam.

---

## 🌟 Tính Năng Chính

1. **Phân vùng ngữ nghĩa vỉa hè & lòng đường (Sidewalk & Road Segmentation):**
   - Tự động tách ranh giới vỉa hè (`sidewalk`), lòng đường (`road`), người đi bộ và không gian công cộng từ góc nhìn UAV chéo/thẳng đứng.
   - Hỗ trợ phép biến đổi chuyển động camera và bù trừ trôi khung hình giữa các keyframe.

2. **Nhận diện đối tượng mục tiêu (Object Detection):**
   - Tích hợp mô hình nhận diện YOLOE chuyên biệt cho các nhóm vi phạm lấn chiếm thường gặp:
     - Biển hiệu / Biển quảng cáo (`billboard / advertising sign`)
     - Bàn, ghế, dù che đặt ngoài trời (`tables, chairs, sunshade umbrellas`)
     - Hàng hóa buôn bán bày ra vỉa hè (`merchandise / display goods`)
     - Phương tiện dừng đỗ trái phép trên vỉa hè (`motorcycles, cars, carts`)
     - Mái che, bạt vươn ra không gian vỉa hè (`awning, canopy`)
     - Cây xanh, cột điện/cột đèn đô thị (`trees, utility poles`).

3. **Công cụ phát hiện sự kiện vi phạm (Encroachment Event Engine):**
   - Đánh giá giao cắt không gian giữa chân đế vật thể với vùng vỉa hè.
   - Theo vết đối tượng (Tracking) qua thời gian để xác nhận vi phạm lâu dài (`encroachment_frames`), loại bỏ các trường hợp đi ngang qua tạm thời.
   - Phát hiện gắn biển hiệu trái phép lên thân cây/cột điện (`sign attachment`).
   - Tự động trích xuất các đoạn video bằng chứng (evidence clips) trước và sau thời điểm vi phạm.

---

## 📂 Cấu Trúc Dự Án

```text
├── configs/                       # Cấu hình hệ thống
│   ├── detection.yaml             # Cấu hình phát hiện vật thể, ngưỡng tin cậy & NMS
│   ├── recognition_context.yaml   # Quy tắc ngữ cảnh đô thị Việt Nam (4 nhóm lớp khó)
│   ├── video.yaml                 # Cấu hình tham số xử lý video & tracking
│   ├── benchmarks/                # Cấu hình kiểm thử benchmark hiệu năng
│   └── segmentation/              # Cấu hình mô hình phân đoạn vỉa hè (CPU / GPU T4)
├── data/
│   └── samples/                   # Video và hình ảnh UAV mẫu phục vụ kiểm thử
├── models/                        # Trọng số mô hình YOLOE & CLIP
├── requirements/                  # Phụ thuộc bổ sung (GIS, COCO evaluation)
├── tests/                         # Bộ kiểm thử đơn vị & tích hợp pipeline
├── uav_pipeline/                  # Mã nguồn chính của pipeline
│   ├── cli/                       # Giao diện dòng lệnh (run, detect, segment, ...)
│   ├── detection/                 # Module suy luận phát hiện vật thể & ngữ cảnh
│   ├── segmentation/              # Module phân vùng không gian vỉa hè
│   ├── tracking/                  # Module theo vết vật thể và ước lượng chuyển động
│   ├── video/                     # Module trích xuất, lập lịch và ghi video đầu ra
│   ├── events.py                  # Bộ phân tích logic vi phạm lấn chiếm vỉa hè
│   └── outputs.py                 # Định dạng xuất kết quả (JSONL, CSV, evidence clips)
├── .gitattributes                 # Cấu hình Git LFS cho file mô hình và video
├── .gitignore                     # Bỏ qua cache tạm, outputs và môi trường ảo
├── pyproject.toml                 # Cấu hình gói và metadata dự án
└── requirements.txt               # Thư viện runtime bắt buộc
```

---

## 🚀 Cài Đặt

### 1. Yêu cầu môi trường
- Python >= 3.10
- GPU NVIDIA (khuyên dùng để đạt tốc độ xử lý video thời gian thực) hoặc CPU.

### 2. Cài đặt thư viện
```powershell
pip install -r requirements.txt
```

Nếu cần xuất bản đồ GIS hoặc chạy benchmark đánh giá COCO:
```powershell
pip install -r requirements/gis.txt
pip install -r requirements/evaluation.txt
```

---

## 💻 Hướng Dẫn Sử Dụng

### 1. Phân tích video phát hiện lấn chiếm vỉa hè
Lệnh toàn diện chạy phân đoạn vỉa hè, phát hiện vật thể, theo vết và ghi nhận các sự kiện vi phạm:

```powershell
python -m uav_pipeline run --input data/samples/1.mp4 --events
```

Các tuỳ chọn quan trọng:
- `--events`: Kích hoạt nhận diện sự kiện vi phạm lấn chiếm vỉa hè và xuất clip bằng chứng.
- `--output <path>`: Thư mục lưu kết quả (mặc định lưu tại `outputs/`).
- `--output-mode {compact,full}`: Chế độ xuất kết quả (`compact`: video annotated + tóm tắt JSONL; `full`: lưu thêm ảnh trích xuất và CSV).
- `--device {cuda,cpu}`: Chọn thiết bị tính toán.

### 2. Kiểm tra nhanh phân đoạn vỉa hè trên ảnh / video
```powershell
python -m uav_pipeline segment --input data/samples/1.png --output outputs/sidewalk_preview.png
```

### 3. Nhận diện vật thể trên ảnh tĩnh
```powershell
python -m uav_pipeline detect --input data/samples/1.png --output outputs/detection_preview.png
```

### 4. Chạy kiểm thử hệ thống
```powershell
python tests/run_tests.py
```
Để chạy kèm kiểm thử tích hợp video:
```powershell
python tests/run_tests.py --video
```

---

## 📊 Kết Quả Đầu Ra

Khi chạy với cờ `--events`, hệ thống tự động sinh:
1. **Video trực quan hóa**: Vẽ ranh giới phân định vỉa hè, hộp bao nhận diện vật thể và nhãn cảnh báo vi phạm màu đỏ.
2. **`events.jsonl`**: Ghi lại từng sự kiện lấn chiếm (thời điểm bắt đầu, kết thúc, loại vi phạm, tọa độ, thời lượng).
3. **Thư mục `evidence_clips/`**: Cắt ngắn các đoạn video vi phạm (khoảng 14 giây gồm 4s trước và 10s sau khi vi phạm) để báo cáo cơ quan chức năng.
