# Dọn phần thử nghiệm không sử dụng

Đã xóa 46 file, khoảng 668.0 MiB theo kích thước file.
Gỡ bộ code/test/hướng dẫn fine-tune không được chọn, script thử prompt dùng một lần, MobileCLIP/model kiểm tra AMP tải cho train, ZIP nhãn trùng và ảnh đối chiếu cũ.

## Kiểm tra hệ thống còn lại

- Trước dọn: 144 test, không lỗi, 3 bỏ qua.
- Sau dọn: 135 test, không lỗi, 3 bỏ qua. Chín test của công cụ training được gỡ cùng công cụ.
- Hash 99 file còn lại khớp trước/sau, gồm code suy luận, cấu hình, checkpoint hiện tại/checkpoint nguồn và sample_data.
- Ba CLI kết hợp, nhận diện video và phân đoạn vẫn chạy được --help.
- Không sửa thuật toán, vocabulary, ngưỡng, lịch keyframe, quy tắc lọc hoặc định dạng xuất của pipeline đang chạy.

## Dữ liệu cần lưu ý

Người dùng xác nhận đã chủ động xóa toàn bộ `datasets/` và `outputs/` ở Terminal trong lúc dọn. Hai thư mục hiện không còn; không tự phục hồi theo yêu cầu giữ bản code gọn.
Dữ liệu `lan_chiem/sample_data/`, checkpoint hiện tại và checkpoint nguồn vẫn giữ nguyên hash.

## Danh sách đã xóa trong thao tác này

| Đường dẫn | Số file | MiB |
| --- | --- | --- |
| `lan_chiem/training` | 11 | 0.08 |
| `lan_chiem/tests/test_training_preparation.py` | 1 | 0.01 |
| `mobileclip_blt.ts` | 1 | 571.98 |
| `yolo26n.pt` | 1 | 5.29 |
| `datasets/uav_annotations_3_6.zip` | 1 | 16.87 |
| `datasets/uav_annotations_all_sources.zip` | 1 | 64.44 |
| `datasets/uav_objects/annotation_sources` | 4 | 0.04 |
| `datasets/uav_objects/annotation_review` | 14 | 9.13 |
| `datasets/uav_objects/ANNOTATION_REPORT.md` | 1 | 0.00 |
| `datasets/uav_objects/ANNOTATION_REPORT_ALL.md` | 1 | 0.00 |
| `datasets/uav_objects/visual_annotations.json` | 1 | 0.17 |
| `datasets/uav_objects/annotation_validation.json` | 1 | 0.00 |
| `lan_chiem/check_1_4_5.py` | 1 | 0.00 |
| `lan_chiem/crop_all_1_4_5.py` | 1 | 0.00 |
| `lan_chiem/test_1_props.py` | 1 | 0.00 |
| `lan_chiem/test_4_cart.py` | 1 | 0.00 |
| `lan_chiem/test_5_props.py` | 1 | 0.00 |
| `lan_chiem/test_distractors_5.py` | 1 | 0.00 |
| `lan_chiem/test_goods_prompt.py` | 1 | 0.00 |
| `lan_chiem/test_sign_prompts.py` | 1 | 0.00 |

## Bỏ trọng số nguồn YOLOE không dùng khi chạy video

Đã xóa checkpoint nguồn `yoloe-v8s-seg.pt`, hướng dẫn trial cũ và công cụ
kiểm tra mask riêng cho checkpoint segmentation. Bỏ fallback alias sang
checkpoint nguồn; các mặc định YOLOE còn lại dùng checkpoint single đang triển khai.
Giữ nguyên trọng số `lan_chiem/models/yoloe-v8s-single.pt` và metadata nguồn gốc
của nó (tên checkpoint cũ trong metadata là lịch sử xuất model, không phải phụ thuộc runtime).
Công cụ xuất checkpoint cần `--source-weights` rõ ràng; script training bên ngoài
cần `--weights` rõ ràng, còn `--prepare-only` không cần model.
Kiểm thử sau dọn: 158 bài, không lỗi, 3 bỏ qua; SHA256 model đang dùng không đổi.

## Dọn cache ngày 2026-11-09

Đã xóa 187 file cache Python trong 13 thư mục `__pycache__`, tổng 1.476.318 byte (khoảng 1,41 MiB), gồm bytecode của nhiều script đã được gỡ trước đây.

Đối chiếu SHA256 trước/sau: 103 file ngoài cache, Git và outputs giữ nguyên nội dung, gồm mã nguồn, cấu hình, model YOLOE và dữ liệu mẫu. Giữ `.hf_cache` vì pipeline dùng để nạp model phân đoạn; giữ toàn bộ kết quả `outputs` và lịch sử `.git`. Không xóa mã nguồn chỉ dựa trên việc chưa thấy import. Không chạy lại suy luận vì thay đổi chỉ xóa cache tự sinh.
