"""
Module predict: Giao diện dòng lệnh (CLI) nhận diện đối tượng UAV 2.2 (Pure CLI, Headless).
Hỗ trợ:
- Chạy ảnh đơn hoặc thư mục ảnh từ thư mục gốc:
  + python -m lan_chiem.predict --input path/to/image.jpg --output lan_chiem/outputs
  + python -m lan_chiem.predict --input path/to/images --output lan_chiem/outputs
- Chế độ cô lập một lớp đối tượng (--class-id hoặc --classes):
  + python -m lan_chiem.predict --input path/to/image.jpg --class-id 6
- Chế độ cô lập một prompt (--prompt):
  + python -m lan_chiem.predict --backend grounding-dino --input path/to/image.jpg --prompt "food vending cart"
- Tự động xóa file review cũ nếu lượt chạy mới không có ứng viên review.
- Định danh file ổn định (MD5 slug, không dùng hash() mặc định).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from PIL import Image

from lan_chiem.registry import ClassRegistry
from lan_chiem.object_classes import CATALOG_VERSION
from lan_chiem.sidewalk import load_sidewalk_from_file
from lan_chiem.visualization import draw_detections, draw_candidates_overlay

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CLI nhận diện đối tượng ảnh UAV bằng YOLOE-v8-S (Headless Pure CLI)."
    )
    parser.add_argument('--backend', choices=['grounding-dino', 'yoloe', 'demo-hybrid'], default='yoloe')
    parser.add_argument('--weights', default=None, help='Override model.weights in config')
    parser.add_argument('--imgsz', type=int, default=None, help='Override model.imgsz in config')
    parser.add_argument('--roi-focus', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Đường dẫn file ảnh đơn lẻ hoặc thư mục ảnh.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Thư mục lưu ảnh và JSON kết quả (mặc định: tự sinh tên trong outputs/).",
    )
    parser.add_argument(
        "--query-mode",
        type=str,
        default=None,
        choices=["compact", "per_class", "grouped"],
        help="Chế độ truy vấn: compact (3 nhóm có đối chứng, mặc định), per_class (10 lớp) hoặc grouped (6 nhóm).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["full", "tiled", "full+tiled"],
        help="Chế độ suy luận: 'full' (toàn ảnh), 'tiled' (chia mảnh), hoặc 'full+tiled'.",
    )
    parser.add_argument(
        "--preset",
        type=float,
        default=None,
        choices=[0.20, 0.25, 0.30],
        help="Preset ngưỡng Box nhanh (0.20, 0.25, hoặc 0.30).",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=None,
        help="Ngưỡng Box tùy chỉnh ghi đè toàn bộ lớp.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=None,
        help="Ngưỡng Text tùy chỉnh (mặc định: 0.20).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Kích thước cạnh mảnh tile (mặc định: 768).",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=None,
        help="Tỉ lệ chồng lấn tile (mặc định: 0.25).",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=None,
        help="Ngưỡng IoU cho NMS nội bộ nhóm (mặc định: 0.50).",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Danh sách class ID cần chạy, phân cách bằng dấu phẩy (vd: '1,2,6').",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Chạy riêng một lớp duy nhất theo ID (1-10, vd: --class-id 6).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Chế độ cô lập một prompt để kiểm tra (vd: --prompt 'food vending cart').",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=None,
        help="Danh sách nhóm prompt cần chạy khi query_mode=grouped (vd: 'people_vehicles,goods').",
    )
    parser.add_argument("--disable-review-rule", action="append", default=[],
                        help="Tắt riêng quy tắc review; có thể truyền nhiều lần.")
    parser.add_argument(
        "--tile-small-only",
        action="store_true",
        help="Chỉ áp dụng chia mảnh tiling cho các nhóm vật thể nhỏ.",
    )
    parser.add_argument(
        "--enable-counter-checks",
        action="store_true",
        default=None,
        help="Bật đối chiếu prompt đối chứng (mặc định: theo config).",
    )
    parser.add_argument(
        "--disable-counter-checks",
        action="store_true",
        help="Tắt đối chiếu prompt đối chứng.",
    )
    parser.add_argument(
        "--enable-cart-merge",
        action="store_true",
        default=None,
        help="Bật cơ chế hợp nhất box xe đẩy trùng.",
    )
    parser.add_argument(
        "--disable-cart-merge",
        action="store_true",
        help="Tắt cơ chế hợp nhất box xe đẩy trùng.",
    )
    parser.add_argument(
        "--enable-leg-filter",
        action="store_true",
        default=None,
        help="Bật bộ lọc hợp nhất box chân người.",
    )
    parser.add_argument(
        "--disable-leg-filter",
        action="store_true",
        help="Tắt bộ lọc hợp nhất box chân người.",
    )
    parser.add_argument(
        "--enable-tree-filter",
        action="store_true",
        default=None,
        help="Bật bộ lọc hình học thân cây.",
    )
    parser.add_argument(
        "--disable-tree-filter",
        action="store_true",
        help="Tắt bộ lọc hình học thân cây.",
    )
    parser.add_argument(
        "--sidewalk-file",
        type=str,
        default=None,
        help="Đường dẫn GeoJSON chứa polygon vỉa hè, lưu riêng ngoài danh mục vật thể.",
    )
    parser.add_argument(
        "--label-mode",
        type=str,
        default="short",
        choices=["short", "full", "id_only", "none"],
        help="Chế độ nhãn khi vẽ: short ([ID] Tên Score), full, id_only, none.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Bật chế độ Debug: ghi chi tiết các giai đoạn loại bỏ ra file JSON debug.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cuda", "cpu"],
        help="Thiết bị thực thi (auto, cuda, cpu).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".hf_cache",
        help="Thư mục cache trọng số Hugging Face.",
    )
    parser.add_argument('--disable-image-feature-cache', action='store_true',
                        help='Tắt cache backbone ảnh để đối chiếu với pipeline gốc.')
    parser.add_argument('--threads', type=int, default=2,
                        help='Số luồng CPU PyTorch; mặc định 2.')
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Đường dẫn file cấu hình YAML.",
    )
    parser.add_argument(
        "--keep-review",
        action="store_true",
        default=False,
        help="Giữ lại các ứng viên review (mặc định đã tắt để loại bỏ các box ảo/box diện rộng).",
    )

    return parser.parse_args()


def process_single_image(
    image_path: Path,
    detector: GroundingDINODetector,
    output_dir: Path,
    args: argparse.Namespace,
    target_classes: Optional[List[int]],
    target_groups: Optional[List[str]],
    sidewalk_regions: Optional[List[Any]],
) -> Dict[str, Any]:
    """Xử lý nhận diện cho 1 ảnh và lưu các kết quả."""
    # Xác định các cờ bật/tắt bổ sung
    enable_counters = False if args.disable_counter_checks else (True if args.enable_counter_checks else None)
    enable_cart_merge = False if args.disable_cart_merge else (True if args.enable_cart_merge else None)
    enable_leg_filter = False if args.disable_leg_filter else (True if args.enable_leg_filter else None)
    enable_tree_filter = False if args.disable_tree_filter else (True if args.enable_tree_filter else None)

    # 1. Thực hiện suy luận
    result = detector.predict(
        image_input=image_path,
        mode=args.mode,
        query_mode=args.query_mode,
        box_threshold_override=args.box_threshold,
        text_threshold_override=args.text_threshold,
        preset=args.preset,
        tile_size=args.tile_size,
        overlap_ratio=args.overlap_ratio,
        nms_iou=args.nms_iou,
        target_classes=target_classes,
        target_groups=target_groups,
        single_prompt=args.prompt,
        enable_counter_checks=enable_counters,
        enable_cart_merge=enable_cart_merge,
        enable_leg_filter=enable_leg_filter,
        enable_tree_filter=enable_tree_filter,
        tile_small_objects_only=args.tile_small_only if args.tile_small_only else None,
        debug=args.debug,
        sidewalk_regions=sidewalk_regions,
    )

    image_pil = result.pop("_image_pil", None)
    predicted_objects = result.pop("_predicted_objects", None)
    review_objects = result.pop("_review_objects", None)
    debug_trace = result.pop("debug_trace", None)
    result.setdefault("provenance", {})["cli_arguments"] = vars(args).copy()
    result["provenance"]["argv"] = list(sys.argv)

    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = image_path.stem

    # Xử lý chế độ cô lập prompt
    if result.get("mode") == "single_prompt_inspection":
        slug = hashlib.md5(args.prompt.encode("utf-8")).hexdigest()[:8]
        out_json_path = output_dir / f"{base_name}_single_prompt_{slug}.json"
        out_img_path = output_dir / f"{base_name}_single_prompt_{slug}.jpg"
        if image_pil is not None:
            annotated = draw_detections(
                image=image_pil,
                detections=result.get("detections", []),
                label_mode=args.label_mode,
            )
            annotated.save(out_img_path, format="JPEG", quality=95)
            result["output_annotated_image"] = str(out_img_path.resolve())

        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        result["output_json"] = str(out_json_path.resolve())
        return result

    # 2. Lưu ảnh kết quả kiểm tra chính duy nhất (chỉ các đối tượng dự đoán được giữ lại)
    out_img_path = output_dir / f"{base_name}_annotated.jpg"
    out_json_path = output_dir / f"{base_name}_result.json"

    keep_rev = getattr(args, "keep_review", False)
    # Drawing preferences must not erase evidence or alter JSON counts.
    review_rows = review_objects if review_objects is not None else result.get("review_candidates", [])

    if image_pil is not None:
        if keep_rev and review_rows:
            annotated = draw_candidates_overlay(
                image=image_pil,
                confirmed_detections=predicted_objects or result.get("object_detections", []),
                review_candidates=review_rows,
                label_mode=args.label_mode,
                sidewalk_regions=result.get("sidewalk_regions", []),
            )
        else:
            annotated = draw_detections(
                image=image_pil,
                detections=predicted_objects or result.get("object_detections", []),
                label_mode=args.label_mode,
                sidewalk_regions=result.get("sidewalk_regions", []),
            )
        annotated.save(out_img_path, format="JPEG", quality=95)

    # Xóa các file ảnh review / candidate cũ nếu còn tồn tại trong thư mục
    for legacy_ext in ("_review.jpg", "_candidates.jpg"):
        old_f = output_dir / f"{base_name}{legacy_ext}"
        if old_f.exists():
            try:
                old_f.unlink()
            except Exception:
                pass

    result["image_path"] = str(image_path.resolve())
    result["output_annotated_image"] = str(out_img_path.resolve())

    # 4. Lưu file JSON kết quả chính
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 5. Lưu debug trace nếu bật debug
    if debug_trace is not None:
        out_debug_path = output_dir / f"{base_name}_debug.json"
        with open(out_debug_path, "w", encoding="utf-8") as f:
            json.dump(debug_trace, f, ensure_ascii=False, indent=2)
        result["output_debug_trace"] = str(out_debug_path.resolve())
    else:
        stale_debug = output_dir / f"{base_name}_debug.json"
        if stale_debug.exists():
            stale_debug.unlink()

    return result


def main() -> int:
    args = parse_args()
    if args.backend in ('yoloe','demo-hybrid') and args.prompt:
        print('YOLOE does not switch to DINO for --prompt. Configure target_phrases and regenerate the single checkpoint, or explicitly choose another backend.',file=sys.stderr)
        return 2
    if args.threads is not None:
        if args.threads < 1:
            print('[!] --threads phải lớn hơn 0.', file=sys.stderr)
            return 1
        import torch
        torch.set_num_threads(args.threads)
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"[!] Lỗi: Không tìm thấy đường dẫn đầu vào: {input_path}", file=sys.stderr)
        return 1

    # Phân giải target_classes
    target_classes = None
    if args.class_id is not None:
        target_classes = [args.class_id]
    elif args.classes is not None:
        try:
            target_classes = [int(c.strip()) for c in args.classes.split(",") if c.strip()]
        except ValueError:
            print(f"[!] Lỗi: Giá trị --classes không hợp lệ: {args.classes}", file=sys.stderr)
            return 1

    target_groups = None
    if args.groups:
        target_groups = [g.strip() for g in args.groups.split(",") if g.strip()]

    # Nạp dữ liệu vỉa hè nếu có
    sidewalk_regions = None
    if args.sidewalk_file:
        sidewalk_regions, sw_msg = load_sidewalk_from_file(args.sidewalk_file)
        print(f"[*] Vỉa hè: {sw_msg}")

    from lan_chiem.output_utils import generate_unique_output_dir
    source_name = input_path.stem if input_path.is_file() else input_path.name
    output_dir = generate_unique_output_dir(args.output, task_name='predict', source_name=source_name)

    try:
        if args.backend != 'yoloe':
            raise ValueError(f"Backend '{args.backend}' has been retired. The production pipeline exclusively uses YOLOE ('--backend yoloe').")
        from lan_chiem.yoloe_detector import YOLOEDetector
        detector = YOLOEDetector(args.weights, args.device, args.imgsz, args.config or Path(__file__).parent/'configs/object_yoloe.yaml', roi_focus=args.roi_focus)
    except Exception as e:
        print(f"[!] Không thể khởi tạo detector: {e}", file=sys.stderr)
        return 1

    if args.disable_image_feature_cache:
        detector.config.setdefault('inference', {})['enable_image_feature_cache'] = False

    for rule in args.disable_review_rule:
        if rule not in detector.config.get("review_rules", {}):
            print(f"[!] Quy tắc không tồn tại: {rule}", file=sys.stderr)
            return 1
        detector.config["review_rules"][rule] = False

    # Kiểm tra target_classes trước khi chạy
    if target_classes is not None:
        for cid in target_classes:
            meta = ClassRegistry.get(cid)
            if meta is None or meta.support_status == "unsupported":
                print(
                    f"[!] Lỗi: Lớp ID {cid} ({meta.display_name if meta else 'Không rõ'}) hiện chưa được hỗ trợ.",
                    file=sys.stderr,
                )
                return 1

    image_files: List[Path] = []
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_files.append(input_path)
    elif input_path.is_dir():
        for p in input_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                image_files.append(p)
        image_files.sort()

    if not image_files:
        print(f"[!] Không tìm thấy file ảnh hợp lệ nào tại: {input_path}", file=sys.stderr)
        return 1

    print(f"[*] Số lượng ảnh: {len(image_files)}")
    print(f"[*] Danh mục: {CATALOG_VERSION} (ID 1–10)")
    print(f"[*] Thiết bị: {detector.device_info}")
    if args.prompt:
        print(f"[*] CHẾ ĐỘ CÔ LẬP PROMPT: '{args.prompt}'")
    elif target_classes is not None:
        print(f"[*] CHẾ ĐỘ CÔ LẬP LỚP: {target_classes}")
    else:
        q_mode_str = ('yoloe_vocabulary' if args.backend in ('yoloe','demo-hybrid') else
                      args.query_mode or detector.config.get("inference", {}).get("query_mode", "per_class"))
        print(f"[*] Chế độ truy vấn: {q_mode_str}")
        print(f"[*] Chế độ suy luận: {args.mode or detector.config.get('inference', {}).get('mode', 'full')}")
        preset_val = args.box_threshold if args.box_threshold is not None else (
            args.preset if args.preset is not None else 'theo cấu hình từng lớp')
        print(f"[*] Ngưỡng Box mặc định: {preset_val}")

    print(f"[*] Chế độ Debug: {'BẬT' if args.debug else 'TẮT'}")
    print(f"[*] Thư mục lưu: {output_dir.resolve()}\n")

    success_count = 0
    fail_count = 0
    total_predicted_counts: Dict[str, int] = {}
    total_review_counts: Dict[str, int] = {}
    total_conflicts = 0
    total_raw_detections = 0
    start_time = time.perf_counter()

    for idx, img_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Đang xử lý: {img_path.name} ...", end=" ", flush=True)
        try:
            res = process_single_image(
                image_path=img_path,
                detector=detector,
                output_dir=output_dir,
                args=args,
                target_classes=target_classes,
                target_groups=target_groups,
                sidewalk_regions=sidewalk_regions,
            )
            success_count += 1
            t_sec = res.get("inference_time_seconds", 0.0)

            if res.get("mode") == "single_prompt_inspection":
                n_det = res.get("total_detections", 0)
                total_raw_detections += n_det
                print(f"HOÀN THÀNH ({t_sec:.2f}s, {n_det} box với prompt '{args.prompt}')")
            else:
                summary = res.get("summary", {})
                n_pred = summary.get("total_predicted_detections", 0)
                n_rev = summary.get("total_review_candidates", 0)
                n_cfl = summary.get("total_conflicts_detected", 0)
                total_conflicts += n_cfl

                keep_rev_flag = getattr(args, "keep_review", False)
                cfl_info = f", {n_cfl} xung đột" if (n_cfl > 0 and keep_rev_flag) else ""
                rev_info = f", {n_rev} ứng viên review" if (n_rev > 0 and keep_rev_flag) else ""
                print(f"HOÀN THÀNH ({t_sec:.2f}s, {n_pred} box dự đoán{rev_info}{cfl_info})")

                for c_key, count in summary.get("predicted_class_counts", {}).items():
                    total_predicted_counts[c_key] = total_predicted_counts.get(c_key, 0) + count
                for c_key, count in summary.get("review_class_counts", {}).items():
                    total_review_counts[c_key] = total_review_counts.get(c_key, 0) + count

        except Exception as e:
            fail_count += 1
            print(f"THẤT BẠI: {e}", file=sys.stderr)

    total_time = time.perf_counter() - start_time

    print("\n" + "=" * 60)
    print("                 TỔNG KẾT XỬ LÝ CLI 2.2")
    print("=" * 60)
    print(f"Tổng số ảnh xử lý  : {len(image_files)}")
    print(f"Thành công         : {success_count}")
    print(f"Thất bại           : {fail_count}")
    print(f"Tổng thời gian     : {total_time:.2f} giây")
    print("-" * 60)
    if args.prompt:
        print(f"Raw predictions (chưa xác minh): {total_raw_detections}")
        print(f"Kết quả lưu tại: {output_dir.resolve()}\n")
        return 0 if fail_count == 0 else 2
    print("Đối tượng dự đoán (Predicted Detections) theo lớp:")
    if not total_predicted_counts:
        print("  (Không có đối tượng dự đoán nào đạt ngưỡng)")
    else:
        for c_key, cnt in sorted(total_predicted_counts.items()):
            print(f"  + [{c_key}]: {cnt}")
    if getattr(args, "keep_review", False) and total_review_counts:
        print("-" * 60)
        print(f"Ứng viên cần kiểm tra lại (Review Candidates): {sum(total_review_counts.values())}")
        for c_key, cnt in sorted(total_review_counts.items()):
            print(f"  ? [{c_key}]: {cnt}")
        if total_conflicts > 0:
            print(f"Xung đột phân lớp phát hiện: {total_conflicts} trường hợp (chi tiết trong file JSON)")
    print("=" * 60)
    print(f"Kết quả lưu tại: {output_dir.resolve()}\n")

    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
