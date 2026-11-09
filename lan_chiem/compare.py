"""
Module compare: Công cụ thực nghiệm so sánh các chế độ suy luận và ngưỡng trên cùng tập ảnh.
Hỗ trợ:
- So sánh 'full' vs 'full+tiled'.
- So sánh các preset ngưỡng: 0.20, 0.25, 0.30.
- Xuất báo cáo Markdown và dữ liệu JSON tổng hợp.
- Cảnh báo rõ ràng: Không coi số lượng box hay điểm số cao hơn là bằng chứng chính xác hơn.
"""

from __future__ import annotations

import argparse
import json
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

from lan_chiem.detector import GroundingDINODetector

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="So sánh hiệu năng phát hiện giữa các chế độ và ngưỡng trên tập ảnh."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Đường dẫn thư mục ảnh cần so sánh.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="lan_chiem/outputs/comparisons",
        help="Thư mục lưu báo cáo so sánh (mặc định: lan_chiem/outputs/comparisons).",
    )
    parser.add_argument(
        "--compare-type",
        type=str,
        default="mode",
        choices=["mode", "thresholds", "all"],
        help="Loại so sánh: 'mode' (full vs full+tiled), 'thresholds' (0.20 vs 0.25 vs 0.30), hoặc 'all'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Thiết bị thực thi.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".hf_cache",
        help="Thư mục cache Hugging Face.",
    )

    return parser.parse_args()


def run_single_experiment(
    detector: GroundingDINODetector,
    images: List[Path],
    exp_name: str,
    mode: str,
    box_threshold: float,
) -> Dict[str, Any]:
    """Chạy thực nghiệm trên toàn bộ danh sách ảnh với 1 cấu hình cụ thể."""
    print(f"[*] Đang chạy cấu hình: {exp_name} (Mode: {mode}, Box Thresh: {box_threshold:.2f}) ...")
    total_time = 0.0
    total_detections = 0
    total_unique_objects = 0
    class_totals: Dict[str, int] = {}
    per_image_results = []

    for img_path in images:
        t0 = time.perf_counter()
        res = detector.predict(
            image_input=img_path,
            mode=mode,
            box_threshold_override=box_threshold,
        )
        t_elapsed = time.perf_counter() - t0
        total_time += t_elapsed

        summary = res.get("summary", {})
        n_conf = summary.get("total_confirmed_detections", summary.get("total_detections", 0))
        n_rev = summary.get("total_review_candidates", 0)
        n_cfl = summary.get("total_conflicts_detected", 0)
        total_detections += n_conf
        total_unique_objects += (n_conf + n_rev)

        conf_counts = summary.get("confirmed_class_counts", summary.get("class_counts", {}))
        for c_key, count in conf_counts.items():
            class_totals[c_key] = class_totals.get(c_key, 0) + count

        per_image_results.append({
            "image": img_path.name,
            "confirmed_detections": n_conf,
            "review_candidates": n_rev,
            "conflicts_detected": n_cfl,
            "time_seconds": round(t_elapsed, 2),
            "class_counts": conf_counts,
        })

    return {
        "experiment_name": exp_name,
        "mode": mode,
        "box_threshold": box_threshold,
        "total_images": len(images),
        "total_time_seconds": round(total_time, 2),
        "avg_time_per_image": round(total_time / max(1, len(images)), 2),
        "total_confirmed_detections": total_detections,
        "total_detections": total_detections,
        "total_hypotheses": total_unique_objects,
        "counting_note": "Predicted + review boxes; may contain duplicates and conflicting hypotheses.",
        "provenance": detector.provenance(),
        "class_totals": class_totals,
        "per_image": per_image_results,
    }


def generate_markdown_report(experiments: List[Dict[str, Any]], output_path: Path) -> None:
    """Tạo báo cáo so sánh định dạng Markdown trực quan."""
    md_lines = [
        "# Báo Cáo So Sánh Hiệu Năng Nhận Diện UAV (lan_chiem)",
        "",
        "> **Lưu ý phương pháp luận quan trọng**:",
        "> Không coi số lượng bounding box hoặc điểm tin cậy trung bình cao hơn là bằng chứng của độ chính xác cao hơn.",
        "> Việc hạ ngưỡng hoặc chia mảnh (tiling) làm tăng độ nhạy phát hiện vật thể nhỏ, nhưng có thể kéo theo báo động giả (false positives).",
        "> Kết quả cần được đối chiếu trực quan với đặc điểm thực tế của hiện trường.",
        "",
        "## 1. Bảng Tổng Hợp Các Cấu Hình Thử Nghiệm",
        "",
        "| Tên thực nghiệm | Chế độ (Mode) | Ngưỡng Box | Box predicted | Box predicted + review | Tổng thời gian (s) | TB mỗi ảnh (s) |",
        "|---|---|---|---|---|---|---|",
    ]

    for exp in experiments:
        n_dets = exp.get("total_confirmed_detections", exp.get("total_detections", 0))
        md_lines.append(
            f"| **{exp['experiment_name']}** | `{exp['mode']}` | `{exp['box_threshold']:.2f}` | "
            f"{n_dets} | {exp.get('total_hypotheses', exp.get('estimated_unique_objects', 0))} | "
            f"{exp['total_time_seconds']}s | {exp['avg_time_per_image']}s |"
        )

    md_lines.extend([
        "",
        "## 2. Phân Bổ Đối Tượng Phát Hiện Theo Từng Lớp",
        "",
    ])

    # Thu thập tất cả các class xuất hiện
    all_classes = sorted(list({c for exp in experiments for c in exp["class_totals"].keys()}))
    if all_classes:
        header = "| Cấu hình | " + " | ".join(all_classes) + " |"
        sep = "|---|" + "|".join(["---"] * len(all_classes)) + "|"
        md_lines.append(header)
        md_lines.append(sep)

        for exp in experiments:
            row_vals = [str(exp["class_totals"].get(c, 0)) for c in all_classes]
            md_lines.append(f"| **{exp['experiment_name']}** | " + " | ".join(row_vals) + " |")
    else:
        md_lines.append("*(Không phát hiện đối tượng nào thỏa mãn ngưỡng)*")

    md_lines.extend([
        "",
        "## 3. Nhận Xét & Khuyến Nghị",
        "",
        "Chưa có annotation được xác nhận: số box và thời gian không chứng minh độ chính xác.",
        "Tiling và ngưỡng thấp có thể thêm cả vật thể đúng, box trùng và nhận nhầm.",
        "Không tự khuyến nghị một preset cho kiểm kê tự động từ các số đếm này.",
        "Dùng lan_chiem.evaluate để lưu đầy đủ JSON/ảnh, nguồn ngưỡng và so sánh không override với preset 0.25.",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[!] Không tìm thấy thư mục: {input_path}", file=sys.stderr)
        return 1

    image_files: List[Path] = []
    if input_path.is_file():
        image_files = [input_path]
    else:
        for p in input_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                image_files.append(p)
        image_files.sort()

    if not image_files:
        print(f"[!] Không tìm thấy file ảnh nào tại: {input_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = GroundingDINODetector(device=args.device, cache_dir=args.cache_dir)
    experiments_plan = []

    if args.compare_type in ("mode", "all"):
        experiments_plan.append(("full_preset_0.25", "full", 0.25))
        experiments_plan.append(("full_plus_tiled_preset_0.25", "full+tiled", 0.25))

    if args.compare_type in ("thresholds", "all"):
        experiments_plan.append(("full_thresh_0.20", "full", 0.20))
        experiments_plan.append(("full_thresh_0.25", "full", 0.25))
        experiments_plan.append(("full_thresh_0.30", "full", 0.30))

    # Loại bỏ cấu hình trùng lặp
    unique_plans = []
    seen = set()
    for name, m, b in experiments_plan:
        if (m, b) not in seen:
            seen.add((m, b))
            unique_plans.append((name, m, b))

    results = []
    for name, m, b in unique_plans:
        exp_res = run_single_experiment(detector, image_files, name, m, b)
        exp_res["provenance"]["cli_arguments"] = vars(args).copy()
        exp_res["provenance"]["argv"] = list(sys.argv)
        results.append(exp_res)

    report_md_path = output_dir / "comparison_report.md"
    report_json_path = output_dir / "comparison_summary.json"

    generate_markdown_report(results, report_md_path)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("                 HOÀN THÀNH SO SÁNH")
    print("=" * 60)
    print(f"Báo cáo Markdown : {report_md_path.resolve()}")
    print(f"Dữ liệu JSON     : {report_json_path.resolve()}")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
