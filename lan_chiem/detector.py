"""
Module detector: Lõi suy luận nhận diện đối tượng Grounding DINO 2.2 (UAV Street Inspector).
Cải tiến phiên bản 2.2:
- Chế độ suy luận theo từng lớp (Per-Class Query Mode): Truy vấn độc lập từng lớp đối tượng,
  loại bỏ hoàn toàn lỗi gán nhầm alias (như fruit, basket crate ở ID 10).
- Chế độ nhóm (Grouped Mode): Thu thập đầy đủ các lớp khớp, đánh dấu ambiguous nếu phrase chứa
  dấu hiệu của nhiều lớp (như 'vendor bicycle vendor pushcart') thay vì tùy tiện chọn một lớp.
- Thống nhất ngưỡng và Debug: Debug chỉ ghi thêm dữ liệu, kết quả chính hoàn toàn bất biến khi bật/tắt debug.
- Thứ tự ưu tiên ngưỡng: CLI > Cấu hình lớp > Cấu hình nhóm > Mặc định chung.
- Trạng thái kết quả chuẩn: "predicted" (dự đoán giữ lại), "review" (cần kiểm tra/xung đột), "rejected" (bị loại).
  Tuyệt đối không dùng nhãn "confirmed".
- Hợp nhất có kiểm soát cho xe đẩy (ID 6): Kết hợp IoU, IoS và khoảng cách tâm để gộp 2 box của cùng 1 xe,
  tránh gộp 2 xe đứng cạnh nhau và không bao cả ghế xung quanh.
- Bộ lọc hình học chân người và thân cây có cờ bật/tắt độc lập trong config và CLI.
- Prompt đối chứng đọc động từ config, so khớp alias có ranh giới từ (không dùng substring tùy tiện).
- Cô lập lớp chuẩn xác: `--class-id 6` chỉ gửi prompt của lớp 6 trước inference; yêu cầu lớp không hỗ trợ sẽ báo lỗi rõ.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageOps
import yaml

from lan_chiem.registry import ClassMeta, ClassRegistry
from lan_chiem.object_classes import CATALOG_VERSION, ClassId

try:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    import transformers
    _TRANSFORMERS_VERSION = transformers.__version__
except ImportError as e:
    raise ImportError(
        "Chưa cài đặt thư viện 'transformers'. Vui lòng chạy: pip install transformers>=4.40.0"
    ) from e


@dataclass
class Detection:
    """Đại diện cho một bounding box đối tượng được nhận diện."""
    id: str
    class_id: Optional[int]
    class_name: str
    class_name_vi: str
    raw_label: str
    prompt_group: str
    source: str  # "full" | "tile_x_y" | "fused"
    score: float
    bbox_xyxy: List[float]  # [x_min, y_min, x_max, y_max]
    subtype: Optional[str] = None  # Cho ID 7 (biển quảng cáo) hoặc ID 10 (hàng hóa)
    status: str = "predicted"  # "predicted" | "review" (không tự gán "confirmed")
    review_reason: Optional[str] = None
    conflict_info: Optional[Dict[str, Any]] = None
    related_detection_ids: List[str] = None
    queried_class_id: Optional[int] = None
    query_prompt: Optional[str] = None
    candidate_class_ids: List[int] = field(default_factory=list)
    status_history: List[Dict[str, Any]] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.related_detection_ids is None:
            self.related_detection_ids = []
        if not self.status_history:
            self.status_history.append({"from": None, "to": self.status,
                                        "reason": self.review_reason or "raw_model_hypothesis"})

    def require_review(self, reason: str) -> None:
        """Record every review decision; never implicitly promote a hypothesis."""
        self.status_history.append({"from": self.status, "to": "review", "reason": reason})
        self.status = "review"
        self.review_reason = "; ".join(filter(None, [self.review_reason, reason]))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "class_name_vi": self.class_name_vi,
            "raw_label": self.raw_label,
            "prompt_group": self.prompt_group,
            "source": self.source,
            "score": round(float(self.score), 4),
            "bbox_xyxy": [round(float(c), 2) for c in self.bbox_xyxy],
            "subtype": self.subtype,
            "status": self.status,
            "review_reason": self.review_reason,
            "conflict_info": self.conflict_info,
            "related_detection_ids": self.related_detection_ids,
            "queried_class_id": self.queried_class_id,
            "query_prompt": self.query_prompt,
            "candidate_class_ids": self.candidate_class_ids,
            "status_history": self.status_history,
            "evidence": self.evidence,
        }


@dataclass
class RejectionRecord:
    """Ghi vết một kết quả bị loại bỏ ở từng giai đoạn suy luận (phục vụ debug trace)."""
    stage: str  # "threshold" | "mapping" | "nms" | "geometry" | "conflict"
    bbox_xyxy: List[float]
    score: float
    raw_label: str
    prompt_group: str
    source: str
    reason: str  # "below_threshold" | "invalid_box" | "unmapped_label" | "ambiguous_label" | "suppressed_duplicate" | "suppressed_duplicate_cart_heuristic" | "subpart_legs" | "canopy_box"
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "bbox_xyxy": [round(float(c), 2) for c in self.bbox_xyxy],
            "score": round(float(self.score), 4),
            "raw_label": self.raw_label,
            "prompt_group": self.prompt_group,
            "source": self.source,
            "reason": self.reason,
            "details": self.details,
        }


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Tính Intersection over Union (IoU) giữa 2 box [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def calculate_ios(box_small: List[float], box_large: List[float]) -> float:
    """Tính Intersection over Smaller Box (IoS) - độ chứa của box nhỏ trong box lớn."""
    x1 = max(box_small[0], box_large[0])
    y1 = max(box_small[1], box_large[1])
    x2 = min(box_small[2], box_large[2])
    y2 = min(box_small[3], box_large[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_small = max(0.0, box_small[2] - box_small[0]) * max(0.0, box_small[3] - box_small[1])
    if area_small <= 0.0:
        return 0.0
    return inter_area / area_small


def calculate_center_distance(box1: List[float], box2: List[float]) -> float:
    """Tính khoảng cách Euclidean giữa tâm của 2 box."""
    c1_x = (box1[0] + box1[2]) / 2.0
    c1_y = (box1[1] + box1[3]) / 2.0
    c2_x = (box2[0] + box2[2]) / 2.0
    c2_y = (box2[1] + box2[3]) / 2.0
    return float(np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2))


def crop_box_to_global(box, offset_xy, crop_size, full_size):
    """Clip at the crop boundary before translating exactly once to image pixels."""
    ox, oy = offset_xy
    cw, ch = crop_size
    fw, fh = full_size
    return [max(0., min(float(v), float(limit))) for v, limit in zip(
        [max(0., min(float(box[0]), cw)) + ox,
         max(0., min(float(box[1]), ch)) + oy,
         max(0., min(float(box[2]), cw)) + ox,
         max(0., min(float(box[3]), ch)) + oy], [fw, fh, fw, fh])]


def determine_sign_subtype(raw_label: str) -> str:
    """Xác định subtype cho ID 7 (biển quảng cáo) dựa trên raw label."""
    rl = raw_label.lower().strip()
    if "traffic" in rl:
        return "traffic_sign"
    elif "advertising" in rl or "billboard" in rl or "banner" in rl:
        return "advertising_board"
    elif "shop" in rl:
        return "shop_sign"
    elif "sandwich" in rl or "standing" in rl:
        return "standing_sign"
    return "unknown"


def determine_goods_subtype(raw_label: str) -> str:
    """Xác định subtype cho ID 10 (hàng hóa) dựa trên raw label."""
    rl = raw_label.lower().strip()
    if "fruit" in rl:
        return "fruit"
    elif "vegetable" in rl:
        return "vegetable"
    elif "crate" in rl:
        return "crate"
    elif "basket" in rl:
        return "basket"
    return "goods_general"


def consolidate_person_boxes(
    person_dets: List[Detection],
    rejections: Optional[List[RejectionRecord]] = None,
) -> List[Detection]:
    """
    Hợp nhất và loại bỏ box người chỉ bao phần chân (subpart legs).
    Nếu box B nằm ở nửa dưới của box A, có cùng trục X và A có diện tích lớn hơn,
    thì loại B để tránh 1 người có nhiều box.
    """
    if len(person_dets) <= 1:
        return person_dets

    person_dets.sort(key=lambda d: (d.bbox_xyxy[2] - d.bbox_xyxy[0]) * (d.bbox_xyxy[3] - d.bbox_xyxy[1]), reverse=True)
    kept: List[Detection] = []

    for cand in person_dets:
        cb = cand.bbox_xyxy
        c_w = cb[2] - cb[0]
        c_h = cb[3] - cb[1]
        is_subpart = False
        winner = None

        for chosen in kept:
            chb = chosen.bbox_xyxy
            ch_w = chb[2] - chb[0]
            ch_h = chb[3] - chb[1]

            x_inter = max(0.0, min(cb[2], chb[2]) - max(cb[0], chb[0]))
            x_overlap_ratio = x_inter / max(1.0, min(c_w, ch_w))

            if cb[1] >= chb[1] + 0.35 * ch_h and abs(cb[3] - chb[3]) <= 0.25 * ch_h and x_overlap_ratio >= 0.60:
                is_subpart = True
                winner = chosen
                break

            ios = calculate_ios(cb, chb)
            if ios >= 0.85 and c_h < 0.70 * ch_h:
                is_subpart = True
                winner = chosen
                break

        if not is_subpart:
            kept.append(cand)
        else:
            if rejections is not None and winner is not None:
                rejections.append(
                    RejectionRecord(
                        stage="geometry",
                        bbox_xyxy=cand.bbox_xyxy,
                        score=cand.score,
                        raw_label=cand.raw_label,
                        prompt_group=cand.prompt_group,
                        source=cand.source,
                        reason="subpart_legs",
                        details=f"Box chỉ bao phần chân người, đã được hợp nhất vào người {winner.id}",
                    )
                )

    return kept


def filter_tree_trunk_geometry(
    trunk_dets: List[Detection],
    img_width: int,
    img_height: int,
    rejections: Optional[List[RejectionRecord]] = None,
) -> List[Detection]:
    """
    Đánh dấu review các phạm vi cây bất thường, giữ nguyên box:
    - Box bao trùm hơn 45% diện tích toàn ảnh (vòm tán cây hoặc khung toàn cảnh).
    - Box nằm ngang (W > 1.8 * H) chiếm diện tích lớn.
    """
    kept: List[Detection] = []
    img_area = img_width * img_height

    for d in trunk_dets:
        b = d.bbox_xyxy
        w = max(1.0, b[2] - b[0])
        h = max(1.0, b[3] - b[1])
        box_area = w * h

        if box_area > 0.45 * img_area or (w > 1.8 * h and box_area > 0.05 * img_area):
            d.evidence["geometry"] = {"width": w, "height": h, "scope_verified": False}
            d.require_review("tree_geometry_scope_unverified")

        kept.append(d)

    return kept


def merge_duplicate_carts(
    cart_dets: List[Detection],
    iou_thresh: float = 0.40,
    ios_thresh: float = 0.65,
    max_center_dist_ratio: float = 0.40,
    rejections: Optional[List[RejectionRecord]] = None,
) -> List[Detection]:
    """
    Cơ chế hợp nhất có kiểm soát cho box xe đẩy bán hàng (ID 6) trùng lặp:
    - Nếu 2 box có IoU >= 0.40 (hoặc IoU >= 0.50), IoS >= 0.65 và tâm gần nhau (khoảng cách tâm / đường chéo box nhỏ <= 0.40)
      -> coi là cùng 1 xe đẩy (ví dụ 1 box bao khung ngoài, 1 box bao thân xe).
    - Giữ box có điểm cao hơn, loại box trùng.
    - Không gộp 2 xe đứng cạnh nhau (khoảng cách tâm xa, IoS nhỏ).
    """
    if len(cart_dets) <= 1:
        return cart_dets

    cart_dets.sort(key=lambda x: x.score, reverse=True)
    kept: List[Detection] = []

    for cand in cart_dets:
        cb = cand.bbox_xyxy
        c_w = cb[2] - cb[0]
        c_h = cb[3] - cb[1]
        c_diag = max(1.0, float(np.sqrt(c_w ** 2 + c_h ** 2)))

        is_dup = False
        winner = None

        for chosen in kept:
            chb = chosen.bbox_xyxy
            ch_w = chb[2] - chb[0]
            ch_h = chb[3] - chb[1]
            ch_diag = max(1.0, float(np.sqrt(ch_w ** 2 + ch_h ** 2)))
            small_diag = min(c_diag, ch_diag)

            iou = calculate_iou(cb, chb)
            # Tính IoS trên box nhỏ hơn
            area_c = c_w * c_h
            area_ch = ch_w * ch_h
            if area_c <= area_ch:
                ios = calculate_ios(cb, chb)
            else:
                ios = calculate_ios(chb, cb)

            dist = calculate_center_distance(cb, chb)
            center_ratio = dist / small_diag

            # Điều kiện trùng lặp xe đẩy của cùng một vật thể
            if (iou >= 0.50) or (iou >= iou_thresh and ios >= ios_thresh and center_ratio <= max_center_dist_ratio):
                is_dup = True
                winner = chosen
                break

        if not is_dup:
            kept.append(cand)
        else:
            if rejections is not None and winner is not None:
                rejections.append(
                    RejectionRecord(
                        stage="nms",
                        bbox_xyxy=cand.bbox_xyxy,
                        score=cand.score,
                        raw_label=cand.raw_label,
                        prompt_group=cand.prompt_group,
                        source=cand.source,
                        reason="suppressed_duplicate_cart_heuristic",
                        details=(
                            f"Hợp nhất xe đẩy trùng với {winner.id} "
                            f"(IoU={calculate_iou(cand.bbox_xyxy, winner.bbox_xyxy):.2f}, "
                            f"IoS={ios:.2f}, tỷ lệ tâm={center_ratio:.2f})"
                        ),
                    )
                )

    return kept


def apply_intra_class_nms(
    detections: List[Detection],
    iou_threshold: float = 0.50,
    rejections: Optional[List[RejectionRecord]] = None,
) -> List[Detection]:
    """
    Áp dụng NMS nội bộ nhóm (Intra-class NMS):
    Chỉ triệt tiêu các box thuộc CÙNG một class_id nếu IoU > iou_threshold.
    Không bao giờ triệt tiêu box giữa các lớp khác nhau.
    """
    if len(detections) <= 1:
        return detections

    by_class: Dict[int, List[Detection]] = {}
    for d in detections:
        by_class.setdefault(d.class_id, []).append(d)

    kept: List[Detection] = []
    for cid, c_dets in by_class.items():
        if len(c_dets) <= 1:
            kept.extend(c_dets)
            continue

        c_dets.sort(key=lambda x: x.score, reverse=True)
        selected: List[Detection] = []
        for candidate in c_dets:
            suppress = False
            winner = None
            for chosen in selected:
                iou = calculate_iou(candidate.bbox_xyxy, chosen.bbox_xyxy)
                if iou > iou_threshold:
                    suppress = True
                    winner = chosen
                    break
            if not suppress:
                selected.append(candidate)
            else:
                if rejections is not None and winner is not None:
                    rejections.append(
                        RejectionRecord(
                            stage="nms",
                            bbox_xyxy=candidate.bbox_xyxy,
                            score=candidate.score,
                            raw_label=candidate.raw_label,
                            prompt_group=candidate.prompt_group,
                            source=candidate.source,
                            reason="suppressed_duplicate",
                            details=f"Bị loại bởi {winner.id} với IoU={calculate_iou(candidate.bbox_xyxy, winner.bbox_xyxy):.2f} > {iou_threshold}",
                        )
                    )
        kept.extend(selected)

    return kept


def generate_tiles(
    image_width: int,
    image_height: int,
    tile_size: int = 768,
    tile_overlap: float = 0.25,
) -> List[Tuple[int, int, int, int]]:
    """Tạo danh sách tile phủ toàn bộ ảnh."""
    if image_width <= tile_size and image_height <= tile_size:
        return [(0, 0, image_width, image_height)]

    step = max(64, int(tile_size * (1.0 - tile_overlap)))

    x_starts = list(range(0, max(1, image_width - tile_size + 1), step))
    if not x_starts or x_starts[-1] + tile_size < image_width:
        x_starts.append(max(0, image_width - tile_size))

    y_starts = list(range(0, max(1, image_height - tile_size + 1), step))
    if not y_starts or y_starts[-1] + tile_size < image_height:
        y_starts.append(max(0, image_height - tile_size))

    x_starts = sorted(list(set(x_starts)))
    y_starts = sorted(list(set(y_starts)))

    tiles: List[Tuple[int, int, int, int]] = []
    for ys in y_starts:
        for xs in x_starts:
            xe = min(image_width, xs + tile_size)
            ye = min(image_height, ys + tile_size)
            tiles.append((xs, ys, xe, ye))

    return tiles


class GroundingDINODetector:
    """
    Bộ nhận diện đối tượng Grounding DINO 2.2 (Pure CLI & Per-Class Inference).
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "config.yaml"
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        if self.config.get("catalog_version") != CATALOG_VERSION:
            raise ValueError("Config phải dùng catalog_version objects-v3-10; không dùng mapping ID 0–12 cũ.")
        if set(self.config.get("classes", {})) != set(ClassRegistry.all_ids()):
            raise ValueError("Config phải khai báo đủ 10 lớp ID 1–10.")

        cfg_model_id = self.config.get("model", {}).get("grounding_dino_model_id") or self.config.get("model", {}).get("model_id")
        if not cfg_model_id or str(cfg_model_id).endswith('.pt') or 'yoloe' in str(cfg_model_id).lower():
            cfg_model_id = "IDEA-Research/grounding-dino-tiny"
        self.model_id = model_id or cfg_model_id
        req_dev = device or self.config.get("model", {}).get("device", "auto")
        self.device = self._resolve_device(req_dev)

        cfg_cache = cache_dir or self.config.get("model", {}).get("cache_dir", ".hf_cache")
        self.cache_dir = self._resolve_cache_dir(cfg_cache)

        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[AutoModelForZeroShotObjectDetection] = None

        self.classes: Dict[int, Dict[str, Any]] = self.config.get("classes", {})
        self.prompt_groups: Dict[str, Any] = self.config.get("prompt_groups", {})
        self.counter_prompts: Dict[str, Any] = self.config.get("counter_prompts", {})
        self._main_counter_evidence = {}
        self.cart_merge_cfg: Dict[str, Any] = self.config.get("cart_merge", {})

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file cấu hình tại: {path}")
        from lan_chiem.configuration import load_config
        return load_config(path)

    @property
    def device_info(self) -> str:
        """Chuỗi mô tả thiết bị thực thi hiện tại (kèm tên GPU nếu có)."""
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                return f"GPU / CUDA ({gpu_name})"
            except Exception:
                return "GPU / CUDA"
        return "CPU"

    @staticmethod
    def _resolve_device(requested: Optional[str] = None) -> str:
        req = (requested or "auto").lower().strip()
        if req in ("cuda", "gpu"):
            if torch.cuda.is_available():
                return "cuda"
            print("Cảnh báo: Yêu cầu GPU (CUDA) nhưng không tìm thấy GPU hoặc PyTorch chưa hỗ trợ CUDA. Tự động chuyển sang CPU.")
            return "cpu"
        elif req == "cpu":
            return "cpu"
        # Mặc định auto: Tự động dùng GPU nếu khả dụng, ngược lại dùng CPU
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _heal_snapshot_pointers(cache_dir: Path) -> None:
        """
        Tự động xử lý các file con trỏ symlink git/zip trên Windows.
        Nếu gặp file text nhỏ trỏ đến blob (../../blobs/...), tự động thay thế bằng hardlink hoặc file copy.
        """
        try:
            for s_dir in cache_dir.glob("models--*/**/snapshots/*"):
                if not s_dir.is_dir():
                    continue
                for item in s_dir.iterdir():
                    if item.is_file() and item.stat().st_size < 300:
                        try:
                            content = item.read_text(encoding="utf-8").strip()
                            if content.startswith("../") and "/blobs/" in content:
                                target = (item.parent / content).resolve()
                                if target.is_file():
                                    item.unlink()
                                    try:
                                        os.link(target, item)
                                    except Exception:
                                        import shutil
                                        shutil.copyfile(target, item)
                        except Exception:
                            pass
        except Exception:
            pass

    @staticmethod
    def _resolve_cache_dir(cache_path: Union[str, Path]) -> str:
        cp = Path(cache_path)
        if not cp.is_absolute():
            workspace_root = Path(__file__).resolve().parent.parent
            resolved = (workspace_root / cp).resolve()
        else:
            resolved = cp.resolve()
        os.makedirs(resolved, exist_ok=True)
        GroundingDINODetector._heal_snapshot_pointers(resolved)
        return str(resolved)

    def load_model(self) -> None:
        """Nạp model và processor nếu chưa có trong bộ nhớ."""
        if self.model is not None and self.processor is not None:
            return

        print(f"[*] Đang nạp model '{self.model_id}' trên thiết bị: {self.device_info}...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
            )
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"[+] Nạp thành công model trên thiết bị: {self.device_info}!")
        except Exception as e:
            raise RuntimeError(f"Lỗi nạp model Grounding DINO: {e}") from e

    @staticmethod
    def preprocess_image(image_input: Union[str, Path, Image.Image, np.ndarray]) -> Tuple[Image.Image, str, str]:
        """Tiền xử lý ảnh, tính sha256 và lấy tên ảnh."""
        image_name = "in_memory_image"
        img_hash = ""
        if isinstance(image_input, (str, Path)):
            ip = Path(image_input)
            if not ip.is_file():
                raise FileNotFoundError(f"Không tìm thấy ảnh tại: {ip}")
            image_name = ip.name
            with open(ip, "rb") as f:
                img_hash = hashlib.sha256(f.read()).hexdigest()
            img = Image.open(ip)
            img.load()
        elif isinstance(image_input, Image.Image):
            img = image_input.copy()
            image_name = getattr(image_input, "filename", "in_memory_image") or "in_memory_image"
            if isinstance(image_name, (str, Path)):
                image_name = Path(image_name).name
            img_hash = hashlib.sha256(img.tobytes()).hexdigest()
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input)
            image_name = "numpy_array"
            img_hash = hashlib.sha256(image_input.tobytes()).hexdigest()
        else:
            raise TypeError(f"Kiểu ảnh không hợp lệ: {type(image_input)}")

        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.width <= 0 or img.height <= 0:
            raise ValueError("Kích thước ảnh không hợp lệ.")
        return img, str(image_name), img_hash

    def resolve_actual_threshold(
        self,
        cli_box_threshold: Optional[float],
        cli_preset: Optional[float],
        class_id: Optional[int],
        group_key: Optional[str],
    ) -> float:
        """
        Xác định ngưỡng box thực tế theo thứ tự ưu tiên nghiêm ngặt:
        1. CLI box_threshold truyền rõ ràng (is not None)
        2. CLI preset truyền rõ ràng (is not None)
        3. Cấu hình lớp (classes[class_id].box_threshold)
        4. Cấu hình nhóm (prompt_groups[group_key].box_threshold)
        5. Mặc định chung (box_threshold_default)
        """
        if cli_box_threshold is not None:
            return float(cli_box_threshold)

        c_thresh = None
        if class_id is not None and class_id in self.classes:
            c_thresh = self.classes[class_id].get("box_threshold")
        elif group_key is not None and group_key in self.prompt_groups:
            c_thresh = self.prompt_groups[group_key].get("box_threshold")

        if cli_preset is not None:
            return float(cli_preset)

        if c_thresh is not None:
            return float(c_thresh)

        return float(self.config.get("inference", {}).get("box_threshold_default", 0.25))

    def threshold_source(self, cli_box_threshold, cli_preset, class_id, group_key):
        if cli_box_threshold is not None:
            return "cli_box_threshold"
        if cli_preset is not None:
            return "cli_preset"
        if self.classes.get(class_id, {}).get("box_threshold") is not None:
            return "class_config"
        if self.prompt_groups.get(group_key, {}).get("box_threshold") is not None:
            return "group_config"
        return "default"

    def provenance(self) -> Dict[str, Any]:
        import platform
        snapshot = yaml.safe_dump(self.config, allow_unicode=True, sort_keys=True)
        return {"config_snapshot": self.config,
                "config_sha256": hashlib.sha256(snapshot.encode("utf-8")).hexdigest(),
                "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                  for name in ('detector.py', 'postprocess.py', 'object_classes.py')},
                "model_id": self.model_id,
                "catalog_version": CATALOG_VERSION,
                "model_revision": getattr(getattr(self.model, "config", None), "_commit_hash", None),
                "torch_num_threads": torch.get_num_threads(),
                "precision": getattr(self, 'precision', 'fp32'),
                "versions": {"python": platform.python_version(), "torch": torch.__version__,
                             "transformers": _TRANSFORMERS_VERSION,
                             "numpy": np.__version__, "pillow": Image.__version__,
                             "pyyaml": yaml.__version__}}

    def _run_model(self, inputs, prompt, source):
        from contextlib import nullcontext
        from lan_chiem.feature_cache import ImageFeatureCache
        if not hasattr(self, '_image_feature_cache'):
            self._image_feature_cache = ImageFeatureCache()
        use_cache = self.config.get('inference', {}).get('enable_image_feature_cache', True)
        previous_hits = self._image_feature_cache.hits
        precision = getattr(self, 'precision', 'fp32')
        cuda = str(self.device).startswith('cuda')
        if cuda:
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        autocast = torch.autocast('cuda', dtype=torch.float16) if cuda and precision == 'fp16' else nullcontext()
        with torch.inference_mode(), autocast, (self._image_feature_cache.use(self.model) if use_cache else nullcontext()):
            outputs = self.model(**inputs)
        if cuda:
            torch.cuda.synchronize(self.device)
        stats = getattr(self, '_inference_stats', None)
        if stats is not None:
            stats.append({'source': source, 'prompt': prompt,
                          'image_feature_cache_hit': self._image_feature_cache.hits > previous_hits,
                          'forward_seconds': round(time.perf_counter()-started, 4)})
            callback = getattr(self, '_forward_progress_callback', None)
            if callback is not None:
                callback(stats[-1])
        return outputs

    def _infer_group_batches(self, jobs, batch_size):
        """Independent captions, padded and batched; original per-query filtering."""
        from types import SimpleNamespace
        collected = []
        for start in range(0, len(jobs), batch_size):
            chunk = jobs[start:start+batch_size]
            if len(chunk) == 1:
                collected.extend(self._infer_crop(**chunk[0]))
                continue
            inputs = self.processor(images=[job['image_crop'] for job in chunk],
                                    text=[job['prompt_text'] for job in chunk],
                                    padding=True, return_tensors='pt')
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            outputs = self._run_model(inputs, [job['prompt_text'] for job in chunk], 'full_prompt_batch')
            for index, job in enumerate(chunk):
                individual_inputs = {key: value[index:index+1] for key, value in inputs.items()}
                individual_output = SimpleNamespace(logits=outputs.logits[index:index+1],
                                                    pred_boxes=outputs.pred_boxes[index:index+1])
                collected.extend(self._infer_crop(**job, _prepared=(individual_inputs, individual_output)))
        return collected

    def _infer_crop(
        self,
        image_crop: Image.Image,
        prompt_text: str,
        target_class_id: Optional[int],  # Nếu per_class thì có class_id, nếu grouped thì None
        group_key: str,
        alias_tuples: List[Tuple[str, int, str, str]],
        source: str,
        offset_xy: Tuple[int, int],
        full_w: int,
        full_h: int,
        actual_box_threshold: float,
        actual_text_threshold: float,
        debug: bool,
        raw_candidate_threshold: float,
        det_counter: List[int],
        rejections: Optional[List[RejectionRecord]],
        _prepared=None,
    ) -> List[Detection]:
        """
        Suy luận trên một ảnh hoặc mảnh cắt.
        Đảm bảo: Tập box giữ lại (score >= actual_box_threshold) hoàn toàn bất biến dù debug bật hay tắt.
        """
        if not prompt_text:
            return []

        if _prepared is None:
            inputs = self.processor(images=image_crop, text=prompt_text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self._run_model(inputs, prompt_text, source)
        else:
            inputs, outputs = _prepared

        # Nếu debug=True, dùng candidate_threshold thấp hơn để thu thập dữ liệu loại bỏ.
        # Nếu debug=False, chỉ lấy từ actual_box_threshold trở lên.
        cand_thresh = min(actual_box_threshold, raw_candidate_threshold) if debug else actual_box_threshold

        target_sizes = [(image_crop.height, image_crop.width)]
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.get("input_ids"),
            threshold=cand_thresh,
            text_threshold=actual_text_threshold,
            target_sizes=target_sizes,
        )

        if not results:
            return []

        res = results[0]
        scores = res.get("scores", torch.tensor([])).cpu().tolist()
        boxes = res.get("boxes", torch.tensor([])).cpu().tolist()
        raw_labels = res.get("text_labels", [])

        ox, oy = offset_xy
        valid_detections: List[Detection] = []
        distractors = self.prompt_groups.get(group_key, {}).get('distractors', {})
        active_distractors = {name: cfg for name, cfg in distractors.items()
                              if cfg.get('prompt', '').lower() in prompt_text.lower()}
        for name in active_distractors:
            self._main_counter_evidence.setdefault(name, [])

        for score, box, raw_label in zip(scores, boxes, raw_labels):
            score_val = float(score)
            x1, y1, x2, y2 = box

            gx1, gy1, gx2, gy2 = crop_box_to_global(
                box, offset_xy, (image_crop.width, image_crop.height), (full_w, full_h))

            # Lọc box không hợp lệ
            if gx2 <= gx1 + 2.0 or gy2 <= gy1 + 2.0:
                if rejections is not None:
                    rejections.append(
                        RejectionRecord(
                            stage="threshold",
                            bbox_xyxy=[gx1, gy1, gx2, gy2],
                            score=score_val,
                            raw_label=raw_label,
                            prompt_group=group_key,
                            source=source,
                            reason="invalid_box",
                            details=f"Kích thước box không hợp lệ (W={gx2-gx1:.1f}, H={gy2-gy1:.1f})",
                        )
                    )
                continue

            # Kiểm tra ngưỡng thực tế
            if score_val < actual_box_threshold:
                if rejections is not None:
                    rejections.append(
                        RejectionRecord(
                            stage="threshold",
                            bbox_xyxy=[gx1, gy1, gx2, gy2],
                            score=score_val,
                            raw_label=raw_label,
                            prompt_group=group_key,
                            source=source,
                            reason="below_threshold",
                            details=f"Điểm số {score_val:.4f} < ngưỡng yêu cầu {actual_box_threshold:.2f}",
                        )
                    )
                continue

            # Ánh xạ nhãn:
            # A. Chế độ Per-Class (mặc định): Gán trực tiếp theo queried class ID!
            if target_class_id is not None:
                cid = target_class_id
                c_meta = ClassRegistry.get(cid)
                cname = c_meta.name if c_meta else f"class_{cid}"
                cname_vi = c_meta.display_name if c_meta else cname

                # Lưu subtype cho ID 9 và ID 10
                subtype = None
                if cid == ClassId.ADVERTISING_SIGN:
                    subtype = determine_sign_subtype(raw_label)
                elif cid == ClassId.GOODS:
                    subtype = determine_goods_subtype(raw_label)

                det_counter[0] += 1
                det_id = f"det_{det_counter[0]}"
                valid_detections.append(
                    Detection(
                        id=det_id,
                        class_id=cid,
                        class_name=cname,
                        class_name_vi=cname_vi,
                        raw_label=raw_label,
                        prompt_group=group_key,
                        source=source,
                        score=score_val,
                        bbox_xyxy=[gx1, gy1, gx2, gy2],
                        subtype=subtype,
                        status="predicted",
                        queried_class_id=cid,
                        query_prompt=prompt_text,
                    )
                )

            # B. Chế độ Grouped (khi chạy nhóm): Thu thập tất cả class ID khớp, xử lý nhãn ghép
            else:
                negative_groups = [name for name, cfg in active_distractors.items()
                    if any(re.search(r'\b' + re.escape(a.lower()) + r'\b', raw_label.lower())
                           for a in cfg.get('aliases', [cfg['prompt']]))]
                if negative_groups:
                    record = {'id': f'main_counter_{group_key}_{source}_{len(self._main_counter_evidence.get("_ambiguous", []))}_{sum(len(v) for v in self._main_counter_evidence.values())}',
                              'score': score_val, 'bbox_xyxy': [gx1, gy1, gx2, gy2],
                              'label': raw_label, 'origin': 'same_caption', 'source': source,
                              'candidate_counter_groups': negative_groups, 'query_prompt': prompt_text,
                              'status': 'ambiguous' if len(negative_groups) > 1 else 'hypothesis'}
                    if len(negative_groups) == 1:
                        self._main_counter_evidence[negative_groups[0]].append(record)
                    else:
                        self._main_counter_evidence.setdefault('_ambiguous', []).append(record)
                cid, cname, cname_vi, is_ambiguous, matching_ids = self._map_grouped_label(raw_label, alias_tuples)

                if cid is None and not is_ambiguous:
                    if rejections is not None:
                        rejections.append(
                            RejectionRecord(
                                stage="mapping",
                                bbox_xyxy=[gx1, gy1, gx2, gy2],
                                score=score_val,
                                raw_label=raw_label,
                                prompt_group=group_key,
                                source=source,
                                reason="non_target_label" if negative_groups else "unmapped_label",
                                details=f"Nhãn '{raw_label}' không khớp với alias nào trong nhóm {group_key}",
                            )
                        )
                    continue

                det_counter[0] += 1
                det_id = f"det_{det_counter[0]}"

                if is_ambiguous:
                    # Nhãn ghép nhiều lớp -> đưa vào review với cảnh báo rõ ràng
                    valid_detections.append(
                        Detection(
                            id=det_id,
                            class_id=None,
                            class_name="ambiguous",
                            class_name_vi="Chưa phân biệt lớp",
                            candidate_class_ids=matching_ids,
                            raw_label=raw_label,
                            prompt_group=group_key,
                            source=source,
                            score=score_val,
                            bbox_xyxy=[gx1, gy1, gx2, gy2],
                            status="review",
                            review_reason=f"Nhãn ghép '{raw_label}' chứa dấu hiệu của nhiều lớp: {matching_ids}. Không tự ý kết luận lớp.",
                            queried_class_id=None,
                            query_prompt=prompt_text,
                        )
                    )
                else:
                    subtype = determine_sign_subtype(raw_label) if cid == ClassId.ADVERTISING_SIGN else None
                    if cid == ClassId.GOODS:
                        subtype = determine_goods_subtype(raw_label)

                    valid_detections.append(
                        Detection(
                            id=det_id,
                            class_id=cid,
                            class_name=cname,
                            class_name_vi=cname_vi,
                            raw_label=raw_label,
                            prompt_group=group_key,
                            source=source,
                            score=score_val,
                            bbox_xyxy=[gx1, gy1, gx2, gy2],
                            subtype=subtype,
                            status="predicted",
                            queried_class_id=None,
                            query_prompt=prompt_text,
                        )
                    )

            if target_class_id is None and negative_groups:
                valid_detections[-1].evidence['competing_non_target_labels'] = negative_groups
                valid_detections[-1].require_review('target_and_non_target_phrase_unresolved')

        return valid_detections

    def _map_grouped_label(
        self,
        raw_label: str,
        alias_tuples: List[Tuple[str, int, str, str]],
    ) -> Tuple[Optional[int], Optional[str], Optional[str], bool, List[int]]:
        """
        Ánh xạ nhãn trong chế độ grouped:
        - Thu thập TẤT CẢ các class ID khớp.
        - Nếu khớp đúng 1 class ID -> trả về class ID đó.
        - Nếu khớp nhiều hơn 1 class ID (nhãn ghép) -> đánh dấu is_ambiguous=True.
        - Không chọn theo thứ tự hoặc độ dài alias.
        """
        cleaned = raw_label.strip().lower()
        cleaned = re.sub(r"^[\.\,\;\:\s\'\"]+|[\.\,\;\:\s\'\"]+$", "", cleaned)
        if not cleaned:
            return None, None, None, False, []

        matching_classes: Dict[int, Tuple[str, str]] = {}

        for alias, cid, cname, cname_vi in alias_tuples:
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, cleaned):
                matching_classes[cid] = (cname, cname_vi)

        m_ids = sorted(list(matching_classes.keys()))
        if len(m_ids) == 1:
            cid = m_ids[0]
            cname, cname_vi = matching_classes[cid]
            return cid, cname, cname_vi, False, m_ids
        elif len(m_ids) > 1:
            return None, None, None, True, m_ids

        return None, None, None, False, []

    def run_counter_checks(
        self,
        image: Image.Image,
        box_thresh: float = 0.20,
        text_thresh: float = 0.20,
        groups: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Chạy các prompt đối chứng đọc động từ config:
        car, barrier_gate, potted_plant, construction_materials, bicycle.
        Lưu đầy đủ query, raw label, score, box của đối chứng.
        """
        counter_groups = {k: v for k, v in self.counter_prompts.get("groups", {}).items()
                          if groups is None or k in groups}
        if not counter_groups:
            return {}

        all_phrases = []
        alias_map: List[Tuple[str, str]] = []  # (alias, group_name)

        for gname, gcfg in counter_groups.items():
            praw = gcfg.get("prompt", "")
            phrases = [s.strip().lower() for s in praw.split(".") if s.strip()]
            for p in phrases:
                if p not in all_phrases:
                    all_phrases.append(p)
            aliases = list(gcfg.get("aliases", [])) + phrases
            for a in aliases:
                a_clean = a.strip().lower()
                if a_clean:
                    alias_map.append((a_clean, gname))

        if not all_phrases:
            return {}

        counter_prompt_str = " . ".join(all_phrases) + " ."
        inputs = self.processor(images=image, text=counter_prompt_str, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._run_model(inputs, counter_prompt_str, "counter")

        target_sizes = [(image.height, image.width)]
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.get("input_ids"),
            threshold=box_thresh,
            text_threshold=text_thresh,
            target_sizes=target_sizes,
        )

        detected_counters: Dict[str, List[Dict[str, Any]]] = {g: [] for g in counter_groups.keys()}

        if not results:
            return detected_counters

        res = results[0]
        scores = res.get("scores", torch.tensor([])).cpu().tolist()
        boxes = res.get("boxes", torch.tensor([])).cpu().tolist()
        labels = res.get("text_labels", [])

        # Keep every matched interpretation, including ambiguous counter phrases.
        detected_counters["_ambiguous"] = []
        for index, (score, box, label) in enumerate(zip(scores, boxes, labels), 1):
            groups = sorted({group for alias, group in alias_map
                             if re.search(r"\b" + re.escape(alias) + r"\b", label.lower())})
            record = {"id": f"counter_{index}", "score": float(score),
                      "bbox_xyxy": [float(c) for c in box], "raw_label": label,
                      "candidate_counter_groups": groups, "query_prompt": counter_prompt_str,
                      "box_threshold": box_thresh, "text_threshold": text_thresh,
                      "status": "ambiguous" if len(groups) != 1 else "hypothesis"}
            if len(groups) == 1:
                record["counter_group"] = groups[0]
                detected_counters[groups[0]].append(record)
            else:
                detected_counters["_ambiguous"].append(record)
        return detected_counters

    def analyze_conflicts_and_partition(
        self, detections: List[Detection],
        counter_objects: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> Tuple[List[Detection], List[Detection], List[Dict[str, Any]]]:
        """Record spatial evidence without treating overlap as semantic verification."""
        from lan_chiem.postprocess import mark_exclusive_overlaps
        reports = mark_exclusive_overlaps(detections) if self.config.get('review_rules', {}).get('exclusive_class_overlap', True) else []
        predicted, review = [], []
        rules = self.config.get("review_rules", {})
        counters = counter_objects or {}

        def enabled(name):
            return rules.get(name, True)

        def relate(det, other, name, iou_limit=.35, containment=.65):
            ob = other.bbox_xyxy if isinstance(other, Detection) else other["bbox_xyxy"]
            oid = other.id if isinstance(other, Detection) else other.get("id")
            iou = calculate_iou(det.bbox_xyxy, ob)
            ios = calculate_ios(det.bbox_xyxy, ob)
            reverse_ios = calculate_ios(ob, det.bbox_xyxy)
            # Same-caption alternatives must describe essentially the same object.
            # A bicycle beside a motorcycle or a small part inside a cart is insufficient.
            if isinstance(other, dict) and other.get('origin') == 'same_caption':
                if other.get('score', 0) < .30 or iou < .60 or min(ios, reverse_ios) < .70:
                    return
            coverage = ios if name == "contained_in_cart_candidate" else max(ios, reverse_ios)
            if iou < iou_limit and coverage < containment:
                return
            report = {"conflict_type": name, "det_id": det.id, "related_id": oid,
                      "bbox_xyxy": det.bbox_xyxy, "related_bbox_xyxy": ob,
                      "iou": iou, "ios": ios, "reverse_ios": reverse_ios,
                      "interpretation": "spatial_overlap_only_not_verified"}
            if isinstance(other, Detection):
                report["related_status_at_observation"] = other.status
                report["related_class_id"] = other.class_id
                if det.id not in other.related_detection_ids:
                    other.related_detection_ids.append(det.id)
            else:
                report["counter_evidence"] = other
            if oid and oid not in det.related_detection_ids:
                det.related_detection_ids.append(oid)
            det.evidence.setdefault("spatial_relations", []).append(report)
            det.conflict_info = det.conflict_info or report
            det.require_review(name)
            reports.append(report)

        for det in detections:
            cid = det.class_id
            if det.candidate_class_ids or cid is None:
                det.require_review("ambiguous_class_unresolved")
                review.append(det)
                continue
            if enabled('upright_object_geometry') and cid in (ClassId.UTILITY_POLE, ClassId.TREE_TRUNK):
                x1, y1, x2, y2 = det.bbox_xyxy
                if x2-x1 > y2-y1:
                    det.evidence['width_height_ratio'] = (x2-x1)/max(y2-y1, 1)
                    det.require_review('upright_object_scope_uncertain')
            if cid in (ClassId.PLASTIC_TABLE, ClassId.PLASTIC_CHAIR):
                det.evidence['material_status'] = 'unverified_prompt_hypothesis'
            if cid == ClassId.VENDING_CART:
                det.evidence["sale_context_status"] = "sale_context_unknown"
                if re.search(r"\b(bicycle|motorcycle)\b", det.raw_label.lower()):
                    det.require_review("vendor_vehicle_sale_context_unknown")
            if cid == ClassId.GOODS:
                words = det.raw_label.lower()
                content = bool(re.search(r"\b(fruit|fruits|vegetable|vegetables|products|packaged)\b", words))
                container = bool(re.search(r"\b(basket|crate|box|container)\b", words))
                det.evidence.update({"observed_subtype": det.subtype,
                    "content_status": "content_candidate" if content else
                                      ("container_only" if container else "content_unknown"),
                    "sale_context_status": "sale_context_unknown",
                    "verification_status": "unverified",
                    "source": "raw_model_phrase_only"})
                # Chỉ đưa vào review nếu không phải sọt/thùng/hoa quả hoặc điểm số quá thấp
                # A text phrase cannot establish goods content or sale context.
                det.require_review(det.evidence["content_status"] + "; sale_context_unknown")

            counter_pairs = {
                2: [("bicycle", "motorcycle_bicycle_overlap")],
                ClassId.PLASTIC_TABLE: [("car", "table_car_overlap"),
                    ("construction_materials", "table_materials_overlap"), ("foam_box", "table_foam_box_overlap")],
                ClassId.PLASTIC_CHAIR: [("foam_box", "chair_foam_box_overlap")],
                6: [("barrier_gate", "cart_barrier_overlap"), ("car", "cart_car_overlap"),
                    ("delivery_vehicle", "cart_delivery_vehicle_overlap")],
                ClassId.UTILITY_POLE: [("tree_trunk", "pole_tree_counter_overlap")],
                ClassId.ADVERTISING_SIGN: [("traffic_sign", "advertising_traffic_overlap")],
                ClassId.GOODS: [("potted_plant", "goods_plant_overlap"),
                     ("construction_materials", "goods_materials_overlap")],
            }
            for group, name in counter_pairs.get(cid, []):
                if enabled(name):
                    for counter in counters.get(group, []):
                        relate(det, counter, name)
            # An ambiguous counter remains ambiguous; it cannot silently become one group.
            if enabled("ambiguous_counter_overlap"):
                applicable = {g for g, _ in counter_pairs.get(cid, [])}
                for counter in counters.get("_ambiguous", []):
                    if applicable.intersection(counter.get("candidate_counter_groups", [])):
                        relate(det, counter, "ambiguous_counter_overlap")

            object_pairs = {
                ClassId.PLASTIC_TABLE: [(ClassId.VENDING_CART, "contained_in_cart_candidate"), (ClassId.CAR, "table_car_overlap")],
                ClassId.VENDING_CART: [(ClassId.MOTORCYCLE, "cart_motorcycle_overlap"), (ClassId.CAR, "cart_car_overlap")],
                ClassId.UTILITY_POLE: [(ClassId.TREE_TRUNK, "pole_trunk_overlap")],
                ClassId.TREE_TRUNK: [(ClassId.UTILITY_POLE, "pole_trunk_overlap")],
                ClassId.GOODS: [(ClassId.PLASTIC_TABLE, "overlap_with_table"), (ClassId.PLASTIC_CHAIR, "goods_chair_overlap")],
            }
            for other_cid, name in object_pairs.get(cid, []):
                if enabled(name):
                    for other in detections:
                        if other.class_id == other_cid and not other.candidate_class_ids:
                            relate(det, other, name,
                                   iou_limit=1.01 if name in ("overlap_with_table", "contained_in_cart_candidate") else .35,
                                   containment=.70 if name in ("overlap_with_table", "contained_in_cart_candidate") else .65)
            if cid == ClassId.TREE_TRUNK and enabled("tree_scope_uncertain"):
                for other in detections:
                    if other is not det and other.class_id == ClassId.TREE_TRUNK:
                        if (calculate_ios(det.bbox_xyxy, other.bbox_xyxy) >= .85 or
                            calculate_ios(other.bbox_xyxy, det.bbox_xyxy) >= .85) and calculate_iou(det.bbox_xyxy, other.bbox_xyxy) < .35:
                            relate(det, other, "tree_scope_uncertain", containment=.85)
            (review if det.status == "review" else predicted).append(det)
        return predicted, review, reports

    def predict_single_prompt(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        prompt_text: str,
        box_threshold: float = 0.20,
        text_threshold: float = 0.20,
    ) -> Dict[str, Any]:
        """Chế độ cô lập một prompt duy nhất phục vụ đối chiếu độc lập."""
        self.load_model()
        image, image_name, img_hash = self.preprocess_image(image_input)
        img_w, img_h = image.width, image.height

        clean_prompt = prompt_text.strip()
        if not clean_prompt.endswith("."):
            clean_prompt += " ."

        self._inference_stats = []
        if hasattr(self, '_image_feature_cache'):
            self._image_feature_cache.clear()
        start_time = time.perf_counter()
        inputs = self.processor(images=image, text=clean_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._run_model(inputs, clean_prompt, "single_prompt")

        target_sizes = [(img_h, img_w)]
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            input_ids=inputs.get("input_ids"),
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )

        elapsed = time.perf_counter() - start_time
        res = results[0] if results else {}
        scores = res.get("scores", torch.tensor([])).cpu().tolist()
        boxes = res.get("boxes", torch.tensor([])).cpu().tolist()
        labels = res.get("text_labels", [])

        dets: List[Dict[str, Any]] = []
        for i, (s, b, l) in enumerate(zip(scores, boxes, labels), 1):
            dets.append({
                "id": f"single_det_{i}",
                "raw_label": l,
                "score": round(float(s), 4),
                "bbox_xyxy": [round(float(c), 2) for c in b],
                "prompt": clean_prompt,
            })

        return {
            "catalog_version": CATALOG_VERSION,
            "image_name": image_name,
            "image_sha256": img_hash,
            "provenance": self.provenance(),
            "image_size": {"width": img_w, "height": img_h},
            "mode": "single_prompt_inspection",
            "prompt_text": clean_prompt,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "inference_time_seconds": round(elapsed, 4),
            "model_calls": len(self._inference_stats),
            "inference_passes": self._inference_stats,
            "total_detections": len(dets),
            "detections": dets,
            "_image_pil": image,
        }

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        mode: Optional[str] = None,  # "full" | "tiled" | "full+tiled"
        query_mode: Optional[str] = None,  # "per_class" | "grouped"
        box_threshold_override: Optional[float] = None,
        text_threshold_override: Optional[float] = None,
        preset: Optional[float] = None,
        tile_size: Optional[int] = None,
        overlap_ratio: Optional[float] = None,
        nms_iou: Optional[float] = None,
        target_classes: Optional[List[int]] = None,
        target_groups: Optional[List[str]] = None,
        single_prompt: Optional[str] = None,
        enable_counter_checks: Optional[bool] = None,
        enable_cart_merge: Optional[bool] = None,
        enable_leg_filter: Optional[bool] = None,
        enable_tree_filter: Optional[bool] = None,
        tile_small_objects_only: Optional[bool] = None,
        debug: bool = False,
        sidewalk_regions: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Quy trình suy luận toàn diện chuẩn 2.2:
        - Hỗ trợ per_class query (mặc định) hoặc grouped query.
        - Đảm bảo tính bất biến của kết quả chính khi bật/tắt debug.
        - Kiểm tra đối chứng đọc động từ config.
        - Hợp nhất có kiểm soát xe đẩy ID 6 trùng lặp.
        """
        if single_prompt:
            b_thresh = box_threshold_override if box_threshold_override is not None else (preset if preset is not None else 0.20)
            t_thresh = text_threshold_override if text_threshold_override is not None else 0.20
            return self.predict_single_prompt(
                image_input=image_input,
                prompt_text=single_prompt,
                box_threshold=b_thresh,
                text_threshold=t_thresh,
            )

        # Kiểm tra lớp yêu cầu có hợp lệ không
        if target_classes is not None:
            for cid in target_classes:
                meta = ClassRegistry.get(cid)
                if meta is None or meta.support_status == "unsupported":
                    raise ValueError(
                        f"Lớp ID {cid} ({meta.display_name if meta else 'Không rõ'}) hiện chưa được hỗ trợ tự động."
                    )

        self.load_model()
        self._inference_stats = []
        if hasattr(self, '_image_feature_cache'):
            self._image_feature_cache.clear()

        # Thiết lập tham số với kiểm tra `is not None`
        inf_cfg = self.config.get("inference", {})
        q_mode = query_mode if query_mode is not None else inf_cfg.get("query_mode", "per_class")
        infer_mode = mode if mode is not None else inf_cfg.get("mode", "full")
        t_size = tile_size if tile_size is not None else inf_cfg.get("tile_size", 768)
        ov_ratio = overlap_ratio if overlap_ratio is not None else inf_cfg.get("overlap_ratio", 0.25)
        iou_thresh = nms_iou if nms_iou is not None else inf_cfg.get("nms_iou", 0.50)
        global_text_def = text_threshold_override if text_threshold_override is not None else inf_cfg.get("text_threshold_default", 0.20)
        raw_cand_thresh = inf_cfg.get("raw_candidate_threshold", 0.15)

        do_counter = enable_counter_checks if enable_counter_checks is not None else inf_cfg.get("enable_counter_checks", True)
        do_cart_merge = enable_cart_merge if enable_cart_merge is not None else inf_cfg.get("enable_cart_merge", True)
        do_leg_filter = enable_leg_filter if enable_leg_filter is not None else inf_cfg.get("enable_leg_filter", True)
        do_tree_filter = enable_tree_filter if enable_tree_filter is not None else inf_cfg.get("enable_tree_filter", True)
        tile_small_only = tile_small_objects_only if tile_small_objects_only is not None else inf_cfg.get("tile_small_objects_only", False)

        image, image_name, img_hash = self.preprocess_image(image_input)
        img_w, img_h = image.width, image.height

        start_time = time.perf_counter()
        det_counter = [0]
        self._main_counter_evidence = {}
        rejections: List[RejectionRecord] = [] if debug else None
        queried_details: List[Dict[str, Any]] = []

        all_collected_detections: List[Detection] = []

        # =========================================================================
        # 1. Chế độ PER_CLASS QUERY (Mặc định)
        # =========================================================================
        if q_mode == "per_class":
            supported_object_ids = [c.id for c in ClassRegistry.object_classes() if c.support_status != "unsupported"]
            active_class_ids = target_classes if target_classes is not None else supported_object_ids

            for cid in active_class_ids:
                if cid not in self.classes:
                    continue

                c_info = self.classes[cid]
                raw_prompt = c_info.get("prompt", "")
                if not raw_prompt:
                    continue

                prompt_text = raw_prompt if raw_prompt.endswith(".") else raw_prompt + " ."
                actual_box_thresh = self.resolve_actual_threshold(
                    cli_box_threshold=box_threshold_override,
                    cli_preset=preset,
                    class_id=cid,
                    group_key=None,
                )
                actual_text_thresh = text_threshold_override if text_threshold_override is not None else c_info.get("text_threshold", global_text_def)
                is_small = c_info.get("is_small_object", False)

                queried_details.append({
                    "class_id": cid,
                    "class_name": c_info.get("name"),
                    "prompt": prompt_text,
                    "threshold_source": self.threshold_source(box_threshold_override, preset, cid, None),
                    "actual_box_threshold": actual_box_thresh,
                    "actual_text_threshold": actual_text_thresh,
                })

                should_run_full = infer_mode in ("full", "full+tiled")
                should_run_tile = infer_mode in ("tiled", "full+tiled")

                if tile_small_only and not is_small:
                    should_run_tile = False
                    should_run_full = True

                # A. Chạy full frame
                if should_run_full:
                    full_dets = self._infer_crop(
                        image_crop=image,
                        prompt_text=prompt_text,
                        target_class_id=cid,
                        group_key=f"class_{cid}",
                        alias_tuples=[],
                        source="full",
                        offset_xy=(0, 0),
                        full_w=img_w,
                        full_h=img_h,
                        actual_box_threshold=actual_box_thresh,
                        actual_text_threshold=actual_text_thresh,
                        debug=debug,
                        raw_candidate_threshold=raw_cand_thresh,
                        det_counter=det_counter,
                        rejections=rejections,
                    )
                    all_collected_detections.extend(full_dets)

                # B. Chạy tiled
                if should_run_tile and (infer_mode == "tiled" or img_w > t_size or img_h > t_size):
                    tiles = generate_tiles(img_w, img_h, tile_size=t_size, tile_overlap=ov_ratio)
                    for tx1, ty1, tx2, ty2 in tiles:
                        tile_crop = image.crop((tx1, ty1, tx2, ty2))
                        tile_dets = self._infer_crop(
                            image_crop=tile_crop,
                            prompt_text=prompt_text,
                            target_class_id=cid,
                            group_key=f"class_{cid}",
                            alias_tuples=[],
                            source=f"tile_{tx1}_{ty1}",
                            offset_xy=(tx1, ty1),
                            full_w=img_w,
                            full_h=img_h,
                            actual_box_threshold=actual_box_thresh,
                            actual_text_threshold=actual_text_thresh,
                            debug=debug,
                            raw_candidate_threshold=raw_cand_thresh,
                            det_counter=det_counter,
                            rejections=rejections,
                        )
                        all_collected_detections.extend(tile_dets)

        # =========================================================================
        # 2. Chế độ GROUPED QUERY (Đối chiếu)
        # =========================================================================
        else:
            active_groups = target_groups if target_groups is not None else inf_cfg.get("compact_group_keys", []) if q_mode == "compact" else [
                gk for gk, gv in self.prompt_groups.items() if gv.get("enabled", True)
            ]

            pending_full_jobs = []
            batch_size = getattr(self, 'prompt_batch_size', 1)
            for gkey in active_groups:
                if gkey not in self.prompt_groups:
                    continue

                ginfo = self.prompt_groups[gkey]
                group_cids = ginfo.get("class_ids", [])
                if target_classes is not None:
                    group_cids = [c for c in group_cids if c in target_classes]
                    if not group_cids:
                        continue

                # Xây dựng prompt group và alias tuples
                phrases: List[str] = []
                alias_tuples: List[Tuple[str, int, str, str]] = []
                for cid in group_cids:
                    if cid not in self.classes:
                        continue
                    c_info = self.classes[cid]
                    cname = c_info.get("name", "")
                    cname_vi = c_info.get("display_name", cname)
                    praw = inf_cfg.get("compact_phrases", {}).get(cid, c_info.get("prompt", "")) if q_mode == "compact" else c_info.get("prompt", "")
                    for iph in [s.strip().lower() for s in praw.split(".") if s.strip()]:
                        if iph not in phrases:
                            phrases.append(iph)
                    for a in (list(c_info.get("aliases", [])) + [v.strip().lower() for v in praw.split(".") if v.strip()]):
                        a_clean = a.strip().lower()
                        if a_clean:
                            alias_tuples.append((a_clean, cid, cname, cname_vi))

                if q_mode == 'compact':
                    for cfg in ginfo.get('distractors', {}).values():
                        phrases.extend(v.strip().lower() for v in cfg['prompt'].split('.') if v.strip())
                prompt_text = " . ".join(phrases) + " ." if phrases else ""
                if not prompt_text:
                    continue

                actual_box_thresh = self.resolve_actual_threshold(
                    cli_box_threshold=box_threshold_override,
                    cli_preset=preset,
                    class_id=None,
                    group_key=gkey,
                )
                actual_text_thresh = text_threshold_override if text_threshold_override is not None else ginfo.get("text_threshold", global_text_def)
                is_small = ginfo.get("is_small_object_group", False)

                queried_details.append({
                    "group_key": gkey,
                    "class_ids": group_cids,
                    "prompt": prompt_text,
                    "threshold_source": self.threshold_source(box_threshold_override, preset, None, gkey),
                    "actual_box_threshold": actual_box_thresh,
                    "actual_text_threshold": actual_text_thresh,
                })

                should_run_full = infer_mode in ("full", "full+tiled")
                should_run_tile = infer_mode in ("tiled", "full+tiled")
                if tile_small_only and not is_small:
                    should_run_tile = False
                    should_run_full = True

                if should_run_full:
                    full_job = dict(
                        image_crop=image,
                        prompt_text=prompt_text,
                        target_class_id=None,
                        group_key=gkey,
                        alias_tuples=alias_tuples,
                        source="full",
                        offset_xy=(0, 0),
                        full_w=img_w,
                        full_h=img_h,
                        actual_box_threshold=actual_box_thresh,
                        actual_text_threshold=actual_text_thresh,
                        debug=debug,
                        raw_candidate_threshold=raw_cand_thresh,
                        det_counter=det_counter,
                        rejections=rejections,
                    )
                    if infer_mode == 'full' and batch_size > 1:
                        pending_full_jobs.append(full_job)
                    else:
                        all_collected_detections.extend(self._infer_crop(**full_job))

                if should_run_tile and (infer_mode == "tiled" or img_w > t_size or img_h > t_size):
                    tiles = generate_tiles(img_w, img_h, tile_size=t_size, tile_overlap=ov_ratio)
                    for tx1, ty1, tx2, ty2 in tiles:
                        tile_crop = image.crop((tx1, ty1, tx2, ty2))
                        tile_dets = self._infer_crop(
                            image_crop=tile_crop,
                            prompt_text=prompt_text,
                            target_class_id=None,
                            group_key=gkey,
                            alias_tuples=alias_tuples,
                            source=f"tile_{tx1}_{ty1}",
                            offset_xy=(tx1, ty1),
                            full_w=img_w,
                            full_h=img_h,
                            actual_box_threshold=actual_box_thresh,
                            actual_text_threshold=actual_text_thresh,
                            debug=debug,
                            raw_candidate_threshold=raw_cand_thresh,
                            det_counter=det_counter,
                            rejections=rejections,
                        )
                        all_collected_detections.extend(tile_dets)

            if pending_full_jobs:
                all_collected_detections.extend(self._infer_group_batches(pending_full_jobs, batch_size))

        from lan_chiem.postprocess import consolidate_nested_boxes
        nested_removed = []
        if inf_cfg.get('enable_nested_dedup', True):
            all_collected_detections, nested_removed = consolidate_nested_boxes(all_collected_detections)
        if rejections is not None:
            for det in nested_removed:
                rejections.append(RejectionRecord('nms', det.bbox_xyxy, det.score, det.raw_label,
                    det.prompt_group, det.source, 'suppressed_nested_duplicate', 'Saved on winner.evidence.duplicate_hypotheses'))

        # =========================================================================
        # 3. HẬU XỬ LÝ (NMS, MERGING, GEOMETRY)
        # =========================================================================
        ambiguous_detections = [d for d in all_collected_detections if d.candidate_class_ids or d.class_id is None or d.evidence.get("contained_detection_ids")]
        by_class: Dict[int, List[Detection]] = {}
        for d in all_collected_detections:
            if not d.candidate_class_ids and d.class_id is not None and not d.evidence.get("contained_detection_ids"):
                by_class.setdefault(d.class_id, []).append(d)

        fused_detections: List[Detection] = []

        for cid, class_dets in by_class.items():
            # A. Bộ lọc hình học thân cây (ID 9) nếu bật
            if cid == ClassId.TREE_TRUNK and do_tree_filter:
                class_dets = filter_tree_trunk_geometry(class_dets, img_w, img_h, rejections)

            # B. Bộ lọc box chân người (ID 1) nếu bật
            if cid == 1 and do_leg_filter:
                class_dets = consolidate_person_boxes(class_dets, rejections)

            # C. Hợp nhất xe đẩy bán hàng (ID 6) trùng lặp nếu bật
            if cid == 6 and do_cart_merge:
                cm_iou = self.cart_merge_cfg.get("iou_threshold", 0.40)
                cm_ios = self.cart_merge_cfg.get("ios_threshold", 0.65)
                cm_center = self.cart_merge_cfg.get("max_center_dist_ratio", 0.40)
                class_dets = merge_duplicate_carts(
                    cart_dets=class_dets,
                    iou_thresh=cm_iou,
                    ios_thresh=cm_ios,
                    max_center_dist_ratio=cm_center,
                    rejections=rejections,
                )

            # D. NMS nội bộ nhóm (Intra-class NMS chuẩn)
            if len(class_dets) > 1:
                class_dets.sort(key=lambda x: x.score, reverse=True)
                selected: List[Detection] = []
                for candidate in class_dets:
                    suppress = False
                    winner = None
                    for chosen in selected:
                        iou = calculate_iou(candidate.bbox_xyxy, chosen.bbox_xyxy)
                        if iou > iou_thresh:
                            suppress = True
                            winner = chosen
                            break
                    if not suppress:
                        selected.append(candidate)
                    else:
                        if rejections is not None and winner is not None:
                            rejections.append(
                                RejectionRecord(
                                    stage="nms",
                                    bbox_xyxy=candidate.bbox_xyxy,
                                    score=candidate.score,
                                    raw_label=candidate.raw_label,
                                    prompt_group=candidate.prompt_group,
                                    source=candidate.source,
                                    reason="suppressed_duplicate",
                                    details=f"Bị loại bởi {winner.id} với IoU={calculate_iou(candidate.bbox_xyxy, winner.bbox_xyxy):.2f} > {iou_thresh}",
                                )
                            )
                fused_detections.extend(selected)
            else:
                fused_detections.extend(class_dets)

        # =========================================================================
        # 4. KIỂM TRA ĐỐI CHỨNG VÀ PHÂN TÁCH PREDICTED VS REVIEW
        # =========================================================================
        counter_objects = {k: list(v) for k, v in self._main_counter_evidence.items()} if do_counter else {}
        counter_plan = {'policy': inf_cfg.get('counter_policy', 'all'), 'queried_groups': [], 'reused_main_car': False, 'same_caption_groups': sorted(self._main_counter_evidence)}
        if do_counter:
            available = self.counter_prompts.get('groups', {})
            groups = list(available)
            if counter_plan['policy'] == 'selective':
                present = {d.class_id for d in fused_detections + ambiguous_detections}
                needed = set(available) - {'car','barrier_gate','potted_plant','construction_materials','bicycle','traffic_sign'}
                if present.intersection({6,7,8}): needed.add('barrier_gate')
                if present.intersection({4,10}): needed.add('construction_materials')
                if 10 in present: needed.add('potted_plant')
                if 7 in present: needed.add('traffic_sign')
                if self.config.get('review_rules', {}).get('motorcycle_bicycle_overlap', True) and 2 in present: needed.add('bicycle')
                queried_ids = {q['class_id'] for q in queried_details if 'class_id' in q}
                queried_ids.update(cid for q in queried_details for cid in q.get('class_ids', []))
                if 3 in queried_ids:
                    counter_plan['reused_main_car'] = True
                    counter_objects['car'] = [dict(d.to_dict(), counter_group='car', origin='main_detection')
                                              for d in fused_detections if d.class_id == 3]
                elif present.intersection({4,6}): needed.add('car')
                groups = [g for g in available if g in needed and g not in self._main_counter_evidence]
            counter_plan['queried_groups'] = groups
            if groups:
                counter_objects.update(self.run_counter_checks(image=image, box_thresh=0.20, text_thresh=0.20, groups=groups))

        predicted_dets, review_dets, conflicts = self.analyze_conflicts_and_partition(
            detections=fused_detections + ambiguous_detections,
            counter_objects=counter_objects,
        )

        # Semantic overlap was already reviewed above. Only suppress same-class
        # duplicates; pixel-position rules and cross-class NMS destroy real objects.
        before_dedup = predicted_dets
        predicted_dets = apply_intra_class_nms(predicted_dets, iou_threshold=iou_thresh)
        dedup_removed = [d for d in before_dedup if d not in predicted_dets]
        if rejections is not None:
            for d in dedup_removed:
                rejections.append(
                    RejectionRecord(
                        stage="deduplication",
                        bbox_xyxy=d.bbox_xyxy,
                        score=d.score,
                        raw_label=d.raw_label,
                        prompt_group=d.prompt_group,
                        source=d.source,
                        reason=d.evidence.get("suppress_reason", "duplicate_or_conflict"),
                        details="Khử trùng lặp đảm bảo chỉ vẽ 1 bounding box cho 1 vật thể",
                    )
                )

        elapsed = time.perf_counter() - start_time

        predicted_class_counts: Dict[str, int] = {}
        for d in predicted_dets:
            c_key = f"{d.class_id}_{d.class_name}"
            predicted_class_counts[c_key] = predicted_class_counts.get(c_key, 0) + 1

        review_class_counts: Dict[str, int] = {}
        for d in review_dets:
            c_key = "ambiguous" if d.candidate_class_ids or d.class_id is None else f"{d.class_id}_{d.class_name}"
            review_class_counts[c_key] = review_class_counts.get(c_key, 0) + 1

        sidewalk_data = []
        if sidewalk_regions:
            for sr in sidewalk_regions:
                if hasattr(sr, "to_dict"):
                    sidewalk_data.append(sr.to_dict())
                elif isinstance(sr, dict):
                    sidewalk_data.append(sr)

        output_result = {
            "catalog_version": CATALOG_VERSION,
            "image_name": image_name,
            "image_sha256": img_hash,
            "image_size": {"width": img_w, "height": img_h},
            "provenance": self.provenance(),
            "model_id": self.model_id,
            "transformers_version": _TRANSFORMERS_VERSION,
            "device": self.device,
            "query_mode": q_mode,
            "inference_mode": infer_mode,
            "config": {
                "box_threshold_default": self.config.get("inference", {}).get("box_threshold_default", 0.25),
                "text_threshold_default": global_text_def,
                "nms_iou": iou_thresh,
                "tile_size": t_size if "tile" in infer_mode else None,
                "overlap_ratio": ov_ratio if "tile" in infer_mode else None,
                "tile_small_objects_only": tile_small_only,
                "enable_counter_checks": do_counter,
                "enable_cart_merge": do_cart_merge,
                "enable_leg_filter": do_leg_filter,
                "enable_tree_filter": do_tree_filter,
                "queried_details": queried_details,
            },
            "inference_time_seconds": round(elapsed, 4),
            "model_calls": len(self._inference_stats),
            "inference_passes": self._inference_stats,
            "summary": {
                "verified_goods_count": 0,
                "goods_content_candidates": sum(d.class_id == ClassId.GOODS and d.evidence.get("content_status") == "content_candidate" for d in review_dets),
                "goods_container_only_candidates": sum(d.class_id == ClassId.GOODS and d.evidence.get("content_status") == "container_only" for d in review_dets),
                "ambiguous_candidates": sum(bool(d.candidate_class_ids) for d in review_dets),
                "total_predicted_detections": len(predicted_dets),
                "total_review_candidates": len(review_dets),
                "total_conflicts_detected": len(conflicts),
                "predicted_class_counts": predicted_class_counts,
                "review_class_counts": review_class_counts,
                # Aliases tương thích ngược
                "total_confirmed_detections": len(predicted_dets),
                "confirmed_class_counts": predicted_class_counts,
                "total_detections": len(predicted_dets),
                "class_counts": predicted_class_counts,
            },
            "class_support_status": ClassRegistry.get_support_status_dict(),
            "object_detections": [d.to_dict() for d in predicted_dets],
            "review_candidates": [d.to_dict() for d in review_dets],
            "conflict_reports": conflicts,
            "counter_evidence": counter_objects,
            "counter_plan": counter_plan,
            "sidewalk_regions": sidewalk_data,
            "action_candidates": [],
            "_predicted_objects": predicted_dets,
            "_review_objects": review_dets,
            "_image_pil": image,
        }

        if debug:
            output_result["debug_trace"] = {
                "raw_candidate_threshold": raw_cand_thresh,
                "total_candidates_collected": len(all_collected_detections) + len(nested_removed) + sum(r.stage in ("threshold", "mapping") for r in (rejections or [])),
                "rejections_by_stage": {
                    "threshold": sum(1 for r in (rejections or []) if r.stage == "threshold"),
                    "mapping": sum(1 for r in (rejections or []) if r.stage == "mapping"),
                    "nms": sum(1 for r in (rejections or []) if r.stage == "nms"),
                    "geometry": sum(1 for r in (rejections or []) if r.stage == "geometry"),
                },
                "rejections_by_reason": {
                    "below_threshold": sum(1 for r in (rejections or []) if r.reason == "below_threshold"),
                    "invalid_box": sum(1 for r in (rejections or []) if r.reason == "invalid_box"),
                    "unmapped_label": sum(1 for r in (rejections or []) if r.reason == "unmapped_label"),
                    "suppressed_duplicate": sum(1 for r in (rejections or []) if r.reason == "suppressed_duplicate"),
                    "suppressed_duplicate_cart_heuristic": sum(1 for r in (rejections or []) if r.reason == "suppressed_duplicate_cart_heuristic"),
                    "subpart_legs": sum(1 for r in (rejections or []) if r.reason == "subpart_legs"),
                    "canopy_box": sum(1 for r in (rejections or []) if r.reason == "canopy_box"),
                },
                "counter_evidence": counter_objects,
                "counter_objects_detected": {
                    k: len(v) for k, v in counter_objects.items()
                },
                "rejected_records": [r.to_dict() for r in (rejections or [])],
            }

        return output_result
