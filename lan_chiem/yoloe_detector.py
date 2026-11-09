"""YOLOE-v8-S box-output backend with optional high-resolution ROI refinement.

Throughput and accuracy must be measured on the deployment GPU and labelled data.
Independent from Grounding DINO (zero transformers/HF dependency).
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps
import torch

from lan_chiem.object_classes import CATALOG_VERSION
from lan_chiem.postprocess import (
    Detection,
    apply_intra_class_nms,
    analyze_conflicts_and_partition,
    reconcile_roi_fragments,
)
from lan_chiem.registry import ClassRegistry
from lan_chiem.video_io import file_hash
from lan_chiem.configuration import load_config


TARGET_PHRASES: Dict[int, List[str]] = {
    1: ['person', 'seated person'],
    2: ['motorcycle', 'motor scooter', 'motorbike'],
    3: ['car', 'passenger car'],
    4: ['plastic table', 'rectangular plastic table', 'small plastic table', 'low folding table', 'folding table'],
    5: [
        'plastic chair', 'plastic stool', 'low plastic stool', 'small plastic stool',
        'blue plastic stool', 'red plastic stool', 'small red plastic stool', 'small blue plastic stool', 'round plastic stool',
    ],
    6: ['street food cart', 'wheeled food vending cart', 'mobile food vending cart', 'fruit food stand'],
    7: ['shop sign', 'advertising signboard', 'hanging sign', 'store signboard'],
    8: ['utility pole', 'electric pole'],
    9: ['tree trunk', 'tree stem'],
    10: ['basket of fruit', 'crate of vegetables', 'packaged products', 'fruit crate', 'stacked merchandise'],
}

COUNTER_PHRASES: Dict[str, str] = {
    # Gates and fences (prevent mistaking gates/storefronts as vending carts)
    'folding gate': 'barrier_gate',
    'metal gate': 'barrier_gate',
    'security gate': 'barrier_gate',
    'iron fence': 'fence',
    # Canopies and awnings (prevent mistaking canopies as shop signs)
    'storefront canopy': 'canopy',
    'fabric awning': 'canopy',
    # Walls and columns (prevent mistaking building columns as poles/carts)
    'concrete wall': 'wall',
    'building column': 'column',
    # Baby strollers (prevent mistaking strollers as motorcycles or carts)
    'baby stroller': 'stroller',
    'pushchair': 'stroller',
    # Vehicles and transit
    'bicycle': 'bicycle',
    'truck': 'car',
    'van': 'car',
    'traffic sign': 'traffic_sign',
    'barrier gate': 'barrier_gate',
    # Street objects
    'potted plant': 'potted_plant',
    'pile of bricks': 'construction_materials',
    'sand pile': 'construction_materials',
    'styrofoam box': 'foam_box',
}


def resolve_yoloe_weights(weights: str) -> str:
    """Resolve an explicit weight path without substituting a different checkpoint."""
    p = Path(weights)
    if p.is_file():
        return str(p)

    workspace_root = Path(__file__).resolve().parent.parent
    if (workspace_root / p).is_file():
        return str(workspace_root / p)

    if p.parent != Path('.'):
        # An explicitly selected experiment checkpoint must not silently fall
        # back to the stock model in the workspace root.
        raise FileNotFoundError(f'YOLOE checkpoint not found: {weights}')

    return str(weights)


class YOLOEDetector:
    """Open-vocabulary YOLOE with box outputs and bounded ROI batches."""

    def __init__(
        self,
        weights: Optional[str] = None,
        device: Optional[str] = None,
        imgsz: Optional[int] = None,
        config_path: Optional[Union[str, Path]] = None,
        precision: Optional[str] = None,
        roi_focus: Optional[bool] = None,
        roi_patch_size: Optional[int] = None,
        roi_max_patches: Optional[int] = None,
    ):
        self.config = self._load_config(config_path)
        model_cfg = self.config.get('model', {})
        weights = weights if weights is not None else model_cfg.get('weights', 'lan_chiem/models/yoloe-v8s-single.pt')
        device = device if device is not None else model_cfg.get('device', 'auto')
        imgsz = imgsz if imgsz is not None else model_cfg.get('imgsz', 960)
        precision = precision if precision is not None else model_cfg.get('precision', 'fp16')
        if imgsz < 320 or imgsz % 32:
            raise ValueError('YOLOE imgsz must be >=320 and divisible by 32')

        self.device = self._resolve_device(device)
        self.weights = resolve_yoloe_weights(weights)
        self.imgsz = imgsz
        if precision not in ('fp16', 'fp32'):
            raise ValueError('precision must be fp16 or fp32')
        self.precision = precision if self.device == 'cuda' else 'fp32'
        self.revision = None
        self.engine = None
        self.model = None
        self._predict_kwargs = {}
        self.trunk_refiner = None
        self.text_embeddings_metadata = {'status':'not_loaded'}
        self.reparameterized = False
        if model_cfg.get('trunk_refiner',{}).get('enabled',False):
            warnings.warn('YOLOE now runs as a single detector: the legacy DINO trunk refiner is disabled.',UserWarning)
        model_cfg = {**model_cfg,'trunk_refiner':{'enabled':False}}

        # Build vocabulary: (phrase, canonical_id, counter_group)
        targets = model_cfg.get('target_phrases', TARGET_PHRASES)
        counters = model_cfg.get('counter_phrases', COUNTER_PHRASES)
        self.vocabulary = self.build_vocabulary(targets, counters)
        self.vocabulary_sha256 = hashlib.sha256(json.dumps(self.vocabulary).encode()).hexdigest()

        # Load config (thresholds, review rules)
        self.roi_focus = model_cfg.get('roi_focus', True) if roi_focus is None else roi_focus
        self.roi_patch_size = model_cfg.get('roi_patch_size', 640) if roi_patch_size is None else roi_patch_size
        self.roi_max_patches = model_cfg.get('roi_max_patches', 2) if roi_max_patches is None else roi_max_patches
        if self.roi_patch_size < 320 or self.roi_patch_size % 32:
            raise ValueError('roi_patch_size must be >=320 and divisible by 32')
        if not isinstance(self.roi_max_patches, int) or self.roi_max_patches < 0:
            raise ValueError('roi_max_patches must be a nonnegative integer')
        if model_cfg.get('roi_strategy', 'seeded') not in ('seeded', 'grid'):
            raise ValueError('roi_strategy must be seeded or grid')
        overlap = model_cfg.get('roi_overlap', .2)
        if not isinstance(overlap, (float, int)) or not np.isfinite(overlap) or not 0 <= overlap < 1:
            raise ValueError('roi_overlap must be finite and in [0, 1)')
        batch_size = model_cfg.get('roi_batch_size', 2)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('roi_batch_size must be a positive integer')
        if self.config.get('review_rules', {}).get('same_head_counter_policy', 'strict') not in ('strict', 'stronger'):
            raise ValueError('same_head_counter_policy must be strict or stronger')

        self.config['model'] = {**model_cfg,
            'backend': 'yoloe',
            'model_id': self.weights,
            'imgsz': self.imgsz,
            'precision': self.precision,
            'vocabulary': self.vocabulary,
            'vocabulary_sha256': self.vocabulary_sha256,
            'quality_verified': False,
            'roi_focus': self.roi_focus,
            'roi_patch_size': self.roi_patch_size,
            'roi_max_patches': self.roi_max_patches,
        }

    @staticmethod
    def build_vocabulary(targets, counters):
        if not isinstance(targets, dict) or set(targets) != set(ClassRegistry.all_ids()):
            raise ValueError('YOLOE target_phrases must declare all canonical IDs 1..10')
        vocabulary, phrases_seen = [], set()
        for cid, phrases in targets.items():
            if not isinstance(phrases, list) or not phrases:
                raise ValueError(f'Class {cid} must have a nonempty list of phrases')
            for phrase in phrases:
                if not isinstance(phrase, str) or not phrase.strip():
                    raise ValueError('Vocabulary phrases must be nonempty strings')
                phrase = phrase.strip().lower()
                if phrase in phrases_seen:
                    raise ValueError(f'Duplicate/ambiguous vocabulary phrase: {phrase}')
                phrases_seen.add(phrase)
                vocabulary.append((phrase, cid, None))
        if not isinstance(counters, dict):
            raise ValueError('counter_phrases must map phrases to counter groups')
        for phrase, group in counters.items():
            if not isinstance(phrase, str) or not phrase.strip() or not isinstance(group, str) or not group.strip():
                raise ValueError('Counter phrases/groups must be nonempty strings')
            phrase = phrase.strip().lower()
            if phrase in phrases_seen:
                raise ValueError(f'Duplicate/ambiguous vocabulary phrase: {phrase}')
            phrases_seen.add(phrase)
            vocabulary.append((phrase, None, group.strip()))
        return vocabulary

    @staticmethod
    def _resolve_device(requested: Optional[str] = None) -> str:
        req = (requested or 'auto').lower().strip()
        if req in ('cuda', 'gpu'):
            if torch.cuda.is_available():
                return 'cuda'
            return 'cpu'
        elif req == 'cpu':
            return 'cpu'
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def device_info(self) -> str:
        if self.device == 'cuda' and torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                return f"GPU / CUDA ({gpu_name}) [FP16: {self.precision == 'fp16'}]"
            except Exception:
                return "GPU / CUDA"
        return "CPU"

    def _load_config(self, config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        cfg_file = Path(config_path) if config_path else Path(__file__).parent/'configs/object_yoloe.yaml'
        config = load_config(cfg_file)
        if config.get('catalog_version') != CATALOG_VERSION:
            raise ValueError('YOLOE config catalog_version does not match the catalogue')
        return config

    def load_model(self) -> None:
        """Load YOLOE model and compute pre-calculated text prompt embeddings."""
        if self.engine is not None:
            return

        from ultralytics import YOLOE

        resolved = Path(self.weights)
        canonical = Path(__file__).parent/'models/yoloe-v8s-single.pt'
        if resolved.resolve() == canonical.resolve():
            checksum = canonical.with_suffix('.sha256')
            expected = checksum.read_text(encoding='ascii').split()[0]
            if file_hash(resolved) != expected:
                raise ValueError('Production YOLOE checkpoint checksum mismatch')
        print(f"[*] Đang nạp mô hình YOLOE: {resolved.name} trên {self.device_info} (imgsz={self.imgsz})...")
        engine = YOLOE(str(resolved))
        engine.to(self.device)

        # Set vocabulary classes and precalculate CLIP text positional embeddings
        names = [phrase for phrase, _, _ in self.vocabulary]
        checkpoint = Path(getattr(engine, 'ckpt_path', None) or self.weights)
        self.revision = file_hash(checkpoint) if checkpoint.is_file() else None
        head = engine.model.model[-1] if isinstance(engine.model,torch.nn.Module) else None
        fixed_head = head is not None and getattr(head,'is_fused',False) is True
        self.reparameterized = fixed_head
        if fixed_head:
            stored_names = list(engine.model.names.values()) if isinstance(engine.model.names,dict) else list(engine.model.names)
            if stored_names != names or getattr(engine.model,'single_vocab_sha256',self.vocabulary_sha256) != self.vocabulary_sha256:
                raise ValueError('Fixed-vocabulary YOLOE checkpoint does not match configured phrases/classes')
            text_pe = getattr(engine.model,'pe',None)
            self.text_embeddings_metadata = {'status':'fixed_checkpoint'}
            self.config['model']['reparameterize'] = True
        else:
            from lan_chiem.embedding_cache import text_embeddings
            text_pe,self.text_embeddings_metadata = text_embeddings(engine,names,self.device,self.revision,
                self.vocabulary_sha256,self.provenance()['versions'],
                enabled=self.config['model'].get('cache_text_embeddings',False) and isinstance(engine.model,torch.nn.Module))
            engine.set_classes(names, text_pe)
        engine.model.eval()
        if self.config['model'].get('reparameterize', False) and not fixed_head and head is not None:
            head = engine.model.model[-1]
            if not hasattr(head, 'is_fused') or not callable(getattr(head, 'fuse', None)):
                raise RuntimeError('This YOLOE head does not support vocabulary reparameterization')
            # Upstream YOLOEDetect/YOLOESegment API. Keep segmentation outputs
            # and all vocabulary channels; only fold constant text projections.
            head.fuse(text_pe)
            self.reparameterized = True
            # Ultralytics creates fused parameters under inference_mode. Some
            # PyTorch backends need normal tensors with version counters when
            # they subsequently prepare/cache these convolutions.
            for module in head.modules():
                for name, parameter in list(module._parameters.items()):
                    if parameter is not None and torch.is_inference(parameter):
                        module._parameters[name] = torch.nn.Parameter(parameter.clone(), requires_grad=False)
                for name, buffer in list(module._buffers.items()):
                    if buffer is not None and torch.is_inference(buffer):
                        module._buffers[name] = buffer.clone()
        if getattr(engine, 'task', None) == 'segment':
            # Keep task=segment so NMS knows the number of class channels, but
            # skip mask reconstruction/upsampling in SegmentationPredictor.
            from lan_chiem.yoloe_predictor import BoxesOnlySegmentationPredictor
            self._predict_kwargs = {'predictor': BoxesOnlySegmentationPredictor}

        self.engine = engine
        self.model = engine.model
        self._provenance = self.provenance()
        print(f"[+] Nạp thành công YOLOE ({len(names)} từ vựng, {self.precision.upper()})!")

    def provenance(self) -> Dict[str, Any]:
        from importlib.metadata import version, PackageNotFoundError
        if not hasattr(self, '_runtime_versions'):
            versions = {'torch': torch.__version__}
            for package in ('ultralytics', 'torchvision'):
                try:
                    versions[package] = version(package)
                except PackageNotFoundError:
                    versions[package] = 'not installed'
            self._runtime_versions = versions
        return {
            'model_id': self.weights,
            'model_revision': self.revision,
            'catalog_version': CATALOG_VERSION,
            'backend': 'yoloe',
            'task': 'detect',
            'imgsz': self.imgsz,
            'precision': self.precision,
            'vocabulary_sha256': self.vocabulary_sha256,
            'reparameterized': self.reparameterized,
            'trunk_refiner': None,
            'detector_count': 1,
            'text_embeddings': dict(self.text_embeddings_metadata),
            'versions': dict(self._runtime_versions),
        }

    @staticmethod
    def preprocess_image(image_input: Union[str, Path, Image.Image, np.ndarray]) -> Tuple[Image.Image, str, Optional[str]]:
        """Tiền xử lý ảnh: chuẩn hóa sang RGB, kiểm tra kích thước, lấy hash."""
        image_name = "in_memory_image"
        img_hash = None
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

    @staticmethod
    def extract_roi_clusters(
        seeds: List[np.ndarray],
        img_w: int,
        img_h: int,
        patch_size: int = 640,
        max_patches: int = 2,
        margin: int = 80,
    ) -> List[Tuple[int, int, int, int]]:
        """Group small object candidate detections into high-resolution RoI patches."""
        if not seeds:
            return []

        clusters: List[Dict[str, Any]] = []
        for s in seeds:
            box = s[:4]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            matched = False
            for cl in clusters:
                if np.hypot(cx - cl['cx'], cy - cl['cy']) < 450:
                    cl['boxes'].append(box)
                    cl['cx'] = np.mean([(b[0] + b[2]) / 2 for b in cl['boxes']])
                    cl['cy'] = np.mean([(b[1] + b[3]) / 2 for b in cl['boxes']])
                    matched = True
                    break
            if not matched:
                clusters.append({'cx': cx, 'cy': cy, 'boxes': [box]})

        clusters.sort(key=lambda c: len(c['boxes']), reverse=True)

        roi_boxes: List[Tuple[int, int, int, int]] = []
        for cl in clusters[:max_patches]:
            boxes = np.array(cl['boxes'])
            x1 = max(0, int(boxes[:, 0].min() - margin))
            y1 = max(0, int(boxes[:, 1].min() - margin))
            x2 = min(img_w, int(boxes[:, 2].max() + margin))
            y2 = min(img_h, int(boxes[:, 3].max() + margin))

            min_dim = int(patch_size * 0.8)
            pw, ph = x2 - x1, y2 - y1
            if pw < min_dim:
                diff = min_dim - pw
                x1 = max(0, x1 - diff // 2)
                x2 = min(img_w, x1 + min_dim)
                x1 = max(0, x2 - min_dim)
            if ph < min_dim:
                diff = min_dim - ph
                y1 = max(0, y1 - diff // 2)
                y2 = min(img_h, y1 + min_dim)
                y1 = max(0, y2 - min_dim)

            if x2 > x1 and y2 > y1:
                roi_boxes.append((x1, y1, x2, y2))

        return roi_boxes

    def decode(
        self,
        data: np.ndarray,
        img_w: int = 1920,
        img_h: int = 1080,
        threshold_override: Optional[float] = None,
        nms_iou: float = .45,
        enable_counters: bool = True,
        sources: Optional[List[str]] = None,
        roi_regions: Optional[Dict[str, Tuple[int,int,int,int]]] = None,
        surface_image=None,
    ) -> Tuple[List[Detection], List[Detection], List[Dict[str, Any]]]:
        """Convert Nx6 YOLO output [x1, y1, x2, y2, conf, cls_idx] and partition using domain rules."""
        detections: List[Detection] = []
        counters: Dict[str, List[Dict[str, Any]]] = {}
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[1] < 6:
            raise RuntimeError('Expected Nx6 YOLO detection output')
        if sources is not None and len(sources) != len(data):
            raise RuntimeError('Detection source count does not match YOLO rows')

        for index, row in enumerate(data):
            if len(row) < 6 or not np.isfinite(row[:6]).all():
                raise RuntimeError('Non-finite/invalid YOLO output; try fp32 and inspect the model')
            *box, score, vocabulary_id = row[:6]
            if not 0 <= score <= 1 or vocabulary_id != int(vocabulary_id):
                raise RuntimeError('Invalid YOLO confidence or fractional vocabulary ID')
            vocabulary_id = int(vocabulary_id)
            if not 0 <= vocabulary_id < len(self.vocabulary):
                raise RuntimeError(f'YOLO vocabulary ID {vocabulary_id} does not match the configured catalogue')

            phrase, cid, group = self.vocabulary[vocabulary_id]
            box = [float(np.clip(box[0], 0, img_w)), float(np.clip(box[1], 0, img_h)),
                   float(np.clip(box[2], 0, img_w)), float(np.clip(box[3], 0, img_h))]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            # Counter phrases (negative classes to prevent false positives)
            if cid is None:
                if enable_counters and score >= 0.25:
                    counters.setdefault(group, []).append({
                        'id': f'yoloe_counter_{index}',
                        'score': float(score),
                        'bbox_xyxy': list(map(float, box)),
                        'raw_label': phrase,
                        'counter_group': group,
                        'origin': 'shared_yoloe_head',
                        'source': sources[index] if sources is not None else 'full',
                    })
                continue

            # Target classes: use class-specific thresholds
            threshold = 0.25
            if 'classes' in self.config and cid in self.config['classes']:
                threshold = self.config['classes'][cid].get('box_threshold', 0.25)
            if threshold_override is not None:
                threshold = threshold_override
            if score < threshold:
                continue

            meta = ClassRegistry.get(cid)
            detections.append(Detection(
                id=f'yoloe_{index}',
                class_id=cid,
                class_name=meta.name if meta else phrase,
                class_name_vi=meta.display_name if meta else phrase,
                raw_label=phrase,
                prompt_group='yoloe_vocabulary',
                source=sources[index] if sources is not None else 'full',
                score=float(score),
                bbox_xyxy=list(map(float, box)),
            ))

        # 1. Intra-class NMS to remove redundant boxes of same class
        detections = reconcile_roi_fragments(detections,roi_regions or {},(img_w,img_h))
        detections = apply_intra_class_nms(detections, iou_threshold=nms_iou)

        # Different classes may legitimately overlap (rider/motorbike, goods/cart).
        # Preserve these hypotheses for conflict review instead of deleting them.

        # 3. Domain spatial conflict resolution
        predicted, review, conflicts = analyze_conflicts_and_partition(
            detections, counters, review_rules=self.config.get('review_rules', {}),
            surface_image=surface_image,
        )
        return predicted, review, conflicts

    def predict_video_frame(self, image_input: Any, color_order: str = 'RGB', **kwargs) -> Dict[str, Any]:
        """Skip hashing; NumPy frames default to RGB, matching predict().

        OpenCV callers can pass color_order='BGR' to avoid RGB/PIL round trips.
        Ultralytics accepts BGR arrays and RGB PIL images.
        """
        if color_order not in ('RGB', 'BGR'):
            raise ValueError('color_order must be RGB or BGR')
        if isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8 or image_input.ndim != 3 or image_input.shape[2] != 3:
                raise ValueError('Video NumPy input must be an HxWx3 uint8 image')
            if not image_input.shape[0] or not image_input.shape[1]:
                raise ValueError('Video frame dimensions must be positive')
            if color_order == 'RGB':
                image_input = np.ascontiguousarray(image_input[..., ::-1])
        elif isinstance(image_input, Image.Image):
            image_input = image_input.convert('RGB')
        else:
            raise TypeError('Video frames must be RGB PIL images or uint8 NumPy arrays')
        return self.predict(image_input, _video_frame=True, **kwargs)

    def _predict_boxes(self, **kwargs):
        predictor = getattr(self.engine, 'predictor', None)
        backend = getattr(predictor, 'model', None)
        actual_half = getattr(backend, 'fp16', None)
        if isinstance(actual_half, bool) and actual_half != bool(kwargs.get('half', False)):
            # Updating predictor.args.half alone does not recast an existing
            # Ultralytics backend. Rebuild it so FP32 audits actually use FP32.
            self.engine.predictor = None
        if self._predict_kwargs:
            # YOLOE.predict consumes `predictor` for visual prompts and does not
            # forward it for text-only inference in several Ultralytics versions.
            from ultralytics.engine.model import Model
            return Model.predict(self.engine, **self._predict_kwargs, **kwargs)
        return self.engine.predict(**kwargs)

    def predict(self, image_input: Any, **kwargs) -> Dict[str, Any]:
        """Global inference followed by optional batched ROI refinements."""
        for opt in ('text_threshold_override',
                    'tile_size', 'overlap_ratio', 'query_mode', 'single_prompt'):
            if kwargs.get(opt) is not None:
                raise ValueError(f'YOLOE detector does not support DINO parameter: {opt}')

        if kwargs.get('mode') not in (None, 'full'):
            raise ValueError('YOLOE supports mode=full with optional ROI; DINO tiling modes are unsupported')
        if kwargs.get('target_groups'):
            raise ValueError('YOLOE uses target_classes, not DINO prompt groups')
        selected = kwargs.get('target_classes')
        if selected is not None and any(cid not in ClassRegistry.all_ids() for cid in selected):
            raise ValueError('target_classes must contain catalogue IDs 1..10')
        threshold_override = kwargs.get('box_threshold_override')
        if threshold_override is None:
            threshold_override = kwargs.get('preset')
        nms_iou = kwargs.get('nms_iou')
        nms_iou = self.config.get('inference', {}).get('nms_iou', .45) if nms_iou is None else nms_iou
        for value in (threshold_override, nms_iou):
            if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('threshold and nms_iou must be finite values in [0, 1]')
        nms_iou = float(nms_iou)
        threshold_override = float(threshold_override) if threshold_override is not None else None

        self.load_model()
        use_roi = kwargs.get('roi_focus', self.roi_focus)
        roi_budget = kwargs.get('roi_patch_budget')
        roi_budget = self.roi_max_patches if roi_budget is None else roi_budget
        if not isinstance(roi_budget,int) or roi_budget < 0:
            raise ValueError('roi_patch_budget must be a nonnegative integer')
        roi_budget = min(roi_budget,self.roi_max_patches)
        raw_conf = min(.12 if use_roi else .15,
                       threshold_override if threshold_override is not None else
                       min((c.get('box_threshold', .25) for c in self.config.get('classes', {}).values()), default=.25))
        if kwargs.get('_video_frame'):
            image = image_input
            name = 'video_frame'
            digest = None
            w, h = getattr(image, 'width', 0), getattr(image, 'height', 0)
            if w == 0 and hasattr(image, 'shape'):
                h, w = image.shape[:2]
        else:
            image, name, digest = self.preprocess_image(image_input)
            w, h = image.width, image.height

        use_half = (self.precision == 'fp16' and self.device == 'cuda')
        if use_half:
            torch.cuda.synchronize()
        started = time.perf_counter()
        model_stages = Counter()
        def record_model_stages(result):
            for key, milliseconds in (getattr(result, 'speed', None) or {}).items():
                if isinstance(milliseconds, (int, float)):
                    model_stages[key] += milliseconds / 1000.

        # Step 1: Global pass
        results = self._predict_boxes(
            source=image,
            device=self.device,
            imgsz=self.imgsz,
            half=use_half,
            conf=raw_conf,
            iou=nms_iou,
            max_det=300,
            agnostic_nms=False,
            verbose=False,
            save=False,
        )[0]
        record_model_stages(results)

        data_global = results.boxes.data.detach().float().cpu().numpy()
        all_data = [data_global]
        sources = ['full'] * len(data_global)
        roi_patches_ran = 0
        roi_model_calls = 0
        valid_roi_boxes = []

        # Step 2: High-Res RoI Focus (Adaptive Local Passes for Sidewalk & Small Objects)
        if use_roi and roi_budget and w > 0 and h > 0:
            seeds = []
            for row in data_global:
                vocab_id = int(row[5])
                if 0 <= vocab_id < len(self.vocabulary):
                    _, cid, _ = self.vocabulary[vocab_id]
                    # A local pass helps only when the crop enlarges the object.
                    # Hạt giống ROI cho các vật thể nhỏ/xa: bàn, ghế, xe đẩy, biển QC, cột điện, thân cây, hàng hóa
                    if cid in (4, 5, 6, 7, 8, 9, 10) and max(row[2]-row[0], row[3]-row[1]) < self.roi_patch_size * .8:
                        seeds.append(row)

            # Support external sidewalk polygons if provided
            sidewalk_regions = kwargs.get('sidewalk_regions') or self.config.get('sidewalk_regions', [])
            for r in sidewalk_regions:
                poly = getattr(r, 'polygon', r.get('polygon', []) if isinstance(r, dict) else [])
                if len(poly) >= 3:
                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    seeds.append(np.array([min(xs), min(ys), max(xs), max(ys), 0.5, 4]))

            roi_boxes = self.extract_roi_clusters(
                seeds, w, h, patch_size=self.roi_patch_size, max_patches=roi_budget
            )
            strategy = self.config['model'].get('roi_strategy', 'seeded')
            if strategy not in ('seeded', 'grid'):
                raise ValueError('roi_strategy must be seeded or grid')
            if strategy == 'grid':
                from lan_chiem.roi import grid_regions, select_regions
                roi_boxes = select_regions(grid_regions(w, h, self.roi_patch_size,
                    self.config['model'].get('roi_overlap', .2)), roi_budget,
                    kwargs.get('roi_offset', 0))

            valid_crops = []
            valid_roi_boxes = []
            for rx1, ry1, rx2, ry2 in roi_boxes:
                if self.roi_patch_size / max(rx2-rx1, ry2-ry1) <= self.imgsz / max(w, h):
                    continue  # This crop cannot add spatial resolution.
                if isinstance(image, Image.Image):
                    crop = image.crop((rx1, ry1, rx2, ry2))
                elif isinstance(image, np.ndarray):
                    crop = image[ry1:ry2, rx1:rx2]
                else:
                    continue
                valid_crops.append(crop)
                valid_roi_boxes.append((rx1, ry1, rx2, ry2))

            if valid_crops:
                batch_size = self.config['model'].get('roi_batch_size', 2)
                if not isinstance(batch_size, int) or batch_size < 1:
                    raise ValueError('roi_batch_size must be a positive integer')
                batch_res = []
                for batch_start in range(0, len(valid_crops), batch_size):
                    batch_res.extend(self._predict_boxes(
                    source=valid_crops[batch_start:batch_start+batch_size],
                    device=self.device,
                    imgsz=self.roi_patch_size,
                    half=use_half,
                    conf=raw_conf,
                    iou=nms_iou,
                    max_det=300,
                    agnostic_nms=False,
                    verbose=False,
                    save=False,
                    ))
                    roi_model_calls += 1
                if len(batch_res) != len(valid_crops):
                    raise RuntimeError('ROI batch result count does not match the requested crops')
                for (rx1, ry1, rx2, ry2), res_crop in zip(valid_roi_boxes, batch_res):
                    record_model_stages(res_crop)
                    data_crop = res_crop.boxes.data.detach().float().cpu().numpy()
                    roi_patches_ran += 1
                    if len(data_crop) > 0:
                        # Crop context: keep small objects and counter-evidence (gates, awnings, foam boxes)
                        # so spatial conflict analysis can suppress false positives in RoI patches.
                        keep_mask = [
                            (self.vocabulary[int(row[5])][1] in (ClassRegistry.all_ids() if strategy == 'grid' else (4, 5, 6, 7, 10)) or
                             self.vocabulary[int(row[5])][2] is not None)
                            for row in data_crop
                        ]
                        data_crop = data_crop[keep_mask].copy()
                        # Artificial crop edges can turn parts of a cart/sign
                        # into confident small-object boxes. Keep full-image
                        # edge objects, but don't promote truncated ROI objects.
                        if len(data_crop):
                            valid = np.ones(len(data_crop), dtype=bool)
                            if rx1 > 0:
                                valid &= data_crop[:, 0] > 2
                            if ry1 > 0:
                                valid &= data_crop[:, 1] > 2
                            if rx2 < w:
                                valid &= data_crop[:, 2] < rx2-rx1-2
                            if ry2 < h:
                                valid &= data_crop[:, 3] < ry2-ry1-2
                            data_crop = data_crop[valid]
                        if len(data_crop) > 0:
                            data_crop[:, [0, 2]] += rx1
                            data_crop[:, [1, 3]] += ry1
                            all_data.append(data_crop)
                            sources.extend([f'roi_{rx1}_{ry1}'] * len(data_crop))

        semantic_started = time.perf_counter()
        combined_data = np.concatenate(all_data, axis=0) if len(all_data) > 1 else data_global
        enable_counters = kwargs.get('enable_counter_checks')
        if enable_counters is None:
            enable_counters = self.config.get('inference', {}).get('enable_counter_checks', True)
        predicted, review, conflicts = self.decode(combined_data, img_w=w, img_h=h,
                                                  threshold_override=threshold_override, nms_iou=nms_iou,
                                                  enable_counters=enable_counters, sources=sources,
                                                  roi_regions={f'roi_{r[0]}_{r[1]}':r for r in valid_roi_boxes},
                                                  surface_image=image)
        observed_classes = list(ClassRegistry.all_ids())

        # Filter by requested target classes if specified
        selected = kwargs.get('target_classes')
        if selected is not None:
            predicted = [d for d in predicted if d.class_id in selected]
            review = [d for d in review if d.class_id in selected]

        inference_time = time.perf_counter() - started

        return {
            'catalog_version': CATALOG_VERSION,
            'backend': 'yoloe',
            'model_id': self.weights,
            'image_name': name,
            'image_sha256': digest,
            'image_size': {'width': w, 'height': h},
            'device': self.device,
            'inference_mode': 'full_roi' if roi_patches_ran else 'full',
            'query_mode': 'yoloe_vocabulary',
            'inference_time_seconds': inference_time,
            'model_stage_seconds': dict(model_stages),
            'semantic_postprocess_seconds': time.perf_counter() - semantic_started,
            'model_calls': 1 + roi_model_calls,
            'images_evaluated': 1 + roi_patches_ran,
            'detector_count': 1,
            'trunk_refiner_ran': False,
            'trunk_refiner_evidence': None,
            'observed_class_ids': observed_classes if selected is None else [cid for cid in observed_classes if cid in selected],
            'roi_patches_evaluated': roi_patches_ran,
            'roi_patch_budget': roi_budget,
            'roi_regions': [list(region) for region in valid_roi_boxes],
            'quality_verified': False,
            'inference_settings': {'imgsz': self.imgsz, 'nms_iou': nms_iou,
                                   'raw_confidence': raw_conf, 'threshold_override': threshold_override,
                                   'roi_enabled': bool(use_roi), 'counter_checks': bool(enable_counters)},
            'provenance': self.provenance(),
            'object_detections': [d.to_dict() for d in predicted],
            'review_candidates': [d.to_dict() for d in review],
            'conflict_reports': conflicts,
            'summary': {
                'total_predicted_detections': len(predicted),
                'total_review_candidates': len(review),
                'total_conflicts_detected': len(conflicts),
                'predicted_class_counts': dict(Counter(str(d.class_id) for d in predicted)),
                'review_class_counts': dict(Counter(str(d.class_id) for d in review)),
            },
            'sidewalk_regions': [r.to_dict() if hasattr(r, 'to_dict') else r
                                 for r in (kwargs.get('sidewalk_regions') or [])],
            '_image_pil': image if isinstance(image, Image.Image) else None,
        }
