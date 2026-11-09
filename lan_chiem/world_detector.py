"""Single-pass open-vocabulary video backend. Accuracy differs from Grounding DINO.

Ultralytics/CLIP are optional video dependencies, imported only when loading.
Canonical IDs remain 1..10; distractors are never exposed as target classes.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from lan_chiem.detector import Detection, GroundingDINODetector, apply_intra_class_nms
from lan_chiem.object_classes import CATALOG_VERSION
from lan_chiem.registry import ClassRegistry
from lan_chiem.video_io import file_hash


TARGET_PHRASES = {
    1: ['person'], 2: ['motorcycle', 'motor scooter'], 3: ['car'],
    4: ['plastic table'], 5: ['plastic chair', 'plastic stool'],
    6: ['street food cart', 'wheeled food vending cart', 'fruit food stand'],
    7: ['shop sign', 'advertising signboard', 'hanging sign'],
    8: ['utility pole'], 9: ['tree trunk'],
    10: ['basket of fruit', 'crate of vegetables', 'packaged products'],
}
COUNTER_PHRASES = {'bicycle': 'bicycle', 'truck': 'car', 'van': 'car',
                   'traffic sign': 'traffic_sign', 'barrier gate': 'barrier_gate',
                   'potted plant': 'potted_plant', 'pile of bricks': 'construction_materials',
                   'sand pile': 'construction_materials'}


class WorldDetector:
    def __init__(self, weights='yolov8s-worldv2.pt', device='cuda', imgsz=640, config_path=None):
        if imgsz < 320 or imgsz % 32:
            raise ValueError('World imgsz must be >=320 and divisible by 32')
        self.device = GroundingDINODetector._resolve_device(device)
        self.weights, self.imgsz = str(weights), imgsz
        self.policy = GroundingDINODetector(config_path=config_path, device=device)
        self.config = deepcopy(self.policy.config)
        self.vocabulary = [(phrase, cid, None) for cid, phrases in TARGET_PHRASES.items() for phrase in phrases]
        self.vocabulary += [(phrase, None, group) for phrase, group in COUNTER_PHRASES.items()]
        self.config['model'] = {'backend': 'yolo_world', 'model_id': self.weights, 'imgsz': imgsz,
                                'vocabulary': self.vocabulary, 'quality_verified': False}
        self.model = self.engine = None
        self.precision = 'fp32'
        self.revision = None

    def load_model(self):
        if self.engine is not None:
            return
        from ultralytics import YOLOWorld
        self.engine = YOLOWorld(self.weights)
        self.engine.to(self.device)
        self.engine.set_classes([phrase for phrase, _, _ in self.vocabulary])
        self.model = self.engine.model
        self.model.eval()
        checkpoint = Path(getattr(self.engine, 'ckpt_path', None) or self.weights)
        self.revision = file_hash(checkpoint) if checkpoint.is_file() else None
        embeddings = getattr(self.model, 'txt_feats', None)
        if isinstance(embeddings, torch.Tensor):
            self.config['model']['text_embedding_sha256'] = hashlib.sha256(
                embeddings.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()

    def provenance(self):
        from importlib.metadata import version, PackageNotFoundError
        versions = {'torch': torch.__version__}
        for package in ('ultralytics', 'transformers'):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = 'not installed'
        return {'model_id': self.weights, 'model_revision': self.revision,
                'catalog_version': CATALOG_VERSION, 'versions': versions}

    def decode(self, data):
        """Convert Nx6 YOLO output and use conservative same-frame conflict rules."""
        detections, counters = [], {}
        for index, row in enumerate(data):
            if len(row) < 6 or not np.isfinite(row[:6]).all():
                raise RuntimeError('Non-finite/invalid YOLO output; try fp32 and inspect the model')
            *box, score, vocabulary_id = row[:6]
            vocabulary_id = int(vocabulary_id)
            if not 0 <= vocabulary_id < len(self.vocabulary):
                raise RuntimeError('YOLO vocabulary ID does not match the configured catalogue')
            phrase, cid, group = self.vocabulary[vocabulary_id]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            if cid is None:
                if score >= .30:
                    counters.setdefault(group, []).append({'id': f'world_counter_{index}',
                        'score': float(score), 'bbox_xyxy': list(map(float, box)), 'raw_label': phrase,
                        'counter_group': group, 'origin': 'same_caption'})
                continue
            threshold = self.config['classes'][cid].get('box_threshold', .25)
            if score < threshold:
                continue
            meta = ClassRegistry.get(cid)
            detections.append(Detection(id=f'world_{index}', class_id=cid,
                class_name=meta.name, class_name_vi=meta.display_name, raw_label=phrase,
                prompt_group='world_vocabulary', source='full', score=float(score),
                bbox_xyxy=list(map(float, box))))
        detections = apply_intra_class_nms(detections, iou_threshold=.5)
        predicted, review, conflicts = self.policy.analyze_conflicts_and_partition(detections, counters)
        return predicted, review, conflicts

    def predict(self, image_input, **kwargs):
        self.load_model()
        if self.precision == 'fp16' and self.device != 'cuda':
            raise ValueError('FP16 requires CUDA')
        if self.device == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        result = self.engine.predict(source=image_input, device=self.device, imgsz=self.imgsz,
            half=self.precision == 'fp16', conf=.15, iou=.5, max_det=300,
            agnostic_nms=False, verbose=False, save=False)[0]
        data = result.boxes.data.detach().float().cpu().numpy()
        predicted, review, conflicts = self.decode(data)
        return {'catalog_version': CATALOG_VERSION, 'object_detections': [d.to_dict() for d in predicted],
                'review_candidates': [d.to_dict() for d in review], 'conflict_reports': conflicts,
                'model_calls': 1, 'inference_time_seconds': time.perf_counter()-started,
                'backend': 'yolo_world', 'quality_verified': False}
