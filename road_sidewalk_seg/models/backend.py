from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from transformers import (
    AutoImageProcessor,
    Mask2FormerImageProcessor,
    Mask2FormerForUniversalSegmentation,
)

from road_sidewalk_seg.core.config import ModelConfig
from road_sidewalk_seg.models.taxonomy import remap_probabilities, resolve_target_ids


def load_image_processor(model_config, cache_dir, local_files_only):
    """Drop obsolete fields that current Mask2Former already ignores.

    Keep size and do_reduce_labels unchanged so normalization is identical to
    the existing effective processor. Never modify the cached HF artifact.
    """
    options = dict(cache_dir=cache_dir, local_files_only=local_files_only)
    if model_config.backend == 'mask2former':
        config, _ = Mask2FormerImageProcessor.get_image_processor_dict(model_config.model_id, **options)
        config = {k: v for k, v in config.items() if k not in ('_max_size', 'reduce_labels')}
        return Mask2FormerImageProcessor.from_dict(config)
    return AutoImageProcessor.from_pretrained(model_config.model_id, use_fast=False, **options)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_taxonomy(model_config: ModelConfig) -> str:
    if model_config.taxonomy != "auto":
        return model_config.taxonomy
    model_id = model_config.model_id.casefold()
    if "mapillary" in model_id:
        return "mapillary_vistas"
    if "cityscapes" in model_id:
        return "cityscapes"
    if "ade" in model_id:
        return "ade20k"
    return "generic"


class SemanticBackend(ABC):
    def __init__(
        self,
        model_config: ModelConfig,
        device: str,
        precision: str,
        cache_dir: str | Path,
    ) -> None:
        self.model_config = model_config
        self.device = resolve_device(device)
        self.precision = precision if self.device.type == "cuda" else "fp32"
        self.cache_dir = str(cache_dir)
        self.taxonomy = resolve_taxonomy(model_config)
        try:
            self.processor = load_image_processor(model_config, self.cache_dir, True)
        except OSError:
            self.processor = load_image_processor(model_config, self.cache_dir, False)
        try:
            self.model = self._load_model(local_files_only=True)
        except OSError:
            self.model = self._load_model(local_files_only=False)
        self.model.to(self.device)
        self.model.eval()

    @abstractmethod
    def _load_model(self, local_files_only: bool) -> torch.nn.Module:
        raise NotImplementedError

    @abstractmethod
    def _source_probabilities(
        self,
        inputs: dict[str, torch.Tensor],
        target_height: int,
        target_width: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _autocast_context(self):
        if self.device.type != "cuda" or self.precision == "fp32":
            return nullcontext()
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _target_probabilities(
        self,
        inputs: dict[str, torch.Tensor],
        target_height: int,
        target_width: int,
    ) -> torch.Tensor:
        probabilities = self._source_probabilities(inputs, target_height, target_width)
        return remap_probabilities(
            probabilities.float(), self.model.config.id2label, self.taxonomy
        )

    def predict_target_scores(self, tile_rgb: np.ndarray) -> np.ndarray:
        target_height, target_width = tile_rgb.shape[:2]
        inputs = self.processor(
            images=tile_rgb,
            return_tensors="pt",
            do_resize=False,
        )
        tensor_inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        with torch.inference_mode(), self._autocast_context():
            target_scores = self._target_probabilities(
                tensor_inputs,
                target_height=target_height,
                target_width=target_width,
            )
        return target_scores[0].cpu().numpy().astype(np.float32, copy=False)


class Mask2FormerBackend(SemanticBackend):
    def _target_probabilities(
        self,
        inputs: dict[str, torch.Tensor],
        target_height: int,
        target_width: int,
    ) -> torch.Tensor:
        outputs = self.model(**inputs)
        class_probabilities = outputs.class_queries_logits.softmax(dim=-1)[..., :-1].float()
        groups = resolve_target_ids(self.model.config.id2label, self.taxonomy)
        class_count = class_probabilities.shape[-1]
        grouped_probabilities = []
        for group in groups:
            valid_ids = [index for index in group if 0 <= index < class_count]
            grouped_probabilities.append(class_probabilities[..., valid_ids].sum(dim=-1))
        grouped_probabilities = torch.stack(grouped_probabilities, dim=-1)
        semantic_scores = torch.einsum(
            "bqg,bqhw->bghw",
            grouped_probabilities,
            outputs.masks_queries_logits.sigmoid().float(),
        )
        semantic_scores = functional.interpolate(
            semantic_scores,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        return semantic_scores / semantic_scores.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _load_model(self, local_files_only: bool) -> torch.nn.Module:
        return Mask2FormerForUniversalSegmentation.from_pretrained(
            self.model_config.model_id,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
        )

    def _source_probabilities(
        self,
        inputs: dict[str, torch.Tensor],
        target_height: int,
        target_width: int,
    ) -> torch.Tensor:
        outputs = self.model(**inputs)
        class_probabilities = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
        mask_probabilities = outputs.masks_queries_logits.sigmoid()
        semantic_scores = torch.einsum(
            "bqc,bqhw->bchw",
            class_probabilities,
            mask_probabilities,
        )
        semantic_scores = functional.interpolate(
            semantic_scores,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        return semantic_scores / semantic_scores.sum(dim=1, keepdim=True).clamp_min(1e-6)


def create_backend(
    model_config: ModelConfig,
    device: str,
    precision: str,
    cache_dir: str | Path,
    cpu_attention: str = "eager",
) -> SemanticBackend:
    backend_types = {
        "mask2former": Mask2FormerBackend,
    }
    try:
        backend_type = backend_types[model_config.backend]
    except KeyError as error:
        raise ValueError(f"Unsupported backend: {model_config.backend}") from error
    backend = backend_type(model_config, device, precision, cache_dir)
    backend.cpu_attention = "eager"
    backend.cpu_attention_modules = 0
    if backend.device.type == "cpu" and cpu_attention == "sdpa":
        from road_sidewalk_seg.models.cpu_attention import configure_cpu_attention

        backend.cpu_attention_modules = configure_cpu_attention(backend.model, cpu_attention)
        if backend.cpu_attention_modules:
            backend.cpu_attention = cpu_attention
    return backend
