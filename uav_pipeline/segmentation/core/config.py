from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uav_pipeline.paths import CACHE_DIR, project_path
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    name: str
    model_id: str
    backend: str
    taxonomy: str = "auto"
    weight: float = 1.0
    enabled: bool = True


@dataclass(slots=True)
class TilingConfig:
    tile_size: int = 768
    overlap: float = 0.25
    weight_floor: float = 0.05
    max_image_side: int = 0
    strategy: str = "tiled"
    whole_image_max_pixels: int = 1048576


@dataclass(slots=True)
class PreprocessingConfig:
    low_light_mode: str = "auto"
    low_light_threshold: float = 0.28
    clahe_clip_limit: float = 1.5
    clahe_grid_size: int = 8


@dataclass(slots=True)
class DecisionConfig:
    min_confidence: float = 0.55
    min_margin: float = 0.12
    max_normalized_entropy: float = 0.85
    unknown_id: int = 255


@dataclass(slots=True)
class PolygonConfig:
    min_area_pixels: float = 500.0
    min_hole_area_pixels: float = 100.0
    min_area_ratio: float = 0.0
    min_hole_area_ratio: float = 0.0
    simplify_epsilon_ratio: float = 0.002
    close_kernel_size: int = 0
    draw_vertices: bool = True
    draw_hole_vertices: bool = False


@dataclass(slots=True)
class CompletionConfig:
    enabled: bool = True
    morphology_backend: str = 'opencv'
    min_component_area_ratio: float = 0.0005
    min_component_relative_area: float = 0.1
    min_component_aspect_ratio: float = 4.5
    closing_kernel_ratio: float = 0.08
    max_closing_kernel_size: int = 121
    fill_holes: bool = True
    max_hole_area_ratio: float = 1.0
    unknown_expansion_ratio: float = 0.0
    max_closing_growth_ratio: float = 1.5
    envelope_enabled: bool = True
    concave_hull_ratio: float = 0.4
    max_envelope_growth_ratio: float = 1.6


@dataclass(slots=True)
class OutputConfig:
    save_scores: bool = False
    overlay_alpha: float = 0.45
    show_unknown: bool = False
    target_crs: str | None = None
    png_compress_level: int = 1


def _default_models() -> list[ModelConfig]:
    return [
        ModelConfig(
            name="mask2former_mapillary",
            model_id="facebook/mask2former-swin-large-mapillary-vistas-semantic",
            backend="mask2former",
            taxonomy="mapillary_vistas",
        )
    ]


@dataclass(slots=True)
class PipelineConfig:
    models: list[ModelConfig] = field(default_factory=_default_models)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    polygon: PolygonConfig = field(default_factory=PolygonConfig)
    completion: CompletionConfig = field(default_factory=CompletionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    device: str = "auto"
    precision: str = "fp16"
    cache_dir: str = str(CACHE_DIR / "huggingface")
    cpu_threads: int = 0
    cpu_attention: str = "eager"

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        with project_path(path).open("r", encoding="utf-8") as file:
            raw = json.load(file)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PipelineConfig":
        values = dict(raw)
        values["cache_dir"] = str(project_path(values.get("cache_dir", str(CACHE_DIR/"huggingface"))))
        values["models"] = [ModelConfig(**item) for item in raw.get("models", _models_as_dict())]
        values["tiling"] = TilingConfig(**raw.get("tiling", {}))
        values["preprocessing"] = PreprocessingConfig(**raw.get("preprocessing", {}))
        values["decision"] = DecisionConfig(**raw.get("decision", {}))
        values["polygon"] = PolygonConfig(**raw.get("polygon", {}))
        values["completion"] = CompletionConfig(**raw.get("completion", {}))
        values["output"] = OutputConfig(**raw.get("output", {}))
        config = cls(**values)
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.cpu_attention not in {"eager", "sdpa"}:
            raise ValueError("cpu_attention must be eager or sdpa")
        if self.tiling.strategy not in {"tiled", "compact", "whole", "auto"}:
            raise ValueError("tiling.strategy must be tiled, compact, whole or auto")
        if (
            isinstance(self.tiling.whole_image_max_pixels, bool)
            or not isinstance(self.tiling.whole_image_max_pixels, int)
            or self.tiling.whole_image_max_pixels < 1024
        ):
            raise ValueError("tiling.whole_image_max_pixels must be an integer >= 1024")
        if (
            isinstance(self.output.png_compress_level, bool)
            or not isinstance(self.output.png_compress_level, int)
            or not 0 <= self.output.png_compress_level <= 9
        ):
            raise ValueError("output.png_compress_level must be an integer in [0, 9]")
        if isinstance(self.cpu_threads, bool) or not isinstance(self.cpu_threads, int) or self.cpu_threads < 0:
            raise ValueError("cpu_threads must be a nonnegative integer; 0 keeps PyTorch defaults")
        if not any(model.enabled for model in self.models):
            raise ValueError("At least one model must be enabled")
        if self.tiling.tile_size <= 0 or self.tiling.tile_size % 32 != 0:
            raise ValueError("tiling.tile_size must be a positive multiple of 32")
        if not 0.0 <= self.tiling.overlap < 0.9:
            raise ValueError("tiling.overlap must be in [0, 0.9)")
        if self.tiling.max_image_side < 0:
            raise ValueError("tiling.max_image_side must be zero or greater")
        if self.preprocessing.low_light_mode not in {"disabled", "auto", "always"}:
            raise ValueError("preprocessing.low_light_mode must be disabled, auto, or always")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if not 0.0 <= self.polygon.min_area_ratio < 1.0:
            raise ValueError("polygon.min_area_ratio must be in [0, 1)")
        if not 0.0 <= self.polygon.min_hole_area_ratio < 1.0:
            raise ValueError("polygon.min_hole_area_ratio must be in [0, 1)")
        if self.completion.morphology_backend not in ('opencv', 'rowwise'):
            raise ValueError('completion.morphology_backend must be opencv or rowwise')
        if not 0.0 <= self.completion.closing_kernel_ratio < 1.0:
            raise ValueError("completion.closing_kernel_ratio must be in [0, 1)")
        if not 0.0 <= self.completion.min_component_area_ratio < 1.0:
            raise ValueError("completion.min_component_area_ratio must be in [0, 1)")
        if not 0.0 <= self.completion.min_component_relative_area < 1.0:
            raise ValueError(
                "completion.min_component_relative_area must be in [0, 1)"
            )
        if self.completion.min_component_aspect_ratio < 1.0:
            raise ValueError(
                "completion.min_component_aspect_ratio must be at least 1"
            )
        if self.completion.max_closing_kernel_size < 0:
            raise ValueError("completion.max_closing_kernel_size must be zero or greater")
        if not 0.0 <= self.completion.max_hole_area_ratio <= 1.0:
            raise ValueError("completion.max_hole_area_ratio must be in [0, 1]")
        if not 0.0 <= self.completion.unknown_expansion_ratio < 1.0:
            raise ValueError("completion.unknown_expansion_ratio must be in [0, 1)")
        if self.completion.max_closing_growth_ratio < 1.0:
            raise ValueError("completion.max_closing_growth_ratio must be at least 1")
        if not 0.0 <= self.completion.concave_hull_ratio <= 1.0:
            raise ValueError("completion.concave_hull_ratio must be in [0, 1]")
        if self.completion.max_envelope_growth_ratio < 1.0:
            raise ValueError("completion.max_envelope_growth_ratio must be at least 1")
        for model in self.models:
            if model.backend != "mask2former":
                raise ValueError(f"Unsupported backend: {model.backend}")
            if model.weight <= 0:
                raise ValueError(f"Model weight must be positive: {model.name}")


def _models_as_dict() -> list[dict[str, Any]]:
    return [asdict(model) for model in _default_models()]
