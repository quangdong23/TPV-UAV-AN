"""Sidewalk segmentation for UAV imagery."""

from road_sidewalk_seg.core.config import PipelineConfig

__all__ = ["PipelineConfig", "SidewalkPipeline"]


def __getattr__(name: str):
    if name == "SidewalkPipeline":
        from road_sidewalk_seg.core.pipeline import SidewalkPipeline

        return SidewalkPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
