"""Sidewalk segmentation for UAV imagery."""

from uav_pipeline.segmentation.core.config import PipelineConfig

__all__ = ["PipelineConfig", "SidewalkPipeline"]


def __getattr__(name: str):
    if name == "SidewalkPipeline":
        from uav_pipeline.segmentation.core.pipeline import SidewalkPipeline

        return SidewalkPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
