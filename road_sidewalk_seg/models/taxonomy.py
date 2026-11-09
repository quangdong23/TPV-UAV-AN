from __future__ import annotations

from collections.abc import Mapping

import torch


TARGET_LABELS = {0: "other", 2: "sidewalk"}

MAPILLARY_DRIVABLE_SURFACE = {
    "road",
    "service lane",
    "bike lane",
    "parking",
    "crosswalk - plain",
    "lane marking - crosswalk",
    "lane marking - general",
}
MAPILLARY_SIDEWALK = {"sidewalk", "pedestrian area", "curb cut"}


def normalize_label(label: str) -> str:
    return " ".join(label.strip().casefold().replace("_", " ").split())


def resolve_target_ids(
    id2label: Mapping[int | str, str], taxonomy: str
) -> tuple[list[int], list[int], list[int]]:
    normalized = {int(class_id): normalize_label(label) for class_id, label in id2label.items()}
    taxonomy_name = taxonomy.casefold()

    if taxonomy_name == "mapillary_vistas":
        competing_surface_names = MAPILLARY_DRIVABLE_SURFACE
        sidewalk_names = MAPILLARY_SIDEWALK
    else:
        competing_surface_names = {"road"}
        sidewalk_names = {"sidewalk"}

    competing_surface_ids = [
        class_id
        for class_id, name in normalized.items()
        if name in competing_surface_names
    ]
    sidewalk_ids = [class_id for class_id, name in normalized.items() if name in sidewalk_names]
    target_ids = set(competing_surface_ids) | set(sidewalk_ids)
    other_ids = [class_id for class_id in normalized if class_id not in target_ids]

    if not sidewalk_ids:
        available = ", ".join(sorted(set(normalized.values())))
        raise ValueError(
            f"Taxonomy '{taxonomy}' did not resolve sidewalk. "
            f"Available labels: {available}"
        )
    return other_ids, competing_surface_ids, sidewalk_ids


def remap_probabilities(
    probabilities: torch.Tensor,
    id2label: Mapping[int | str, str],
    taxonomy: str,
) -> torch.Tensor:
    if probabilities.ndim != 4:
        raise ValueError("Expected probabilities with shape [B, C, H, W]")

    other_ids, competing_surface_ids, sidewalk_ids = resolve_target_ids(id2label, taxonomy)
    class_count = probabilities.shape[1]

    def valid_ids(class_ids: list[int]) -> list[int]:
        return [class_id for class_id in class_ids if 0 <= class_id < class_count]

    groups = [
        valid_ids(other_ids),
        valid_ids(competing_surface_ids),
        valid_ids(sidewalk_ids),
    ]
    target_scores = []
    for class_ids in groups:
        if class_ids:
            target_scores.append(probabilities[:, class_ids].sum(dim=1))
        else:
            target_scores.append(torch.zeros_like(probabilities[:, 0]))

    scores = torch.stack(target_scores, dim=1)
    return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-6)
