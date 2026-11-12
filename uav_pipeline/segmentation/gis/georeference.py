from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uav_pipeline.segmentation.media.image_io import GeoReference



def georeference_feature_collection(
    feature_collection: dict,
    georeference: "GeoReference",
    target_crs: str | None = None,
) -> dict:
    try:
        from pyproj import Transformer
        from shapely import affinity
        from shapely.geometry import mapping, shape
        from shapely.ops import transform as transform_geometry
    except ImportError as error:
        raise RuntimeError(
            "Georeferencing GeoTIFF output requires pyproj and shapely. "
            "Install them with: python -m pip install pyproj shapely"
        ) from error

    source_crs = georeference.crs
    destination_crs = target_crs or source_crs
    a, b, c, d, e, f = georeference.transform
    transformer = None
    if destination_crs != source_crs:
        transformer = Transformer.from_crs(
            source_crs,
            destination_crs,
            always_xy=True,
        )

    output = deepcopy(feature_collection)
    for feature in output["features"]:
        geometry = shape(feature["geometry"])
        geometry = affinity.affine_transform(geometry, [a, b, d, e, c, f])
        if transformer is not None:
            geometry = transform_geometry(transformer.transform, geometry)
        feature["geometry"] = mapping(geometry)

    output["crs"] = {
        "type": "name",
        "properties": {"name": destination_crs},
    }
    return output
