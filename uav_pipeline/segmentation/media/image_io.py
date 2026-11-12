from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True, slots=True)
class GeoReference:
    crs: str
    transform: tuple[float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class ImageRecord:
    path: Path
    image_rgb: np.ndarray
    georeference: GeoReference | None


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    output = np.empty(image.shape, dtype=np.uint8)
    for channel_index in range(image.shape[2]):
        channel = image[..., channel_index].astype(np.float32)
        low, high = np.percentile(channel, (2.0, 98.0))
        if high <= low:
            output[..., channel_index] = np.clip(channel, 0, 255).astype(np.uint8)
        else:
            scaled = (channel - low) * (255.0 / (high - low))
            output[..., channel_index] = np.clip(scaled, 0, 255).astype(np.uint8)
    return output


def _load_geotiff(path: Path) -> ImageRecord:
    try:
        import rasterio
    except ImportError as error:
        raise RuntimeError(
            "Reading georeferenced TIFF files requires rasterio. "
            "Install dependencies from requirements.txt."
        ) from error

    with rasterio.open(path) as dataset:
        if dataset.count >= 3:
            bands = dataset.read([1, 2, 3])
        else:
            single_band = dataset.read(1)
            bands = np.stack([single_band, single_band, single_band])
        image_rgb = _to_uint8(np.moveaxis(bands, 0, -1))
        georeference = None
        if dataset.crs is not None and dataset.transform is not None:
            transform = dataset.transform
            georeference = GeoReference(
                crs=dataset.crs.to_string(),
                transform=(
                    float(transform.a),
                    float(transform.b),
                    float(transform.c),
                    float(transform.d),
                    float(transform.e),
                    float(transform.f),
                ),
            )
    return ImageRecord(path=path, image_rgb=image_rgb, georeference=georeference)


def load_image(path: str | Path) -> ImageRecord:
    image_path = Path(path)
    if image_path.suffix.casefold() in {".tif", ".tiff"}:
        return _load_geotiff(image_path)

    with Image.open(image_path) as pil_image:
        image = ImageOps.exif_transpose(pil_image).convert("RGB")
        image_rgb = np.asarray(image).copy()
    return ImageRecord(path=image_path, image_rgb=image_rgb, georeference=None)


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def discover_images(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.casefold() in SUPPORTED_SUFFIXES
    )

