"""High-recall zero-shot road/sidewalk segmentation with CLIPSeg.

No task-specific training is used. Recall is improved with prompt ensembles,
overlapping tiles and geometry-aware hole/gap filling.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

OUTPUT_DIR = Path("outputs")
MODEL_ID = "CIDAS/clipseg-rd64-refined"
IMAGE_FILE_TYPES = [
    ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
    ("All files", "*.*"),
]

PROMPT_GROUPS = {
    "road": [
        "aerial view of an asphalt road surface",
        "aerial view of a paved street",
        "roadway seen from directly above",
    ],
    "sidewalk": [
        "aerial view of a paved sidewalk",
        "pedestrian pavement beside a road",
        "tiled footpath seen from directly above",
    ],
}
PROMPTS = [prompt for group in PROMPT_GROUPS.values() for prompt in group]
COLORS_BGR = np.array([[255, 80, 0], [0, 190, 255]], dtype=np.uint8)
MIN_CONFIDENCE = 0.30
TILE_SIZE = 720
TILE_OVERLAP = 180


def predict_crop(image: Image.Image, processor, model, device: str) -> np.ndarray:
    """Return two score maps (road, sidewalk) for one crop."""
    width, height = image.size
    batch = processor(
        text=PROMPTS,
        images=[image] * len(PROMPTS),
        return_tensors="pt",
        padding=True,
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        logits = model(**batch).logits
    prompt_scores = torch.sigmoid(logits).cpu().numpy()
    prompt_scores = np.stack(
        [cv2.resize(score, (width, height), interpolation=cv2.INTER_CUBIC) for score in prompt_scores]
    )
    group_size = len(next(iter(PROMPT_GROUPS.values())))
    return np.stack(
        [prompt_scores[index * group_size : (index + 1) * group_size].mean(axis=0) for index in range(2)]
    )


def tiled_prediction(image: Image.Image, processor, model, device: str) -> np.ndarray:
    """Blend global context and overlapping local predictions."""
    width, height = image.size
    global_scores = predict_crop(image, processor, model, device)
    score_sum = np.zeros((2, height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    stride = TILE_SIZE - TILE_OVERLAP

    x_starts = list(range(0, max(width - TILE_SIZE, 0) + 1, stride))
    y_starts = list(range(0, max(height - TILE_SIZE, 0) + 1, stride))
    if not x_starts or x_starts[-1] != max(width - TILE_SIZE, 0):
        x_starts.append(max(width - TILE_SIZE, 0))
    if not y_starts or y_starts[-1] != max(height - TILE_SIZE, 0):
        y_starts.append(max(height - TILE_SIZE, 0))

    for y0 in y_starts:
        for x0 in x_starts:
            x1, y1 = min(x0 + TILE_SIZE, width), min(y0 + TILE_SIZE, height)
            crop = image.crop((x0, y0, x1, y1))
            local_scores = predict_crop(crop, processor, model, device)
            crop_h, crop_w = y1 - y0, x1 - x0
            wy = np.hanning(crop_h) if crop_h > 2 else np.ones(crop_h)
            wx = np.hanning(crop_w) if crop_w > 2 else np.ones(crop_w)
            weight = np.maximum(np.outer(wy, wx), 0.08).astype(np.float32)
            score_sum[:, y0:y1, x0:x1] += local_scores * weight
            weight_sum[y0:y1, x0:x1] += weight

    local_scores = score_sum / np.maximum(weight_sum[None, :, :], 1e-6)
    return 0.45 * global_scores + 0.55 * local_scores


def remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    cleaned = np.zeros_like(binary, dtype=np.uint8)
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == component] = 1
    return cleaned


def _interpolate_and_smooth(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Interpolate a 1-D boundary through occluded columns."""
    columns = np.arange(values.size)
    interpolated = np.interp(columns, columns[valid], values[valid]).astype(np.float32)
    # Smooth only along the road direction; this preserves curb position.
    return cv2.GaussianBlur(interpolated[None, :], (101, 1), 0)[0]


def restore_occluded_sidewalk(road: np.ndarray, sidewalk: np.ndarray) -> np.ndarray:
    """Infer sidewalk hidden by parking/vendor objects along both road edges.

    The CLIPSeg mask supplies visible sidewalk samples. Road boundaries supply the
    inner edge; interpolated visible samples supply the outer edge. Extrapolation
    also handles an occlusion that continues to the image border.
    """
    height, width = road.shape
    rows = np.arange(height)[:, None]
    road_present = road.any(axis=0)
    if road_present.sum() < width * 0.25:
        return sidewalk

    road_top_raw = np.where(road, rows, height).min(axis=0).astype(np.float32)
    road_bottom_raw = np.where(road, rows, -1).max(axis=0).astype(np.float32)
    road_top = _interpolate_and_smooth(road_top_raw, road_present)
    road_bottom = _interpolate_and_smooth(road_bottom_raw, road_present)

    top_outer = np.zeros(width, dtype=np.float32)
    bottom_outer = np.zeros(width, dtype=np.float32)
    top_valid = np.zeros(width, dtype=bool)
    bottom_valid = np.zeros(width, dtype=bool)

    for x in range(width):
        sidewalk_rows = np.flatnonzero(sidewalk[:, x])
        upper = sidewalk_rows[sidewalk_rows < road_top[x]]
        lower = sidewalk_rows[sidewalk_rows > road_bottom[x]]
        if upper.size >= 5:
            top_outer[x] = np.percentile(upper, 5)
            top_valid[x] = True
        if lower.size >= 5:
            bottom_outer[x] = np.percentile(lower, 95)
            bottom_valid[x] = True

    restored = sidewalk.copy()
    road_columns = road_present[None, :]

    # Require broad evidence before treating either side as a continuous sidewalk.
    if top_valid.sum() >= width * 0.12:
        outer = _interpolate_and_smooth(top_outer, top_valid)
        envelope_width = max(51, int(width * 0.22)) | 1
        # Propagate the nearest visible outer edge underneath wide vendor canopies.
        outer = cv2.erode(outer[None, :], np.ones((1, envelope_width), np.uint8))[0]
        outer = cv2.GaussianBlur(outer[None, :], (101, 1), 0)[0]
        restored |= (
            (rows >= outer[None, :])
            & (rows < road_top[None, :])
            & road_columns
        )
    if bottom_valid.sum() >= width * 0.12:
        outer = _interpolate_and_smooth(bottom_outer, bottom_valid)
        envelope_width = max(51, int(width * 0.22)) | 1
        # The maximum visible extent reconstructs parking that reaches an image edge.
        outer = cv2.dilate(outer[None, :], np.ones((1, envelope_width), np.uint8))[0]
        outer = cv2.GaussianBlur(outer[None, :], (101, 1), 0)[0]
        restored |= (
            (rows <= outer[None, :])
            & (rows > road_bottom[None, :])
            & road_columns
        )

    # Never overwrite the visible/inferred road region.
    restored &= ~road
    return restored


def complete_surfaces(scores: np.ndarray) -> np.ndarray:
    """Fill occlusions and short gaps while retaining model-derived boundaries."""
    confidence = scores.max(axis=0)
    labels = scores.argmax(axis=0)
    height, width = labels.shape
    min_area = max(500, int(height * width * 0.0015))
    completed = []

    for class_index in range(2):
        binary = ((labels == class_index) & (confidence >= MIN_CONFIDENCE)).astype(np.uint8)
        binary = remove_small_components(binary, min_area)

        # Horizontal closing reconstructs surfaces hidden by cars, people and canopies.
        # Aerial scenes often contain wide cars/canopies. Bridge gaps up to roughly
        # one quarter of the frame so the underlying traversable surface is retained.
        gap_width = max(31, int(width * (0.28 if class_index == 0 else 0.18)))
        horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (gap_width | 1, 17))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, horizontal)
        smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, smooth)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        binary = remove_small_components(binary, min_area)
        completed.append(binary.astype(bool))

    road, sidewalk = completed
    overlap = road & sidewalk
    road[overlap] = scores[0][overlap] >= scores[1][overlap]
    sidewalk[overlap] = ~road[overlap]
    sidewalk = restore_occluded_sidewalk(road, sidewalk)
    return np.stack([road, sidewalk])


def choose_images() -> list[Path]:
    """Open the native file picker and return all selected image paths."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError(
            "Tkinter is unavailable. Pass image paths on the command line instead."
        ) from exc

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilenames(
        parent=root,
        title="Chọn ảnh UAV cần nhận diện",
        filetypes=IMAGE_FILE_TYPES,
    )
    root.destroy()
    return [Path(path) for path in selected]


def unique_output_paths(image_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    """Build input-based output names and avoid overwriting older results."""
    stem = image_path.stem
    suffix_number = 1
    while True:
        unique_stem = stem if suffix_number == 1 else f"{stem}_{suffix_number}"
        overlay_path = output_dir / f"{unique_stem}_zero_shot_overlay.png"
        mask_path = output_dir / f"{unique_stem}_zero_shot_mask.png"
        scores_path = output_dir / f"{unique_stem}_zero_shot_scores.npy"
        if not any(path.exists() for path in (overlay_path, mask_path, scores_path)):
            return overlay_path, mask_path, scores_path
        suffix_number += 1


def save_png(path: Path, image_bgr: np.ndarray) -> None:
    """Save PNG reliably even when the Windows path contains Unicode characters."""
    success, encoded = cv2.imencode(".png", image_bgr)
    if not success:
        raise RuntimeError(f"Cannot encode output image: {path}")
    encoded.tofile(str(path))


def process_image(
    image_path: Path,
    output_dir: Path,
    processor,
    model,
    device: str,
) -> None:
    image = Image.open(image_path).convert("RGB")
    scores = tiled_prediction(image, processor, model, device)
    class_masks = complete_surfaces(scores)
    source_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    color_mask = np.zeros_like(source_bgr)
    for class_index, color in enumerate(COLORS_BGR):
        color_mask[class_masks[class_index]] = color

    detected = class_masks.any(axis=0)
    overlay = source_bgr.copy()
    overlay[detected] = cv2.addWeighted(source_bgr, 0.42, color_mask, 0.58, 0)[detected]

    overlay_path, mask_path, scores_path = unique_output_paths(image_path, output_dir)
    save_png(overlay_path, overlay)
    save_png(mask_path, color_mask)
    np.save(scores_path, scores)
    total = image.width * image.height
    print(f"\nImage: {image_path}")
    print(f"Road coverage: {class_masks[0].sum() / total:.1%}")
    print(f"Sidewalk coverage: {class_masks[1].sum() / total:.1%}")
    print(f"Overlay: {overlay_path}")
    print(f"Mask: {mask_path}")
    print("Legend: blue = road; yellow = sidewalk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot road and sidewalk segmentation for UAV images."
    )
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        help="Image paths. When omitted, a file picker opens.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for generated overlays, masks and score files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = args.images or choose_images()
    if not image_paths:
        print("No image selected.")
        return

    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Image file not found:\n{missing_text}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model once for {len(image_paths)} image(s)...")
    processor = CLIPSegProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = CLIPSegForImageSegmentation.from_pretrained(
        MODEL_ID, local_files_only=True
    ).to(device).eval()

    for index, image_path in enumerate(image_paths, start=1):
        print(f"Processing {index}/{len(image_paths)}: {image_path.name}")
        process_image(
            image_path=image_path,
            output_dir=args.output_dir,
            processor=processor,
            model=model,
            device=device,
        )


if __name__ == "__main__":
    main()
