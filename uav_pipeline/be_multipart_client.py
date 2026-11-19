import argparse
import mimetypes
import os
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


def ensure_browser_compatible_video(video_path: Path) -> None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError(
            "ffprobe is required to verify browser-compatible H.264 video. "
            "Install FFmpeg before uploading."
        )

    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    codec = streams[0].get("codec_name") if streams else None
    pixel_format = streams[0].get("pix_fmt") if streams else None
    if codec != "h264" or pixel_format != "yuv420p":
        raise RuntimeError(
            f"Video codec {codec or 'unknown'}/{pixel_format or 'unknown'} is not browser compatible. "
            "Convert it with FFmpeg to H.264 yuv420p before uploading."
        )


def api_request(
    method: str,
    url: str,
    api_key: str,
    *,
    json: dict[str, Any] | None = None,
    timeout: int = 60,
) -> dict[str, Any]:
    response = requests.request(
        method,
        url,
        headers={"x-api-key": api_key},
        json=json,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def upload_part(
    video_path: Path,
    part_number: int,
    part_size: int,
    upload_url: str,
    retries: int = 3,
) -> dict[str, Any]:
    offset = (part_number - 1) * part_size
    with video_path.open("rb") as video_file:
        video_file.seek(offset)
        data = video_file.read(part_size)

    for attempt in range(1, retries + 1):
        try:
            response = requests.put(upload_url, data=data, timeout=(15, 900), allow_redirects=False)
            response.raise_for_status()
            if not 200 <= response.status_code < 300:
                raise RuntimeError("Video part PUT did not return HTTP 2xx")
            etag = response.headers.get("ETag")
            if not etag:
                raise RuntimeError("MinIO did not return an ETag")
            return {"part_number": part_number, "etag": etag}
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2**attempt)

    raise RuntimeError("Upload failed")


def upload_image(
    image_path: Path,
    upload_url: str,
    content_type: str,
    retries: int = 3,
    expected_size: int | None = None,
) -> None:
    # Freeze bytes before PUT: retries send exactly the same declared image.
    image_data = image_path.read_bytes()
    if expected_size is not None and len(image_data) != expected_size:
        raise RuntimeError("Evidence image size differs from initiate metadata")
    for attempt in range(1, retries + 1):
        try:
            response = requests.put(
                upload_url,
                data=image_data,
                headers={"Content-Type": content_type, "Content-Length": str(len(image_data))},
                timeout=(15, 300),
                allow_redirects=False,
            )
            response.raise_for_status()
            if not 200 <= response.status_code < 300:
                raise RuntimeError("Image PUT did not return HTTP 2xx")
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2**attempt)


def image_payloads(image_paths: list[Path]) -> list[dict[str, Any]]:
    supported_types = {"image/jpeg", "image/png", "image/webp"}
    payloads = []
    if len(image_paths) > 50:
        raise ValueError("At most 50 evidence images are supported")
    if len({path.name for path in image_paths}) != len(image_paths):
        raise ValueError("Evidence image filenames must be unique")
    for image_path in image_paths:
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not 0 < image_path.stat().st_size <= 25 * 1024 * 1024:
            raise ValueError("Evidence image must be between 1 byte and 25 MB")
        content_type = mimetypes.guess_type(image_path.name)[0]
        if content_type not in supported_types:
            raise ValueError(
                f"Unsupported image {image_path.name}; use JPG, JPEG, PNG or WEBP"
            )
        payloads.append(
            {
                "filename": image_path.name,
                "file_size": image_path.stat().st_size,
                "content_type": content_type,
            }
        )
    return payloads


def parse_violators(value: str | None) -> list[dict[str, Any]] | None:
    if value is None:
        return None
    try:
        violators = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("--violators-json must be valid JSON") from exc
    if not isinstance(violators, list):
        raise ValueError("--violators-json must contain a JSON array")
    return violators or None


def upload_video(args: argparse.Namespace) -> dict[str, Any]:
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    ensure_browser_compatible_video(video_path)

    api_url = args.api_url.rstrip("/")
    content_type = mimetypes.guess_type(video_path.name)[0] or "video/mp4"
    detected_at = args.detected_at or datetime.now().astimezone().isoformat()
    image_paths = [Path(value).expanduser().resolve() for value in args.image]
    images = image_payloads(image_paths)

    initiate_payload = {
        "filename": video_path.name,
        "file_size": video_path.stat().st_size,
        "content_type": content_type,
        "detected_at": detected_at,
        "location": args.location,
        "behavior_name": args.behavior,
        "camera_id": args.camera_id,
        "confidence": args.confidence,
        "model_version": args.model_version,
        "violators": parse_violators(args.violators_json),
        "images": images,
    }

    initiated = api_request(
        "POST",
        f"{api_url}/api/uploads/initiate",
        args.api_key,
        json=initiate_payload,
    )
    violation_id = initiated["violation_id"]
    part_size = initiated["part_size"]
    total_parts = initiated["total_parts"]

    print(f"Upload {violation_id}: {total_parts} parts")

    try:
        if (type(part_size) is not int or part_size <= 0 or
                type(total_parts) is not int or total_parts != (video_path.stat().st_size + part_size - 1) // part_size):
            raise RuntimeError("Invalid multipart upload plan")
        if len(initiated.get("images", [])) != len(image_paths):
            raise RuntimeError("API image upload plan mismatch")
        for image_path, image, target in zip(
            image_paths,
            images,
            initiated.get("images", []),
            strict=True,
        ):
            content_type = target.get("content_type", image["content_type"])
            if content_type != image["content_type"]:
                raise RuntimeError("API image Content-Type differs from declared type")
            if image_path.stat().st_size != image["file_size"]:
                raise RuntimeError("Evidence image changed after initiate")
            if target.get("filename", image["filename"]) != image["filename"]:
                raise RuntimeError("API image filename differs from declared image order")
            upload_image(image_path, target["upload_url"], content_type, expected_size=image["file_size"])
            print(f"Uploaded image {image_path.name}", flush=True)

        urls: dict[int, str] = {}
        batch_size = 100
        for start in range(1, total_parts + 1, batch_size):
            part_numbers = list(range(start, min(start + batch_size, total_parts + 1)))
            result = api_request(
                "POST",
                f"{api_url}/api/uploads/{violation_id}/part-urls",
                args.api_key,
                json={"part_numbers": part_numbers},
            )
            urls.update({item["part_number"]: item["url"] for item in result["parts"]})

        completed_parts = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    upload_part,
                    video_path,
                    part_number,
                    part_size,
                    urls[part_number],
                ): part_number
                for part_number in range(1, total_parts + 1)
            }
            for future in as_completed(futures):
                completed = future.result()
                completed_parts.append(completed)
                print(
                    f"Uploaded part {completed['part_number']}/{total_parts}",
                    flush=True,
                )

        if (len(completed_parts) != total_parts or
                {part["part_number"] for part in completed_parts} != set(range(1, total_parts + 1)) or
                any(not part.get("etag") for part in completed_parts)):
            raise RuntimeError("Video parts are incomplete; complete will not be called")
        return api_request(
            "POST",
            f"{api_url}/api/uploads/{violation_id}/complete",
            args.api_key,
            json={"parts": sorted(completed_parts, key=lambda item: item["part_number"])},
            timeout=120,
        )
    except Exception:
        try:
            api_request(
                "POST",
                f"{api_url}/api/uploads/{violation_id}/abort",
                args.api_key,
            )
        except Exception:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload AI violation video")
    parser.add_argument("--api-url", required=True)
    parser.add_argument(
        "--api-key",
        default=os.getenv("TPV_UPLOAD_API_KEY"),
        required=os.getenv("TPV_UPLOAD_API_KEY") is None,
    )
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Evidence image path; repeat --image for multiple JPG/PNG/WEBP files",
    )
    parser.add_argument("--location", required=True)
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--detected-at")
    parser.add_argument("--camera-id")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--model-version")
    parser.add_argument(
        "--violators-json",
        help='JSON array, for example [{"full_name":"Nguyen Van A","citizen_id":"001203004567"}]',
    )
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    return parser


if __name__ == "__main__":
    try:
        response = upload_video(build_parser().parse_args())
        print(response)
    except Exception as exc:
        print(f"Upload failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
