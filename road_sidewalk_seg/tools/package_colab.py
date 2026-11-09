from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package current source for Colab without caches or model weights")
    parser.add_argument("--output", default="UAV_code_optimized.zip")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sources = list((root / "road_sidewalk_seg").rglob("*.py"))
    sources += list((root / "configs").glob("*.json"))
    sources += [root / name for name in (
        "requirements.txt", "requirements-gis.txt", "README.md", "UAV_Sidewalk_Colab.ipynb",
    )]
    payloads = {}
    for source in sorted(sources):
        if source.is_file():
            if not source.resolve().is_relative_to(root):
                raise ValueError(f"Source outside project root: {source}")
            payloads[source.relative_to(root).as_posix()] = source.read_bytes()
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    manifest = {"selected_profile": "swin-large-k8-work960", "code_layout": "functional-v1", "sha256": hashes, "dependencies": "requirements.txt", "excluded": ["caches", "weights", "outputs", "videos"]}
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in payloads.items():
            archive.writestr(name, data)
        archive.writestr("build_manifest.json", json.dumps(manifest, indent=2))
    print(f"Packaged {len(payloads)} files: {destination}")


if __name__ == "__main__":
    main()
