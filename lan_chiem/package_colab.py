"""Package object video source for Colab, excluding data, caches and weights."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('UAV_lan_chiem_realtime_v3.zip'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    package = root / 'lan_chiem'
    paths = list(package.glob('*.py')) + list(package.glob('*.yaml'))
    paths += list((package / 'tests').glob('*.py'))
    paths += [package / 'README.md'] + list(package.glob('requirements*.txt'))
    paths += [root / 'UAV_LanChiem_Video_T4.ipynb']
    payload = {}
    for path in paths:
        if path.is_file():
            if not path.resolve().is_relative_to(root):
                raise ValueError(f'Source outside project: {path}')
            payload[path.relative_to(root).as_posix()] = path.read_bytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            archive.writestr(name, data)
        archive.writestr('build_manifest.json', json.dumps({
            'build_version': 'video-v3-quality-gated',
            'sha256': {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()},
            'excluded': ['weights', '.hf_cache', 'sample_data', 'outputs', 'videos'],
            'entrypoint': 'python -m lan_chiem.video_predict'}, indent=2))
    print(f'Packaged {len(payload)} files: {args.output.resolve()}')


if __name__ == '__main__':
    main()
