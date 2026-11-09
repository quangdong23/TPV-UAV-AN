"""Export review crops for current catalogue JSON; never reinterpret legacy IDs."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image
from lan_chiem.object_classes import CATALOG_VERSION


def main():
    root = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-results', type=Path, default=root / 'outputs/catalog_v3')
    parser.add_argument('--output', type=Path, default=root / 'outputs/catalog_v3/review_crops')
    args = parser.parse_args()
    results = sorted(args.input_results.glob('*_result.json'))
    if not results:
        parser.error('No result JSON files; pass --input-results for the new catalogue run.')
    loaded = [(p, json.loads(p.read_text(encoding='utf-8'))) for p in results]
    for path, result in loaded:
        if result.get('catalog_version') != CATALOG_VERSION:
            parser.error(f'{path}: legacy/missing catalogue version; cannot use IDs 1-10.')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for source, result in loaded:
        image_path = Path(result.get('image_path') or root / 'sample_data' / result['image_name'])
        image_bytes = image_path.read_bytes()
        digest = hashlib.sha256(image_bytes).hexdigest()
        if digest != result['image_sha256']:
            raise ValueError(f'Image hash mismatch: {image_path}')
        image = Image.open(image_path).convert('RGB')
        for detection in result.get('review_candidates', []):
            x1, y1, x2, y2 = detection['bbox_xyxy']
            box = [max(0, int(x1)-24), max(0, int(y1)-24),
                   min(image.width, int(x2)+25), min(image.height, int(y2)+25)]
            filename = f'{source.stem}_{detection["id"]}.png'
            image.crop(box).save(args.output / filename)
            manifest.append({'catalog_version': CATALOG_VERSION, 'source_result': str(source.resolve()),
                             'image': str(image_path.resolve()), 'image_sha256': digest,
                             'hypothesis': detection, 'crop_bbox_xyxy': box, 'crop': filename,
                             'review_status': 'unverified_candidate_not_confirmed_error'})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Exported {len(manifest)} unverified review crops to {args.output}')


if __name__ == '__main__':
    main()
