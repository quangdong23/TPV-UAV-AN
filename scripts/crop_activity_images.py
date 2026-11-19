"""Khoanh cặp người–xe trên ảnh tĩnh bằng YOLOE; không nạp MobileNet."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from uav_pipeline.detection.detector import YOLOEDetector


def candidate_pairs(rows, shape):
    """Giữ từng cặp gần nhau; ảnh tĩnh không áp quy tắc đứng yên của video."""
    h, w = shape[:2]
    people = [r for r in rows if r['class_id'] == 1]
    vehicles = [r for r in rows if r['class_id'] in (2, 3)]
    pairs = []
    for person in people:
        a = np.asarray(person['bbox_xyxy'])
        for vehicle in vehicles:
            b = np.asarray(vehicle['bbox_xyxy'])
            scale = max(np.linalg.norm(b[2:]-b[:2]), 1.)
            gap = np.maximum(0, np.maximum(a[:2]-b[2:], b[:2]-a[2:]))
            feet = np.array([(a[0]+a[2])/2, a[3]]) - [(b[0]+b[2])/2, b[3]]
            if np.linalg.norm(gap)/scale > .25 or np.linalg.norm(feet)/scale > 1.:
                continue
            box = np.r_[np.minimum(a[:2], b[:2]), np.maximum(a[2:], b[2:])]
            margin = .1*(box[2:]-box[:2])
            crop = np.r_[np.maximum(0, np.floor(box[:2]-margin)),
                         np.minimum([w,h], np.ceil(box[2:]+margin))].astype(int).tolist()
            pairs.append({'person_bbox': a.tolist(), 'vehicle_bbox': b.tolist(),
                          'bbox': crop, 'person_score': person['score'], 'vehicle_score': vehicle['score']})
    return pairs


def sheets(records, output, per_page=12):
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 15)
    for start in range(0, len(records), per_page):
        canvas = Image.new('RGB', (1600, 4*270), 'white')
        draw = ImageDraw.Draw(canvas)
        for j, row in enumerate(records[start:start+per_page]):
            image = Image.open(output/'annotated'/row['category']/row['name'])
            image.thumbnail((530,235))
            x, y = (j%3)*533, (j//3)*270
            canvas.paste(image,(x,y))
            draw.text((x+2,y+237), f"{start+j:03d} | {row['name']} | {len(row['pairs'])} crop", fill='black', font=font)
        canvas.save(output/'review'/f'page_{start//per_page:02d}.jpg', quality=90)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()
    import torch
    torch.set_num_threads(4)
    detector = YOLOEDetector(weights=args.weights, device='cpu', roi_max_patches=12)
    output = args.output
    (output/'review').mkdir(parents=True,exist_ok=True)
    records=[]
    for category in ('rửa xe','sửa xe'):
        for kind in ('annotated','crops'):
            (output/kind/category).mkdir(parents=True,exist_ok=True)
        for path in sorted((Path('data')/category).glob('*.jpg')):
            cache = output/'detections'/category/(path.stem+'.json')
            cache.parent.mkdir(parents=True,exist_ok=True)
            if cache.exists():
                result=json.loads(cache.read_text())
            else:
                result=detector.predict(path, target_classes=[1,2,3])
                result = {key:value for key,value in result.items() if not key.startswith('_')}
                cache.write_text(json.dumps(result,ensure_ascii=False))
            frame=cv2.imread(str(path))
            rows=result['object_detections']
            pairs=candidate_pairs(rows,frame.shape)
            rendered=frame.copy()
            for row in rows:
                box=np.rint(row['bbox_xyxy']).astype(int)
                color=(255,180,0) if row['class_id']==1 else (0,220,0)
                cv2.rectangle(rendered,tuple(box[:2]),tuple(box[2:]),color,2)
            for i,pair in enumerate(pairs):
                x1,y1,x2,y2=pair['bbox']
                cv2.rectangle(rendered,(x1,y1),(x2,y2),(0,200,255),3)
                cv2.putText(rendered,f'C{i+1}',(x1,max(20,y1)),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,200,255),2)
                crop=output/'crops'/category/f'{path.stem}__crop_{i+1:02d}.jpg'
                cv2.imwrite(str(crop),frame[y1:y2,x1:x2],[cv2.IMWRITE_JPEG_QUALITY,95])
            cv2.imwrite(str(output/'annotated'/category/path.name),rendered)
            records.append({'source':str(path),'category':category,'name':path.name,'pairs':pairs})
            (output/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
            print(f'{len(records)}/290 {path.name}: {len(rows)} box, {len(pairs)} crop',flush=True)
            if args.limit and len(records)>=args.limit:
                sheets(records,output)
                return
    sheets(records,output)


if __name__=='__main__':
    main()
