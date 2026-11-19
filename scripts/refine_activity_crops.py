"""Tìm lại xe trong crop quanh người, lưu đề xuất độc lập để rà soát."""
import json
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from uav_pipeline.detection.detector import YOLOEDetector
from scripts.crop_activity_images import candidate_pairs, sheets

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--weights',required=True)
parser.add_argument('--baseline',type=Path,default=Path('outputs/activity_image_crops'))
parser.add_argument('--output',type=Path,default=Path('outputs/activity_image_crops_refined'))
args=parser.parse_args()
root=args.baseline
out=args.output
(out/'review').mkdir(parents=True,exist_ok=True)
torch.set_num_threads(3)
d=YOLOEDetector(weights=args.weights,device='cpu',imgsz=640,roi_focus=False)
paths=sorted(Path('data/rửa xe').glob('*.jpg'))+sorted(Path('data/sửa xe').glob('*.jpg'))
records=[]
for index,path in enumerate(paths):
    cat=path.parent.name
    cache=root/'detections'/cat/(path.stem+'.json')
    if not cache.exists():
        raise FileNotFoundError(f'Chạy script baseline trước: {cache}')
    r=json.loads(cache.read_text())
    frame=cv2.imread(str(path));h,w=frame.shape[:2]
    rows=r['object_detections']
    pairs=[]
    refined_cache=out/'detections'/cat/(path.stem+'.json')
    refined_cache.parent.mkdir(parents=True,exist_ok=True)
    if refined_cache.exists():
        rows=json.loads(refined_cache.read_text())
    else:
        extra=[]
        for person in rows:
            if person['class_id']!=1:continue
            a=np.array(person['bbox_xyxy']);side=max(300,2.5*max(a[2:]-a[:2]))
            x=max(0,min(w-int(side),int((a[0]+a[2]-side)/2)))
            y=max(0,min(h-int(side),int((a[1]+a[3]-side)/2)))
            patch=frame[y:min(h,int(y+side)),x:min(w,int(x+side))]
            result=d.predict_video_frame(patch,color_order='BGR',target_classes=[1,2,3],box_threshold_override=.2)
            for row in result['object_detections']:
                b=(np.asarray(row['bbox_xyxy'])+[x,y,x,y]).tolist()
                if row['class_id'] in (2,3):
                    extra.append(dict(row,bbox_xyxy=b,source='person_context'))
        rows=rows+extra
        refined_cache.write_text(json.dumps(rows,ensure_ascii=False))
    pairs=candidate_pairs(rows,frame.shape)
    # Mỗi người chỉ giữ xe gần nhất; loại cặp trùng giữa full-frame và crop.
    selected=[]
    for pair in sorted(pairs,key=lambda p:np.linalg.norm(np.array(p['person_bbox'][:2])-p['vehicle_bbox'][:2])):
        if any(np.linalg.norm(np.array(pair['person_bbox'])-a['person_bbox'])<5 for a in selected):continue
        selected.append(pair)
    for folder in ('annotated','crops'):(out/folder/cat).mkdir(parents=True,exist_ok=True)
    rendered=frame.copy()
    for i,pair in enumerate(selected):
        for key,color in [('person_bbox',(255,180,0)),('vehicle_bbox',(0,220,0)),('bbox',(0,200,255))]:
            box=np.array(pair[key]).astype(int);cv2.rectangle(rendered,tuple(box[:2]),tuple(box[2:]),color,3)
        x1,y1,x2,y2=pair['bbox']
        cv2.imwrite(str(out/'crops'/cat/f'{path.stem}__crop_{i+1:02d}.jpg'),frame[y1:y2,x1:x2],[cv2.IMWRITE_JPEG_QUALITY,95])
    cv2.imwrite(str(out/'annotated'/cat/path.name),rendered)
    records.append({'source':str(path),'category':cat,'name':path.name,'pairs':selected})
    (out/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    if (index+1)%12==0:sheets(records,out)
    print(index+1,path.name,len(selected),'crop',flush=True)
sheets(records,out)
