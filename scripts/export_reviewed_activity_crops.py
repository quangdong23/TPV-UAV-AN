"""Xuất crop đã rà soát; giữ tọa độ và nguồn hiệu chỉnh để tái lập."""
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--corrections',type=Path,default=Path('outputs/activity_image_crops/review_corrections.json'))
parser.add_argument('--output',type=Path,default=Path('outputs/activity_crops'))
args=parser.parse_args()
source=args.corrections
out=args.output
font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',max(16,20))
records=json.loads(source.read_text())
for record in records:
    path=Path(record['source']);image=Image.open(path).convert('RGB');w,h=image.size
    refw,refh=record['reference_size'];scale=[w/refw,h/refh,w/refw,h/refh]
    category=path.parent.name
    for folder in ('crops','annotated'):(out/folder/category).mkdir(parents=True,exist_ok=True)
    annotated=image.copy();draw=ImageDraw.Draw(annotated)
    record['source_size']=[w,h];record['crops']=[]
    for i,reference in enumerate(record['boxes']):
        box=[round(v*s) for v,s in zip(reference,scale)]
        box=[max(0,box[0]),max(0,box[1]),min(w,box[2]),min(h,box[3])]
        assert box[2]>box[0] and box[3]>box[1]
        crop=out/'crops'/category/f'{path.stem}__crop_{i+1:02d}.jpg'
        image.crop(box).save(crop,quality=95)
        draw.rectangle(box,outline=(255,200,0),width=max(3,w//500))
        draw.text((box[0],max(0,box[1]-28)),f'C{i+1} - reviewed',fill=(255,200,0),font=font)
        record['crops'].append({'bbox_xyxy':box,'file':str(crop),'origin':'visual_review_correction'})
    annotated.save(out/'annotated'/category/path.name,quality=90)
(out/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
(out/'review').mkdir(exist_ok=True)
for start in range(0,len(records),12):
    canvas=Image.new('RGB',(1920,4*310),'white');draw=ImageDraw.Draw(canvas)
    for j,record in enumerate(records[start:start+12]):
        path=Path(record['source']);im=Image.open(out/'annotated'/path.parent.name/path.name);im.thumbnail((640,275))
        x=j%3*640;y=j//3*310;canvas.paste(im,(x,y))
        draw.text((x,y+278),f"{start+j:03d} {path.name} | {len(record['crops'])} crop",font=font,fill='black')
    canvas.save(out/'review'/f'page_{start//12:02d}.jpg',quality=90)
print('Ảnh nguồn:',len(records),'Crop:',sum(len(r['crops']) for r in records),flush=True)
