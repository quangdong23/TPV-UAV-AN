"""Compare fast backend with DINO on sampled frames. Agreement is not accuracy."""
import os
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch

from lan_chiem.video_io import atomic_json, file_hash
from lan_chiem.video_tracking import iou
from lan_chiem.visualization import draw_detections


def agreement(reference, candidate, threshold=.5):
    classes = sorted({d['class_id'] for d in reference+candidate})
    rows = {}
    for cid in classes:
        a = [d for d in reference if d['class_id'] == cid]
        b = [d for d in candidate if d['class_id'] == cid]
        overlap = np.array([[iou(np.array(x['bbox_xyxy']), np.array(y['bbox_xyxy'])) for y in b] for x in a]).reshape(len(a), len(b))
        matched = 0
        if overlap.size:
            # Match valid edges first, then maximize their total IoU.
            cost = -(overlap >= threshold).astype(float)*1000-overlap
            x, y = linear_sum_assignment(cost)
            matched = int((overlap[x,y] >= threshold).sum())
        rows[str(cid)] = {'reference': len(a), 'candidate': len(b), 'matched': matched}
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--backend', choices=['world', 'yoloe'], default='world')
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--work-max-side', type=int, default=960)
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    args = parser.parse_args(argv)
    args.weights = args.weights or ('yoloe-v8s.pt' if args.backend == 'yoloe' else 'yolov8m-worldv2.pt')
    if args.samples < 1:
        parser.error('--samples must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    cap = cv2.VideoCapture(str(args.input))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    try:
        for index in sorted(set(np.linspace(0, max(count-1,0), args.samples).astype(int).tolist())):
            cap.set(cv2.CAP_PROP_POS_FRAMES,index)
            ok, frame = cap.read()
            if ok:
                frames.append((index, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
    finally:
        cap.release()
    if not frames:
        raise ValueError('No decodable audit frames')
    args.output.mkdir(parents=True,exist_ok=True)
    from lan_chiem.detector import GroundingDINODetector
    reference = GroundingDINODetector(device=args.device)
    refs = []
    for index,image in frames:
        print(f'[audit] Grounding DINO reference frame={index}',flush=True)
        result = reference.predict(image)
        refs.append(result['object_detections'])
        draw_detections(image, result['object_detections']).save(args.output/f'{index:08d}_reference.jpg')
    del reference
    gc.collect()
    if args.device == 'cuda':
        torch.cuda.empty_cache()
    if args.backend == 'yoloe':
        from lan_chiem.yoloe_detector import YOLOEDetector
        fast = YOLOEDetector(args.weights, args.device, args.imgsz)
    else:
        from lan_chiem.world_detector import WorldDetector
        fast = WorldDetector(args.weights,args.device,args.imgsz)
    fast.precision = 'fp16' if args.device == 'cuda' else 'fp32'
    totals, details = {}, []
    for (index,image),ref in zip(frames,refs):
        from lan_chiem.video_predict import fit_work
        print(f'[audit] {args.backend} candidate frame={index}',flush=True)
        working=fit_work(np.asarray(image),args.work_max_side)
        result=fast.predict(Image.fromarray(working))
        scale=[image.width/working.shape[1],image.height/working.shape[0]]*2
        candidate=[{**d,'bbox_xyxy':(np.array(d['bbox_xyxy'])*scale).tolist()} for d in result['object_detections']]
        draw_detections(image,candidate).save(args.output/f'{index:08d}_candidate.jpg')
        counts=agreement(ref,candidate)
        for cid,row in counts.items():
            total=totals.setdefault(cid,{'reference':0,'candidate':0,'matched':0})
            for key in total:total[key]+=row[key]
        details.append({'frame_index':index,'reference':ref,'candidate':candidate,'counts':counts})
    total_ref=sum(v['reference'] for v in totals.values())
    total_candidate=sum(v['candidate'] for v in totals.values())
    matched=sum(v['matched'] for v in totals.values())
    recall=matched/total_ref if total_ref else None
    precision=matched/total_candidate if total_candidate else None
    for row in totals.values():
        row['reference_retention']=row['matched']/row['reference'] if row['reference'] else None
    report={'source_sha256':file_hash(args.input),'backend':'yoloe' if args.backend == 'yoloe' else 'yolo_world','weights':args.weights,'imgsz':args.imgsz,
            'work_max_side':args.work_max_side,
            'reference_retention':recall,'candidate_agreement':precision,'per_class':totals,
            'agreement_screen_passed':bool(recall is not None and precision is not None and recall>=.85 and precision>=.70
                and all(v['reference_retention']>=.70 for v in totals.values() if v['reference']>=3)),
            'classes_without_reference':[cid for cid in range(1,11) if not totals.get(str(cid),{}).get('reference',0)],
            'quality_verified':False,'note':'DINO agreement is NOT ground-truth precision/recall or tracking validation.',
            'frames':details}
    atomic_json(args.output/'audit.json',report)
    print('[audit] Saved comparison images and audit.json; quality_verified=False',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
