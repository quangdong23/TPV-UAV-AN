"""Preload once, then process demo videos; preparation time stays explicit."""
import argparse,json,time
from pathlib import Path
import cv2
import torch
from lan_chiem.configuration import load_config
from lan_chiem.video_predict import process_video,VideoOptions,fit_work
from lan_chiem.video_io import atomic_json,file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--single-model',action='store_true')
    p.add_argument('--evaluate-images',action='store_true')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--config',type=Path)
    p.add_argument('--video-config',type=Path)
    args=p.parse_args();root=Path(__file__).resolve().parent
    if not args.output.resolve().is_relative_to(root):p.error('Output must stay inside lan_chiem')
    if args.output.exists() and any(args.output.iterdir()):p.error('Choose an empty output directory')
    if any(not x.is_file() for x in args.input):p.error('Input video does not exist')
    if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable')
    torch.set_num_threads(2)
    if args.single_model:
        from lan_chiem.yoloe_detector import YOLOEDetector
        detector=YOLOEDetector(config_path=args.config or root/'yoloe_single.yaml',device=args.device)
    else:
        from lan_chiem.demo_hybrid import DemoHybridDetector
        detector=DemoHybridDetector(config_path=args.config or root/'yoloe_demo.yaml',device=args.device)
    options=VideoOptions(**load_config(args.video_config or root/('video_yoloe_single.yaml' if args.single_model else 'video_demo.yaml')))
    options.warmup_runs=0
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();detector.load_model();load_seconds=time.perf_counter()-started
    cap=cv2.VideoCapture(str(args.input[0]));ok,frame=cap.read();cap.release()
    if not ok:raise ValueError('No decodable first frame')
    warm=time.perf_counter()
    frame=fit_work(frame,options.work_max_side)
    for i in range(3):
        detector.predict_video_frame(frame,color_order='BGR',refine_trunks=i==0,roi_patch_budget=options.roi_patch_budget)
    preparation={'model_load_seconds':load_seconds,'warmup_seconds':time.perf_counter()-warm,
                 'total_seconds':time.perf_counter()-started,'provenance':detector.provenance(),
                 'note':'Prepared once before playback; video FPS excludes this time. Python import time is not measured.'}
    atomic_json(args.output/'preparation.json',preparation)
    print('[READY] Both initialization and warmup completed. Starting video processing.',flush=True)
    summaries=[]
    for i,source in enumerate(args.input,1):
        folder=args.output/f'video_{i}'
        process_video(source,folder,detector,options)
        summaries.append(json.loads((folder/'summary.json').read_text(encoding='utf-8')))
    total=sum(s['processed_frames'] for s in summaries)
    seconds=sum(s['wall_seconds'] for s in summaries)
    atomic_json(args.output/'session.json',{'preparation':preparation,'videos':summaries,
        'video_frames':total,'video_pipeline_fps':total/seconds,
        'fps_including_preparation':total/(seconds+preparation['total_seconds']),
        'quality_verified':False,'input_frame_skipping':False})
    if args.evaluate_images:
        from lan_chiem.visualization import draw_candidates_overlay
        folder=args.output/'images';folder.mkdir()
        for source in sorted((root/'sample_data').glob('*.png')):
            result=detector.predict(source)
            image=result.pop('_image_pil')
            atomic_json(folder/f'{source.stem}_result.json',result)
            draw_candidates_overlay(image,result['object_detections'],result['review_candidates']).save(folder/f'{source.stem}.jpg')
    print('[DONE] Session metadata includes startup costs separately.',flush=True)


if __name__=='__main__':main()
