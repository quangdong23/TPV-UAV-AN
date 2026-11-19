"""Replay a local video through live evidence processing and real API upload.

Detection dates refer to this test run, not original camera capture dates.
"""
import argparse
import os
import time
os.environ['TZ'] = 'Asia/Ho_Chi_Minh'
if hasattr(time, 'tzset'):
    time.tzset()
from pathlib import Path
import cv2
import torch
from uav_pipeline.paths import PROJECT_ROOT
from uav_pipeline.config import load_config
from uav_pipeline.video.pipeline import VideoOptions,process_video
from uav_pipeline.segmentation import PipelineConfig,SidewalkPipeline
from uav_pipeline.segmentation.media.video import VideoPreviewOptions
from uav_pipeline.segmentation.session import SidewalkSession
from uav_pipeline.detection.detector import YOLOEDetector
from uav_pipeline.events import ViolationMonitor
from uav_pipeline.activities import ActivityClassifier
from uav_pipeline.uploads import EvidenceUploader
from uav_pipeline.outputs import generate_unique_output_dir


class ReplayCapture:
    input_mode='file_live_simulation'
    epoch=0
    reconnects=0
    interrupted=False
    def __init__(self,path):
        self.source_path=str(path.resolve())
        self.url=path.resolve().as_uri()
        self.capture=cv2.VideoCapture(str(path))
    def isOpened(self):return self.capture.isOpened()
    def get(self,prop):return self.capture.get(prop)
    def read(self):return self.capture.read()
    def release(self):self.capture.release()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not args.input.is_file():parser.error('Input video missing')
    if not torch.cuda.is_available():parser.error('CUDA unavailable')
    torch.set_num_threads(2)
    options=VideoOptions(**load_config(PROJECT_ROOT/'configs/video.yaml'))
    options.preview_interval=2.
    options.warmup_runs=3
    if options.frame_step!=1 or not 1<=options.keyframe_interval<=5:
        parser.error('Test must preserve every frame and detector interval <=5')
    config=PipelineConfig.from_json(PROJECT_ROOT/'configs/segmentation/gpu_t4.json')
    config.device='cuda'
    detector=YOLOEDetector(device='cuda',config_path=PROJECT_ROOT/'configs/detection.yaml')
    activity=ActivityClassifier(PROJECT_ROOT/'configs/activity.yaml',device='cuda')
    events=ViolationMonitor()
    uploader=EvidenceUploader.from_config(PROJECT_ROOT/'configs/upload_api.json')
    output=generate_unique_output_dir(args.output,task_name='upload_test',source_name=args.input.stem)
    capture=ReplayCapture(args.input)
    print('[test] File replay with REAL evidence upload. Event dates refer to this run.',flush=True)
    try:
        with SidewalkPipeline(config) as pipeline:
            session=SidewalkSession(pipeline,VideoPreviewOptions(keyframe_interval=5,cv_threads=options.cv_threads),
                                    sample_step=0,crisp_overlay=True)
            process_video(args.input,output,detector,options,sidewalk=session,events=events,
                activity=activity,live_capture=capture,event_uploader=uploader,output_mode='full')
    finally:capture.release()
    print('[saved] '+str(output),flush=True)


if __name__=='__main__':main()
