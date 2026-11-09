import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
import torch

from lan_chiem.colab_stream import run_streaming
from lan_chiem.tests.test_video import FakeDetector
from lan_chiem.video_audit import agreement
from lan_chiem.video_predict import VideoOptions, process_video
from lan_chiem.video_tracking import association_costs, bbox, iou, measurement
from lan_chiem.world_detector import WorldDetector
from lan_chiem.quality_gate import require_candidate_audit


class RealtimeTests(unittest.TestCase):
    def test_rejects_failed_missing_or_mismatched_fast_candidate_audit(self):
        arguments=('world',None,'image_hash','weights.pt',960,960)
        with self.assertRaises(ValueError):require_candidate_audit(*arguments)
        report={'agreement_screen_passed':False}
        with self.assertRaises(ValueError):require_candidate_audit('world',report,*arguments[2:])
        report={'agreement_screen_passed':True,'source_sha256':'image_hash','weights':'weights.pt',
                'imgsz':960,'work_max_side':960,'backend':'yolo_world'}
        require_candidate_audit('world',report,*arguments[2:])
        with self.assertRaises(ValueError):require_candidate_audit('world',report,'different_video','weights.pt',960,960)
        require_candidate_audit('grounding-dino',None,'image_hash','',0,0)
    def test_vectorized_assignment_preserves_original_gates_and_costs(self):
        rng=np.random.default_rng(21)
        tracks=[{'class_id':int(rng.integers(1,4)),'x':np.r_[rng.uniform(20,200,2),rng.uniform(10,80,2),np.zeros(4)]} for _ in range(30)]
        detections=[]
        for _ in range(35):
            a=rng.uniform(0,200,2);b=a+rng.uniform(10,80,2)
            detections.append({'class_id':int(rng.integers(1,4)),'bbox_xyxy':np.r_[a,b]})
        expected=np.full((len(tracks),len(detections)),1e6)
        for a,t in enumerate(tracks):
            for b,d in enumerate(detections):
                if t['class_id']!=d['class_id']:continue
                z=measurement(d['bbox_xyxy'])
                overlap=iou(bbox(t['x']),np.array(d['bbox_xyxy']))
                distance=np.linalg.norm(z[:2]-t['x'][:2])/max(np.linalg.norm(t['x'][2:4]),1.)
                size_change=np.max(np.abs(np.log(z[2:4]/np.maximum(t['x'][2:4],1.))))
                if size_change<.8 and (overlap>=.1 or distance<=.75):expected[a,b]=1-overlap+.2*distance
        np.testing.assert_allclose(association_costs(tracks,detections),expected,rtol=1e-12,atol=1e-12)
    def test_world_maps_aliases_and_counters_without_changing_catalog_ids(self):
        detector=WorldDetector(device='cpu')
        rows=[]
        seen=set()
        for index,(_,cid,group) in enumerate(detector.vocabulary):
            if cid and cid not in seen:
                seen.add(cid)
                x=cid*100
                rows.append([x,20,x+25,80,.8,index])
        predicted,review,_=detector.decode(np.asarray(rows))
        self.assertEqual({d.class_id for d in predicted+review},set(range(1,11)))
        self.assertTrue(all(d.status=='review' for d in review))
        index=next(i for i,(_,cid,_) in enumerate(detector.vocabulary) if cid==2)
        predicted,_,_=detector.decode(np.array([[0,0,20,40,.8,index],[0,0,20,40,.7,index+1]]))
        self.assertEqual(len(predicted),1)
        counter_index=next(i for i,(_,cid,_) in enumerate(detector.vocabulary) if cid is None)
        self.assertEqual(detector.decode(np.array([[0,0,20,40,.9,counter_index]]))[0],[])

    def test_world_single_forward_and_nonfinite_output_rejected(self):
        detector=WorldDetector(device='cpu')
        engine=SimpleNamespace(predict=lambda **kwargs:[SimpleNamespace(boxes=SimpleNamespace(data=torch.tensor([[0,0,20,40,.8,0]])))])
        detector.engine=engine
        result=detector.predict(Image.new('RGB',(100,100)))
        self.assertEqual(result['model_calls'],1)
        self.assertEqual(result['object_detections'][0]['class_id'],1)
        with self.assertRaisesRegex(RuntimeError,'Non-finite'):
            detector.decode(np.array([[0,0,20,40,np.nan,0]]))

    def test_audit_matching_is_class_specific_and_one_to_one(self):
        reference=[{'class_id':1,'bbox_xyxy':[0,0,20,20]}]
        candidate=[{'class_id':1,'bbox_xyxy':[0,0,20,20]},
                   {'class_id':1,'bbox_xyxy':[0,0,20,20]},
                   {'class_id':2,'bbox_xyxy':[0,0,20,20]}]
        rows=agreement(reference,candidate)
        self.assertEqual(rows['1'],{'reference':1,'candidate':2,'matched':1})
        self.assertEqual(rows['2']['matched'],0)

    def test_scaled_video_exports_source_coordinates_and_restores_work_coordinates(self):
        root=Path(__file__).resolve().parents[1]/'outputs'/'video_tests'
        root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            folder=Path(folder)
            source=folder/'source.avi'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'MJPG'),10.,(320,240))
            self.assertTrue(writer.isOpened())
            image=np.random.default_rng(15).integers(0,255,(240,320,3),dtype=np.uint8)
            for _ in range(6):writer.write(image)
            writer.release()
            options=VideoOptions(work_max_side=160,fast_render=True,min_hits=1,segment_frames=2,
                                 warmup_runs=1,log_interval=2.,preview_interval=0.)
            with redirect_stdout(io.StringIO()),patch('lan_chiem.video_io.shutil.which',return_value=None):
                process_video(source,folder/'out',FakeDetector(),options,max_frames=3)
                process_video(source,folder/'out',FakeDetector(),options,resume=True)
            records=[json.loads(line) for line in (folder/'out/tracks.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual([r['frame_index'] for r in records],list(range(6)))
            self.assertEqual(records[0]['working_size'],[160,120])
            self.assertEqual(records[0]['source_size'],[320,240])
            np.testing.assert_allclose(records[0]['tracks'][0]['bbox_xyxy'],[40,40,100,100])
            summary=json.loads((folder/'out/summary.json').read_text(encoding='utf-8'))
            self.assertTrue(summary['all_source_frames_processed'])
            self.assertIn('end_to_end_target_met',summary)
            self.assertFalse(summary['live_latency_verified'])
            self.assertFalse((folder/'out/latest_preview.jpg').exists())
            capture=cv2.VideoCapture(str(folder/'out/segments/part_000000.mp4'))
            try:
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),160)
            finally:capture.release()
