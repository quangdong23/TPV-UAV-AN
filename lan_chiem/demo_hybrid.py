"""Explicit demo hybrid: YOLOE primary and local Grounding DINO trunk assistance."""
from pathlib import Path
import os
import re
import numpy as np
import torch
from PIL import Image
from lan_chiem.postprocess import Detection, apply_intra_class_nms, calculate_ios
from lan_chiem.registry import ClassRegistry
from lan_chiem.video_io import file_hash


class LocalTrunkRefiner:
    caption = 'tree trunk . flag . utility pole . building column .'
    def __init__(self, config, device, precision):
        self.config, self.device, self.precision = config, device, precision
        root = Path(__file__).resolve().parent.parent
        snapshot = config.get('snapshot')
        if snapshot:
            self.snapshot = (root / snapshot).resolve()
        else:
            cache = root / '.hf_cache' / ('models--' + config.get('model_id', 'IDEA-Research/grounding-dino-tiny').replace('/', '--')) / 'snapshots'
            available = sorted(p for p in cache.glob('*') if (p/'config.json').is_file() and (p/'model.safetensors').is_file())
            if len(available) != 1:
                raise FileNotFoundError('Select an existing local DINO snapshot in model.trunk_refiner.snapshot; no download is attempted')
            self.snapshot = available[0]
        self.model = self.processor = None
        self.revision = None
        self.last_output = None

    def load(self):
        if self.model is not None:
            return
        os.environ.setdefault('USE_TF','0')
        os.environ.setdefault('USE_FLAX','0')
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        cache = Path(__file__).parent / 'outputs' / 'demo_model_cache'
        self.processor = AutoProcessor.from_pretrained(self.snapshot, local_files_only=True, cache_dir=str(cache))
        self.processor.image_processor.size = {'shortest_edge':int(self.config.get('shortest_edge',640)),'longest_edge':int(self.config.get('longest_edge',1066))}
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.snapshot, local_files_only=True, cache_dir=str(cache)).eval().to(self.device)
        self.revision = file_hash(self.snapshot/'model.safetensors')

    def predict(self, image, threshold=.2):
        self.load()
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image[:,:,::-1])  # detector's native NumPy input is BGR
        image = image.convert('RGB')
        inputs = self.processor(images=image, text=self.caption, return_tensors='pt').to(self.device)
        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.float16,
                enabled=self.device == 'cuda' and self.precision == 'fp16'):
            output = self.model(**inputs)
        raw = self.processor.post_process_grounded_object_detection(output, inputs.input_ids,
            threshold=threshold, text_threshold=.2, target_sizes=[(image.height,image.width)])[0]
        meta = ClassRegistry.get(9)
        labels = raw.get('text_labels')
        if labels is None:
            labels = raw.get('labels')
        if labels is None and len(raw['boxes'])==0:
            labels=[]  # HF may omit label arrays for a normal empty result.
        invalid = labels is None or len(labels) != len(raw['boxes'])
        if invalid:
            self.last_output={'caption':self.caption,'invalid_semantic_output':True,
                              'reason':'grounded_label_count_mismatch','boxes':[],'scores':[],'labels':[]}
            return []  # Never guess a class or stop playback on malformed assistance.
        invalid_labels=sum(not isinstance(label,str) for label in labels)
        labels=[label if isinstance(label,str) else None for label in labels]
        self.last_output = {'caption':self.caption,'boxes':raw['boxes'].float().cpu().tolist(),
                            'scores':raw['scores'].float().cpu().tolist(),'labels':list(labels),
                            'invalid_semantic_output':bool(invalid_labels),'invalid_label_count':invalid_labels}
        detections = []
        for i,(box,score) in enumerate(zip(raw['boxes'].float().cpu().tolist(), raw['scores'].float().cpu().tolist())):
            if not np.isfinite(box).all() or not np.isfinite(score):
                continue
            if labels[i] is None:
                continue
            words = set(re.findall('[a-z]+',labels[i].lower()))
            if not words.intersection({'tree','trunk'}):
                continue  # Counter categories are not catalogue class 9.
            box = np.clip(box,[0,0,0,0],[image.width,image.height,image.width,image.height]).tolist()
            if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                continue
            d = Detection(f'trunk_{i}',9,meta.name,meta.display_name,'tree trunk','trunk_refiner','trunk_refiner',score,box)
            d.evidence['backend'] = 'grounding_dino_local_trunk_query'
            d.evidence['grounded_label'] = labels[i]
            if words.intersection({'flag','utility','pole','building','column'}):
                d.require_review('trunk_counter_class_ambiguity')
            if 'trunk' not in words:
                d.require_review('tree_without_grounded_trunk_scope')
            if box[2]-box[0] > box[3]-box[1]:
                d.require_review('upright_object_scope_uncertain')
            detections.append(d)
        candidates = apply_intra_class_nms(detections,.5)
        kept = []
        for d in sorted(candidates,key=lambda d:-d.score):
            same = None
            for other in kept:
                a,b = d.bbox_xyxy,other.bbox_xyxy
                if (max(calculate_ios(a,b),calculate_ios(b,a)) >= .9 and
                    abs((a[0]+a[2]-b[0]-b[2])/2) <= .5*max(a[2]-a[0],b[2]-b[0]) and
                    abs(a[3]-b[3]) <= .15*min(a[3]-a[1],b[3]-b[1])):
                    same = other
                    break
            if same:
                same.evidence.setdefault('alternative_trunk_scopes',[]).append(d.to_dict())
            else:
                kept.append(d)
        return kept

    def provenance(self):
        return {'backend':'grounding_dino_trunks','snapshot':str(self.snapshot),'model_revision':self.revision,
                'prompt':self.caption,'quality_verified':False}


from collections import Counter
import time
from lan_chiem.yoloe_detector import YOLOEDetector
from lan_chiem.postprocess import calculate_iou


class DemoHybridDetector(YOLOEDetector):
    """Opt-in only. Sparse assistance never reuses an old box as a detection."""
    def __init__(self,*args,**kwargs):
        if kwargs.get('config_path') is None:
            kwargs['config_path']=Path(__file__).with_name('yoloe_demo.yaml')
        super().__init__(*args,**kwargs)
        cfg=self.config.setdefault('hybrid_demo',{})
        self.assistant=LocalTrunkRefiner(cfg,self.device,self.precision)
        self.config['model']['detector_mode']='explicit_demo_hybrid'
        if not np.isfinite(cfg.get('threshold',.2)) or not 0<=cfg.get('threshold',.2)<=1:
            raise ValueError('Demo DINO threshold must be in [0,1]')
        short,long=cfg.get('shortest_edge',640),cfg.get('longest_edge',1066)
        if not isinstance(short,int) or not isinstance(long,int) or short<320 or long<short:
            raise ValueError('Demo DINO resize must have longest >= shortest >=320')

    def load_model(self):
        super().load_model()
        self.assistant.precision=self.precision
        self.assistant.load()
        self.config['hybrid_demo']['loaded_provenance']=self.assistant.provenance()

    def provenance(self):
        from importlib.metadata import version,PackageNotFoundError
        base=super().provenance()
        try:base['versions']['transformers']=version('transformers')
        except PackageNotFoundError:base['versions']['transformers']='not installed'
        return {**base,'backend':'yoloe_dino_demo','detector_count':2,
                'trunk_refiner':self.assistant.provenance() if hasattr(self,'assistant') else None}

    def predict(self,image_input,**kwargs):
        refine=kwargs.pop('refine_trunks',not kwargs.get('_video_frame',False))
        result=super().predict(image_input,**kwargs)
        selected=kwargs.get('target_classes')
        refine=bool(refine and (selected is None or 9 in selected))
        if not refine:
            result['observed_class_ids']=[cid for cid in result['observed_class_ids'] if cid!=9]
        else:
            started=time.perf_counter()
            assistant_image=result.get('_image_pil')
            if assistant_image is None:assistant_image=image_input
            additions=self.assistant.predict(assistant_image,threshold=self.config.get('hybrid_demo',{}).get('threshold',.2))
            if (self.assistant.last_output or {}).get('invalid_semantic_output'):
                result['observed_class_ids']=[cid for cid in result['observed_class_ids'] if cid!=9]
            if self.device=='cuda':torch.cuda.synchronize()
            seconds=time.perf_counter()-started
            predicted=result['object_detections']
            review=result['review_candidates']
            for d in additions:
                item=d.to_dict()
                if d.status=='review':
                    review.append(item)
                    continue
                same=[p for p in predicted if p['class_id']==9 and calculate_iou(p['bbox_xyxy'],d.bbox_xyxy)>=.5]
                if same:item['evidence']['primary_alternatives']=same
                predicted[:]=[p for p in predicted if p not in same]
                predicted.append(item)
            result['model_calls']+=1
            result['images_evaluated']+=1
            result['inference_time_seconds']+=seconds
            result['model_stage_seconds']['dino_trunks']=seconds
            result['trunk_refiner_evidence']={**(self.assistant.last_output or {}),'seconds':seconds,
                                              'predicted_trunks':sum(d.status=='predicted' for d in additions)}
        result.update(backend='yoloe_dino_demo',detector_count=2,trunk_refiner_ran=refine,provenance=self.provenance())
        result['summary']={'total_predicted_detections':len(result['object_detections']),
            'total_review_candidates':len(result['review_candidates']),
            'total_conflicts_detected':len(result['conflict_reports']),
            'predicted_class_counts':dict(Counter(str(d['class_id']) for d in result['object_detections'])),
            'review_class_counts':dict(Counter(str(d['class_id']) for d in result['review_candidates']))}
        return result
