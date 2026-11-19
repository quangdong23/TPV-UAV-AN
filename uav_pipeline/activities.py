"""MobileNet activity classification of tracked sidewalk interaction crops.

Object IDs remain in the detector catalogue. Activity labels are separate,
image-based hypotheses; events require repeated independent classifications.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from uav_pipeline.events import on_sidewalk
from uav_pipeline.paths import PROJECT_ROOT


CLASSES = ['other_activity', 'repairing_vehicle', 'washing_vehicle']
LABELS = {'repairing_vehicle': 'Nghi sửa xe', 'street_vending': 'Nghi bán hàng rong',
          'washing_vehicle': 'Nghi rửa xe', 'other_activity': 'Hoạt động khác'}
ACTIVITY_KINDS = frozenset({'motorcycle_repair_suspected', 'vehicle_repair_suspected',
                           'street_vending_suspected', 'vehicle_washing_suspected'})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ActivityClassifier:
    def __init__(self, config_path=None, device='cpu'):
        from uav_pipeline.config import load_config
        self.config_path = Path(config_path or PROJECT_ROOT/'configs/activity.yaml')
        if not self.config_path.is_absolute():
            self.config_path = PROJECT_ROOT/self.config_path
        self.config = load_config(self.config_path)
        self._validate()
        self.device = torch.device(device)
        self.model = None
        self.states = {}
        self.previous_index = None
        self.model_calls = self.images_evaluated = 0
        self._metadata = None

    def _validate(self):
        c=self.config
        if type(c.get('enabled')) is not bool:
            raise ValueError('Activity enabled must be boolean')
        for name in ('confidence_threshold', 'min_margin', 'ema_alpha'):
            v=c.get(name)
            if type(v) not in (int,float) or not math.isfinite(v) or not 0 < v <= 1:
                raise ValueError(f'Invalid activity {name}')
        for name in ('sample_interval_seconds','max_evidence_age_seconds','confirmation_seconds',
                     'pair_gap_ratio','max_crop_area_ratio'):
            v=c.get(name)
            if type(v) not in (int,float) or not math.isfinite(v) or v <= 0:
                raise ValueError(f'Invalid activity {name}')
        if 'repair_confirmation_seconds' in c and (type(c['repair_confirmation_seconds']) not in (int,float) or not math.isfinite(c['repair_confirmation_seconds']) or c['repair_confirmation_seconds'] <= 0):
            raise ValueError('Invalid repair confirmation duration')
        if c['max_crop_area_ratio'] > 1 or c['max_evidence_age_seconds'] < c['sample_interval_seconds']:
            raise ValueError('Invalid activity crop/evidence bounds')
        if type(c.get('crop_padding_ratio')) not in (int,float) or not 0 <= c['crop_padding_ratio'] <= .5:
            raise ValueError('Invalid activity crop padding')
        frame_interval=c.get('max_sample_frame_interval',5)
        if type(frame_interval) is not int or not 1 <= frame_interval <= 5:
            raise ValueError('Activity maximum sample interval must be between 1 and 5 frames')
        for name in ('min_samples','min_detector_observations','max_crops_per_batch','min_crop_side'):
            if type(c.get(name)) is not int or c[name] < 1:
                raise ValueError(f'Invalid activity {name}')

    def load_model(self):
        if self.model is not None:
            return
        from torchvision.models import mobilenet_v3_small
        weights=PROJECT_ROOT/self.config['weights']
        manifest_path=PROJECT_ROOT/self.config['metadata']
        manifest=json.loads(manifest_path.read_text())
        checkpoint=torch.load(weights,map_location='cpu',weights_only=True)
        if (checkpoint.get('classes') != CLASSES or manifest.get('classes') != CLASSES
                or checkpoint.get('class_to_idx') != {name:i for i,name in enumerate(CLASSES)}):
            raise ValueError('Activity checkpoint class order does not match its manifest')
        preprocessing=manifest.get('preprocessing')
        if checkpoint.get('config',{}).get('preprocessing') != preprocessing:
            raise ValueError('Activity preprocessing differs between checkpoint and manifest')
        expected={'size':224,'base_letterbox_size':224,'resize':'contain_and_pad',
                  'fill':[114,114,114],'mean':[.485,.456,.406],'std':[.229,.224,.225]}
        if preprocessing != expected:
            raise ValueError('Unsupported activity preprocessing; expected RGB letterbox 224 and ImageNet normalization')
        model=mobilenet_v3_small(weights=None)
        model.classifier[3]=torch.nn.Linear(model.classifier[3].in_features,len(CLASSES))
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        self.model=model.to(self.device).eval()
        self._metadata={'architecture':'mobilenet_v3_small','weights':str(weights),
            'weights_sha256':sha256(weights),'metadata_sha256':sha256(manifest_path),
            'classes':CLASSES,'preprocessing':preprocessing,'color_order':'RGB',
            'resize_interpolation':'PIL_BILINEAR','device':str(self.device),'precision':'fp32',
            'training_macro_f1':checkpoint.get('macro_f1'),'training_epoch':checkpoint.get('epoch'),
            'deployment_quality_verified':False,'config':deepcopy(self.config)}

    def provenance(self):
        self.load_model()
        return deepcopy(self._metadata)

    @staticmethod
    def preprocess(frame, box):
        """Take an unannotated BGR crop, letterbox in RGB, then normalize."""
        h,w=frame.shape[:2]
        x1,y1=np.maximum(0,np.floor(box[:2]).astype(int))
        x2,y2=np.minimum([w,h],np.ceil(box[2:]).astype(int))
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Empty activity crop')
        image=Image.fromarray(cv2.cvtColor(frame[y1:y2,x1:x2],cv2.COLOR_BGR2RGB))
        ratio=224/max(image.size)
        resized=image.resize((max(1,round(image.width*ratio)),max(1,round(image.height*ratio))),Image.Resampling.BILINEAR)
        canvas=Image.new('RGB',(224,224),(114,114,114))
        canvas.paste(resized,((224-resized.width)//2,(224-resized.height)//2))
        array=np.asarray(canvas,dtype=np.float32)/255.
        array=(array-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        return torch.from_numpy(array.transpose(2,0,1).copy())

    def candidates(self, observations, mask, shape):
        h,w=shape[:2]
        rows=[]
        for row in observations:
            b=np.asarray(row.get('bbox_xyxy',[]),float)
            if (b.shape != (4,) or not np.isfinite(b).all() or np.any(b[2:] <= b[:2])
                    or not row.get('track_confirmed',False)
                    or row.get('position_source') not in ('detection','optical_flow','camera_motion')
                    or row.get('detection_age_seconds',1e9) > .5):
                continue
            if on_sidewalk(b,mask):
                rows.append(row)
        people=[row for row in rows if row['class_id']==1]
        result=[]
        for person in people:
            a=np.asarray(person['bbox_xyxy'],float)
            for context,ids in [('vehicle',{2,3})]:
                choices=[]
                for support in rows:
                    if support['class_id'] not in ids:
                        continue
                    b=np.asarray(support['bbox_xyxy'],float)
                    scale=max(np.linalg.norm(b[2:]-b[:2]),1.)
                    gap=np.maximum(0,np.maximum(a[:2]-b[2:],b[:2]-a[2:]))
                    feet=np.array([[(a[0]+a[2])/2,a[3]],[(b[0]+b[2])/2,b[3]]])
                    distance=float(np.linalg.norm(gap)/scale)
                    if distance > self.config['pair_gap_ratio'] or np.linalg.norm(feet[0]-feet[1])/scale > 1.25:
                        continue
                    choices.append((distance,support['track_id'],support))
                if not choices:
                    continue
                _,_,support=min(choices,key=lambda v:(v[0],v[1]))
                b=np.asarray(support['bbox_xyxy'],float)
                box=np.r_[np.minimum(a[:2],b[:2]),np.maximum(a[2:],b[2:])]
                pad=(box[2:]-box[:2])*self.config['crop_padding_ratio']
                box=np.r_[np.maximum(0,box[:2]-pad),np.minimum([w,h],box[2:]+pad)]
                extent=box[2:]-box[:2]
                if min(extent) < self.config['min_crop_side'] or np.prod(extent)/(w*h) > self.config['max_crop_area_ratio']:
                    continue
                result.append({'key':(context,person['track_id'],support['track_id']),
                    'context':context,'track_ids':[person['track_id'],support['track_id']],
                    'participant_boxes_normalized': [
                        {'track_id': item['track_id'], 'class_id': item['class_id'],
                         'bbox_xyxy': (np.asarray(item['bbox_xyxy'])/[w,h,w,h]).tolist()}
                        for item in (person,support)],
                    'class_ids':[1,support['class_id']],'bbox_xyxy':box.tolist(),
                    'fresh_detection':all(t['position_source']=='detection' for t in (person,support))})
        return result

    def _classify(self, frame, candidates):
        self.load_model()
        batch=torch.stack([self.preprocess(frame,row['bbox_xyxy']) for row in candidates]).to(self.device)
        with torch.inference_mode():
            probabilities=self.model(batch).float().softmax(dim=1).cpu().numpy()
        if probabilities.shape != (len(candidates),len(CLASSES)) or not np.isfinite(probabilities).all():
            raise RuntimeError('Invalid activity classifier output')
        self.model_calls+=1;self.images_evaluated+=len(candidates)
        return probabilities

    def update(self, index, timestamp, frame, observations, mask, motion, detector_ran):
        if self.previous_index is not None and index <= self.previous_index:
            raise ValueError('Activity frames must increase strictly')
        if motion.get('scene_cut') or (self.previous_index is not None and index != self.previous_index+1):
            self.states.clear()
        self.previous_index=index
        candidates=self.candidates(observations,mask,frame.shape)
        present={row['key'] for row in candidates}
        self.states={key:value for key,value in self.states.items() if key in present
                     and 0 <= timestamp-value['sample_timestamp'] <= self.config['max_evidence_age_seconds']}
        due=[row for row in candidates if detector_ran and
             (timestamp-self.states.get(row['key'],{}).get('sample_timestamp',-1e9) >= self.config['sample_interval_seconds']-1e-9
              or index-self.states.get(row['key'],{}).get('sample_frame',-10**9) >= self.config.get('max_sample_frame_interval',5))]
        due.sort(key=lambda row:(self.states.get(row['key'],{}).get('sample_timestamp',-1e9),row['key']))
        due=due[:self.config['max_crops_per_batch']]
        if due:
            for row,raw in zip(due,self._classify(frame,due)):
                old=self.states.get(row['key'])
                alpha=self.config['ema_alpha']
                smooth=alpha*raw+(1-alpha)*old['probabilities'] if old else raw.copy()
                top=int(np.argmax(smooth));label=CLASSES[top]
                score=float(smooth[top]);margin=float(score-np.sort(smooth)[-2])
                allowed=({'repairing_vehicle','washing_vehicle'} if row['context']=='vehicle' else {'street_vending'})
                positive=(label in allowed and CLASSES[int(np.argmax(raw))]==label
                          and score >= self.config['confidence_threshold']
                          and float(raw[top]) >= self.config['confidence_threshold']
                          and margin >= self.config['min_margin'])
                same=positive and old and old['positive'] and old['label']==label
                samples=old['samples']+1 if same else int(positive)
                first=old['first_timestamp'] if same else timestamp
                first_frame=old['first_frame'] if same else index
                detector_count=old['detector_count'] if same else 0
                last_detector_frame=old['last_detector_frame'] if same else None
                if positive and row['fresh_detection'] and last_detector_frame != index:
                    detector_count += 1
                    last_detector_frame = index
                self.states[row['key']]={'sample_timestamp':timestamp,'sample_frame':index,
                    'probabilities':smooth,'raw_probabilities':raw,'label':label,'score':score,'margin':margin,
                    'positive':positive,'samples':samples,'first_timestamp':first,'first_frame':first_frame,
                    'detector_count':detector_count,'last_detector_frame':last_detector_frame}
        predictions=[]
        for row in candidates:
            state=self.states.get(row['key'])
            if state is None:
                continue
            duration=state['sample_timestamp']-state['first_timestamp']
            required_seconds=(self.config.get('repair_confirmation_seconds',self.config['confirmation_seconds'])
                              if state['label']=='repairing_vehicle' else self.config['confirmation_seconds'])
            confirmed=bool(state['positive'] and state['samples']>=self.config['min_samples']
                           and duration+1e-9>=required_seconds
                           and state['detector_count']>=self.config['min_detector_observations'])
            label=state['label']
            kind={'repairing_vehicle':('motorcycle_repair_suspected' if row['class_ids'][1]==2 else 'vehicle_repair_suspected'),
                  'street_vending':'street_vending_suspected','washing_vehicle':'vehicle_washing_suspected'}.get(label)
            predictions.append({k:v for k,v in row.items() if k not in ('key','fresh_detection')} | {
                'activity':label,'kind':kind,'label':LABELS[label],'score':state['score'],'margin':state['margin'],
                'probabilities':{name:float(value) for name,value in zip(CLASSES,state['probabilities'])},
                'raw_probabilities':{name:float(value) for name,value in zip(CLASSES,state['raw_probabilities'])},
                'status':'temporally_supported_requires_review' if confirmed else ('candidate' if state['positive'] else 'other_or_uncertain'),
                'confirmed':confirmed,'requires_review':True,'model':'mobilenet_v3_small',
                'model_sample_frame':state['sample_frame'],'model_sample_timestamp_seconds':state['sample_timestamp'],
                'evidence_age_seconds':timestamp-state['sample_timestamp'],
                'max_evidence_age_seconds':self.config['max_evidence_age_seconds'],'independent_model_samples':state['samples'],
                'independent_detector_observations':state['detector_count'],
                'duration_seconds':duration,'first_frame':state['first_frame'],
                'confirmation_progress': {
                    'samples':state['samples'], 'required_samples':self.config['min_samples'],
                    'seconds':duration, 'required_seconds':required_seconds,
                    'detector_observations':state['detector_count'],
                    'required_detector_observations':self.config['min_detector_observations']},
                'first_timestamp_seconds':state['first_timestamp']})
        return predictions

    @staticmethod
    def render(frame, predictions):
        from uav_pipeline.video.pipeline import _put_label
        for row in predictions:
            h,w=frame.shape[:2]
            for participant in row.get('participant_boxes_normalized',[]):
                inner=np.rint(np.asarray(participant['bbox_xyxy'])*[w,h,w,h]).astype(int)
                shade=(255,200,0) if participant['class_id']==1 else (0,220,0)
                cv2.rectangle(frame,tuple(inner[:2]),tuple(inner[2:]),shade,2)
            box=np.rint(row['bbox_xyxy']).astype(int)
            color=(0,0,255) if row['confirmed'] else (0,180,255)
            cv2.rectangle(frame,tuple(box[:2]),tuple(box[2:]),color,2)
            prefix='' if row['confirmed'] else 'Chưa xác nhận: '
            progress=row.get('confirmation_progress')
            if progress and not row['confirmed'] and row.get('status')=='candidate':
                prefix=(f"Chờ xác nhận {min(progress['samples'],progress['required_samples'])}"
                        f"/{progress['required_samples']} mẫu, "
                        f"{min(progress['seconds'],progress['required_seconds']):.1f}"
                        f"/{progress['required_seconds']:.1f}s: ")
            _put_label(frame,f"{prefix}{row['label']} {row['score']:.0%}",int(box[0]),max(0,int(box[1])-22),color)
        return frame
