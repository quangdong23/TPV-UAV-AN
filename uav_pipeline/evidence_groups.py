"""Ảnh crop có padding và mô tả gộp cho snapshot cùng frame."""
from pathlib import Path
import math
import cv2
import numpy as np
from uav_pipeline.detection.classes import ClassId


class EvidenceGroups:
    def __init__(self):
        self.links={}

    def update(self, observations, timestamp, scene_cut=False):
        if scene_cut:self.links.clear()
        valid=[t for t in observations if t.get('track_confirmed',True)
               and t.get('detection_age_seconds',0)<=.5
               and t.get('position_source') in ('detection','optical_flow','camera_motion')]
        people=[t for t in valid if t['class_id']==ClassId.PERSON]
        current={}
        # Mutual nearest association: one nearby person cannot validate all
        # surrounding objects. Normalize by the smaller participant, never a
        # large billboard's diagonal.
        choices={}
        for obj in valid:
            if obj['class_id']==ClassId.PERSON:continue
            b=np.asarray(obj['bbox_xyxy'],float)
            candidates=[]
            for person in people:
                a=np.asarray(person['bbox_xyxy'],float)
                height=max(a[3]-a[1],1.)
                scale=max(1.,min(height,float(np.linalg.norm(b[2:]-b[:2]))))
                gap=np.maximum(0,np.maximum(a[:2]-b[2:],b[:2]-a[2:]))
                distance=float(np.linalg.norm(gap))/scale
                # Reject overhead/facade signs and distant floor objects.
                feet_distance=float(np.linalg.norm(
                    np.array([(a[0]+a[2])/2,a[3]])-
                    np.array([(b[0]+b[2])/2,b[3]])))/height
                if distance>.15 or feet_distance>.75:continue
                if obj['class_id']==ClassId.ADVERTISING_SIGN and b[3]-b[1]>2*height:continue
                score=distance+feet_distance
                candidates.append((score,person))
                choices.setdefault(person['track_id'],[]).append((score,obj['track_id']))
            candidates.sort(key=lambda item:item[0])
            if not candidates or (len(candidates)>1 and candidates[1][0]-candidates[0][0]<.15):continue
            current[obj['track_id']]={'candidate':candidates[0]}
        accepted={}
        for obj_id,row in current.items():
            score,person=row['candidate']
            ranked=sorted(choices[person['track_id']])
            if ranked[0][1]!=obj_id or (len(ranked)>1 and ranked[1][0]-ranked[0][0]<.1):continue
            old=self.links.get(obj_id)
            same=old and old['person_id']==person['track_id'] and 0<=timestamp-old['last']<=.5
            first=old['first'] if same else timestamp
            samples=old['samples'] if same else 0
            last_frame=old['last_detection'] if same else None
            obj=next(t for t in valid if t['track_id']==obj_id)
            marker=obj.get('last_detection_frame', timestamp)
            if obj['position_source']==person['position_source']=='detection' and marker!=last_frame:
                samples+=1;last_frame=marker
            accepted[obj_id]={'person_id':person['track_id'],'first':first,'last':timestamp,
                'samples':samples,'last_detection':last_frame,
                'supported':samples>=3 and timestamp-first>=.4}
        current=accepted
        self.links=current

    def capture(self, event, observations, original, output, working_shape, model_version, image_frame=None):
        by_id={t['track_id']:t for t in observations}
        members=[by_id[tid] for tid in event['track_ids'] if tid in by_id
                 and (not event.get('class_ids') or by_id[tid]['class_id'] in event['class_ids'])]
        event['images']=[]
        if not members:
            event['evidence_status']='skipped_no_related_person'
            return False
        related=[]
        family=event.get('evidence',{}).get('family')
        if family is None and not any(t['class_id']==ClassId.PERSON for t in members):
            # Tương thích sự kiện cũ; vật liệu và biển không ghép thêm người.
            for obj in members:
                if obj['class_id'] in (ClassId.CONSTRUCTION_MATERIALS,ClassId.ADVERTISING_SIGN):continue
                link=self.links.get(obj['track_id'])
                if link and link['supported'] and link['person_id'] in by_id:
                    related.append(by_id[link['person_id']])
        members=list({t['track_id']:t for t in members+related}.values())
        from uav_pipeline.demo_people import attach_demo_people
        attach_demo_people(event,members)
        boxes=np.array([t['bbox_xyxy'] for t in members],float)
        oh,ow=original.shape[:2];wh,ww=working_shape[:2]
        box=np.r_[boxes[:,:2].min(axis=0),boxes[:,2:].max(axis=0)]*np.array([ow/ww,oh/wh,ow/ww,oh/wh])
        padding=np.maximum((box[2:]-box[:2])*.15,8)
        x1,y1=np.maximum(0,np.floor(box[:2]-padding)).astype(int)
        x2,y2=np.minimum([ow,oh],np.ceil(box[2:]+padding)).astype(int)
        if x2<=x1 or y2<=y1:raise RuntimeError('Invalid evidence crop')
        folder=Path(output)/'event_images';folder.mkdir(exist_ok=True)
        paths=[]
        for label,image in [('crop',original[y1:y2,x1:x2]),('context',original)]:
            frame=event['confirmation_frame'] if image_frame is None else image_frame
            path=folder/(f"{event['event_id']}_crop.jpg" if label=='crop' else f'snapshot_{frame:09d}.jpg')
            temporary=path.with_name(path.stem+'.tmp.jpg')
            if not cv2.imwrite(str(temporary),image,[cv2.IMWRITE_JPEG_QUALITY,92]):
                raise RuntimeError('Cannot save evidence image')
            temporary.replace(path)
            paths.append({'path':str(path.relative_to(output)),'role':label})
        from uav_pipeline.uploads import behavior_name
        from uav_pipeline.video.io import atomic_json
        import json
        snapshot=folder/f'snapshot_{frame:09d}.json'
        metadata=json.loads(snapshot.read_text()) if snapshot.exists() else {'frame':frame,'alerts':[]}
        metadata['alerts']=[r for r in metadata['alerts'] if r['event_id']!=event['event_id']]
        metadata['alerts'].append({'event_id':event['event_id'],'description':behavior_name({**event,'kind':event.get('kind','sidewalk_encroachment')}),
                                  'bbox_source_pixels':[int(x1),int(y1),int(x2),int(y2)],
                                  **({'demo_people':event['demo_people']} if event.get('demo_people') else {}),
                                  **({k:event[k] for k in ('confidence','confidence_source')}
                                     if event.get('confidence_source')=='demo' else {})})
        metadata['description']='; '.join(dict.fromkeys(r['description'] for r in metadata['alerts']))
        atomic_json(snapshot,metadata)
        event['snapshot_metadata']=str(snapshot.relative_to(output))
        event['images']=paths
        event['evidence_status']='ready'
        event['model_version']=model_version
        event['group']={'track_ids':[t['track_id'] for t in members],
                        'person_track_ids':[t['track_id'] for t in members if t['class_id']==ClassId.PERSON],
                        'bbox_source_pixels':[int(x1),int(y1),int(x2),int(y2)],
                        'interpretation':'grouped_region_requires_review',
                        'rendered':bool(family),'image_frame':event['confirmation_frame'] if image_frame is None else image_frame}
        return True
