"""Temporal rule alerts, not an automatic legal or physical-attachment verdict."""
from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from uav_pipeline.detection.classes import ClassId
from uav_pipeline.detection.postprocess import ENCROACHMENT_CLASSES, SIDEWALK_AREA_CLASSES, annotate_encroachment
from uav_pipeline.video.io import atomic_json, open_writer, write_frame


@dataclass(frozen=True)
class EventOptions:
    encroachment_frames: int = 10
    crowd_min_people: int = 4
    crowd_gap_ratio: float = .5
    attachment_frames: int = 10
    min_detector_observations: int = 2
    pre_seconds: float = 2.
    post_seconds: float = 8.
    cooldown_seconds: float = 14.
    event_hold_seconds: float = 2.
    max_clip_seconds: float = 15.
    contact_margin_pixels: int = 0
    attachment_gap_ratio: float = .15
    attachment_relative_tolerance: float = .2
    attachment_camera_motion_ratio: float = .01
    repair_seconds: float = 2.
    repair_gap_ratio: float = .25
    repair_stationary_ratio: float = .2

    def validate(self):
        for name in ('encroachment_frames', 'attachment_frames', 'min_detector_observations', 'crowd_min_people'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not isinstance(self.contact_margin_pixels, int) or self.contact_margin_pixels < 0:
            raise ValueError('Invalid contact margin')
        for name in ('pre_seconds','post_seconds','cooldown_seconds','attachment_gap_ratio',
                     'attachment_relative_tolerance','attachment_camera_motion_ratio'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid event option {name}')
        if self.post_seconds <= 0:
            raise ValueError('post_seconds must be positive')
        for name in ('repair_seconds', 'repair_gap_ratio', 'repair_stationary_ratio',
                     'event_hold_seconds', 'max_clip_seconds', 'crowd_gap_ratio'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')


def on_sidewalk(box, mask, margin=0):
    """Conservative contact test: the center and adjacent foot pixels must be sidewalk."""
    b = np.asarray(box, float)
    if b.shape != (4,) or not np.isfinite(b).all() or np.any(b[2:] <= b[:2]):
        return False
    h, w = mask.shape
    x, y = math.floor((b[0]+b[2])/2), math.ceil(b[3])-1
    if not (margin <= x < w-margin and margin <= y < h-margin):
        return False
    return bool(np.all(mask[y-margin:y+margin+1, x-margin:x+margin+1] == 2))


def attachment_candidates(sign, supports, gap_limit):
    """Image-space candidates only; a nearby support does not establish attachment."""
    a=np.asarray(sign['bbox_xyxy'],float);size=a[2:]-a[:2]
    if np.any(size<=0):return []
    candidates=[]
    for support in supports:
        b=np.asarray(support['bbox_xyxy'],float);extent=b[2:]-b[:2]
        if np.any(extent<=0):continue
        gap=np.maximum(0,np.maximum(a[:2]-b[2:],b[:2]-a[2:]))
        ratio=float(np.linalg.norm(gap)/max(np.linalg.norm(size),1))
        vertical=max(0,min(a[3],b[3])-max(a[1],b[1]))/size[1]
        if ratio<=gap_limit and vertical>=.5:
            candidates.append((ratio,support,b,extent))
    return sorted(candidates,key=lambda row:row[0])


class ViolationMonitor:
    def __init__(self, options=None):
        self.options = options or EventOptions()
        self.options.validate()
        self.presence, self.pairs, self.last_alert = {}, {}, {}
        self.events = []
        self.event_sequence = 0
        self.active_events = []
        self.confirmed_track_ids = set()
        self.confirmed_last_seen = {}
        self.previous_index = None
        self.scene = 0
        self.repairs = {}
        self.repair_regions = []
        self.violation_regions = []
        self.model_activities = {}
        self.current_detected_at = None

    def _step(self, mapping, key, index, timestamp, observed):
        state = mapping.get(key)
        if state is None or state['last_index'] not in (index-1,index):
            state = {'first_frame': index, 'first_timestamp': timestamp, 'count': 0,
                     'detector_count': 0, 'last_detector_frame': None, 'alerted': False, 'camera_motion': 0., 'relative': None,
                     'detected_at': self.current_detected_at}
            mapping[key] = state
        if state.get('last_index') != index:
            state['count'] += 1
        state['last_index'] = index
        if observed and state['last_detector_frame'] != index:
            state['detector_count'] += 1
            state['last_detector_frame'] = index
        return state

    def _merge_event(self, kind, tracks, index, timestamp, family=None):
        # Nối đợt cũ theo ID hoặc giao bbox sau khi bù chuyển động camera.
        from uav_pipeline.violation_regions import union_box
        boxes={cid:union_box([t for t in tracks if t['class_id']==cid])
               for cid in {t['class_id'] for t in tracks if t['class_id']!=ClassId.PERSON or family=='crowd'}}
        for episode in self.active_events:
            event = episode['event']
            if event['last_frame'] == index or event['kind'] != kind:
                continue
            if family is not None and family != event.get('evidence',{}).get('family'):
                continue
            if family is None and set(boxes) != set(episode['boxes']):
                continue
            current_ids={t['track_id'] for t in tracks if t['class_id']!=ClassId.PERSON or family=='crowd'}
            shared=set(event['track_ids']) & current_ids
            same_ids=bool(shared)
            if family=='crowd':same_ids=len(shared)>=math.ceil(min(len(event['track_ids']),len(current_ids))/2)
            overlaps = []
            compared=boxes
            previous=episode['boxes']
            if family is not None:
                def envelope(values):
                    values=np.asarray(list(values),float)
                    return np.r_[values[:,:2].min(axis=0),values[:,2:].max(axis=0)]
                compared={0:envelope(boxes.values())};previous={0:envelope(previous.values())}
            for cid, box in compared.items():
                old = previous[cid]
                intersection = np.prod(np.maximum(0, np.minimum(box[2:], old[2:])-np.maximum(box[:2], old[:2])))
                union = np.prod(box[2:]-box[:2])+np.prod(old[2:]-old[:2])-intersection
                overlaps.append(intersection/max(union, 1.) >= .5)
            if not same_ids and not all(overlaps):
                continue
            episode['boxes'] = boxes
            event['last_frame'] = index
            event['last_timestamp_seconds'] = timestamp
            ids = [t['track_id'] for t in tracks]
            if ids not in event['track_id_history']:
                event['track_id_history'].append(ids)
                del event['track_id_history'][:-64]
            return event
        return None

    def _emit(self, kind, key, state, index, timestamp, tracks, extra=None):
        options = self.options
        merged = self._merge_event(kind, tracks, index, timestamp,(extra or {}).get('family'))
        if merged is not None:
            if extra:
                ids=[t['track_id'] for t in tracks]
                changed=(extra.get('family') and (set(ids)!=set(merged['track_ids']) or
                         extra.get('behavior_name')!=merged['evidence'].get('behavior_name')))
                merged['evidence'].update(extra)
                if changed:
                    merged['track_ids']=ids;merged['class_ids']=[t['class_id'] for t in tracks]
                    merged['evidence_status']='pending';merged.pop('evidence_image_error',None)
                    from uav_pipeline.demo_people import attach_demo_people
                    attach_demo_people(merged,tracks)
            state['alerted'] = True
            if kind in ('sidewalk_encroachment', 'sign_attachment_suspected'):
                self.confirmed_track_ids.update(t['track_id'] for t in tracks)
            return None
        if state['alerted'] or timestamp-self.last_alert.get((kind,key), -1e9) < options.cooldown_seconds:
            return None
        if state['detector_count'] < options.min_detector_observations:
            return None
        state['alerted'] = True
        self.last_alert[(kind,key)] = timestamp
        self.event_sequence += 1
        event = {'event_id': f'event_{self.event_sequence:06d}', 'kind': kind,
                 'status': 'rule_triggered_requires_review', 'requires_review': True,
                 'track_ids': [t['track_id'] for t in tracks], 'class_ids': [t['class_id'] for t in tracks],
                 'first_frame': state['first_frame'], 'first_timestamp_seconds': state['first_timestamp'],
                 'detected_at': state.get('detected_at'),
                 'confirmation_frame': index, 'confirmation_timestamp_seconds': timestamp,
                 'consecutive_valid_frames': state['count'],
                 'independent_detector_observations': state['detector_count'],
                 'scene_index': self.scene,
                 'requested_pre_seconds': options.pre_seconds, 'requested_post_seconds': options.post_seconds,
                 'evidence': extra or {}, 'clip': None}
        from uav_pipeline.demo_people import attach_demo_people
        attach_demo_people(event,tracks)
        from uav_pipeline.demo_confidence import demo_confidence
        event.update(demo_confidence((event['event_id'],event.get('detected_at'))))
        event['evidence']['confirmation_boxes_source_pixels'] = [
            {'track_id':t['track_id'],'class_id':t['class_id'],
             'bbox_xyxy':(np.asarray(t['bbox_xyxy'])*self.source_factor).tolist()} for t in tracks]
        if kind not in ('motorcycle_repair_suspected','vehicle_repair_suspected',
                         'street_vending_suspected','vehicle_washing_suspected','crowd_gathering_suspected'):
            self.confirmed_track_ids.update(event['track_ids'])
        event.update(last_frame=index, last_timestamp_seconds=timestamp,
                     track_id_history=[event['track_ids']])
        from uav_pipeline.violation_regions import union_box
        self.active_events.append({'event':event,'boxes':{
            cid:union_box([t for t in tracks if t['class_id']==cid])
            for cid in set(event['class_ids']) if cid!=ClassId.PERSON or (extra or {}).get('family')=='crowd'}})
        self.events.append(event)
        return event

    def update(self, index, timestamp, observations, mask, motion, source_size=None, activity_regions=None, detected_at=None):
        if detected_at is not None:
            from datetime import datetime
            moment = datetime.fromisoformat(detected_at)
            if moment.tzinfo is None:
                raise ValueError('Event wall time must include timezone')
        self.current_detected_at = detected_at
        h,w=mask.shape
        sw,sh=source_size or (w,h)
        self.source_factor=np.array([sw/w,sh/h,sw/w,sh/h])
        if self.previous_index is not None and index <= self.previous_index:
            raise ValueError('Event input must be strictly increasing source frames')
        if motion.get('scene_cut') or (self.previous_index is not None and index != self.previous_index+1):
            self.presence.clear(); self.pairs.clear(); self.confirmed_track_ids.clear(); self.scene += 1
            self.repairs.clear(); self.model_activities.clear()
            self.active_events.clear()
        self.active_events = [e for e in self.active_events
            if timestamp-e['event']['last_timestamp_seconds'] <= self.options.event_hold_seconds]
        if motion.get('reliable'):
            matrix = np.asarray(motion['matrix'], float)
            for episode in self.active_events:
                for cid, box in episode['boxes'].items():
                    corners = np.array([[box[0],box[1],1], [box[2],box[1],1],
                                        [box[0],box[3],1], [box[2],box[3],1]]) @ matrix.T
                    episode['boxes'][cid] = np.r_[corners.min(axis=0), corners.max(axis=0)]
        self.last_alert = {k:v for k,v in self.last_alert.items()
                           if timestamp-v < self.options.cooldown_seconds}
        for track in observations:
            if track['track_id'] in self.confirmed_track_ids:
                self.confirmed_last_seen[track['track_id']] = timestamp
        self.confirmed_last_seen = {k:v for k,v in self.confirmed_last_seen.items()
            if k in self.confirmed_track_ids and timestamp-v <= max(1.,self.options.cooldown_seconds)}
        self.confirmed_track_ids.intersection_update(self.confirmed_last_seen)
        self.previous_index = index
        o = self.options
        valid = [t for t in observations if t['position_source'] in ('detection','optical_flow','camera_motion')
                 and t.get('detection_age_seconds',0) <= .5
                 and np.isfinite(t['bbox_xyxy']).all()]
        valid = [annotate_encroachment(t, mask, 20.*h/sh) for t in valid]
        fresh = lambda t: t['position_source'] == 'detection'
        signs=[t for t in valid if t['class_id']==ClassId.ADVERTISING_SIGN]
        supports=[t for t in valid if t['class_id'] in (ClassId.UTILITY_POLE,ClassId.TREE_TRUNK)]
        candidates_by_sign={t['track_id']:attachment_candidates(t,supports,o.attachment_gap_ratio)
                            for t in signs}
        from uav_pipeline.violation_regions import grouped_regions,crowd_regions
        eligible=[]
        for track in valid:
            if track['class_id'] not in ENCROACHMENT_CLASSES | {ClassId.ADVERTISING_SIGN}:
                continue
            # A projected sign foot on the mask cannot distinguish hanging from standing.
            # Defer nearby signs to the temporal attachment rule, including ambiguous pairs.
            if track['class_id']==ClassId.ADVERTISING_SIGN and candidates_by_sign.get(track['track_id']):
                continue
            is_sign = track['class_id'] == ClassId.ADVERTISING_SIGN
            if is_sign and (track.get('subtype') != 'sidewalk_standee' or not track['is_encroaching']):
                continue
            area_class = track['class_id'] in SIDEWALK_AREA_CLASSES
            if not is_sign and not (track['is_encroaching'] if area_class else on_sidewalk(track['bbox_xyxy'],mask,o.contact_margin_pixels)):
                continue
            eligible.append(track)
        people=[t for t in valid if t['class_id']==ClassId.PERSON and t.get('track_confirmed',False)]
        self.violation_regions=grouped_regions(eligible,people)+crowd_regions(people,o.crowd_min_people,o.crowd_gap_ratio)
        created,present=[],set()
        for region in self.violation_regions:
            objects=region['objects'];members=region['members']
            key=(region['family'],min(t['track_id'] for t in objects))
            present.add(key)
            state=self._step(self.presence,key,index,timestamp,all(fresh(t) for t in objects))
            region['confirmed']=(state['count']>=o.encroachment_frames and
                state['detector_count']>=o.min_detector_observations and
                all(t.get('track_confirmed',True) for t in objects))
            if region['confirmed']:
                extra={'behavior_name':region['behavior_name'],'family':region['family'],
                    'person_count':len(members) if region['family']=='crowd' else None,
                    'contact_rule':('people_proximity' if region['family']=='crowd' else 'bbox_sidewalk_overlap_at_least_0.1' if region['family'] in ('materials','machinery')
                                    else 'sign_foot_within_20_source_pixels' if region['family']=='sign'
                                    else 'bottom_center_with_sidewalk_margin'),
                    'contact_evidence':[t.get('contact_evidence') for t in objects],
                    'combined_bbox_source_pixels':(np.asarray(region['bbox_xyxy'])*self.source_factor).tolist(),
                    'association_rule':region['association_rule']}
                event=self._emit(region.get('kind','sidewalk_encroachment'),key,state,index,timestamp,members,extra)
                if event:created.append(event)
        self.presence = {k:v for k,v in self.presence.items() if k in present}

        active = set()
        if motion.get('reliable'):
            matrix = np.asarray(motion['matrix'],float)
            h,w = mask.shape
            camera = float(np.linalg.norm(matrix[:,2])/max(math.hypot(w,h),1))
            for sign in signs:
                a=np.asarray(sign['bbox_xyxy'],float)
                candidates=candidates_by_sign[sign['track_id']]
                # Ambiguous two supports: don't choose a convenient pole/tree.
                if not candidates or (len(candidates)>1 and abs(candidates[0][0]-candidates[1][0])<.03):
                    continue
                ratio,support,b,extent = candidates[0]
                key = (sign['track_id'],support['track_id'])
                relative = ((a[:2]+a[2:])/2-(b[:2]+b[2:])/2)/np.maximum(extent,1)
                previous = self.pairs.get(key)
                if previous is not None and previous['relative'] is not None:
                    if np.max(np.abs(relative-previous['relative'])) > o.attachment_relative_tolerance:
                        self.pairs.pop(key,None)
                state = self._step(self.pairs,key,index,timestamp,fresh(sign) and fresh(support))
                if state['relative'] is None:state['relative']=relative
                state['camera_motion'] += camera
                active.add(key)
                if (state['count'] >= o.attachment_frames
                        and state['camera_motion'] >= o.attachment_camera_motion_ratio
                        and sign.get('track_confirmed',True) and support.get('track_confirmed',True)):
                    event = self._emit('sign_attachment_suspected',key,state,index,timestamp,[sign,support],
                        {'normalized_box_gap':ratio,'camera_motion_ratio':state['camera_motion'],
                         'interpretation':'persistent_2d_proximity_not_proof_of_attachment'})
                    if event:created.append(event)
        self.pairs={k:v for k,v in self.pairs.items() if k in active}
        if activity_regions is None:
            self.model_activities.clear()
            created.extend(self._update_repairs(index, timestamp, valid, mask, motion))
        else:
            self.repairs.clear()
            created.extend(self._update_model_activities(index,timestamp,valid,activity_regions))
        for tid in self.confirmed_track_ids:
            self.confirmed_last_seen.setdefault(tid,timestamp)
        return created

    def _update_model_activities(self,index,timestamp,observations,regions):
        """Model-supported alerts replace proximity-only repair guesses when enabled."""
        from uav_pipeline.activities import ACTIVITY_KINDS
        by_id={row['track_id']:row for row in observations}
        self.repair_regions=[]
        created=[];active=set()
        for region in regions:
            if (not region.get('confirmed') or region.get('kind') not in ACTIVITY_KINDS
                    or region.get('evidence_age_seconds',1e9) > region.get('max_evidence_age_seconds',.75)):
                continue
            tracks=[by_id[tid] for tid in region['track_ids'] if tid in by_id]
            if len(tracks)!=len(region['track_ids']):
                continue
            key=(region['kind'],*region['track_ids'])
            active.add(key)
            observed=all(row['position_source']=='detection' for row in tracks)
            state=self._step(self.model_activities,key,index,timestamp,observed)
            evidence={k:v for k,v in region.items() if k!='bbox_xyxy'}
            evidence['combined_bbox_source_pixels']=(np.asarray(region['bbox_xyxy'])*self.source_factor).tolist()
            evidence['interpretation']='temporal_image_classifier_hypothesis_requires_review'
            event=self._emit(region['kind'],key,state,index,timestamp,tracks,evidence)
            if event:
                created.append(event)
            self.repair_regions.append(region)
        self.model_activities={key:value for key,value in self.model_activities.items() if key in active}
        return created

    def _update_repairs(self, index, timestamp, observations, mask, motion):
        """Persistent stationary proximity is a review candidate, not action recognition."""
        self.repair_regions = []
        matrix = np.asarray(motion.get('matrix', []), float)
        if (not motion.get('reliable') or matrix.shape != (2, 3)
                or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:, :2])) < 1e-8):
            self.repairs.clear()
            return []
        o = self.options
        people = [t for t in observations if t['class_id'] == ClassId.PERSON
                  and on_sidewalk(t['bbox_xyxy'], mask, o.contact_margin_pixels)]
        bikes = [t for t in observations if t['class_id'] == ClassId.MOTORCYCLE
                 and on_sidewalk(t['bbox_xyxy'], mask, o.contact_margin_pixels)]
        active, created = set(), []
        for person in people:
            a = np.asarray(person['bbox_xyxy'], float)
            for bike in bikes:
                b = np.asarray(bike['bbox_xyxy'], float)
                scale = max(float(np.linalg.norm(b[2:] - b[:2])), 1.)
                gap = np.maximum(0, np.maximum(a[:2] - b[2:], b[:2] - a[2:]))
                feet = np.array([[(a[0]+a[2])/2, a[3]], [(b[0]+b[2])/2, b[3]]])
                if (np.linalg.norm(gap)/scale > o.repair_gap_ratio
                        or np.linalg.norm(feet[0]-feet[1])/scale > 1.):
                    continue
                key = (person['track_id'], bike['track_id'])
                previous = self.repairs.get(key)
                anchors = None
                if previous is not None:
                    anchors = previous['anchors'] @ matrix[:, :2].T + matrix[:, 2]
                    # Compare with the first position, not only the preceding frame:
                    # slow walking together must not accumulate stationary time.
                    if (not 0 < timestamp-previous['last_timestamp'] <= .5
                            or np.max(np.linalg.norm(feet-anchors, axis=1))/scale > o.repair_stationary_ratio):
                        self.repairs.pop(key, None)
                        anchors = None
                observed = all(t['position_source'] == 'detection' for t in (person, bike))
                state = self._step(self.repairs, key, index, timestamp, observed)
                state['anchors'] = feet.copy() if anchors is None else anchors
                state['last_timestamp'] = timestamp
                active.add(key)
                duration = timestamp-state['first_timestamp']
                if (duration + 1e-9 < o.repair_seconds
                        or not all(t.get('track_confirmed', True) for t in (person, bike))
                        or state['detector_count'] < o.min_detector_observations):
                    continue
                box = np.concatenate((np.minimum(a[:2], b[:2]), np.maximum(a[2:], b[2:])))
                evidence = {'duration_seconds': duration, 'normalized_box_gap': float(np.linalg.norm(gap)/scale),
                            'combined_bbox_source_pixels': (box*self.source_factor).tolist(),
                            'interpretation': 'stationary_sidewalk_proximity_not_proof_of_repair'}
                event = self._emit('motorcycle_repair_suspected', key, state, index, timestamp,
                                   [person, bike], evidence)
                if event:
                    created.append(event)
                # Cooldown suppresses duplicate events, not a currently valid box.
                self.repair_regions.append({'kind': 'motorcycle_repair_suspected',
                    'label': 'Nghi sửa xe', 'requires_review': True,
                    'track_ids': list(key), 'bbox_xyxy': box.tolist(), 'duration_seconds': duration})
        self.repairs = {k:v for k,v in self.repairs.items() if k in active}
        return created

    def render(self, image, observations):
        active=self.confirmed_track_ids.intersection(t['track_id'] for t in observations
                                                     if t.get('is_encroaching', True))
        grouped_ids={t['track_id'] for r in self.violation_regions for t in r['members']}
        for track in observations:
            if track['track_id'] in active and track['track_id'] not in grouped_ids:
                box=np.rint(track['bbox_xyxy']).astype(int)
                cv2.rectangle(image,tuple(box[:2]),tuple(box[2:]),(0,0,255),3)
        from uav_pipeline.video.pipeline import _put_label
        for region in self.violation_regions:
            box=np.rint(region['bbox_xyxy']).astype(int)
            confirmed=region['confirmed'] or any(t['track_id'] in active for t in region['objects'])
            color=(0,0,255) if confirmed else (0,180,255)
            cv2.rectangle(image,tuple(box[:2]),tuple(box[2:]),color,2)
            import textwrap
            width=max(18,min(65,(image.shape[1]-max(0,int(box[0])))//10))
            for line,text in enumerate(textwrap.wrap(('' if confirmed else 'Chưa xác nhận: ')+region['behavior_name'],width)):
                _put_label(image,text,max(0,int(box[0])),max(0,int(box[1])-22)+line*22,color)
        for region in self.repair_regions:
            if region.get('model') == 'mobilenet_v3_small':
                continue  # The classifier renderer includes confidence and pending states.
            from uav_pipeline.video.pipeline import _put_label
            box = np.rint(region['bbox_xyxy']).astype(int)
            cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (0,0,255), 3)
            _put_label(image, region['label'], int(box[0]), max(0, int(box[1])-22), (0,0,255))
        return image

    def export(self, output, fps, frames, video_available=True, merge_clips=True):
        """Gộp các cửa sổ chồng nhau khi clip chung không vượt giới hạn."""
        output=Path(output); clips=[]; errors=[]; jobs=[]
        limit=max(1,int(math.floor(self.options.max_clip_seconds*fps)))
        windows=[]
        for event in self.events:
            anchor=event['first_frame']
            first=max(0,int(math.floor(anchor-self.options.pre_seconds*fps)))
            requested=int(math.ceil(anchor+self.options.post_seconds*fps))
            last=min(frames,requested,first+limit)
            windows.append((first,last,requested,event))
        for first,last,requested,event in sorted(windows,key=lambda row:(row[0],row[1])):
            group=next((g for g in reversed(clips)
                        if merge_clips and first < g['last'] and max(last,g['last'])-g['first'] <= limit),None)
            if group is None:
                group={'first':first,'last':last,'members':[]}
                clips.append(group)
            group['last']=max(group['last'],last)
            group['members'].append((event,requested))
        from uav_pipeline.uploads import behavior_name
        clip_reports=[]
        for number,group in enumerate(clips,1):
            first,last=group['first'],group['last']
            members=group['members']
            alerts=[{'event_id':e.get('event_id',f'event_{self.events.index(e)+1:06d}'),
                     'kind':e['kind'],'track_ids':e['track_ids'],
                     'description':behavior_name(e),
                     **({'demo_people':e['demo_people']} if e.get('demo_people') else {})} for e,_ in members]
            info={'clip_id':f'clip_{number:06d}',
                  'start_output_seconds':first/fps,'end_output_seconds':last/fps,
                  'actual_duration_seconds':(last-first)/fps,
                  'max_duration_seconds':self.options.max_clip_seconds,
                  'description':'; '.join(dict.fromkeys(a['description'] for a in alerts)),
                  'alerts':alerts,'shared':len(members)>1,
                  'timebase':'annotated_CFR_source_frames; source_PTS_is_in_event_metadata'}
            if video_available:
                (output/'event_clips').mkdir(exist_ok=True)
                info.update(path=f'event_clips/clip_{number:06d}.mp4',status='saved')
                jobs.append((output/info['path'],first,last))
            else:
                info.update(status='unavailable',error='Final annotated video unavailable')
                errors.append(info['error'])
            from uav_pipeline.demo_confidence import demo_confidence
            info.update(demo_confidence((info['clip_id'],min(e.get('detected_at') or '' for e,_ in members))))
            clip_reports.append(info)
            for event,requested in members:
                event['clip']={**info,'anchor_output_seconds':event['first_frame']/fps,
                               'truncated_start':event['first_frame']<self.options.pre_seconds*fps,
                               'truncated_end':requested>last}
        failures=self._write_clips(output/'annotated.mp4',jobs,fps) if jobs else {}
        for event in self.events:
            info=event['clip']
            if info.get('path') in failures:
                info.update(status='error',error=failures[info['path']])
        for info in clip_reports:
            if info.get('path') in failures:
                info.update(status='error',error=failures[info['path']])
        errors.extend(failures.values())
        report={'options':asdict(self.options),'alert_count':len(self.events),
                'counts_by_kind':{kind:sum(e['kind']==kind for e in self.events)
                                  for kind in ('sidewalk_encroachment','sign_attachment_suspected',
                                               'motorcycle_repair_suspected','vehicle_repair_suspected',
                                               'street_vending_suspected','vehicle_washing_suspected','crowd_gathering_suspected')},
                'unique_clips':len(clips)-len(failures),'clip_errors':errors,
                'requires_human_review':True,'clips':clip_reports,'events':self.events}
        atomic_json(output/'events.json',report)
        return {key:value for key,value in report.items() if key!='events'}

    @staticmethod
    def _write_clips(source, jobs, fps):
        """Export independent clips with at most two decoders/writers in memory."""
        def write_job(job):
            target,first,last=job
            temporary=target.with_suffix('.tmp.mp4')
            try:
                ViolationMonitor._write_clip(source,temporary,first,last,fps)
                temporary.replace(target)
                return None
            except (OSError,RuntimeError,cv2.error) as error:
                temporary.unlink(missing_ok=True)
                return f'event_clips/{target.name}',str(error)
        if not jobs:return {}
        with ThreadPoolExecutor(max_workers=min(2,len(jobs)),thread_name_prefix='event-clip') as pool:
            return dict(result for result in pool.map(write_job,jobs) if result is not None)

    @staticmethod
    def _write_clip(source,target,first,last,fps):
        """Always export H264/yuv420p MP4 with faststart for browser playback.

        Fast timestamp seeking can lose an EOF frame for rounded CFR rates.
        Retry with exact decoded frame trimming, never silently switch codec.
        """
        from uav_pipeline.video.io import _recording_ffmpeg
        executable=_recording_ffmpeg()
        if not executable:
            raise RuntimeError('Evidence clips require FFmpeg with H264 support')
        target=Path(target)
        common=[executable,'-nostdin','-hide_banner','-loglevel','error','-y']
        encoding=['-an','-c:v','libx264','-preset','veryfast','-crf','20',
                  '-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(target)]
        commands=[common+['-ss',format(first/fps,'.12f'),'-i',str(source),'-map','0:v:0',
                         '-frames:v',str(last-first),'-vf','setpts=PTS-STARTPTS']+encoding,
                  common+['-i',str(source),'-map','0:v:0','-vf',
                         f'trim=start_frame={first}:end_frame={last},setpts=PTS-STARTPTS',
                         '-fps_mode','passthrough']+encoding]
        for command in commands:
            try:
                result=subprocess.run(command,capture_output=True,
                    timeout=max(120,last/fps*2),stdin=subprocess.DEVNULL)
                if result.returncode==0:
                    probe=cv2.VideoCapture(str(target))
                    try:
                        count=int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
                        codec=int(probe.get(cv2.CAP_PROP_FOURCC))
                        tag=''.join(chr((codec>>(8*i))&255) for i in range(4)).lower()
                        ok,_=probe.read()
                    finally:probe.release()
                    if count==last-first and tag in ('h264','avc1') and ok:return
            except (OSError,subprocess.TimeoutExpired):
                pass
            target.unlink(missing_ok=True)
        raise RuntimeError('Cannot create exact-frame H264 evidence clip; no incompatible fallback was saved')
