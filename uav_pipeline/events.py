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
from uav_pipeline.detection.postprocess import ENCROACHMENT_CLASSES, annotate_encroachment
from uav_pipeline.video.io import atomic_json, open_writer, write_frame


@dataclass(frozen=True)
class EventOptions:
    encroachment_frames: int = 10
    attachment_frames: int = 10
    min_detector_observations: int = 2
    pre_seconds: float = 4.
    post_seconds: float = 10.
    cooldown_seconds: float = 14.
    contact_margin_pixels: int = 0
    attachment_gap_ratio: float = .15
    attachment_relative_tolerance: float = .2
    attachment_camera_motion_ratio: float = .01
    repair_seconds: float = 2.
    repair_gap_ratio: float = .25
    repair_stationary_ratio: float = .2

    def validate(self):
        for name in ('encroachment_frames', 'attachment_frames', 'min_detector_observations'):
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
        for name in ('repair_seconds', 'repair_gap_ratio', 'repair_stationary_ratio'):
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
        self.confirmed_track_ids = set()
        self.previous_index = None
        self.scene = 0
        self.repairs = {}
        self.repair_regions = []

    def _step(self, mapping, key, index, timestamp, observed):
        state = mapping.get(key)
        if state is None or state['last_index'] != index-1:
            state = {'first_frame': index, 'first_timestamp': timestamp, 'count': 0,
                     'detector_frames': set(), 'alerted': False, 'camera_motion': 0., 'relative': None}
            mapping[key] = state
        state['count'] += 1
        state['last_index'] = index
        if observed:
            state['detector_frames'].add(index)
        return state

    def _emit(self, kind, key, state, index, timestamp, tracks, extra=None):
        options = self.options
        if state['alerted'] or timestamp-self.last_alert.get((kind,key), -1e9) < options.cooldown_seconds:
            return None
        if len(state['detector_frames']) < options.min_detector_observations:
            return None
        state['alerted'] = True
        self.last_alert[(kind,key)] = timestamp
        event = {'event_id': f'event_{len(self.events)+1:06d}', 'kind': kind,
                 'status': 'rule_triggered_requires_review', 'requires_review': True,
                 'track_ids': [t['track_id'] for t in tracks], 'class_ids': [t['class_id'] for t in tracks],
                 'first_frame': state['first_frame'], 'first_timestamp_seconds': state['first_timestamp'],
                 'confirmation_frame': index, 'confirmation_timestamp_seconds': timestamp,
                 'consecutive_valid_frames': state['count'],
                 'independent_detector_observations': len(state['detector_frames']),
                 'scene_index': self.scene,
                 'requested_pre_seconds': options.pre_seconds, 'requested_post_seconds': options.post_seconds,
                 'evidence': extra or {}, 'clip': None}
        event['evidence']['confirmation_boxes_source_pixels'] = [
            {'track_id':t['track_id'],'class_id':t['class_id'],
             'bbox_xyxy':(np.asarray(t['bbox_xyxy'])*self.source_factor).tolist()} for t in tracks]
        if kind != 'motorcycle_repair_suspected':
            self.confirmed_track_ids.update(event['track_ids'])
        self.events.append(event)
        return event

    def update(self, index, timestamp, observations, mask, motion, source_size=None):
        h,w=mask.shape
        sw,sh=source_size or (w,h)
        self.source_factor=np.array([sw/w,sh/h,sw/w,sh/h])
        if self.previous_index is not None and index <= self.previous_index:
            raise ValueError('Event input must be strictly increasing source frames')
        if motion.get('scene_cut') or (self.previous_index is not None and index != self.previous_index+1):
            self.presence.clear(); self.pairs.clear(); self.confirmed_track_ids.clear(); self.scene += 1
            self.repairs.clear()
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
        created, present = [], set()
        for track in valid:
            tid = track['track_id']
            if track['class_id'] not in ENCROACHMENT_CLASSES | {ClassId.ADVERTISING_SIGN}:
                continue
            # A projected sign foot on the mask cannot distinguish hanging from standing.
            # Defer nearby signs to the temporal attachment rule, including ambiguous pairs.
            if track['class_id']==ClassId.ADVERTISING_SIGN and candidates_by_sign.get(tid):
                continue
            is_sign = track['class_id'] == ClassId.ADVERTISING_SIGN
            if is_sign and (track.get('subtype') != 'sidewalk_standee' or not track['is_encroaching']):
                continue
            if not is_sign and not on_sidewalk(track['bbox_xyxy'],mask,o.contact_margin_pixels):
                continue
            present.add(tid)
            state = self._step(self.presence,tid,index,timestamp,fresh(track))
            if state['count'] >= o.encroachment_frames and track.get('track_confirmed',True):
                event = self._emit('sidewalk_encroachment',tid,state,index,timestamp,[track],
                    {'contact_rule': 'sign_foot_within_20_source_pixels' if is_sign else 'bottom_center_with_sidewalk_margin',
                     'allowed_class_ids': sorted(ENCROACHMENT_CLASSES | {ClassId.ADVERTISING_SIGN}),
                     'sign_rule': 'sidewalk_contact_without_nearby_support' if track['class_id']==ClassId.ADVERTISING_SIGN else None})
                if event: created.append(event)
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
        created.extend(self._update_repairs(index, timestamp, valid, mask, motion))
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
                        or len(state['detector_frames']) < o.min_detector_observations):
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
        for track in observations:
            if track['track_id'] in active:
                box=np.rint(track['bbox_xyxy']).astype(int)
                cv2.rectangle(image,tuple(box[:2]),tuple(box[2:]),(0,0,255),3)
        if active:
            cv2.putText(image,f'RULE ALERT: {len(active)} tracks - review required',(12,52),
                        cv2.FONT_HERSHEY_SIMPLEX,.65,(0,0,255),2)
        for region in self.repair_regions:
            from uav_pipeline.video.pipeline import _put_label
            box = np.rint(region['bbox_xyxy']).astype(int)
            cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (0,0,255), 3)
            _put_label(image, region['label'], int(box[0]), max(0, int(box[1])-22), (0,0,255))
        return image

    def export(self, output, fps, frames, video_available=True):
        """Cut annotated clips after analysis. Constant memory; exact shared windows reuse one clip."""
        output=Path(output); duration=frames/fps; clips={}; errors=[]; jobs=[]
        for event in self.events:
            anchor=event['first_frame']/fps
            start=max(0.,anchor-self.options.pre_seconds)
            end=min(duration,anchor+self.options.post_seconds)
            first=max(0,int(math.floor(start*fps))); last=min(frames,int(math.ceil(end*fps)))
            key=(first,last)
            info={'anchor_output_seconds':anchor,'start_output_seconds':first/fps,'end_output_seconds':last/fps,
                  'actual_duration_seconds':(last-first)/fps,
                  'truncated_start':anchor<self.options.pre_seconds,
                  'truncated_end':anchor+self.options.post_seconds>duration,
                  'timebase':'annotated_CFR_source_frames; source_PTS_is_in_event_metadata'}
            if not video_available:
                info.update(status='unavailable',error='Final annotated video unavailable')
                errors.append(info['error'])
            elif key in clips:
                info.update(clips[key],shared=True)
            else:
                folder=output/'event_clips';folder.mkdir(exist_ok=True)
                relative=f'event_clips/clip_{len(clips)+1:06d}.mp4'
                clips[key]={'path':relative,'status':'saved'}
                jobs.append((output/relative,first,last))
                info.update(clips[key],shared=False)
            event['clip']=info
        failures=self._write_clips(output/'annotated.mp4',jobs,fps) if jobs else {}
        for event in self.events:
            info=event['clip']
            if info.get('path') in failures:
                info.update(status='error',error=failures[info['path']])
        errors.extend(failures.values())
        report={'options':asdict(self.options),'alert_count':len(self.events),
                'counts_by_kind':{kind:sum(e['kind']==kind for e in self.events)
                                  for kind in ('sidewalk_encroachment','sign_attachment_suspected',
                                               'motorcycle_repair_suspected')},
                'unique_clips':len(clips)-len(failures),'clip_errors':errors,
                'requires_human_review':True,'events':self.events}
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
        executable=shutil.which('ffmpeg')
        if executable:
            try:
                result=subprocess.run([executable,'-nostdin','-hide_banner','-loglevel','error','-y',
                    '-ss',format(first/fps,'.12f'),'-i',str(source),'-map','0:v:0',
                    '-frames:v',str(last-first),'-an','-c:v','libx264','-preset','ultrafast',
                    '-crf','20','-threads','2',str(target)],capture_output=True,
                    timeout=max(60,(last-first)/fps*5),stdin=subprocess.DEVNULL)
                if result.returncode==0:
                    probe=cv2.VideoCapture(str(target))
                    count=int(probe.get(cv2.CAP_PROP_FRAME_COUNT));probe.release()
                    if count==last-first:return
            except (OSError,subprocess.TimeoutExpired):
                pass
            target.unlink(missing_ok=True)
        capture=cv2.VideoCapture(str(source));writer=None
        try:
            if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES,first):
                raise RuntimeError('Cannot seek annotated video for event clip')
            for _ in range(first,last):
                ok,frame=capture.read()
                if not ok:raise RuntimeError('Unexpected end while writing event clip')
                if writer is None:writer=open_writer(target,fps,frame.shape)
                write_frame(writer,frame)
            if writer is None:raise RuntimeError('Empty event clip')
        finally:
            capture.release()
            if writer is not None:writer.release()
