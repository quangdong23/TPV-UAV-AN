"""Class-aware Kalman/Hungarian baseline with GMC and foreground flow.

This is a lightweight tracker, not an implementation of BoT-SORT/ReID.
Only actual detections confirm tracks; flow never creates a new identity.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from lan_chiem.video_motion import warp_box


def measurement(box):
    b = np.asarray(box, float)
    return np.concatenate(((b[:2] + b[2:]) / 2, np.maximum(1., b[2:] - b[:2])))


def bbox(state):
    center, size = state[:2], np.maximum(1., state[2:4])
    return np.concatenate((center - size / 2, center + size / 2))


def iou(a, b):
    intersection = np.maximum(0., np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0., a[2:] - a[:2]).prod() + np.maximum(0., b[2:] - b[:2]).prod() - intersection
    return float(intersection / max(union, 1e-6))


# Nhóm các lớp đối tượng tương đồng dễ dao động nhãn do góc nhìn UAV:
# 4: bàn nhựa, 5: ghế nhựa
# 6: xe đẩy bán hàng, 10: hàng hóa
# 7: biển quảng cáo, 8: cột điện, 9: thân cây

# Các lớp vật thể tĩnh trên vỉa hè / mặt đất không tự di chuyển:
# 4: bàn, 5: ghế, 6: xe đẩy, 7: biển, 8: cột, 9: cây, 10: hàng hóa
STATIC_CLASSES = {4, 5, 6, 7, 8, 9, 10}


def association_costs(tracks, detections):
    costs=np.full((len(tracks),len(detections)),1e6)
    if not costs.size:
        return costs
    state=np.stack([t['x'][:4] for t in tracks])
    boxes=np.stack([bbox(t['x']) for t in tracks])
    target=np.stack([measurement(d['bbox_xyxy']) for d in detections])
    observed=np.array([d['bbox_xyxy'] for d in detections],float)
    intersection=np.maximum(0.,np.minimum(boxes[:,None,2:],observed[None,:,2:])
                            -np.maximum(boxes[:,None,:2],observed[None,:,:2])).prod(-1)
    union=np.maximum(0.,boxes[:,2:]-boxes[:,:2]).prod(-1)[:,None]+np.maximum(0.,observed[:,2:]-observed[:,:2]).prod(-1)[None,:]-intersection
    overlap=intersection/np.maximum(union,1e-6)
    distance=np.linalg.norm(target[None,:,:2]-state[:,None,:2],axis=-1)/np.maximum(np.linalg.norm(state[:,2:4],axis=-1),1.)[:,None]
    size_change=np.abs(np.log(target[None,:,2:4]/np.maximum(state[:,None,2:4],1.))).max(-1)
    
    t_cids = np.array([t['class_id'] for t in tracks])
    d_cids = np.array([d['class_id'] for d in detections])
    same_class = t_cids[:, None] == d_cids[None, :]

    valid = same_class & (size_change < .8) & ((overlap >= .1) | (distance <= .75))
    costs[valid] = (1 - overlap + .2 * distance)[valid]
    return costs


class ObjectTracker:
    def __init__(
        self,
        min_hits=2,
        max_lost_seconds=1.,
        max_prediction_seconds=.5,
        max_static_lost_seconds=None,
        max_static_prediction_seconds=None,
        class_persistence_boost=True,
        persistence_keyframes=3,
        persistence_missed_keyframes=2,
    ):
        if not isinstance(class_persistence_boost, bool):
            raise ValueError('class_persistence_boost must be boolean')
        if (not isinstance(persistence_keyframes, int) or persistence_keyframes < 3
                or not isinstance(persistence_missed_keyframes, int) or persistence_missed_keyframes < 0):
            raise ValueError('Invalid class persistence keyframe limits')
        self.class_persistence_boost = class_persistence_boost
        self.persistence_keyframes = persistence_keyframes
        self.persistence_missed_keyframes = persistence_missed_keyframes
        self.min_hits = min_hits
        self.max_lost_seconds = max_lost_seconds
        self.max_prediction_seconds = max_prediction_seconds
        self.max_static_lost_seconds = max_lost_seconds if max_static_lost_seconds is None else max_static_lost_seconds
        self.max_static_prediction_seconds = max_prediction_seconds if max_static_prediction_seconds is None else max_static_prediction_seconds
        self.next_id = 1
        self.tracks = []
        self.confirmed_ids = set()
        self.events = []

    def boxes(self, timestamp=None):
        # Lost/expired tracks still participate in re-association, but should
        # not consume foreground LK work or mask useful camera-motion points.
        return {t['id']: bbox(t['x']) for t in self.tracks
                if not t.get('missed_detection', False) and
                (timestamp is None or timestamp - t['last_detection'] <= (
                    self.max_static_prediction_seconds
                    if t['class_id'] in STATIC_CLASSES and t['hits'] >= self.min_hits
                    else self.max_prediction_seconds))}

    @staticmethod
    def correct(t, box, noise):
        z = measurement(box)
        p = t['p']
        gain = np.linalg.solve(p[:4, :4] + np.eye(4) * noise, p[:4, :]).T
        t['x'] += gain @ (z - t['x'][:4])
        # Joseph form preserves a positive covariance under repeated updates.
        residual = np.eye(8)
        residual[:, :4] -= gain
        t['p'] = residual @ p @ residual.T + noise * gain @ gain.T

    def update(self, timestamp, detections, motion, observed_regions=None, observed_classes=None):
        self.events = []
        def event(t, state, reason):
            self.events.append({'track_id': t['id'], 'class_id': t['class_id'],
                                'track_state': state, 'reason': reason})
        if motion['scene_cut']:
            for t in self.tracks:
                event(t, 'ended', 'scene_cut')
            self.tracks = []  # IDs remain globally unique within the video.
        for t in self.tracks:
            timeout = (self.max_static_lost_seconds
                       if t['class_id'] in STATIC_CLASSES and (t['hits'] >= self.min_hits or t.get('roi_dependent', False) or t.get('sparse_observation',False))
                       else self.max_lost_seconds)
            if timestamp - t['last_detection'] > timeout:
                event(t, 'ended', 'detection_timeout')
        self.tracks = [
            t for t in self.tracks
            if timestamp - t['last_detection'] <= (
                self.max_static_lost_seconds
                if t['class_id'] in STATIC_CLASSES and (t['hits'] >= self.min_hits or t.get('roi_dependent', False) or t.get('sparse_observation',False))
                else self.max_lost_seconds
            )
        ]
        for t in self.tracks:
            dt = max(0., timestamp - t['timestamp'])
            matrix = motion['matrix']
            if motion['reliable']:
                t['x'][:4] = measurement(warp_box(bbox(t['x']), matrix))
                if t.get('smooth_box') is not None:
                    t['smooth_box'] = warp_box(t['smooth_box'], matrix)
                scale = np.hypot(matrix[0, 0], matrix[1, 0])
                transform = np.eye(8)
                transform[:2, :2] = matrix[:, :2]
                transform[4:6, 4:6] = matrix[:, :2]
                for index in (2, 3, 6, 7):
                    transform[index, index] = scale
                t['x'][4:] = transform[4:, 4:] @ t['x'][4:]
                t['p'] = transform @ t['p'] @ transform.T
            # Damping velocity for confirmed static objects so they don't drift
            if t['class_id'] in STATIC_CLASSES and t['hits'] >= self.min_hits:
                t['x'][4:] *= 0.5
            flow_anchor = t['x'][:4].copy()
            transition = np.eye(8)
            transition[:4, 4:] = np.eye(4) * dt
            t['x'] = transition @ t['x']
            t['p'] = transition @ t['p'] @ transition.T + np.eye(8) * max(dt, .001) * 4
            t['timestamp'] = timestamp
            t['source'] = 'prediction'
            if t['id'] in motion['local_boxes']:
                # LK measures displacement from the previous tracked box.
                # Filtering that absolute box again accumulates position lag
                # because repeated pseudo-measurements collapse covariance.
                previous = flow_anchor
                observed = measurement(motion['local_boxes'][t['id']])
                if dt > 0 and t['class_id'] not in STATIC_CLASSES:
                    velocity = (observed[:2] - previous[:2]) / dt
                    t['x'][4:6] = .5 * t['x'][4:6] + .5 * velocity
                t['x'][:4] = observed
                t['source'] = 'optical_flow'
        if detections is not None:
            detections = [d for d in detections if d.get('class_id') is not None
                          and not d.get('candidate_class_ids') and d.get('status', 'predicted') == 'predicted'
                          and np.isfinite(d['bbox_xyxy']).all()
                          and np.all(np.asarray(d['bbox_xyxy'])[2:] > np.asarray(d['bbox_xyxy'])[:2])]
            costs = association_costs(self.tracks, detections)
            matched_tracks, matched_dets = set(), set()
            if costs.size:
                rows, cols = linear_sum_assignment(costs)
                for a, b in zip(rows, cols):
                    if costs[a, b] >= 1e5:
                        continue
                    t, d = self.tracks[a], detections[b]
                    if t.get('missed_detection', False):
                        t['smooth_box'] = None
                    self.correct(t, d['bbox_xyxy'], 4.)
                    # The matched detector is the absolute observation, not a
                    # weak correction to an already drifting flow trajectory.
                    t['x'][:4] = measurement(d['bbox_xyxy'])
                    t.update(last_detection=timestamp, hits=t['hits'] + 1,
                             score=float(d['score']), source='detection',
                             missed_detection=False, miss_count=0,
                             roi_dependent=d.get('source', '').startswith('roi_'))
                    t['sparse_observation'] = d.get('source') == 'trunk_refiner'
                    t['consecutive_observations'] = t.get('consecutive_observations', 1) + 1
                    if t['class_id'] in (7, 8, 9) and t['consecutive_observations'] >= self.persistence_keyframes:
                        t['persistent_class'] = self.class_persistence_boost
                    t['raw_label'] = d.get('raw_label', '')
                    t['class_name_vi'] = d.get('class_name_vi', str(d['class_id']))
                    matched_tracks.add(a)
                    matched_dets.add(b)
            is_empty_scene = (len(detections) == 0)
            for a, t in enumerate(self.tracks):
                if a not in matched_tracks:
                    if observed_classes is not None and t['class_id'] not in observed_classes:
                        continue
                    # A global pass has already missed this ROI-only object before.
                    # Only a crop containing its complete predicted box supplies
                    # comparable negative evidence. Timeouts still apply normally.
                    if observed_regions is not None and t.get('roi_dependent', False):
                        box = bbox(t['x'])
                        if not any(box[0] > r[0]+2 and box[1] > r[1]+2 and
                                   box[2] < r[2]-2 and box[3] < r[3]-2
                                   for r in observed_regions):
                            continue
                    if t['hits'] < self.min_hits:
                        # Two isolated guesses must not confirm a tentative ID.
                        t['hits'] = 0
                    t['miss_count'] = t.get('miss_count', 0) + 1
                    t['consecutive_observations'] = 0
                    persistent = (t.get('persistent_class', False)
                                  and t['miss_count'] <= self.persistence_missed_keyframes
                                  and (motion['reliable'] or t['id'] in motion['local_boxes']))
                    if persistent:
                        t['source'] = ('optical_flow' if t['id'] in motion['local_boxes'] else 'camera_motion')
                        continue
                    is_confirmed_static = (t['class_id'] in STATIC_CLASSES and t['hits'] >= self.min_hits)
                    max_miss = 4 if is_confirmed_static else 2
                    if is_empty_scene or t['miss_count'] >= max_miss:
                        t['missed_detection'] = True
            for b, d in enumerate(detections):
                if b in matched_dets:
                    continue
                self.tracks.append({'id': self.next_id, 'class_id': int(d['class_id']),
                                    'class_name_vi': d.get('class_name_vi', str(d['class_id'])),
                                    'x': np.r_[measurement(d['bbox_xyxy']), np.zeros(4)],
                                    'p': np.diag([10.] * 4 + [100.] * 4), 'timestamp': timestamp,
                                    'last_detection': timestamp, 'hits': 1, 'score': float(d['score']),
                                    'source': 'detection', 'missed_detection': False, 'miss_count': 0,
                                    'consecutive_observations': 1, 'persistent_class': False,
                                    'raw_label': d.get('raw_label', ''),
                                    'roi_dependent': d.get('source', '').startswith('roi_'),
                                    'sparse_observation': d.get('source') == 'trunk_refiner',
                                    'smooth_box': np.asarray(d['bbox_xyxy'], float)})
                self.next_id += 1
        output = []
        for t in self.tracks:
            if t['hits'] >= self.min_hits:
                self.confirmed_ids.add(t['id'])
            age = timestamp - t['last_detection']
            is_confirmed_static = (t['class_id'] in STATIC_CLASSES and t['hits'] >= self.min_hits)
            max_pred = self.max_static_prediction_seconds if is_confirmed_static else self.max_prediction_seconds
            state = ('tentative' if t['hits'] < self.min_hits else
                     'lost' if t.get('missed_detection', False) or age > max_pred else 'active')
            if t.get('reported_state') != state:
                event(t, state, 'detector_or_motion_update')
                t['reported_state'] = state
            # No detector call != detector called and found nothing.
            if (t['hits'] < self.min_hits or t.get('missed_detection', False)
                    or age > max_pred):
                continue
            curr_b = bbox(t['x'])
            t['smooth_box'] = curr_b.copy()
            output.append({'track_id': t['id'], 'class_id': t['class_id'],
                           'class_name_vi': t['class_name_vi'], 'bbox_xyxy': t['smooth_box'].tolist(),
                           'detection_score': t['score'], 'last_detection_timestamp': t['last_detection'],
                           'detection_age_seconds': age,
                           'raw_label': t.get('raw_label', ''), 'is_encroaching': False,
                           'class_persistence_active': t.get('persistent_class', False),
                           'position_source': t['source'], 'track_state': 'active'})
        return output

    def state_dict(self):
        tracks = [{**t, 'x': t['x'].tolist(), 'p': t['p'].tolist(),
                   'smooth_box': (t['smooth_box'].tolist() if isinstance(t.get('smooth_box'), np.ndarray)
                                  else t.get('smooth_box', bbox(t['x']).tolist()))} for t in self.tracks]
        return {'tracks': tracks, 'next_id': self.next_id, 'confirmed_ids': sorted(self.confirmed_ids)}

    def event_observations(self, timestamp, shape, motion=None):
        """Include tentative IDs so an event's start is not shifted by confirmation.

        These rows never promote a tentative ID; the rule engine still requires
        independent detector observations and valid measured motion. Static
        objects can use reliable camera motion between recent detections.
        """
        h,w=shape[:2];rows=[]
        for track in self.tracks:
            if track.get('missed_detection',False):continue
            age=timestamp-track['last_detection']
            limit=self.max_static_prediction_seconds if track['class_id'] in STATIC_CLASSES else self.max_prediction_seconds
            if age>limit:continue
            box=np.clip(bbox(track['x']),[0,0,0,0],[w,h,w,h])
            if not np.isfinite(box).all() or np.any(box[2:]<=box[:2]):continue
            source=track['source']
            if (source=='prediction' and track['class_id'] in STATIC_CLASSES
                    and motion is not None and motion.get('reliable') and not motion.get('scene_cut')):
                source='camera_motion'
            rows.append({'track_id':track['id'],'class_id':track['class_id'],
                         'bbox_xyxy':box.tolist(),'position_source':source,
                         'detection_age_seconds':age,'detection_score':track['score'],
                         'raw_label':track.get('raw_label', ''), 'is_encroaching':False,
                         'track_confirmed':track['hits']>=self.min_hits})
        return rows

    def load_state_dict(self, state):
        self.tracks = [{**t, 'x': np.asarray(t['x'], float), 'p': np.asarray(t['p'], float),
                        'smooth_box': np.asarray(t['smooth_box'], float) if 'smooth_box' in t and t['smooth_box'] is not None else bbox(np.asarray(t['x'], float))} for t in state['tracks']]
        self.next_id = state['next_id']
        self.confirmed_ids = set(state['confirmed_ids'])
