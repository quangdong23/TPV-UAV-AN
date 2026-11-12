"""Temporal rules and evidence window contracts, independent of model predictions."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from uav_pipeline.events import EventOptions, ViolationMonitor, on_sidewalk


def observation(tid=1, cls=4, box=(20,20,40,60), source='detection'):
    return dict(track_id=tid,class_id=cls,bbox_xyxy=list(box),position_source=source,
                detection_age_seconds=0.,track_confirmed=True)


class EventTests(unittest.TestCase):
    def setUp(self):
        self.mask=np.full((100,100),2,np.uint8)
        self.motion=dict(reliable=True,matrix=np.array([[1.,0.,1.],[0.,1.,0.]]))

    def update(self,m,i,obs=None,mask=None,motion=None):
        return m.update(i,i/30,obs if obs is not None else [observation()],
                        self.mask if mask is None else mask,self.motion if motion is None else motion)

    def test_ten_frames_allowed_classes_and_first_anchor(self):
        for cls in (4,5,6,7,10):
            m=ViolationMonitor()
            for i in range(9):self.assertEqual(self.update(m,i,[observation(cls=cls)]),[])
            alerts=self.update(m,9,[observation(cls=cls)])
            self.assertEqual(len(alerts),1)
            self.assertEqual(alerts[0]['first_frame'],0)
            self.assertEqual(alerts[0]['confirmation_frame'],9)
            self.assertEqual(alerts[0]['consecutive_valid_frames'],10)
            self.assertTrue(alerts[0]['requires_review'])
            self.assertEqual(self.update(m,10,[observation(cls=cls)]),[])

    def test_person_never_encroaches(self):
        m=ViolationMonitor()
        for i in range(60):self.assertEqual(self.update(m,i,[observation(cls=1)]),[])
        self.assertEqual(m.presence,{})

    def test_ground_sign_and_hanging_sign_have_distinct_rules(self):
        m=ViolationMonitor()
        ground=observation(1,7,(5,20,15,40))
        hanging=observation(2,7,(60,20,80,40))
        pole=observation(3,8,(68,5,75,90))
        for i in range(20):self.update(m,i,[ground,hanging,pole])
        encroachment=[e for e in m.events if e['kind']=='sidewalk_encroachment']
        attachment=[e for e in m.events if e['kind']=='sign_attachment_suspected']
        self.assertEqual([e['track_ids'] for e in encroachment],[[1]])
        self.assertEqual([e['track_ids'] for e in attachment],[[2,3]])
        self.assertNotIn(2,m.presence)

    def test_requires_three_independent_detections(self):
        m=ViolationMonitor(EventOptions(encroachment_frames=20,min_detector_observations=3))
        for i in range(20):
            source='detection' if i in (0,10) else 'optical_flow'
            self.assertEqual(self.update(m,i,[observation(source=source)]),[])
        alerts=self.update(m,20)
        self.assertEqual(alerts[0]['independent_detector_observations'],3)

    def test_outside_missing_prediction_and_cut_reset(self):
        for interruption in ('outside','missing','prediction','cut','gap'):
            m=ViolationMonitor()
            for i in range(9):self.update(m,i)
            kwargs={}
            if interruption=='outside':kwargs['mask']=np.zeros_like(self.mask)
            if interruption=='missing':kwargs['obs']=[]
            if interruption=='prediction':kwargs['obs']=[observation(source='prediction')]
            if interruption=='cut':kwargs['motion']=dict(self.motion,scene_cut=True)
            index=10 if interruption=='gap' else 9
            self.assertEqual(self.update(m,index,**kwargs),[])
            for i in range(index+1,index+9):self.assertEqual(self.update(m,i),[])

    def test_unknown_and_edge_contact_rejected(self):
        self.mask[59,30]=0
        self.assertFalse(on_sidewalk((20,20,40,60),self.mask))
        self.assertFalse(on_sidewalk((0,0,2,2),self.mask,margin=2))
        self.assertFalse(on_sidewalk((40,20,20,60),self.mask))

    def test_attachment_needs_motion_persistence_and_unambiguous_support(self):
        pair=[observation(1,7,(30,20,50,40)),observation(2,9,(38,5,45,90))]
        for mode in ('moving','static','unreliable','ambiguous','separated'):
            m=ViolationMonitor()
            for i in range(10):
                obs=list(pair);motion=self.motion
                if mode=='static':motion=dict(reliable=True,matrix=np.array([[1.,0.,0.],[0.,1.,0.]]))
                if mode=='unreliable':motion=dict(reliable=False)
                if mode=='ambiguous':obs.append(observation(3,8,(38,5,45,90)))
                if mode=='separated' and i==5:obs=[pair[0]]
                self.update(m,i,obs,motion=motion)
            events=[e for e in m.events if e['kind']=='sign_attachment_suspected']
            self.assertEqual(len(events),int(mode=='moving'))
            if events:self.assertIn('not_proof',events[0]['evidence']['interpretation'])

    def test_red_persists_after_tracking_interruption_until_scene_cut(self):
        m=ViolationMonitor()
        for i in range(10):self.update(m,i)
        self.update(m,10,[])
        self.update(m,11)
        image=np.zeros((100,100,3),np.uint8)
        red=m.render(image.copy(),[observation()])
        np.testing.assert_array_equal(red[40,20],(0,0,255))
        self.update(m,12,motion=dict(self.motion,scene_cut=True))
        clear=m.render(image.copy(),[observation()])
        np.testing.assert_array_equal(clear[40,20],(0,0,0))

    def test_default_contact_matches_visible_mask_boundary(self):
        mask=np.zeros_like(self.mask);mask[59,30]=2
        self.assertTrue(on_sidewalk((20,20,40,60),mask))
        self.assertFalse(on_sidewalk((20,20,40,60),mask,margin=2))

    def test_two_detections_with_reliable_camera_motion_can_confirm_static_object(self):
        from uav_pipeline.tracking.objects import ObjectTracker
        from tests.test_video import detection, movement
        tracker=ObjectTracker(min_hits=2)
        m=ViolationMonitor()
        for i in range(10):
            motion=movement(shift=0)
            tracker.update(i/30,[detection(cid=5)] if i in (0,5) else None,motion)
            rows=tracker.event_observations(i/30,(100,100,3),motion)
            m.update(i,i/30,rows,self.mask,motion)
        self.assertEqual(m.events[0]['confirmation_frame'],9)
        self.assertEqual(m.events[0]['independent_detector_observations'],2)
        unreliable=tracker.event_observations(9/30,(100,100,3),dict(motion,reliable=False))
        self.assertEqual(unreliable[0]['position_source'],'prediction')

    def test_clip_window_first_appearance_shared_and_edges(self):
        m=ViolationMonitor()
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            for tid,frame in ((1,60),(2,60),(3,0),(4,190)):
                m.events.append(dict(first_frame=frame,kind='sidewalk_encroachment',track_ids=[tid]))
            with patch.object(m,'_write_clips',return_value={}) as write:
                summary=m.export(output,10,200)
            self.assertEqual(summary['unique_clips'],3)
            self.assertEqual(write.call_args.args[1][0][1:],(20,160))
            self.assertEqual(m.events[0]['clip']['actual_duration_seconds'],14)
            self.assertTrue(m.events[1]['clip']['shared'])
            self.assertTrue(m.events[2]['clip']['truncated_start'])
            self.assertTrue(m.events[3]['clip']['truncated_end'])
            self.assertEqual(json.loads((output/'events.json').read_text())['alert_count'],4)

    def test_real_clip_frame_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.mp4';target=Path(tmp)/'clip.mp4'
            writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(32,32))
            if not writer.isOpened():self.skipTest('mp4v unavailable')
            for i in range(200):writer.write(np.full((32,32,3),i,np.uint8))
            writer.release()
            other=Path(tmp)/'other.mp4'
            with patch('uav_pipeline.events.shutil.which',return_value=None):
                failures=ViolationMonitor._write_clips(source,[(target,20,160),(other,30,170)],10)
            self.assertEqual(failures,{})
            for path in (target,other):
                cap=cv2.VideoCapture(str(path))
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),140)
                cap.release()


class MotorcycleRepairTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.full((240,320), 2, np.uint8)
        self.motion = dict(reliable=True, matrix=np.eye(2,3))

    def pair(self, shift=0, source='detection'):
        return [observation(1,1,(20+shift,20,40+shift,80),source),
                observation(2,2,(45+shift,45,95+shift,80),source)]

    def step(self, monitor, i, fps=10, rows=None, **kwargs):
        return monitor.update(i,i/fps,self.pair() if rows is None else rows,
                              kwargs.pop('mask',self.mask),kwargs.pop('motion',self.motion),**kwargs)

    def test_two_seconds_at_different_frame_rates_and_source_box(self):
        for fps in (10,30):
            monitor = ViolationMonitor()
            for i in range(2*fps):
                self.assertEqual(self.step(monitor,i,fps,source_size=(640,480)),[])
                self.assertEqual(monitor.repair_regions,[])
            events = self.step(monitor,2*fps,fps,source_size=(640,480))
            self.assertEqual(len(events),1)
            event = events[0]
            self.assertEqual(event['kind'],'motorcycle_repair_suspected')
            self.assertEqual(event['track_ids'],[1,2])
            self.assertEqual(event['first_timestamp_seconds'],0)
            self.assertEqual(event['confirmation_timestamp_seconds'],2)
            self.assertEqual(event['evidence']['combined_bbox_source_pixels'],[40,40,190,160])
            self.assertTrue(event['requires_review'])
            self.assertEqual(self.step(monitor,2*fps+1,fps),[])
            image = monitor.render(np.zeros((240,320,3),np.uint8),self.pair())
            np.testing.assert_array_equal(image[60,20],(0,0,255))
            np.testing.assert_array_equal(image[80,70],(0,0,255))

    def test_both_must_be_near_and_on_sidewalk_with_valid_observations(self):
        for mode in ('person_road','bike_road','unknown','far','prediction','stale','unconfirmed','single'):
            monitor = ViolationMonitor()
            mask = self.mask.copy()
            rows = self.pair()
            if mode == 'person_road': mask[79,30] = 0
            if mode == 'bike_road': mask[79,70] = 0
            if mode == 'unknown': mask[79,70] = 255
            if mode == 'far': rows[1]['bbox_xyxy'] = [220,45,270,80]
            if mode == 'prediction': rows[0]['position_source'] = 'prediction'
            if mode == 'stale': rows[0]['detection_age_seconds'] = .6
            if mode == 'unconfirmed': rows[0]['track_confirmed'] = False
            if mode == 'single': rows = rows[:1]
            for i in range(31): self.assertEqual(self.step(monitor,i,rows=rows,mask=mask),[])
            self.assertEqual(monitor.repair_regions,[],mode)

    def test_walking_pair_does_not_accumulate_stationary_time(self):
        monitor = ViolationMonitor()
        for i in range(40):
            self.assertEqual(self.step(monitor,i,rows=self.pair(shift=i)),[])
        self.assertEqual(monitor.repair_regions,[])

    def test_camera_pan_is_compensated_and_bad_motion_clears(self):
        monitor = ViolationMonitor()
        motion = dict(reliable=True,matrix=np.array([[1.,0.,1.],[0.,1.,0.]]))
        for i in range(21): self.step(monitor,i,rows=self.pair(shift=i),motion=motion)
        self.assertEqual(len(monitor.events),1)
        self.step(monitor,21,motion=dict(reliable=False))
        self.assertEqual(monitor.repair_regions,[])
        self.assertEqual(monitor.repairs,{})

    def test_interruption_and_scene_cut_restart_the_duration(self):
        for mode in ('missing','separated','cut','gap','changed_id'):
            monitor = ViolationMonitor()
            for i in range(20): self.step(monitor,i)
            index = 21 if mode == 'gap' else 20
            rows = self.pair()
            motion = self.motion
            if mode == 'missing': rows = []
            if mode == 'separated': rows[1]['bbox_xyxy'] = [220,45,270,80]
            if mode == 'cut': motion = dict(self.motion,scene_cut=True)
            if mode == 'changed_id': rows[0]['track_id'] = 3
            self.assertEqual(self.step(monitor,index,rows=rows,motion=motion),[])
            for i in range(index+1,index+20): self.assertEqual(self.step(monitor,i),[],mode)

    def test_red_group_disappears_when_person_leaves_or_pair_moves(self):
        for mode in ('missing','moving','outside'):
            monitor = ViolationMonitor()
            for i in range(21): self.step(monitor,i)
            self.assertEqual(len(monitor.repair_regions),1)
            rows = [] if mode == 'missing' else self.pair(shift=30 if mode == 'moving' else 0)
            mask = np.zeros_like(self.mask) if mode == 'outside' else self.mask
            self.step(monitor,21,rows=rows,mask=mask)
            self.assertEqual(monitor.repair_regions,[])
            self.assertEqual(monitor.confirmed_track_ids,set())

    def test_independent_detections_and_configurable_duration(self):
        monitor = ViolationMonitor(EventOptions(repair_seconds=1))
        for i in range(11):
            rows = self.pair(source='detection' if i == 0 else 'optical_flow')
            self.assertEqual(self.step(monitor,i,rows=rows),[])
        self.assertEqual(len(self.step(monitor,11)),1)
        for value in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError): EventOptions(repair_seconds=value).validate()

    def test_timestamp_gap_does_not_count_unobserved_time(self):
        monitor = ViolationMonitor()
        for i in range(20): self.step(monitor,i)
        self.assertEqual(monitor.update(20,10,self.pair(),self.mask,self.motion),[])
        self.assertEqual(monitor.repair_regions,[])

    def test_cooldown_suppresses_duplicate_event_but_keeps_current_group_visible(self):
        monitor = ViolationMonitor()
        for i in range(21): self.step(monitor,i)
        self.step(monitor,21,rows=[])
        for i in range(22,43): self.step(monitor,i)
        self.assertEqual(len(monitor.events),1)
        self.assertEqual(len(monitor.repair_regions),1)

    def test_repair_event_is_exported_with_kind_count(self):
        monitor = ViolationMonitor()
        for i in range(21): self.step(monitor,i)
        with tempfile.TemporaryDirectory() as tmp:
            summary = monitor.export(tmp,10,21,video_available=False)
            self.assertEqual(summary['counts_by_kind']['motorcycle_repair_suspected'],1)
            event = json.loads((Path(tmp)/'events.json').read_text())['events'][0]
            self.assertEqual(event['evidence']['combined_bbox_source_pixels'],[20,20,95,80])


if __name__=='__main__':unittest.main()
