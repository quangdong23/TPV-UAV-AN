"""Gom cụm từ 5 người; chỉ là cảnh báo cần kiểm duyệt."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from uav_pipeline.violation_regions import crowd_regions
from uav_pipeline.events import ViolationMonitor,EventOptions
from uav_pipeline.evidence_groups import EvidenceGroups
from uav_pipeline.uploads import behavior_name
from tests.test_events import observation


def people(offset=0,first_id=1):
    return [observation(first_id+i,1,(offset+10+i*18,10,offset+25+i*18,70)) for i in range(5)]


class CrowdTests(unittest.TestCase):
    def test_four_people_or_five_distant_people_do_not_alert(self):
        self.assertEqual(crowd_regions(people()[:4]),[])
        self.assertEqual(crowd_regions([observation(i,1,(i*100,10,i*100+15,70)) for i in range(5)]),[])
        self.assertEqual(crowd_regions(people()[:4]+[people()[0]]),[])

    def test_two_separate_crowds_and_seated_person_excluded(self):
        self.assertEqual(len(crowd_regions(people()+people(300,10))),2)
        rows=people();rows[-1]['raw_label']='seated person'
        self.assertEqual(crowd_regions(rows),[])
        # Một hàng dài không bị nối thành cụm nhờ chuỗi hàng xóm.
        rows=[observation(i,1,(i*50,10,i*50+15,70)) for i in range(5)]
        self.assertEqual(crowd_regions(rows),[])

    def test_one_alert_then_id_change_reuses_event_and_person_crop(self):
        monitor=ViolationMonitor(EventOptions(crowd_gap_ratio=.9));mask=np.zeros((120,600),np.uint8)
        for frame in range(12):
            monitor.update(frame,frame/30,people(),mask,{'reliable':False},activity_regions=[])
        self.assertEqual(len(monitor.events),1)
        event=monitor.events[0]
        self.assertEqual(event['kind'],'crowd_gathering_suspected')
        self.assertEqual(event['evidence']['person_count'],5)
        self.assertEqual(behavior_name(event),'tụ tập trái phép')
        self.assertTrue(event['requires_review'])
        for frame in range(12,24):
            monitor.update(frame,frame/30,people(first_id=20),mask,{'reliable':False},activity_regions=[])
        self.assertEqual(len(monitor.events),1)
        with tempfile.TemporaryDirectory() as directory:
            groups=EvidenceGroups();image=np.zeros((120,600,3),np.uint8)
            self.assertTrue(groups.capture(event,people(first_id=20),image,Path(directory),image.shape,'test'))
            self.assertEqual(len(event['group']['person_track_ids']),5)
            self.assertEqual(len(event['images']),2)

    def test_stale_or_unconfirmed_tracks_do_not_count(self):
        monitor=ViolationMonitor();rows=people();rows[-1]['track_confirmed']=False
        for frame in range(15):monitor.update(frame,frame/30,rows,np.zeros((100,200),np.uint8),{'reliable':False})
        self.assertEqual(monitor.events,[])
