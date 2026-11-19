"""Thông tin giả chỉ đi kèm cảnh báo có người, retry không đổi hồ sơ."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from uav_pipeline.demo_people import attach_demo_people,profiles,api_demo_identity
from uav_pipeline.events import ViolationMonitor
from uav_pipeline.uploads import EvidenceUploader
from tests.test_events import observation
from tests.test_uploads import API


class DemoPeopleTests(unittest.TestCase):
    def test_no_person_has_no_identity_and_people_get_explicit_demo_profiles(self):
        event={};attach_demo_people(event,[observation(cls=4)])
        self.assertNotIn('demo_people',event)
        tracks=[observation(1,1),observation(2,1),observation(2,1)]
        attach_demo_people(event,tracks)
        self.assertEqual(len(event['demo_people']),2)
        before=[dict(p) for p in event['demo_people']]
        attach_demo_people(event,tracks)
        self.assertEqual(event['demo_people'],before)
        for person in before:
            self.assertTrue(person['is_demo']);self.assertFalse(person['identity_verified'])
            self.assertNotIn('(DEMO)',person['full_name'])
            self.assertIn(person['cccd'],[p['cccd'] for p in profiles()])
            self.assertIsInstance(person['cccd'],str)

    def test_rule_attaches_identity_only_to_person_members(self):
        monitor=ViolationMonitor();mask=np.full((100,100),2,np.uint8)
        rows=[observation(1,4,(10,20,40,50)),observation(2,1,(35,25,55,50))]
        for i in range(12):monitor.update(i,i/30,rows,mask,{'reliable':False})
        self.assertEqual(len(monitor.events),1)
        self.assertEqual(monitor.events[0]['demo_people'][0]['person_track_id'],2)

    def test_api_receives_people_and_cccd_and_retry_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'video.mp4').write_bytes(b'123456')
            event={'event_id':'e1','kind':'motorcycle_repair_suspected',
                   'detected_at':'2026-11-18T12:00:00+07:00','clip':{'status':'saved','path':'video.mp4'}}
            attach_demo_people(event,[observation(1,1),observation(2,2)])
            api=API();client=EvidenceUploader('https://example.test','test',location='20, 105',http=api,sleep=lambda _:None)
            client._client.ensure_browser_compatible_video=lambda path:None
            self.assertEqual(client.upload_one(root,event),'ready')
            payload=api.calls[0][2]['json']
            self.assertEqual(payload['violators'],[{'full_name':p['full_name'],'citizen_id':p['cccd']} for p in event['demo_people']])
            self.assertNotIn('people',payload)
            self.assertNotIn('cccd',payload)
            self.assertNotIn('demo_people',payload)
            self.assertEqual(client.upload_one(root,event),'already_ready')
            self.assertEqual(sum(url.endswith('/initiate') for _,url,_ in api.calls),1)

    def test_merged_clip_keeps_person_when_first_event_has_no_people(self):
        clip={'status':'saved','path':'same.mp4','clip_id':'clip1','alerts':[{},{}],'description':'nhóm'}
        first={'event_id':'a','clip':clip,'kind':'sidewalk_encroachment'}
        second={'event_id':'b','clip':clip,'kind':'motorcycle_repair_suspected'}
        attach_demo_people(second,[observation(7,1)])
        client=EvidenceUploader('https://example.test','test')
        with tempfile.TemporaryDirectory() as folder,patch.object(client,'upload_one',return_value='ready') as upload:
            client.upload_events(folder,[first,second])
        upload.assert_called_once()
        payload=api_demo_identity(upload.call_args.args[1])
        self.assertEqual(payload['violators'][0]['full_name'],second['demo_people'][0]['full_name'])

    def test_api_maps_all_members_and_empty_list_to_null(self):
        self.assertEqual(api_demo_identity({}),{'violators':None})
        event={}
        attach_demo_people(event,[observation(1,1),observation(2,1)])
        rows=api_demo_identity(event)['violators']
        self.assertEqual(len(rows),2)
        for row in rows:
            self.assertNotIn('DEMO',row['full_name'])
            self.assertTrue(row['citizen_id'].startswith('0'))
        self.assertTrue(all(p['is_demo'] and p['identity_source']=='synthetic'
                            for p in event['demo_people']))
