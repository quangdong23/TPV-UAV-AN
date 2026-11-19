"""Kiểm tra gom vùng và ảnh minh chứng theo yêu cầu nghiệp vụ."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from uav_pipeline.violation_regions import grouped_regions,violation_mask
from uav_pipeline.events import ViolationMonitor
from uav_pipeline.evidence_groups import EvidenceGroups
from tests.test_events import observation


class RegionTests(unittest.TestCase):
    def test_furniture_group_with_seated_person_and_distant_objects(self):
        rows=[observation(1,4,(10,20,40,50)),observation(2,5,(42,25,55,50)),
              observation(3,5,(80,20,95,50))]
        person=observation(4,1,(35,25,55,50))
        groups=grouped_regions(rows,[person])
        self.assertEqual(len(groups),2)
        self.assertIn('dịch vụ ăn uống',groups[0]['behavior_name'])
        self.assertEqual([t['track_id'] for t in groups[0]['members']],[1,2,4])
        monitor=ViolationMonitor()
        mask=np.full((100,100),2,np.uint8)
        for i in range(12):monitor.update(i,i/30,rows+[person],mask,{'reliable':False})
        self.assertEqual(len(monitor.events),2)
        self.assertEqual(monitor.events[0]['track_ids'],[1,2,4])

    def test_person_arriving_later_updates_existing_alert_and_crop_members(self):
        monitor=ViolationMonitor();mask=np.full((100,100),2,np.uint8)
        table=observation(1,4,(10,20,40,50));person=observation(2,1,(35,25,55,50))
        for frame in range(12):monitor.update(frame,frame/30,[table],mask,{'reliable':False})
        monitor.events[0]['evidence_status']='ready'
        monitor.update(12,12/30,[table,person],mask,{'reliable':False})
        self.assertEqual(len(monitor.events),1)
        self.assertEqual(monitor.events[0]['track_ids'],[1,2])
        self.assertEqual(monitor.events[0]['evidence_status'],'pending')
        self.assertIn('dịch vụ ăn uống',monitor.events[0]['evidence']['behavior_name'])

    def test_adjacent_chairs_with_gap_are_one_region(self):
        rows=[observation(1,5,(10,10,30,50)),observation(2,5,(50,10,70,50))]
        self.assertEqual(len(grouped_regions(rows,[])),1)

    def test_goods_materials_and_machinery_person_rules(self):
        standing=observation(9,1,(40,10,55,50))
        crouched=observation(8,1,(35,30,55,50))
        for cid,person,expected in [(10,standing,True),(11,crouched,False),
                                    (12,standing,False),(12,crouched,True)]:
            region=grouped_regions([observation(1,cid,(20,20,40,50))],[person])[0]
            self.assertEqual(len(region['members'])==2,expected)

    def test_road_edge_excludes_distant_road_and_unknown(self):
        mask=np.zeros((40,100),np.uint8);mask[:,:10]=2;mask[:,10:]=1
        result=violation_mask(mask,10)
        self.assertEqual(result[20,15],2)
        self.assertEqual(result[20,80],1)
        mask[:,10:]=255
        self.assertEqual(violation_mask(mask,10)[20,15],255)

    def test_shared_clip_upload_collects_images_and_description_once(self):
        from unittest.mock import patch
        from uav_pipeline.uploads import EvidenceUploader,behavior_name
        client=EvidenceUploader('https://example.test','test')
        clip={'path':'shared.mp4','clip_id':'clip1','alerts':[{},{}],
              'description':'bàn ghế; hàng hoá'}
        events=[{'event_id':f'e{i}','kind':'sidewalk_encroachment','clip':clip,
                 'images':[{'path':f'crop{i}.jpg'},{'path':'snapshot.jpg'}]} for i in range(2)]
        with tempfile.TemporaryDirectory() as folder, patch.object(client,'upload_one',return_value='ready') as upload:
            result=client.upload_events(folder,events)
        self.assertEqual(result['ready'],1)
        upload.assert_called_once()
        event=upload.call_args.args[1]
        self.assertEqual(len(event['images']),3)
        self.assertEqual(behavior_name(event),'bàn ghế; hàng hoá')

    def test_material_crop_without_person_and_shared_snapshot_descriptions(self):
        rows=[observation(1,11,(10,10,30,40)),observation(2,7,(60,10,80,40))]
        events=[{'event_id':f'e{i}','kind':'sidewalk_encroachment','track_ids':[row['track_id']],
                 'class_ids':[row['class_id']],'confirmation_frame':9,
                 'evidence':{'family':family,'behavior_name':name}}
                for i,(row,family,name) in enumerate(zip(rows,['materials','sign'],['vật liệu','biển']))]
        with tempfile.TemporaryDirectory() as folder:
            groups=EvidenceGroups();image=np.zeros((100,100,3),np.uint8)
            for event in events:
                self.assertTrue(groups.capture(event,rows,image,Path(folder),image.shape,'test'))
            self.assertEqual(events[0]['images'][1]['path'],events[1]['images'][1]['path'])
            snapshot=json.loads((Path(folder)/events[0]['snapshot_metadata']).read_text())
            self.assertEqual(len(snapshot['alerts']),2)
            self.assertEqual(snapshot['description'],'vật liệu; biển')
            self.assertLess(events[0]['group']['bbox_source_pixels'][0],10)
