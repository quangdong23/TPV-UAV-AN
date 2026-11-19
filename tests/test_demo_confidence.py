"""Confidence demo được gắn nhãn, giữ nguyên qua retry và gộp clip."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from uav_pipeline.demo_confidence import demo_confidence,api_demo_confidence
from uav_pipeline.events import ViolationMonitor
from uav_pipeline.uploads import EvidenceUploader
from tests.test_uploads import API
from tests.test_events import observation


class DemoConfidenceTests(unittest.TestCase):
    def test_range_label_and_stability(self):
        values=[demo_confidence((f'e{i}','2026-11-18T12:00:00+07:00')) for i in range(100)]
        self.assertTrue(all(.85<=v['confidence']<=.98 and v['confidence_source']=='demo' for v in values))
        self.assertGreater(len({v['confidence'] for v in values}),1)
        self.assertEqual(values[0],demo_confidence(('e0','2026-11-18T12:00:00+07:00')))

    def test_model_scores_remain_untouched_and_clip_has_its_own_demo_score(self):
        event={'event_id':'e1','evidence':{'score':.42},
               'confidence':.87,'confidence_source':'demo',
               'clip':{'confidence':.95,'confidence_source':'demo'}}
        self.assertEqual(api_demo_confidence(event)['confidence'],.95)
        self.assertEqual(event['evidence']['score'],.42)

    def test_api_and_new_client_retry_keep_score_and_do_not_reupload(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'clip.mp4').write_bytes(b'123456')
            event={'event_id':'e1','kind':'sidewalk_encroachment','class_ids':[4],
                   'detected_at':'2026-11-18T12:00:00+07:00','clip':{'status':'saved','path':'clip.mp4'}}
            api=API();client=EvidenceUploader('https://example.test','test',http=api,sleep=lambda _:None)
            self.assertEqual(client.upload_one(root,event),'ready')
            payload=api.calls[0][2]['json']
            self.assertEqual(payload['confidence_source'],'demo')
            state=json.loads(next((root/'uploads').glob('*.json')).read_text())
            self.assertEqual(state['demo_confidence']['confidence'],payload['confidence'])
            calls=len(api.calls)
            other=EvidenceUploader('https://example.test','test',http=api)
            self.assertEqual(other.upload_one(root,json.loads(json.dumps(event))),'already_ready')
            self.assertEqual(len(api.calls),calls)

    def test_event_and_shared_clip_manifest_store_demo_fields(self):
        monitor=ViolationMonitor();mask=np.full((100,100),2,np.uint8)
        for frame in range(12):
            monitor.update(frame,frame/30,[observation()],mask,{'reliable':False})
        self.assertEqual(monitor.events[0]['confidence_source'],'demo')
        with tempfile.TemporaryDirectory() as folder,patch.object(monitor,'_write_clips',return_value={}):
            monitor.export(folder,30,12)
            report=json.loads((Path(folder)/'events.json').read_text())
        self.assertEqual(report['clips'][0]['confidence_source'],'demo')
        self.assertEqual(api_demo_confidence(report['events'][0]),
                         {key:report['clips'][0][key] for key in ('confidence','confidence_source')})
