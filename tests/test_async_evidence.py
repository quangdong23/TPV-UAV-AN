import json
import tempfile
import threading
import time
from pathlib import Path
import unittest
from unittest.mock import patch
from uav_pipeline.events import ViolationMonitor, EventOptions
from uav_pipeline.video.evidence import LiveEvidence

class AsyncEvidenceTests(unittest.TestCase):
    def test_cutting_does_not_block_analysis_and_pins_rotating_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);monitor=ViolationMonitor(EventOptions(pre_seconds=.2,post_seconds=.4))
            monitor.events=[{'event_id':'e1','kind':'sidewalk_encroachment',
                'first_frame':3,'track_ids':[1],'class_ids':[4],'images':[]}]
            live=LiveEvidence(root,monitor,10,async_clips=True)
            source=root/'source.mp4';source.write_bytes(b'source')
            segments=[{'video':'source.mp4','frames':10,'start_frame':0}]
            gate=threading.Event();entered=threading.Event()
            def cut(src,dst,*args):
                entered.set();gate.wait(4);Path(dst).write_bytes(b'clip')
            try:
                with patch('uav_pipeline.video.evidence.consolidate_recording'),patch.object(monitor,'_write_clip',side_effect=cut):
                    live.flush(segments,10,1.)
                    self.assertTrue(entered.wait(2))
                    self.assertFalse(gate.is_set())
                    self.assertEqual(monitor.events,[])
                    jobs=list((root/'clip_jobs').glob('*/job.json'))
                    self.assertEqual(len(jobs),1)
                    source.unlink()
                    self.assertTrue((jobs[0].parent/'source.mp4').exists())
                    gate.set();live.finish()
                saved=json.loads((root/'closed_events.jsonl').read_text())
                self.assertEqual(saved['clip']['status'],'saved')
                self.assertEqual(list((root/'clip_jobs').glob('*/job.json')),[])
            finally:gate.set();live.close()

    def test_upload_waits_for_batch_interval_and_shutdown_drains(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            class Uploader:
                def upload_events(self,root,events):
                    calls.append(time.monotonic());return {'ready':1,'already_ready':0,'failed':0}
            live=LiveEvidence(Path(tmp),ViolationMonitor(),10,Uploader(),upload_interval_seconds=.4)
            try:
                start=time.monotonic()
                (live.pending/'e1.json').write_text(json.dumps({'event_id':'e1'}))
                live.wake.set();time.sleep(.1)
                self.assertEqual(calls,[])
                limit=time.monotonic()+2
                while not calls and time.monotonic()<limit:time.sleep(.02)
                self.assertEqual(len(calls),1)
                self.assertGreaterEqual(calls[0]-start,.3)
                (live.pending/'e2.json').write_text(json.dumps({'event_id':'e2'}))
                live.close()
                self.assertEqual(len(calls),2)
                self.assertEqual(list(live.pending.glob('*.json')),[])
            finally:live.close()
