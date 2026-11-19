import json
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
from uav_pipeline.events import ViolationMonitor, EventOptions
from uav_pipeline.video.evidence import LiveEvidence
from uav_pipeline.video.io import open_writer,write_frame


class LiveLifecycleTests(unittest.TestCase):
    def test_simultaneous_events_share_live_clip_and_images(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);monitor=ViolationMonitor(EventOptions(pre_seconds=.2,post_seconds=.4))
            for frame,cid in [(3,4),(4,10)]:
                monitor.events.append({'event_id':f'e{frame}','kind':'sidewalk_encroachment',
                    'first_frame':frame,'track_ids':[frame],'class_ids':[cid],
                    'images':[{'path':f'e{frame}.jpg'}]})
            live=LiveEvidence(root,monitor,10)
            (root/'source.mp4').write_bytes(b'test')
            segments=[{'video':'source.mp4','frames':10,'start_frame':0}]
            def fake_cut(source,target,*args):Path(target).write_bytes(b'clip')
            with patch('uav_pipeline.video.evidence.consolidate_recording'), patch.object(monitor,'_write_clip',side_effect=fake_cut):
                live.flush(segments,10,1.,final=True)
            rows=[json.loads(line) for line in (root/'closed_events.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['clip']['path'],rows[1]['clip']['path'])
            self.assertEqual(len(rows[0]['clip']['alerts']),2)
            self.assertEqual(live.counts['saved_clips'],1)
            live.close()

    def test_observation_state_is_constant_size_and_deduplicates_frame(self):
        monitor=ViolationMonitor()
        mapping={}
        for index in range(10000):monitor._step(mapping,1,index,index/10,True)
        monitor._step(mapping,1,9999,999.9,True)
        state=mapping[1]
        self.assertEqual(state['detector_count'],10000)
        self.assertEqual(state['last_detector_frame'],9999)
        self.assertFalse(any(isinstance(v,set) for v in state.values()))

    def test_clip_crosses_segments_uploads_while_running_and_prunes_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);(output/'segments').mkdir()
            monitor=ViolationMonitor(EventOptions(pre_seconds=.2,post_seconds=.4))
            event={'event_id':'event_000001','kind':'sidewalk_encroachment',
                   'first_frame':3,'track_ids':[1],'class_ids':[4],
                   'evidence_status':'skipped_no_related_person','images':[],
                   'detected_at':'2026-11-17T12:00:00+07:00'}
            monitor.events.append(event)
            uploaded=[]
            class Uploader:
                def upload_events(self,root,events):
                    self_test.assertTrue((root/events[0]['clip']['path']).is_file())
                    uploaded.extend(events)
                    return {'ready':1,'already_ready':0,'failed':0}
            self_test=self
            live=LiveEvidence(output,monitor,10,Uploader(),retention_seconds=.3,upload_interval_seconds=.1)
            segments=[]
            self.assertFalse(live.ready(6))
            self.assertTrue(live.ready(7))
            try:
                for part in range(3):
                    video=f'segments/part_{part:06d}.mp4'
                    writer=open_writer(output/video,10,(32,32,3))
                    for i in range(4):write_frame(writer,np.full((32,32,3),part*50,np.uint8))
                    writer.release()
                    records=f'segments/part_{part:06d}.jsonl'
                    (output/records).write_text('')
                    segments.append({'video':video,'records':records,'frames':4,'start_frame':part*4,'part':part})
                    live.flush(segments,(part+1)*4,(part+1)*.4)
                deadline=time.monotonic()+3
                while not uploaded and time.monotonic()<deadline:time.sleep(.01)
                self.assertTrue(uploaded)
                self.assertEqual(monitor.events,[])
                self.assertFalse((output/'segments/part_000000.mp4').exists())
                self.assertEqual(len(segments),1)
                saved=json.loads((output/'closed_events.jsonl').read_text())
                self.assertEqual(saved['clip']['status'],'saved')
                self.assertFalse(saved['clip']['truncated_end'])
            finally:live.close()

class IndependentEvidenceBufferTests(unittest.TestCase):
    def test_evidence_finalizes_without_touching_main_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);(output/'segments').mkdir()
            main=output/'segments/original_000000.mp4';main.write_bytes(b'main-recording-open')
            monitor=ViolationMonitor(EventOptions(pre_seconds=.2,post_seconds=.4))
            monitor.events.append({'event_id':'event_000001','kind':'sidewalk_encroachment',
                'first_frame':3,'track_ids':[1],'class_ids':[4],
                'detected_at':'2026-11-17T12:00:00+07:00','images':[]})
            live=LiveEvidence(output,monitor,10)
            try:
                for index in range(25):live.write(np.zeros((32,32,3),np.uint8),index/10)
                self.assertEqual(monitor.events,[])
                self.assertTrue((output/'event_clips/event_000001.mp4').is_file())
                self.assertEqual(main.read_bytes(),b'main-recording-open')
                self.assertTrue(all(s['video'].startswith('evidence_buffer/') for s in live.segments))
                live.finish()
            finally:live.close()
