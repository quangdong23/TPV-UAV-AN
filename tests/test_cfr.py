import tempfile
from pathlib import Path
import threading
import unittest
import av
import numpy as np
from uav_pipeline.video.cfr import TimestampCFRClock,open_cfr_writer,cut_cfr
from uav_pipeline.video.io import ThreadedVideoWriter
from uav_pipeline.video.evidence import LiveEvidence
from uav_pipeline.events import ViolationMonitor,EventOptions


def read_video(path):
    with av.open(str(path)) as video:
        fps=float(video.streams.video[0].average_rate)
        rows=[(float(f.pts*f.time_base),int(f.to_ndarray(format='bgr24')[32,32,0])) for f in video.decode(video=0)]
    return fps,rows


class CFRTests(unittest.TestCase):
    def test_source_13_then_20_fps_produces_paired_fixed_15_fps(self):
        timestamps=[i/13 for i in range(26)]+[2+i/20 for i in range(40)]
        clock=TimestampCFRClock(15)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            raw=open_cfr_writer(root/'raw.mp4',15,(64,64,3))
            boxes=open_cfr_writer(root/'bbox.mp4',15,(64,64,3))
            previous=None;selected=[]
            def emit(value):
                selected.append(value)
                raw.write(np.full((64,64,3),value,np.uint8))
                boxes.write(np.full((64,64,3),value+100,np.uint8))
            try:
                for index,t in enumerate(timestamps):
                    for tick,current in clock.push(t):emit(index if current or previous is None else previous)
                    previous=index
                for tick in clock.finish(4):emit(previous)
            finally:raw.release();boxes.release()
            self.assertEqual(len(selected),60)
            self.assertGreater(len(selected)-len(set(selected)),0)
            self.assertLess(len(set(selected)),len(timestamps))
            for path in (root/'raw.mp4',root/'bbox.mp4'):
                fps,rows=read_video(path)
                self.assertAlmostEqual(fps,15)
                np.testing.assert_allclose([t for t,_ in rows],np.arange(60)/15,atol=.00002)
            _,a=read_video(root/'raw.mp4');_,b=read_video(root/'bbox.mp4')
            np.testing.assert_allclose([t for t,_ in a],[t for t,_ in b],atol=.00002)
            errors=[(y-x)-100 for (_,x),(_,y) in zip(a,b)]
            # CRF H.264 is lossy; dark first frames differ by up to five levels.
            # Exact synchronization is checked independently through all PTS.
            self.assertTrue(all(abs(value)<=6 for value in errors),f'Lossy encode differences: {errors}')
            cut_cfr(root/'bbox.mp4',root/'clip.mp4',1.8,2.2,15)
            fps,clip=read_video(root/'clip.mp4')
            self.assertAlmostEqual(fps,15);self.assertEqual(len(clip),6)

    def test_reconnect_gap_is_filled_without_resetting_clock(self):
        clock=TimestampCFRClock(15)
        a=list(clock.push(0));b=list(clock.push(3))
        self.assertEqual(len(a)+len(b),46)
        self.assertEqual(b[-1],(3.,True))
        self.assertEqual(sum(not current for _,current in b),44)
        self.assertEqual(len(list(clock.finish(3+1/15))),0)
        with self.assertRaises(ValueError):list(clock.push(3))

    def test_encoder_queue_is_bounded_and_owns_frame_copies(self):
        gate=threading.Event();entered=threading.Event();done=threading.Event();frames=[]
        class Writer:
            def write(self,frame):
                entered.set();gate.wait(3);frames.append(frame.copy())
            def release(self):pass
        writer=ThreadedVideoWriter(Writer())
        frame=np.zeros((4,4,3),np.uint8)
        writer.write(frame);self.assertTrue(entered.wait(1))
        writer.write(frame);writer.write(frame)
        def produce():writer.write(frame);done.set()
        thread=threading.Thread(target=produce);thread.start()
        try:
            self.assertEqual(writer.queue.maxsize,2)
            self.assertFalse(done.wait(.05))
            frame[:]=255
        finally:
            gate.set();thread.join(3);writer.release()
        self.assertTrue(done.is_set());self.assertFalse(np.any(frames[0]))

    def test_cfr_evidence_window_uses_timestamp_and_fixed_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);monitor=ViolationMonitor(EventOptions(pre_seconds=.2,post_seconds=.4))
            monitor.events=[{'event_id':'e1','kind':'sidewalk_encroachment','class_ids':[4],
                'track_ids':[1],'first_frame':10,'first_timestamp_seconds':1.,'images':[]}]
            live=LiveEvidence(root,monitor,15,vfr=True,cfr=True)
            try:
                for index in range(30):live.write(np.zeros((64,64,3),np.uint8),index/15)
                live.finish()
                import json
                event=json.loads((root/'closed_events.jsonl').read_text().splitlines()[0])
                self.assertEqual(event['clip']['status'],'saved')
                self.assertAlmostEqual(event['clip']['start_output_seconds'],.8)
                self.assertAlmostEqual(event['clip']['end_output_seconds'],1.4)
                fps,rows=read_video(root/event['clip']['path'])
                self.assertAlmostEqual(fps,15);self.assertEqual(len(rows),9)
            finally:live.close()

    def test_worker_failure_propagates_to_caller(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer=open_cfr_writer(Path(tmp)/'invalid.mp4',15,(64,64,3))
            with self.assertRaises(RuntimeError):
                writer.write(np.zeros((32,32,3),np.uint8));writer.release()
