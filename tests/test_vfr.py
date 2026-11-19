"""Video live đổi nhịp vẫn giữ đúng PTS, thời lượng và clip minh chứng."""
import tempfile
import unittest
from pathlib import Path
import av
import numpy as np
from uav_pipeline.video.vfr import VFRWriter,cut_vfr
from uav_pipeline.video.io import consolidate_recording
from uav_pipeline.video.evidence import LiveEvidence
from uav_pipeline.events import ViolationMonitor,EventOptions


def timestamps():
    return [i/15.2 for i in range(152)]+[10+i/13 for i in range(130)]


def decoded(path):
    with av.open(str(path)) as video:
        return [float(frame.pts*frame.time_base) for frame in video.decode(video=0)]


class VFRTests(unittest.TestCase):
    def write(self,path,points):
        writer=VFRWriter(path,15.2,(64,64,3))
        try:
            for t in points:writer.write_at(np.zeros((64,64,3),np.uint8),t)
        finally:writer.release()

    def test_15_2_to_13_fps_keeps_twenty_seconds_without_dropping_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'vfr.mp4';points=timestamps();self.write(path,points)
            actual=decoded(path)
            self.assertEqual(len(actual),282)
            self.assertLess(max(abs(a-b) for a,b in zip(actual,points)),.0001)
            with av.open(str(path)) as video:
                self.assertAlmostEqual(float(video.duration/av.time_base),20,delta=.08)
            cut=Path(folder)/'cut.mp4';cut_vfr(path,cut,9,11)
            times=decoded(cut);gaps=np.diff(times)
            self.assertTrue(np.any(abs(gaps-1/15.2)<.001))
            self.assertTrue(np.any(abs(gaps-1/13)<.001))
            self.assertLessEqual(times[-1],2.001)

    def test_segment_concat_preserves_reconnect_gap(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'segments').mkdir()
            paths=['segments/a.mp4','segments/b.mp4']
            for name in paths:self.write(root/name,[i/10 for i in range(10)])
            consolidate_recording(root,paths,'original.mp4',20,start_timestamps=[0,3])
            times=decoded(root/'original.mp4')
            self.assertEqual(len(times),20)
            self.assertAlmostEqual(times[10],3,delta=.001)

    def test_live_clip_window_uses_seconds_not_initial_fps(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);monitor=ViolationMonitor(EventOptions(pre_seconds=.2,post_seconds=.4))
            monitor.events=[{'event_id':'e1','kind':'sidewalk_encroachment','class_ids':[4],
                             'track_ids':[1],'first_frame':152,'first_timestamp_seconds':10.}]
            live=LiveEvidence(root,monitor,15.2,vfr=True)
            try:
                for t in timestamps():live.write(np.zeros((64,64,3),np.uint8),t)
                live.finish()
                import json
                event=json.loads((root/'closed_events.jsonl').read_text().splitlines()[0])
                self.assertEqual(event['clip']['status'],'saved')
                self.assertAlmostEqual(event['clip']['start_output_seconds'],9.8)
                self.assertAlmostEqual(event['clip']['end_output_seconds'],10.4)
                self.assertTrue(decoded(root/event['clip']['path']))
            finally:live.close()
