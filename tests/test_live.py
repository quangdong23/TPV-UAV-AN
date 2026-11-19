import contextlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from uav_pipeline.video.live import RTMPCapture, safe_source
from uav_pipeline.video.pipeline import VideoOptions, process_video
from tests.test_video import FakeDetector
from tests.test_combined_video import FakeSidewalk
from uav_pipeline.segmentation.session import SidewalkSession
from uav_pipeline.segmentation.media.video import VideoPreviewOptions


class Source:
    def __init__(self, frames, fps=10., opened=True):
        self.frames=list(frames);self.fps=fps;self.opened=opened;self.index=0;self.closed=False
    def isOpened(self):return self.opened
    def read(self):
        if not self.frames:return False,None
        self.index+=1
        return True,self.frames.pop(0)
    def get(self,key):
        if key==cv2.CAP_PROP_FPS:return self.fps
        if key==cv2.CAP_PROP_POS_MSEC:return max(0,self.index-1)*100.
        return 0.
    def release(self):self.closed=True


class LiveTests(unittest.TestCase):
    def test_retry_reconnect_keeps_all_frames_and_monotonic_time(self):
        frames=[np.full((80,100,3),v,np.uint8) for v in range(4)]
        sources=[Source([],opened=False),Source(frames[:2]),Source(frames[2:])]
        calls=[]
        def factory(*args):calls.append(args);return sources.pop(0)
        cap=RTMPCapture('rtmp://host/live/x?pass=secret',retry_seconds=.001,factory=factory)
        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertTrue(cap.open())
            times=[];values=[]
            for _ in range(4):
                ok,frame=cap.read();self.assertTrue(ok)
                values.append(int(frame[0,0,0]));times.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        self.assertEqual(values,[0,1,2,3]);self.assertTrue(all(b>a for a,b in zip(times,times[1:])))
        self.assertEqual(cap.epoch,1);self.assertEqual(cap.reconnects,1)
        self.assertIn(cv2.CAP_PROP_READ_TIMEOUT_MSEC,calls[0][2])
        self.assertNotIn('secret',log.getvalue())
        cap.release();self.assertFalse(cap.read()[0])

    def test_wait_can_be_stopped_and_invalid_options_rejected(self):
        stopped=threading.Event();stopped.set()
        cap=RTMPCapture('rtmp://host/live/x',stop=stopped)
        self.assertFalse(cap.open())
        for value in (0,-1,float('nan')):
            with self.assertRaises(ValueError):RTMPCapture('rtmp://host/x',retry_seconds=value)
        self.assertEqual(safe_source('rtmp://user:secret@host:1935/live/x?pass=hidden'),'rtmp://host:1935/live/x')

    def test_reconnect_accepts_fps_change_but_rejects_resolution_change(self):
        frame=np.zeros((80,100,3),np.uint8)
        for changed,accepted in [(Source([frame],fps=13),True),
                                 (Source([np.zeros((90,100,3),np.uint8)]),False)]:
            sources=[Source([frame],fps=15.2),changed]
            cap=RTMPCapture('rtmp://host/x',retry_seconds=.001,factory=lambda *args:sources.pop(0))
            with contextlib.redirect_stdout(io.StringIO()):
                cap.open();cap.read()
                if accepted:
                    before=cap.timestamp
                    self.assertTrue(cap.read()[0]);self.assertEqual(cap.fps,13)
                    self.assertGreater(cap.timestamp,before)
                else:
                    with self.assertRaises(RuntimeError):cap.read()
            cap.release()

    def test_pipeline_records_matching_cfr_timestamps_after_fps_change(self):
        import av
        class TimedSource(Source):
            def get(self,key):
                if key==cv2.CAP_PROP_POS_MSEC:return max(0,self.index-1)*1000/self.fps
                return super().get(key)
        frame=np.zeros((120,160,3),np.uint8)
        sources=[TimedSource([frame.copy() for _ in range(3)],15.2),
                 TimedSource([frame.copy() for _ in range(4)],13)]
        cap=RTMPCapture('rtmp://host/x',retry_seconds=.001,factory=lambda *args:sources.pop(0))
        session=SidewalkSession(FakeSidewalk(),VideoPreviewOptions(keyframe_interval=1),sample_step=0)
        options=VideoOptions(min_hits=1,keyframe_interval=1,min_detection_interval=1,
                             live_segment_seconds=.15,save_tracking=True)
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            root=Path(folder);cap.open()
            process_video('rtmp://host/x',root,FakeDetector(),options,max_frames=7,
                          sidewalk=session,output_mode='full',live_capture=cap)
            rows=[json.loads(line) for line in (root/'tracks.jsonl').read_text().splitlines()]
            actual=[]
            for name in ('original.mp4','annotated.mp4'):
                with av.open(str(root/name)) as video:
                    actual.append([float(f.pts*f.time_base) for f in video.decode(video=0)])
            expected=__import__('math').ceil((rows[-1]['timestamp_seconds']+1/13)*15-1e-8)
            self.assertEqual(len(actual[0]),expected)
            np.testing.assert_allclose(actual[0],actual[1],atol=.0001)
            np.testing.assert_allclose(np.diff(actual[0]),1/15,atol=.0001)
            self.assertEqual(rows[-1]['source_fps_reported'],13)
            summary=json.loads((root/'summary.json').read_text())
            self.assertEqual(summary['recording_frame_rate_mode'],'CFR')
            self.assertEqual(summary['processed_frames'],7)
            self.assertEqual(summary['recording_fps'],15)

    def test_full_pipeline_reconnect_resets_identity_masks_and_hides_credentials(self):
        frame=np.random.default_rng(22).integers(0,255,(120,160,3),np.uint8)
        sources=[Source([frame.copy() for _ in range(3)]),Source([frame.copy() for _ in range(3)])]
        cap=RTMPCapture('rtmp://host/live/x?pass=secret',retry_seconds=.001,
                        factory=lambda *args:sources.pop(0))
        session=SidewalkSession(FakeSidewalk(),VideoPreviewOptions(keyframe_interval=5),sample_step=0)
        options=VideoOptions(min_hits=1,keyframe_interval=5,min_detection_interval=1,
                             segment_frames=2,decode_prefetch=2,live_segment_seconds=.3,recording_fps=10)
        with tempfile.TemporaryDirectory() as directory,contextlib.redirect_stdout(io.StringIO()):
            cap.open()
            out=Path(directory)/'out'
            process_video('rtmp://host/live/x?pass=secret',out,FakeDetector(),options,
                          max_frames=6,sidewalk=session,output_mode='full',live_capture=cap)
            rows=[json.loads(x) for x in (out/'tracks.jsonl').read_text().splitlines()]
            self.assertEqual([r['frame_index'] for r in rows],list(range(6)))
            self.assertNotEqual(rows[0]['tracks'][0]['track_id'],rows[3]['tracks'][0]['track_id'])
            self.assertEqual(rows[3]['motion_reason'],'stream_reconnected')
            self.assertTrue(rows[3]['detector_ran']);self.assertTrue(rows[3]['sidewalk']['ai_ran'])
            self.assertTrue(all(b['timestamp_seconds']>a['timestamp_seconds'] for a,b in zip(rows,rows[1:])))
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['input_mode'],'rtmp_live');self.assertEqual(summary['live_reconnects'],1)
            self.assertFalse(summary['intentional_frame_dropping'])
            self.assertEqual(summary['original_video'],'original.mp4')
            self.assertTrue(summary['original_recording_enabled'])
            self.assertEqual(summary['video'],'annotated.mp4')
            self.assertTrue((out/'original.mp4').exists())
            self.assertTrue(list((out/'segments').glob('original_*.mp4')))
            video=cv2.VideoCapture(str(out/'annotated.mp4'))
            try:
                self.assertTrue(video.isOpened())
                self.assertEqual(int(video.get(cv2.CAP_PROP_FRAME_COUNT)),6)
                self.assertAlmostEqual(video.get(cv2.CAP_PROP_FPS),10.)
                decoded=0
                while True:
                    ok,image=video.read()
                    if not ok:break
                    decoded+=1
                self.assertEqual(decoded,6)
            finally:
                video.release()
            checkpoint=json.loads((out/'checkpoint.json').read_text())
            self.assertEqual([part['frames'] for part in checkpoint['segments']],[3,3])
            self.assertTrue(all('original_video' in part for part in checkpoint['segments']))
            original=cv2.VideoCapture(str(out/'original.mp4'))
            try:
                self.assertEqual(int(original.get(cv2.CAP_PROP_FRAME_COUNT)),6)
                self.assertEqual(int(original.get(cv2.CAP_PROP_FRAME_WIDTH)),160)
                ok,raw=original.read()
                self.assertTrue(ok)
                self.assertLess(float(np.abs(raw.astype(float)-frame.astype(float)).mean()),50)
            finally:original.release()
            for p in out.rglob('*'):
                if p.suffix in ('.json','.jsonl'):self.assertNotIn('secret',p.read_text())


class PublishTests(unittest.TestCase):
    def test_output_queue_is_bounded_and_does_not_mutate_source(self):
        from uav_pipeline.video.publish import RTMPPublisher
        # Block network setup while proving the viewer buffer is bounded.
        gate=threading.Event()
        def run(instance):gate.wait(2)
        with patch.object(RTMPPublisher,'_run',run):
            publisher=RTMPPublisher('rtmp://host/live/result?pass=hidden',10,executable='unused')
            frame=np.zeros((80,100,3),np.uint8)
            for _ in range(10):publisher.write(frame)
            frame[:]=255
            self.assertEqual(publisher.queue.qsize(),2)
            self.assertEqual(publisher.status()['frames_dropped_for_viewing'],8)
            self.assertFalse(np.any(publisher.queue.get_nowait()))
            self.assertNotIn('hidden',json.dumps(publisher.status()))
            gate.set();publisher.release()

    def test_real_loopback_rtmp_publish_and_decode(self):
        import socket,subprocess,time
        from uav_pipeline.video.publish import RTMPPublisher,ffmpeg_path
        executable=ffmpeg_path()
        with socket.socket() as probe:
            probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
        url='rtmp://127.0.0.1:'+str(port)+'/live/result'
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'received.flv'
            server=subprocess.Popen([executable,'-nostdin','-hide_banner','-loglevel','error',
                '-listen','1','-i',url,'-c','copy','-f','flv',str(target)],
                stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            publisher=None
            try:
                time.sleep(.5)
                publisher=RTMPPublisher(url,10,executable=executable,retry_seconds=.1)
                for index in range(20):
                    frame=np.full((80,100,3),(index*10)%255,np.uint8)
                    publisher.write(frame);time.sleep(.1)
                time.sleep(.2)
                publisher.release()
                server.wait(timeout=6)
                video=cv2.VideoCapture(str(target))
                try:
                    self.assertTrue(video.isOpened())
                    decoded=0
                    while True:
                        ok,frame=video.read()
                        if not ok:break
                        self.assertEqual(frame.shape,(80,100,3));decoded+=1
                    self.assertGreaterEqual(decoded,5)
                finally:video.release()
                self.assertGreaterEqual(publisher.status()['frames_written_to_encoder'],5)
            finally:
                if publisher is not None:publisher.release()
                if server.poll() is None:
                    server.kill();server.wait()


class HLSTests(unittest.TestCase):
    def test_player_url_resolves_to_master_playlist_and_keeps_query(self):
        from uav_pipeline.video.live import normalize_live_url,is_live_url
        url='http://host:8888/live/stream/?token=secret'
        self.assertTrue(is_live_url(url))
        self.assertEqual(normalize_live_url(url),'http://host:8888/live/stream/index.m3u8?token=secret')
        direct='https://host/live/index.m3u8?token=secret'
        self.assertEqual(normalize_live_url(direct),direct)
        self.assertFalse(is_live_url('https://host/movie.mp4'))
        frame=np.zeros((80,100,3),np.uint8)
        calls=[]
        def factory(*args):calls.append(args);return Source([frame])
        cap=RTMPCapture(url,factory=factory)
        with contextlib.redirect_stdout(io.StringIO()):
            cap.open();self.assertTrue(cap.read()[0])
        self.assertEqual(calls[0][0],normalize_live_url(url))
        cap.release()

    def test_http_publish_destination_is_rejected(self):
        from uav_pipeline.video.publish import RTMPPublisher
        with self.assertRaises(ValueError):
            RTMPPublisher('http://host/live/stream/',10,executable='unused')
