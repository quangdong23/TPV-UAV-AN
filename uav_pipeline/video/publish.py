"""Bounded RTMP viewing output. Local recordings keep every processed frame.

Only the viewer queue may discard old frames when publishing cannot keep up.
No URL/password is written to application status or logs.
"""
import math
import os
import queue
import select
import shutil
import subprocess
import threading
import time
import numpy as np
import cv2
from urllib.parse import urlsplit
from uav_pipeline.video.live import safe_source


def ffmpeg_path():
    executable=shutil.which('ffmpeg')
    if executable:return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError('Live output requires ffmpeg or imageio-ffmpeg') from None


def publish_command(executable,url,fps,shape):
    h,w=shape[:2]
    return [executable,'-nostdin','-hide_banner','-loglevel','error','-y',
        '-f','rawvideo','-pixel_format','bgr24','-video_size',str(w)+'x'+str(h),
        '-framerate',str(fps),'-i','pipe:0','-an','-vf','pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-c:v','libx264','-preset','ultrafast','-tune','zerolatency','-crf','23',
        '-pix_fmt','yuv420p','-threads','2','-g',str(max(1,round(fps))),
        '-bf','0','-flvflags','no_duration_filesize','-rw_timeout','3000000','-f','flv',url]


class RTMPPublisher:
    def __init__(self,url,fps,executable=None,retry_seconds=3.):
        if urlsplit(url).scheme.lower() not in ('rtmp','rtmps'):raise ValueError('Live output must use RTMP or RTMPS')
        if not math.isfinite(fps) or fps<=0:raise ValueError('Invalid publish FPS')
        self.url,self.fps=url,fps
        self.executable=executable or ffmpeg_path()
        self.retry_seconds=retry_seconds
        self.queue=queue.Queue(maxsize=2)
        self.stop=threading.Event()
        self.state={'enabled':True,'destination':safe_source(url),'status':'waiting_for_frames',
                    'frames_submitted':0,'frames_written_to_encoder':0,'frames_dropped_for_viewing':0,
                    'connection_attempts':0,'publisher_failures':0,
                    'note':'Encoder writes do not confirm delivery to viewers. Local recordings retain all processed frames.'}
        self.worker=threading.Thread(target=self._run,name='uav-rtmp-publish',daemon=True)
        self.worker.start()

    def write(self,frame):
        if self.stop.is_set():return
        self.state['frames_submitted']+=1
        value=np.ascontiguousarray(frame).copy()
        try:self.queue.put_nowait(value)
        except queue.Full:
            try:self.queue.get_nowait();self.state['frames_dropped_for_viewing']+=1
            except queue.Empty:pass
            try:self.queue.put_nowait(value)
            except queue.Full:self.state['frames_dropped_for_viewing']+=1

    def status(self):return dict(self.state)

    @staticmethod
    def _close(process):
        if process is None:return
        try:
            process.terminate()
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill();process.wait(timeout=2)
        except ProcessLookupError:pass
        finally:
            if process.stdin is not None:process.stdin.close()

    def _write(self,process,frame):
        data=memoryview(frame).cast('B')
        deadline=time.monotonic()+3.
        fd=process.stdin.fileno()
        while data and not self.stop.is_set():
            if process.poll() is not None:raise BrokenPipeError()
            if time.monotonic()>deadline:raise TimeoutError()
            _,ready,_=select.select([],[fd],[],.1)
            if ready:
                try:written=os.write(fd,data[:65536])
                except BlockingIOError:continue
                if written<=0:raise BrokenPipeError()
                data=data[written:]
        return not data

    def _run(self):
        process=None
        shape=None
        retry_at=0.
        try:
            while not self.stop.is_set():
                try:frame=self.queue.get(timeout=.1)
                except queue.Empty:continue
                if time.monotonic()<retry_at:
                    self.state['frames_dropped_for_viewing']+=1
                    continue
                try:
                    if process is None or shape!=frame.shape:
                        self._close(process);process=None
                        shape=frame.shape
                        self.state['status']='connecting'
                        self.state['connection_attempts']+=1
                        process=subprocess.Popen(publish_command(self.executable,self.url,self.fps,shape),
                            stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,bufsize=0)
                        os.set_blocking(process.stdin.fileno(),False)
                    if self._write(process,frame):
                        self.state['frames_written_to_encoder']+=1
                        self.state['status']='sending_to_encoder'
                except (OSError,TimeoutError,ValueError):
                    self.state['publisher_failures']+=1
                    self.state['frames_dropped_for_viewing']+=1
                    self.state['status']='reconnecting'
                    print('[live-output] Connection unavailable; retrying (local recording continues)',flush=True)
                    self._close(process);process=None
                    retry_at=time.monotonic()+self.retry_seconds
        finally:
            self._close(process)
            self.state['frames_dropped_for_viewing']+=self.queue.qsize()
            self.state['status']='stopped'

    def release(self):
        self.stop.set()
        self.worker.join(timeout=8.)
        if self.worker.is_alive():raise RuntimeError('Live output worker failed to stop')
