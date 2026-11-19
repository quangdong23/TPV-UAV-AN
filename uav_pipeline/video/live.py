"""RTMP/HLS ingest with bounded network calls and explicit reconnect boundaries.

Frames are consumed in order; no latest-frame replacement or intentional drops.
Backpressure can accumulate upstream latency if inference is slower than source.
"""
import math
import threading
import time
from urllib.parse import urlsplit, urlunsplit
import cv2


def is_live_url(value):
    parts = urlsplit(str(value))
    return parts.scheme.lower() in ('rtmp', 'rtmps') or (
        parts.scheme.lower() in ('http', 'https') and
        (parts.path.endswith('/') or parts.path.lower().endswith('.m3u8')))


def normalize_live_url(value):
    parts = urlsplit(str(value))
    if parts.scheme.lower() in ('http', 'https') and parts.path.endswith('/'):
        return urlunsplit((parts.scheme,parts.netloc,parts.path+'index.m3u8',parts.query,''))
    return str(value)


def safe_source(value):
    parts = urlsplit(str(value))
    host = parts.hostname or 'unknown'
    port = ':'+str(parts.port) if parts.port else ''
    return parts.scheme+'://'+host+port+parts.path


class RTMPCapture:
    def __init__(self, url, stop=None, open_timeout=10., read_timeout=10.,
                 retry_seconds=3., fps_fallback=25., factory=None):
        if not is_live_url(url):
            raise ValueError('Live input must be RTMP or an HTTP(S) HLS playlist/player directory')
        for value in (open_timeout, read_timeout, retry_seconds, fps_fallback):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Live timeout/retry/FPS values must be finite and positive')
        self.url = normalize_live_url(url)
        self.stop = stop if stop is not None else threading.Event()
        self.factory = factory or cv2.VideoCapture
        self.open_timeout, self.read_timeout = open_timeout, read_timeout
        self.retry_seconds, self.fps = retry_seconds, fps_fallback
        self.capture = None
        self.position = 0
        self.timestamp = -1.
        self.pending = None
        self.epoch = 0
        self.reconnects = 0
        self.interrupted = False
        self._base = 0.
        self._last_pts = None
        self._epoch_frames = 0
        self._connected_once = False
        self.shape = None
        self._last_arrival = None
        self._source_pts = None

    def _disconnect(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _connect(self):
        while not self.stop.is_set():
            print('[live] Waiting for live source (credentials hidden)', flush=True)
            # Native FFmpeg diagnostics may include a URL. Exception text is
            # deliberately suppressed; Python logs/JSON never contain credentials.
            try:
                cap = self.factory(self.url, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, round(self.open_timeout*1000),
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, round(self.read_timeout*1000)])
                if cap.isOpened():
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        reported = cap.get(cv2.CAP_PROP_FPS)
                        if math.isfinite(reported) and reported > 0:
                            self.fps = reported
                        if self.shape is not None and frame.shape != self.shape:
                            cap.release()
                            raise RuntimeError('Live resolution changed; restart live session')
                        self.shape = frame.shape
                        self.capture, self.pending = cap, frame
                        if self._connected_once:
                            self.epoch += 1
                            self.reconnects += 1
                        self._connected_once = True
                        source_pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000.
                        gap=source_pts-self._source_pts if self._source_pts is not None else 0.
                        if not math.isfinite(gap) or gap<=0:
                            gap=max(1/self.fps,time.monotonic()-self._last_arrival) if self._last_arrival is not None else 0.
                        self._base=max(0.,self.timestamp+gap) if self.position else 0.
                        self._last_pts = None
                        self._epoch_frames = 0
                        print('[live] Connected; FPS='+str(self.fps)+' epoch='+str(self.epoch), flush=True)
                        return True
                cap.release()
            except RuntimeError:
                raise
            except Exception:
                pass
            if self.stop.wait(self.retry_seconds):
                break
        self.interrupted = True
        return False

    def open(self):
        return self._connect()

    def isOpened(self):
        return self.capture is not None and not self.stop.is_set()

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:return 0
        if prop == cv2.CAP_PROP_POS_MSEC:return self.timestamp*1000
        if prop == cv2.CAP_PROP_POS_FRAMES:return self.position
        if self.capture is not None:return self.capture.get(prop)
        return 0.

    def read(self):
        while not self.stop.is_set():
            if self.capture is None and not self._connect():break
            try:
                if self.pending is not None:
                    frame,self.pending=self.pending,None
                    ok=True
                else:
                    ok,frame=self.capture.read()
            except Exception:
                ok,frame=False,None
            if not ok or frame is None:
                self._disconnect()
                print('[live] Source interrupted; reconnecting',flush=True)
                if self.stop.wait(self.retry_seconds):break
                continue
            if frame.shape != self.shape:
                raise RuntimeError('Live resolution changed; restart live session')
            pts=self.capture.get(cv2.CAP_PROP_POS_MSEC)/1000.
            if self._last_pts is None:
                delta=0.
            else:
                delta=pts-self._last_pts
                if not math.isfinite(delta) or delta<=0:delta=1/self.fps
            self.timestamp=self._base if self._epoch_frames==0 else self.timestamp+delta
            self._last_pts=pts if math.isfinite(pts) else 0.
            self._last_arrival=time.monotonic()
            self._source_pts=pts if math.isfinite(pts) else None
            self._epoch_frames+=1
            self.position+=1
            return True,frame
        self.interrupted=True
        return False,None

    def release(self):
        self.stop.set()
        self._disconnect()
