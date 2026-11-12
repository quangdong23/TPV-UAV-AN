"""Bounded file decode prefetch; the worker owns the positioned capture.

No frames are skipped. POS_MSEC/POS_FRAMES describe the last consumed frame,
never the decoder's future position. Position/seek the capture before wrapping.
"""
import queue
import threading

import cv2


class PrefetchCapture:
    def __init__(self, capture, capacity=2):
        if type(capacity) is not int or not 1 <= capacity <= 8:
            raise ValueError('Prefetch capacity must be an integer in 1..8')
        self.capture = capture
        self.opened = capture.isOpened()
        self.metadata = {p: capture.get(p) for p in (
            cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT,
            cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)}
        self.position = capture.get(cv2.CAP_PROP_POS_FRAMES)
        self.timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
        self.queue = queue.Queue(maxsize=capacity)
        self.stop = threading.Event()
        self.ended = False
        self.error = None
        self.worker = threading.Thread(target=self._run, name='uav-file-decode', daemon=True)
        self.worker.start()

    def _put(self, value):
        while not self.stop.is_set():
            try:
                self.queue.put(value, timeout=.1)
                return
            except queue.Full:
                pass

    def _run(self):
        try:
            while not self.stop.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    break
                self._put((frame, self.capture.get(cv2.CAP_PROP_POS_MSEC),
                           self.capture.get(cv2.CAP_PROP_POS_FRAMES)))
        except Exception as error:
            self.error = error
        finally:
            try:
                self.capture.release()
            except Exception as error:
                if self.error is None:
                    self.error = error
            finally:
                self._put(None)

    def isOpened(self):
        return self.opened and not self.stop.is_set()

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.timestamp
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return self.position
        return self.metadata.get(prop, 0.)

    def read(self):
        if self.ended:
            return False, None
        while not self.stop.is_set():
            try:
                item = self.queue.get(timeout=.1)
                break
            except queue.Empty:
                continue
        else:
            self.ended = True
            return False, None
        if item is None:
            self.ended = True
            if self.error is not None:
                raise RuntimeError('Prefetch video decoder failed') from self.error
            return False, None
        frame, self.timestamp, self.position = item
        return True, frame

    def grab(self):
        return self.read()[0]

    def release(self):
        self.ended = True
        self.opened = False
        self.stop.set()
        self.worker.join(timeout=5)
        if self.worker.is_alive():
            raise RuntimeError('Video decode worker failed to stop')
