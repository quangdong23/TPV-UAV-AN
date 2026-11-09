"""Streaming exports and resumable video segments; no video held in RAM."""
import csv
import hashlib
import json
import shutil
import subprocess
import queue
import threading
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


class NVENCWriter:
    """Bounded raw-frame pipe to FFmpeg; no queued frames retained in Python.

    Encoder failure must abort the segment before its checkpoint is committed.
    stderr goes to a file so FFmpeg cannot deadlock on a full stderr pipe.
    """
    def __init__(self, path, fps, size):
        executable = shutil.which('ffmpeg')
        if executable is None:
            raise RuntimeError('NVENC output requires ffmpeg with h264_nvenc support')
        self.size = size
        self.closed = False
        self.log_path = Path(path).with_suffix('.encoder.log')
        self.log = self.log_path.open('w+b')
        w, h = size
        command = [executable, '-hide_banner', '-loglevel', 'error', '-y',
                   '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', f'{w}x{h}',
                   '-framerate', str(fps), '-i', 'pipe:0', '-an',
                   '-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '20',
                   '-pix_fmt', 'yuv420p', '-threads', '2', '-filter_threads', '2',
                   '-movflags', '+faststart', str(path)]
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                            stderr=self.log,
                                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except Exception:
            self.log.close()
            raise

    def write(self, frame):
        w, h = self.size
        if self.closed:
            raise RuntimeError('Video encoder is already closed')
        if frame.dtype != np.uint8 or frame.shape != (h, w, 3):
            raise ValueError('Encoder input must match the padded HxWx3 uint8 frame size')
        try:
            self.process.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as error:
            self.release()
            raise RuntimeError(f'Video encoder pipe failed; see {self.log_path}') from error

    def release(self):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                code = self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
                raise RuntimeError(f'Video encoder did not finalize; see {self.log_path}')
            if code:
                self.log.flush()
                self.log.seek(0)
                detail = self.log.read().decode('utf-8', errors='replace')[-2000:]
                raise RuntimeError(f'NVENC encoder failed ({code}): {detail}')
        finally:
            self.log.close()


class ThreadedVideoWriter:
    """Overlap encoding with inference with at most two queued frame copies."""
    def __init__(self, writer):
        self.writer = writer
        self.queue = queue.Queue(maxsize=2)
        self.error = None
        self.closed = False
        self.worker = threading.Thread(target=self._run, name='video-encoder', daemon=True)
        self.worker.start()

    def _run(self):
        try:
            while True:
                frame = self.queue.get()
                if frame is None:
                    return
                self.writer.write(frame)
        except Exception as error:
            self.error = error

    def _check_error(self):
        if self.error is not None:
            raise RuntimeError(f'Background video encoder failed: {self.error}') from self.error

    def _put(self, frame):
        while True:
            self._check_error()
            try:
                self.queue.put(frame, timeout=.1)
                return
            except queue.Full:
                continue

    def write(self, frame):
        if self.closed:
            raise RuntimeError('Video encoder is already closed')
        self._put(frame.copy())  # The caller may reuse or modify its array.

    def release(self):
        if self.closed:
            return
        self.closed = True
        try:
            self._put(None)
        finally:
            # All queued frames must reach the encoder before checkpoint/hash.
            self.worker.join()
            self.writer.release()
        self._check_error()


def open_writer(path, fps, shape, encoder='opencv', async_encode=False):
    h, w = shape[:2]
    size = (w + w % 2, h + h % 2)
    if encoder == 'nvenc':
        writer = NVENCWriter(path, fps, size)
    elif encoder == 'opencv':
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f'Cannot create video: {path}')
    else:
        raise ValueError(f'Unknown video encoder: {encoder}')
    return ThreadedVideoWriter(writer) if async_encode else writer


def write_frame(writer, frame):
    h, w = frame.shape[:2]
    if h % 2 or w % 2:
        frame = cv2.copyMakeBorder(frame, 0, h % 2, 0, w % 2, cv2.BORDER_REPLICATE)
    writer.write(frame)


def consolidate(output, segments, export_csv=True, consume_single=False):
    """Rebuild public exports only from committed segments."""
    output = Path(output)
    fields = ['frame_index', 'timestamp_seconds', 'track_id', 'class_id', 'class_name_vi',
              'x1', 'y1', 'x2', 'y2', 'detection_score', 'last_detection_timestamp',
              'position_source', 'track_state']
    json_temp = output / 'tracks.jsonl.tmp'
    csv_temp = output / 'tracks.csv.tmp'
    csv_stream = csv_temp.open('w', encoding='utf-8', newline='') if export_csv else nullcontext(None)
    with json_temp.open('w', encoding='utf-8') as target, csv_stream as table:
        writer = csv.DictWriter(table, fieldnames=fields) if table is not None else None
        if writer is not None:
            writer.writeheader()
        for segment in segments:
            with (output / segment['records']).open(encoding='utf-8') as source:
                for line in source:
                    target.write(line)
                    record = json.loads(line)
                    for track in record['tracks'] if writer is not None else []:
                        row = {k: track[k] for k in fields if k in track}
                        row.update(frame_index=record['frame_index'], timestamp_seconds=record['timestamp_seconds'])
                        row.update(zip(('x1', 'y1', 'x2', 'y2'), track['bbox_xyxy']))
                        writer.writerow(row)
    json_temp.replace(output / 'tracks.jsonl')
    if export_csv:
        csv_temp.replace(output / 'tracks.csv')
    if not segments:
        return 'no_frames'
    if len(segments) == 1:
        # A single committed video is already final: copy bytes, never re-encode.
        temporary = output / 'annotated.tmp.mp4'
        if consume_single:
            (output / segments[0]['video']).replace(temporary)
        else:
            shutil.copyfile(output / segments[0]['video'], temporary)
        temporary.replace(output / 'annotated.mp4')
        return 'annotated.mp4'
    if shutil.which('ffmpeg') is None:
        return 'ffmpeg_missing: video available in segments/'
    # Relative generated names contain no user-supplied quoting characters.
    listing = output / 'segments' / 'concat.txt'
    listing.write_text(''.join(f"file '{Path(s['video']).name}'\n" for s in segments), encoding='utf-8')
    temporary = output / 'annotated.tmp.mp4'
    result = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat',
                             '-safe', '1', '-i', str(listing), '-c', 'copy', str(temporary)],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'Cannot concatenate segments: {result.stderr[-2000:]}')
    temporary.replace(output / 'annotated.mp4')
    return 'annotated.mp4'


def remove_committed_intermediates(output, segments):
    """Remove only generated files after final video/data/summary are durable.

    Never remove a directory recursively or discard unknown files in it.
    """
    output = Path(output)
    for name in ('annotated.mp4', 'tracks.jsonl', 'summary.json'):
        if not (output/name).is_file() or (output/name).stat().st_size == 0:
            raise RuntimeError(f'Final output missing; intermediates preserved: {name}')
    paths = []
    for index, segment in enumerate(segments):
        for key in ('video', 'records'):
            name = segment[key]
            expected = f'segments/part_{index:06d}.{"mp4" if key == "video" else "jsonl"}'
            if name != expected:
                raise ValueError('Refusing cleanup of unexpected segment path')
            paths.append(output/name)
        paths.extend([output/f'segments/previous_{index:06d}.png',
                      output/f'segments/part_{index:06d}.encoder.log'])
    paths.extend(output/name for name in ('segments/concat.txt', 'checkpoint.json', 'status.json', 'latest_preview.jpg'))
    for path in paths:
        path.unlink(missing_ok=True)
    directory = output/'segments'
    if directory.is_dir() and not any(directory.iterdir()):
        directory.rmdir()
