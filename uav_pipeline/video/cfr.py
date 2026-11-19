"""Fixed-rate recording slots driven by source timestamps, not AI frame count."""
from fractions import Fraction
import math
import subprocess


class TimestampCFRClock:
    def __init__(self,fps=15):
        if not math.isfinite(fps) or fps<=0:
            raise ValueError('Recording FPS must be finite and positive')
        self.fps=float(fps);self.origin=None;self.next_index=0;self.last_timestamp=None

    def push(self,timestamp):
        if not math.isfinite(timestamp) or (self.last_timestamp is not None and timestamp<=self.last_timestamp):
            raise ValueError('Recording timestamps must increase strictly')
        if self.origin is None:self.origin=timestamp
        self.last_timestamp=timestamp
        stop=math.floor((timestamp-self.origin)*self.fps+1e-8)+1
        start=self.next_index;self.next_index=max(self.next_index,stop)
        # A range, not a list: reconnect gaps never allocate a frame backlog.
        return ((self.origin+i/self.fps,abs(self.origin+i/self.fps-timestamp)<1e-8)
                for i in range(start,stop))

    def finish(self,end_timestamp):
        if self.origin is None:return iter(())
        if not math.isfinite(end_timestamp) or end_timestamp<self.last_timestamp:
            raise ValueError('Recording end precedes last frame')
        stop=math.ceil((end_timestamp-self.origin)*self.fps-1e-8)
        start=self.next_index;self.next_index=max(self.next_index,stop)
        return (self.origin+i/self.fps for i in range(start,stop))


class _CFRFrameWriter:
    def __init__(self,path,fps,shape,encoder):
        from uav_pipeline.video.vfr import VFRWriter
        self.writer=VFRWriter(path,fps,shape,encoder)
        self.fps=float(fps);self.frames=0
    def write(self,frame):
        self.writer.write_at(frame,self.frames/self.fps)
        self.frames+=1
    def release(self):self.writer.release()


def open_cfr_writer(path,fps,shape,encoder='opencv'):
    from uav_pipeline.video.io import ThreadedVideoWriter
    # At most two copied frames waiting per stream; errors propagate to caller.
    return ThreadedVideoWriter(_CFRFrameWriter(path,fps,shape,encoder))


def cut_cfr(source,target,start,end,fps):
    from uav_pipeline.video.io import _recording_ffmpeg
    executable=_recording_ffmpeg()
    if not executable:raise RuntimeError('CFR evidence cutting requires FFmpeg')
    first=max(0,math.ceil(start*fps-1e-8))
    last=max(first+1,math.ceil(end*fps-1e-8))
    rate=Fraction(str(fps)).limit_denominator(100000)
    result=subprocess.run([executable,'-nostdin','-hide_banner','-loglevel','error','-y',
        '-i',str(source),'-an','-vf',f'trim=start_frame={first}:end_frame={last},setpts=PTS-STARTPTS',
        '-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',
        '-r',str(rate),'-fps_mode','cfr','-threads','2','-movflags','+faststart',str(target)],
        capture_output=True,timeout=max(120,(end-start)*4))
    if result.returncode:raise RuntimeError('Unable to cut timestamp-aligned CFR evidence')
