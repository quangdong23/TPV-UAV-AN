"""Ghi H264 VFR theo timestamp; không giữ lịch sử frame trong RAM."""
from fractions import Fraction
import math
import numpy as np


class VFRWriter:
    def __init__(self,path,fps,shape,encoder='opencv'):
        import av
        self.av=av
        self.container=av.open(str(path),'w',options={'movflags':'+faststart'})
        self.stream=self.container.add_stream('h264_nvenc' if encoder=='nvenc' else 'libx264',
                                             rate=Fraction(str(fps)).limit_denominator(100000))
        h,w=shape[:2]
        self.stream.width=w+w%2;self.stream.height=h+h%2
        self.stream.pix_fmt='yuv420p'
        self.stream.time_base=Fraction(1,90000)
        self.stream.codec_context.time_base=self.stream.time_base
        self.stream.codec_context.max_b_frames=0
        self.stream.codec_context.thread_count=2
        self.stream.options={'preset':'p4','cq':'20'} if encoder=='nvenc' else {'preset':'veryfast','crf':'20','tune':'zerolatency'}
        self.base=None;self.last_pts=-1;self.closed=False

    def write_at(self,frame,timestamp):
        if self.closed or not math.isfinite(timestamp):raise ValueError('Timestamp/frame không hợp lệ')
        if self.base is None:self.base=timestamp
        pts=round((timestamp-self.base)*90000)
        if pts<=self.last_pts:raise ValueError('Timestamp video phải tăng liên tục')
        h,w=frame.shape[:2]
        if (h+h%2,w+w%2)!=(self.stream.height,self.stream.width):
            raise ValueError('Kích thước video thay đổi')
        if h%2 or w%2:frame=np.pad(frame,((0,h%2),(0,w%2),(0,0)),mode='edge')
        image=self.av.VideoFrame.from_ndarray(frame,format='bgr24')
        image.pts=pts;image.time_base=Fraction(1,90000)
        for packet in self.stream.encode(image):self.container.mux(packet)
        self.last_pts=pts

    def release(self):
        if self.closed:return
        self.closed=True
        try:
            for packet in self.stream.encode():self.container.mux(packet)
        finally:self.container.close()


def cut_vfr(source,target,start,end):
    """Cắt theo giây và giữ nhịp PTS; không chuyển về FPS cố định."""
    import subprocess
    from uav_pipeline.video.io import _recording_ffmpeg
    executable=_recording_ffmpeg()
    if not executable:raise RuntimeError('Cắt video VFR cần FFmpeg')
    result=subprocess.run([executable,'-nostdin','-hide_banner','-loglevel','error','-y',
        '-i',str(source),'-ss',str(start),'-t',str(end-start),'-an',
        '-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',
        '-fps_mode','vfr','-enc_time_base','1:90000','-threads','2','-movflags','+faststart',str(target)],
        capture_output=True,timeout=max(120,(end-start)*4))
    if result.returncode:raise RuntimeError('Không cắt được clip VFR')
