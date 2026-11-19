"""Codec validation uses actual encoded media; API tests never contact services."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import av
import numpy as np
from uav_pipeline.uploads import ensure_browser_compatible_video, UploadError
from tests import test_uploads as fixtures


class CodecTests(unittest.TestCase):
    def test_pyav_fallback_accepts_h264_and_rejects_mpeg4(self):
        with tempfile.TemporaryDirectory() as tmp, patch('uav_pipeline.uploads.shutil.which',return_value=None):
            for codec in ('libx264','mpeg4'):
                path=Path(tmp)/(codec+'.mp4')
                with av.open(str(path),'w') as container:
                    stream=container.add_stream(codec,rate=25)
                    stream.width=64;stream.height=64;stream.pix_fmt='yuv420p'
                    for _ in range(2):
                        frame=av.VideoFrame.from_ndarray(np.zeros((64,64,3),np.uint8),format='bgr24')
                        for packet in stream.encode(frame):container.mux(packet)
                    for packet in stream.encode():container.mux(packet)
                if codec=='libx264':ensure_browser_compatible_video(path)
                else:
                    with self.assertRaisesRegex(UploadError,'H.264/yuv420p'):
                        ensure_browser_compatible_video(path)


class RecoveryTests(unittest.TestCase):
    setUp=fixtures.UploadTests.setUp
    tearDown=fixtures.UploadTests.tearDown
    def test_abort_failure_never_initiates_duplicate(self):
        self.api.fail_parts=True
        original=self.api.request
        def request(method,url,**kwargs):
            if url.endswith('/abort'):return fixtures.Response({'detail':'abort unavailable'},503)
            return original(method,url,**kwargs)
        self.api.request=request
        with self.assertRaisesRegex(UploadError,'abort not confirmed'):
            self.client.upload_one(self.root,self.event)
        with self.assertRaisesRegex(UploadError,'outcome uncertain'):
            self.client.upload_one(self.root,self.event)
        self.assertEqual(sum(url.endswith('/initiate') for _,url,_ in self.api.calls),1)

    def test_complete_timeout_does_not_abort_or_create_duplicate(self):
        self.api.uncertain=True
        with self.assertRaises(UploadError):self.client.upload_one(self.root,self.event)
        self.assertFalse(any(url.endswith('/abort') for _,url,_ in self.api.calls))
