import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import requests
from uav_pipeline.uploads import EvidenceUploader, UploadError


def response(data,code=200,headers=None):
    result=requests.Response();result.status_code=code
    result._content=json.dumps(data).encode()
    result.headers.update(headers or {})
    return result


class Transport:
    def __init__(self):
        self.calls=[];self.puts=[];self.fail_complete=False;self.fail_abort=False
    def request(self,method,url,**kwargs):
        self.calls.append((method,url,kwargs))
        payload=kwargs.get('json') or {}
        if url.endswith('/initiate'):
            return response({'violation_id':'be-id','part_size':4,'total_parts':2,
                'images':[{'upload_url':'https://storage/image'} for _ in payload['images']]})
        if url.endswith('/part-urls'):
            return response({'parts':[{'part_number':n,'url':f'https://storage/part/{n}'} for n in payload['part_numbers']]})
        if url.endswith('/complete'):
            return response({'detail':'Evidence image has not been uploaded'},409) if self.fail_complete else response({'success':True,'status':'ready','violation_id':'be-id'})
        if url.endswith('/abort'):
            return response({},503 if self.fail_abort else 200)
        raise AssertionError(url)
    def put(self,url,**kwargs):
        if hasattr(kwargs['data'],'read'):
            kwargs={**kwargs,'data':kwargs['data'].read()}
        self.puts.append((url,kwargs))
        return response({},headers={'ETag':'"etag"'})


class BEIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        (self.root/'clip.mp4').write_bytes(b'12345678')
        (self.root/'crop.jpg').write_bytes(b'jpeg-evidence')
        self.api=Transport()
        self.client=EvidenceUploader('https://api.example','test-key',location='20, 105',http=self.api,sleep=lambda _:None)
        self.client._client.ensure_browser_compatible_video=Mock()
        self.event={'event_id':'e1','kind':'sidewalk_encroachment',
            'detected_at':'2026-11-18T12:00:00+07:00',
            'clip':{'status':'saved','path':'clip.mp4'},'images':[{'path':'crop.jpg'}],
            'demo_people':[{'full_name':'Sample','cccd':'000000000001','person_track_id':1,'is_demo':True}]}

    def test_original_client_uploads_images_parts_completes_and_deduplicates(self):
        self.assertEqual(self.client.upload_one(self.root,self.event),'ready')
        payload=self.api.calls[0][2]['json']
        self.assertEqual(payload['violators'],[{'full_name':'Sample','citizen_id':'000000000001'}])
        self.assertNotIn('people',payload);self.assertNotIn('cccd',payload)
        self.assertEqual(len(payload['images']),1)
        self.assertEqual(self.api.puts[0][1]['headers']['Content-Type'],'image/jpeg')
        self.assertEqual(self.api.puts[0][1]['headers']['Content-Length'],'13')
        self.assertEqual(len(self.api.puts),3)
        self.assertEqual(self.api.calls[-1][2]['json'],{'parts':[{'part_number':1,'etag':'"etag"'},{'part_number':2,'etag':'"etag"'}]})
        calls=len(self.api.calls)
        self.assertEqual(self.client.upload_one(self.root,self.event),'already_ready')
        self.assertEqual(len(self.api.calls),calls)

    def test_failure_uses_be_abort_then_can_retry(self):
        self.api.fail_complete=True
        with self.assertRaisesRegex(UploadError,'Evidence image'):
            self.client.upload_one(self.root,self.event)
        self.assertTrue(self.api.calls[-1][1].endswith('/abort'))
        state=json.loads(next((self.root/'uploads').glob('*.json')).read_text())
        self.assertEqual(state['state'],'retry_pending')
        self.api.fail_complete=False
        self.assertEqual(self.client.upload_one(self.root,self.event),'ready')

    def test_unconfirmed_abort_blocks_duplicate(self):
        self.api.fail_complete=self.api.fail_abort=True
        with self.assertRaises(UploadError):self.client.upload_one(self.root,self.event)
        calls=len(self.api.calls)
        with self.assertRaisesRegex(UploadError,'outcome uncertain'):
            self.client.upload_one(self.root,self.event)
        self.assertEqual(len(self.api.calls),calls)

    def test_no_identity_or_images_sends_explicit_null_and_empty_list(self):
        event={k:v for k,v in self.event.items() if k not in ('demo_people','images')}
        self.client.upload_one(self.root,event)
        payload=self.api.calls[0][2]['json']
        self.assertIsNone(payload['violators']);self.assertEqual(payload['images'],[])

    def test_preflight_failure_retains_job_without_network(self):
        self.client._client.ensure_browser_compatible_video.side_effect=RuntimeError('ffprobe missing')
        with self.assertRaises(UploadError):self.client.upload_one(self.root,self.event)
        self.assertEqual(self.api.calls,[])
        state=json.loads(next((self.root/'uploads').glob('*.json')).read_text())
        self.assertEqual(state['state'],'retry_pending')


if __name__=='__main__':unittest.main()
