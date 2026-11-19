import json
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest
import hashlib
from types import SimpleNamespace
from tests import test_be_client_integration as fixtures
from uav_pipeline.uploads import UploadError


class BEUploadFixTests(unittest.TestCase):
    setUp=fixtures.BEIntegrationTests.setUp

    def test_image_fingerprint_and_storage_response_are_logged_without_signed_query(self):
        original=self.api.put
        def put(url,**kwargs):
            result=original(url,**kwargs)
            if url.endswith('/image'):
                raw=(self.root/'crop.jpg').read_bytes()
                result.headers['ETag']='"'+hashlib.md5(raw).hexdigest()+'"'
                result.headers['x-amz-request-id']='storage-request-id'
                result.request=SimpleNamespace(method='PUT',headers={'Content-Length':str(len(raw))})
            return result
        self.client._client.requests.put=put
        self.client.upload_one(self.root,self.event)
        rows=[json.loads(line) for line in (self.root/'upload_http.jsonl').read_text().splitlines()]
        image=next(r for r in rows if r.get('kind')=='image')
        self.assertTrue(image['etag_matches_uploaded_bytes'])
        self.assertEqual(image['sent_bytes'],13)
        self.assertEqual(image['final_method'],'PUT')
        self.assertEqual(image['storage_request_id'],'storage-request-id')

    def test_redirected_get_is_not_accepted_as_successful_put(self):
        original=self.api.put
        def put(url,**kwargs):
            result=original(url,**kwargs)
            result.request=SimpleNamespace(method='GET',headers={})
            return result
        self.client._client.requests.put=put
        with self.assertRaises(UploadError):self.client.upload_one(self.root,self.event)
        self.assertFalse(any(url.endswith('/complete') for _,url,_ in self.api.calls))

    def test_complete_waits_for_image_put(self):
        entered=threading.Event();release=threading.Event()
        original=self.api.put
        def put(url,**kwargs):
            if url.endswith('/image'):
                entered.set()
                if not release.wait(5):raise RuntimeError('Test release timeout')
            return original(url,**kwargs)
        self.api.put=put
        self.client._client.requests.put=put
        with ThreadPoolExecutor(max_workers=1) as executor:
            future=executor.submit(self.client.upload_one,self.root,self.event)
            try:
                self.assertTrue(entered.wait(3))
                self.assertFalse(any(url.endswith(('/part-urls','/complete')) for _,url,_ in self.api.calls))
            finally:release.set()
            self.assertEqual(future.result(timeout=5),'ready')

    def test_missing_etag_aborts_without_complete(self):
        original=self.api.put
        def put(url,**kwargs):
            result=original(url,**kwargs)
            if '/part/' in url:result.headers.pop('ETag',None)
            return result
        self.api.put=put
        self.client._client.requests.put=put
        with self.assertRaises(UploadError):self.client.upload_one(self.root,self.event)
        self.assertFalse(any(url.endswith('/complete') for _,url,_ in self.api.calls))
        self.assertTrue(self.api.calls[-1][1].endswith('/abort'))

    def test_image_plan_mismatch_aborts_without_put_or_complete(self):
        original=self.api.request
        def request(method,url,**kwargs):
            result=original(method,url,**kwargs)
            if url.endswith('/initiate'):
                body=result.json();body['images']=[]
                result=fixtures.response(body)
            return result
        self.api.request=request
        self.client._client.requests.request=request
        with self.assertRaisesRegex(UploadError,'RuntimeError'):
            self.client.upload_one(self.root,self.event)
        self.assertEqual(self.api.puts,[])
        self.assertFalse(any(url.endswith('/complete') for _,url,_ in self.api.calls))

    def test_audit_has_each_http_status_and_complete_called_once(self):
        self.api.fail_complete=True
        with self.assertRaisesRegex(UploadError,'Evidence image'):
            self.client.upload_one(self.root,self.event)
        lines=(self.root/'upload_http.jsonl').read_text()
        self.assertNotIn('test-key',lines)
        self.assertNotIn('https://storage',lines)
        rows=[json.loads(line) for line in lines.splitlines()]
        puts=[r for r in rows if r['event']=='put_response']
        self.assertEqual(len(puts),3)
        self.assertTrue(all(r['status']==200 for r in puts))
        parts=[r for r in puts if r['kind']=='video_part']
        self.assertTrue(all(r['etag_present'] for r in parts))
        complete=[r for r in rows if r.get('endpoint','').endswith('/complete') and r['event']=='api_response']
        self.assertEqual(len(complete),1)
        self.assertEqual(complete[0]['status'],409)
        self.assertEqual(complete[0]['body']['detail'],'Evidence image has not been uploaded')
        self.assertTrue(any(r['event']=='upload_exception' and 'traceback' in r for r in rows))

    def test_non_200_abort_does_not_allow_new_session(self):
        self.api.fail_complete=True
        original=self.api.request
        def request(method,url,**kwargs):
            result=original(method,url,**kwargs)
            if url.endswith('/abort'):result.status_code=201
            return result
        self.api.request=request
        self.client._client.requests.request=request
        with self.assertRaises(UploadError):self.client.upload_one(self.root,self.event)
        calls=len(self.api.calls)
        with self.assertRaisesRegex(UploadError,'outcome uncertain'):
            self.client.upload_one(self.root,self.event)
        self.assertEqual(len(self.api.calls),calls)
