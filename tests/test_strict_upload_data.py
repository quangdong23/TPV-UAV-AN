import json
import unittest
from tests import test_be_client_integration as fixtures
from uav_pipeline.uploads import UploadError


class StrictUploadDataTests(unittest.TestCase):
    setUp=fixtures.BEIntegrationTests.setUp

    def test_image_put_is_exact_bytes_with_explicit_length_and_no_redirect(self):
        self.client.upload_one(self.root,self.event)
        url,kwargs=self.api.puts[0]
        self.assertEqual(kwargs['data'],(self.root/'crop.jpg').read_bytes())
        self.assertEqual(kwargs['headers']['Content-Length'],'13')
        self.assertEqual(kwargs['headers']['Content-Type'],'image/jpeg')
        self.assertIs(kwargs['allow_redirects'],False)
        self.assertTrue(all(kw['allow_redirects'] is False for _,kw in self.api.puts))

    def test_storage_redirect_aborts_without_complete(self):
        original=self.api.put
        def put(url,**kwargs):
            result=original(url,**kwargs);result.status_code=302
            return result
        self.client._client.requests.put=put
        with self.assertRaises(UploadError):self.client.upload_one(self.root,self.event)
        self.assertFalse(any(url.endswith('/complete') for _,url,_ in self.api.calls))
        self.assertTrue(self.api.calls[-1][1].endswith('/abort'))

    def test_supplied_violators_and_event_gps_override_demo_and_config(self):
        people=[{'full_name':'Test supplied record','citizen_id':'000000000099'}]
        self.client.upload_one(self.root,{**self.event,'violators':people,'location':'21.123456789, 105.987654321'})
        payload=self.api.calls[0][2]['json']
        self.assertEqual(payload['violators'],people)
        self.assertEqual(payload['location'],'21.123456789, 105.987654321')

    def test_invalid_gps_or_numeric_citizen_id_sends_no_request(self):
        for extra in ({'location':'Ha Noi'},{'location':'91, 105'},
                      {'violators':[{'full_name':'Test','citizen_id':123}]}):
            with self.assertRaises(UploadError):
                self.client.upload_one(self.root,{**self.event,**extra})
        self.assertEqual(self.api.calls,[])

    def test_image_bytes_changed_after_declaration_are_not_sent(self):
        backend=self.client._client
        with self.assertRaisesRegex(RuntimeError,'size differs'):
            backend.upload_image(self.root/'crop.jpg','https://storage/image','image/jpeg',expected_size=999)
        self.assertEqual(self.api.puts,[])

    def test_wrong_image_filename_in_plan_aborts_without_put(self):
        original=self.api.request
        def request(method,url,**kwargs):
            result=original(method,url,**kwargs)
            if url.endswith('/initiate'):
                body=result.json();body['images'][0]['filename']='wrong.jpg'
                return fixtures.response(body)
            return result
        self.client._client.requests.request=request
        with self.assertRaises(UploadError):self.client.upload_one(self.root,self.event)
        self.assertEqual(self.api.puts,[])
        self.assertTrue(self.api.calls[-1][1].endswith('/abort'))
