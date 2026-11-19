import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from uav_pipeline.video.evidence import LiveEvidence


class ShutdownTests(unittest.TestCase):
    def test_waits_for_slow_request_and_final_job(self):
        entered = threading.Event()
        calls = []
        class Uploader:
            def upload_events(self, output, events):
                calls.append(events[0]['event_id'])
                entered.set()
                time.sleep(1.2)
                return {'ready': 1, 'already_ready': 0, 'failed': 0}
        with tempfile.TemporaryDirectory() as tmp:
            evidence = LiveEvidence(tmp, None, 30, Uploader())
            for name in ('first',):
                (evidence.pending/(name+'.json')).write_text(json.dumps({'event_id': name}))
            evidence.wake.set()
            self.assertTrue(entered.wait(5))
            (evidence.pending/'last.json').write_text(json.dumps({'event_id': 'last'}))
            evidence.close()
            self.assertFalse(evidence.worker.is_alive())
            self.assertEqual(calls, ['first', 'last'])
            self.assertEqual(list(evidence.pending.glob('*.json')), [])
            self.assertEqual(json.loads((Path(tmp)/'upload_summary.json').read_text())['ready'], 2)

    def test_failed_request_is_retained_without_endless_retry(self):
        class Uploader:
            def upload_events(self, output, events):
                return {'ready': 0, 'already_ready': 0, 'failed': 1}
        with tempfile.TemporaryDirectory() as tmp:
            evidence = LiveEvidence(tmp, None, 30, Uploader())
            (evidence.pending/'failed.json').write_text('{}')
            evidence.close()
            self.assertFalse(evidence.worker.is_alive())
            self.assertTrue((evidence.pending/'failed.json').exists())
            self.assertEqual(json.loads((Path(tmp)/'upload_summary.json').read_text())['pending'], 1)


if __name__ == '__main__':
    unittest.main()
