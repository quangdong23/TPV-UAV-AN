import io
import hashlib
import inspect
import json
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from lan_chiem.colab_stream import run_streaming, read_log_tail


class StreamTests(unittest.TestCase):
    def test_notebook_ignores_cached_helper_from_an_older_project(self):
        root=Path(__file__).resolve().parents[2]
        notebook=json.loads((root/'UAV_LanChiem_Video_T4.ipynb').read_text(encoding='utf-8'))
        source=next(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code'
                    and 'import types, inspect' in ''.join(c['source']))
        block=source[source.index('import types, inspect'):source.index("print('Build:'")]
        helper=root/'lan_chiem/colab_stream.py'
        scope={'PROJECT':root,'manifest':{'sha256':{'lan_chiem/colab_stream.py':hashlib.sha256(helper.read_bytes()).hexdigest()}},
               'hashlib':hashlib}
        stale=SimpleNamespace(run_streaming=lambda command:None)
        with patch.dict(sys.modules,{'lan_chiem.colab_stream':stale}),redirect_stdout(io.StringIO()):
            exec(compile(block,'<notebook-helper>','exec'),scope)
        self.assertIn('echo',inspect.signature(scope['run_streaming']).parameters)
        self.assertEqual(scope['run_streaming'].__code__.co_filename,str(helper))
        self.assertIsNot(scope['run_streaming'],stale.run_streaming)

    def test_missing_log_does_not_mask_original_exception(self):
        self.assertIn('Log unavailable', read_log_tail(Path(__file__).with_name('absent_setup_log.txt')))
    def test_notebook_receives_stdout_stderr_and_durable_log(self):
        root = Path(__file__).resolve().parents[1] / 'outputs' / 'video_tests'
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            log = Path(directory) / 'run.log'
            lines, stdout = [], io.StringIO()
            with redirect_stdout(stdout):
                run_streaming([sys.executable, '-u', '-c',
                               "import sys; print('frame=1 avg_fps=2.00',flush=True); print('stderr visible',file=sys.stderr,flush=True)"],
                              log_path=log, on_line=lines.append)
            self.assertEqual(lines, ['frame=1 avg_fps=2.00', 'stderr visible'])
            self.assertEqual(stdout.getvalue(), log.read_text(encoding='utf-8'))

    def test_failure_does_not_look_like_success(self):
        with self.assertRaises(subprocess.CalledProcessError):
            run_streaming([sys.executable, '-c', 'raise SystemExit(2)'])

    def test_compact_mode_still_delivers_progress_without_printing_every_line(self):
        lines,stdout=[],io.StringIO()
        with redirect_stdout(stdout):
            run_streaming([sys.executable,'-u','-c',"print('[video] frame=1 avg_fps=12.5')"],
                          on_line=lines.append,echo=False)
        self.assertEqual(stdout.getvalue(),'')
        self.assertEqual(lines,['[video] frame=1 avg_fps=12.5'])
