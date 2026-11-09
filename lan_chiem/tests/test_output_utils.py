"""Unit tests for unified output directory management and non-colliding auto-naming."""
import tempfile
import unittest
from pathlib import Path

from lan_chiem.output_utils import generate_unique_output_dir, increment_path, UNIFIED_OUTPUTS_DIR


class OutputUtilsTests(unittest.TestCase):
    def test_auto_generate_output_in_unified_root(self):
        out = generate_unique_output_dir(None, task_name='pipeline', source_name='test_video.mp4')
        self.assertEqual(out.parent, UNIFIED_OUTPUTS_DIR)
        self.assertTrue(out.name.startswith('pipeline_test_video_'))
        self.assertTrue(out.exists())
        # Clean up created directory
        out.rmdir()

    def test_increment_path_on_existing_nonempty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'exp'
            base.mkdir()
            (base / 'log.txt').write_text('content')
            
            inc1 = increment_path(base)
            self.assertEqual(inc1.name, 'exp_1')
            inc1.mkdir()
            (inc1 / 'log.txt').write_text('content')
            
            inc2 = increment_path(base)
            self.assertEqual(inc2.name, 'exp_2')

    def test_bare_folder_name_redirected_to_unified_root(self):
        out = generate_unique_output_dir('custom_test_exp', allow_existing=True)
        self.assertEqual(out.parent, UNIFIED_OUTPUTS_DIR)
        self.assertEqual(out.name, 'custom_test_exp')
        if out.exists():
            out.rmdir()

    def test_runs_path_redirected_to_unified_root(self):
        out = generate_unique_output_dir('runs/my_test_run', allow_existing=True)
        self.assertEqual(out.parent, UNIFIED_OUTPUTS_DIR)
        self.assertEqual(out.name, 'my_test_run')
        if out.exists():
            out.rmdir()


if __name__ == '__main__':
    unittest.main()

