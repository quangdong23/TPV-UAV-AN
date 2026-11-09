"""Output policy checks: compact exports preserve analytical results and failures."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from lan_chiem.tests.test_combined_video import make_video, FakeSidewalk
from lan_chiem.tests.test_video import FakeDetector
from lan_chiem.video_predict import process_video, VideoOptions
from lan_chiem.video_io import remove_committed_intermediates
from road_sidewalk_seg import PipelineConfig, SidewalkPipeline
from road_sidewalk_seg.media.video import VideoPreviewOptions
from uav_pipeline.sidewalk import SidewalkSession


def records(folder):
    return [json.loads(line) for line in (folder/'tracks.jsonl').read_text().splitlines()]


class VideoOutputTests(unittest.TestCase):
    def test_compact_and_full_keep_same_analysis_with_only_three_final_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source.avi'; make_video(source)
            options = VideoOptions(min_hits=1, segment_frames=2, preview_interval=.01)
            with redirect_stdout(io.StringIO()):
                for mode in ('full', 'compact'):
                    session = SidewalkSession(FakeSidewalk(), VideoPreviewOptions(keyframe_interval=1), sample_step=0)
                    process_video(source, root/mode, FakeDetector(), options, sidewalk=session, output_mode=mode)
            self.assertEqual({p.name for p in (root/'compact').iterdir()},
                             {'annotated.mp4', 'tracks.jsonl', 'summary.json'})
            full, compact = records(root/'full'), records(root/'compact')
            for rows in (full, compact):
                for row in rows:
                    for key in ('frame_seconds', 'processing_fps', 'detector_seconds'):
                        row.pop(key)
            self.assertEqual(full, compact)
            summary = json.loads((root/'compact/summary.json').read_text())
            self.assertFalse(summary['intermediates_retained'])
            self.assertFalse(summary['resume_supported'])
            self.assertTrue((root/'full/tracks.csv').is_file())
            self.assertTrue((root/'full/segments').is_dir())

    def test_single_video_compact_finishes_without_ffmpeg(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source.avi'; make_video(source)
            with redirect_stdout(io.StringIO()), patch('lan_chiem.video_io.shutil.which', return_value=None):
                process_video(source, root/'out', FakeDetector(), VideoOptions(segment_frames=1), output_mode='compact')
            self.assertEqual({p.name for p in (root/'out').iterdir()},
                             {'annotated.mp4', 'tracks.jsonl', 'summary.json'})
            self.assertEqual(len(records(root/'out')), 4)

    def test_partial_run_keeps_checkpoint_and_resume_keeps_every_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source.avi'; make_video(source)
            options = VideoOptions(min_hits=1, segment_frames=1)
            with redirect_stdout(io.StringIO()):
                process_video(source, root/'out', FakeDetector(), options, max_frames=2, output_mode='compact')
                self.assertTrue((root/'out/checkpoint.json').is_file())
                # Without FFmpeg, two existing segments must remain available.
                with patch('lan_chiem.video_io.shutil.which', return_value=None):
                    process_video(source, root/'out', FakeDetector(), options, resume=True, output_mode='compact')
            self.assertEqual([r['frame_index'] for r in records(root/'out')], list(range(4)))
            summary = json.loads((root/'out/summary.json').read_text())
            self.assertTrue(summary['intermediates_retained'])
            self.assertTrue((root/'out/segments/part_000001.mp4').is_file())

    def test_failed_render_does_not_delete_in_progress_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source.avi'; make_video(source)
            with redirect_stdout(io.StringIO()), patch('lan_chiem.video_predict.render', side_effect=RuntimeError('render failed')):
                with self.assertRaisesRegex(RuntimeError, 'render failed'):
                    process_video(source, root/'out', FakeDetector(), output_mode='compact')
            self.assertTrue((root/'out/segments/part_000000.mp4').is_file())
            self.assertFalse((root/'out/summary.json').exists())

    def test_combined_explicit_samples_are_still_available(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source.avi'; make_video(source, 1)
            session = SidewalkSession(FakeSidewalk(), VideoPreviewOptions(keyframe_interval=1), sample_step=1)
            with redirect_stdout(io.StringIO()):
                process_video(source, root/'out', FakeDetector(), sidewalk=session, output_mode='compact')
            self.assertTrue((root/'out/images/00000000.jpg').is_file())
            self.assertTrue((root/'out/sidewalk_masks/00000000.png').is_file())
            self.assertEqual(records(root/'out')[0]['sidewalk']['mask_path'], 'sidewalk_masks/00000000.png')

    def test_cleanup_requires_final_outputs_and_does_not_remove_unknown_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'segments').mkdir()
            unknown = root/'segments/user_notes.txt'; unknown.write_text('keep')
            with self.assertRaisesRegex(RuntimeError, 'intermediates preserved'):
                remove_committed_intermediates(root, [])
            for name in ('annotated.mp4', 'tracks.jsonl', 'summary.json'):
                (root/name).write_text('complete')
            remove_committed_intermediates(root, [])
            self.assertEqual(unknown.read_text(), 'keep')

    def test_compact_image_rasters_preserve_final_mask_and_overlay_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); pipeline = SidewalkPipeline(PipelineConfig())
            mask = np.zeros((20, 30), np.uint8); mask[:, :10] = 2
            result = SimpleNamespace(mask=mask, confidence=np.ones_like(mask, dtype=float))
            completion = SimpleNamespace(mask=mask)
            overlay = np.full((20, 30, 3), 123, np.uint8)
            for mode in ('full', 'compact'):
                (root/mode).mkdir()
                pipeline._save_raster_outputs(result, completion, overlay, root/mode, diagnostics=mode == 'full')
            self.assertEqual({p.name for p in (root/'compact').iterdir()}, {'mask.png', 'overlay.png'})
            for name in ('mask.png', 'overlay.png'):
                self.assertEqual((root/'compact'/name).read_bytes(), (root/'full'/name).read_bytes())
