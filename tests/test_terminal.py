"""Prevent video subprocesses from changing or consuming the shell terminal."""
import io
import os
from pathlib import Path
import select
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from uav_pipeline.cli.terminal import preserve_terminal
from uav_pipeline.video.io import consolidate, NVENCWriter
from uav_pipeline.events import ViolationMonitor


class TerminalTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'posix', 'POSIX terminal test')
    def test_terminal_restores_on_success_error_and_ctrl_c_without_flushing_input(self):
        import pty
        import termios
        for error in (None, RuntimeError, KeyboardInterrupt):
            master, slave = pty.openpty()
            try:
                original = termios.tcgetattr(slave)
                with os.fdopen(os.dup(slave), 'r') as stream, patch('sys.stdin', stream):
                    try:
                        with preserve_terminal():
                            modified = termios.tcgetattr(slave)
                            modified[3] &= ~(termios.ECHO | termios.ICANON)
                            termios.tcsetattr(slave, termios.TCSANOW, modified)
                            os.write(master, b'next command\n')
                            if error:
                                raise error()
                    except (RuntimeError, KeyboardInterrupt):
                        pass
                self.assertEqual(termios.tcgetattr(slave), original)
                self.assertTrue(select.select([slave], [], [], 1)[0])
                self.assertEqual(os.read(slave, 100), b'next command\n')
            finally:
                os.close(master)
                os.close(slave)

    def test_redirected_stdin_does_not_require_a_terminal(self):
        with patch('sys.stdin', io.StringIO('unchanged')):
            with preserve_terminal():
                pass

    def test_clip_ffmpeg_cannot_read_terminal(self):
        capture = Mock()
        capture.get.return_value = 2
        with patch('uav_pipeline.events.shutil.which', return_value='ffmpeg'), \
             patch('uav_pipeline.events.subprocess.run', return_value=SimpleNamespace(returncode=0)) as run, \
             patch('uav_pipeline.events.cv2.VideoCapture', return_value=capture):
            ViolationMonitor._write_clip('source.mp4', Path('/tmp/terminal_test_clip.mp4'), 0, 2, 10.)
        self.assertIn('-nostdin', run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs['stdin'], subprocess.DEVNULL)

    def test_concat_ffmpeg_cannot_read_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'segments').mkdir()
            segments = []
            for i in range(2):
                name = f'segments/part_{i}.jsonl'
                (root/name).write_text('{"tracks": []}\n')
                segments.append({'records': name, 'video': f'segments/part_{i}.mp4'})

            def run(command, **kwargs):
                self.assertIn('-nostdin', command)
                self.assertEqual(kwargs['stdin'], subprocess.DEVNULL)
                Path(command[-1]).write_bytes(b'video')
                return SimpleNamespace(returncode=0)

            with patch('uav_pipeline.video.io.shutil.which', return_value='ffmpeg'), \
                 patch('uav_pipeline.video.io.subprocess.run', side_effect=run):
                self.assertEqual(consolidate(root, segments, export_csv=False), 'annotated.mp4')

    def test_nvenc_retains_raw_video_pipe_but_disables_keyboard_commands(self):
        process = Mock()
        process.stdin = io.BytesIO()
        process.wait.return_value = 0
        with tempfile.TemporaryDirectory() as directory, \
             patch('uav_pipeline.video.io.shutil.which', return_value='ffmpeg'), \
             patch('uav_pipeline.video.io.subprocess.Popen', return_value=process) as spawn:
            writer = NVENCWriter(Path(directory)/'video.mp4', 10., (4, 2))
            self.assertIn('-nostdin', spawn.call_args.args[0])
            self.assertEqual(spawn.call_args.kwargs['stdin'], subprocess.PIPE)
            writer.release()
