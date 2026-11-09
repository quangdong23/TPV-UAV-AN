"""Offline production-pipeline tests for active YOLOE and UAV video pipeline."""
import argparse
import importlib
import inspect
import sys
import unittest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', action='store_true', help='Include video integration tests')
    args = parser.parse_args()
    suite = unittest.TestSuite()
    modules = [
        'test_refactor',
        'test_yoloe',
        'test_single_model',
        'test_sidewalk',
        'test_tracking_trunks',
        'test_roi_fragments',
        'test_review3',
        'test_output_utils',
    ]
    if args.video:
        modules.extend([
            'test_combined_video',
            'test_video',
            'test_video_outputs',
            'test_motion_inputs',
            'test_events',
            'test_segmentation_parity',
        ])
    for name in modules:
        module = importlib.import_module('lan_chiem.tests.' + name)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        for fn_name, fn in inspect.getmembers(module, inspect.isfunction):
            if fn_name.startswith('test_'):
                suite.addTest(unittest.FunctionTestCase(fn))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    raise SystemExit(main())
