"""Offline production-pipeline tests for active YOLOE and UAV video pipeline."""
import argparse
import importlib
import inspect
import sys
import unittest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', action='store_true', help='Include video integration tests')
    parser.add_argument('--evaluation', action='store_true', help='Include optional COCO/TrackEval checks')
    args = parser.parse_args()
    from uav_pipeline.paths import OUTPUTS_DIR
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    suite = unittest.TestSuite()
    modules = [
        'test_layout',
        'test_refactor',
        'test_yoloe',
        'test_single_model',
        'test_sidewalk',
        'test_tracking_trunks',
        'test_roi_fragments',
        'test_review3',
        'test_output_utils',
        'test_upgrade',
        'test_stage2',
        'test_recognition_context',
        'test_terminal',
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
    if args.evaluation:
        modules.extend(['test_coco', 'test_tools'])
    for name in modules:
        module = importlib.import_module('tests.' + name)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        for fn_name, fn in inspect.getmembers(module, inspect.isfunction):
            if fn_name.startswith('test_'):
                suite.addTest(unittest.FunctionTestCase(fn))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    raise SystemExit(main())
