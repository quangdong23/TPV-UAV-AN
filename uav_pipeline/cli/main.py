"""One command for the unified UAV system."""
import argparse
import sys


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    commands = {
        'run': 'Process video with segmentation, detection, tracking and events',
        'detect': 'Detect objects in images',
        'segment': 'Inspect sidewalk segmentation in images or videos',
        'benchmark': 'Run and compare the production video benchmark',
        'export-model': 'Export a fixed-vocabulary YOLOE checkpoint',
    }
    # Preserve the original python -m uav_pipeline --input ... invocation.
    if args and args[0].startswith('-') and args[0] not in ('-h', '--help'):
        command, remaining = 'run', args
    elif args and args[0] in commands:
        command, remaining = args[0], args[1:]
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('command', nargs='?', choices=list(commands))
        parser.epilog = '\n'.join(f'{name}: {description}' for name, description in commands.items())
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.parse_args(args)
        parser.print_help()
        return 0
    if command == 'run':
        from .run import main as entry
        return entry(remaining)
    if command == 'segment':
        from uav_pipeline.segmentation.app.cli import main as entry
        return entry(remaining)
    if command == 'benchmark':
        from .benchmark import run_pipeline
        return run_pipeline(remaining)
    from importlib import import_module
    module = import_module('uav_pipeline.cli.' + ('detect' if command == 'detect' else 'export_model'))
    original = sys.argv
    try:
        sys.argv = [f'uav {command}', *remaining]
        return module.main()
    finally:
        sys.argv = original
