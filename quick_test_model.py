"""Short pipeline smoke test with local video; never uploads or publishes."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Test nhanh AI pipeline tren video, khong upload API.')
    parser.add_argument('--input', type=Path, default=Path('data/samples/8.mp4'))
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--weights', help='Checkpoint YOLOE; mac dinh theo config he thong')
    args = parser.parse_args()
    if args.frames < 1:
        parser.error('--frames phai lon hon 0')
    source = args.input if args.input.is_absolute() else root / args.input
    if not source.is_file():
        parser.error(f'Khong tim thay video: {source}')
    output = root / 'outputs' / ('quick_test_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    command = [sys.executable, '-m', 'uav_pipeline', 'run',
               '--input', str(source), '--device', args.device,
               '--max-frames', str(args.frames), '--events', '--no-upload-events',
               '--output', str(output)]
    if args.weights:
        command += ['--weights', args.weights]
    print(f'[quick-test] frames={args.frames} device={args.device}', flush=True)
    print('[quick-test] Model va pipeline day du; FPS khong phai benchmark rieng YOLO.', flush=True)
    print(f'[quick-test] Output: {output}', flush=True)
    result = subprocess.run(command, cwd=root)
    print('[quick-test] ' + ('PASS' if result.returncode == 0 else 'FAIL'), flush=True)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
