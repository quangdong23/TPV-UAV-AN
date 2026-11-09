"""Relay child stdout/stderr through notebook stdout and stop it on interruption."""
from contextlib import nullcontext
import os
from pathlib import Path
import subprocess


def read_log_tail(path, lines=30):
    """Diagnostics must never replace the original setup/process exception."""
    try:
        return ''.join(Path(path).read_text(encoding='utf-8', errors='replace').splitlines(True)[-lines:])
    except OSError as error:
        return f'Log unavailable ({path}): {error}'


def run_streaming(command, cwd=None, log_path=None, on_line=None, echo=True):
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging = Path(log_path).open('a', encoding='utf-8') if log_path else nullcontext(None)
    with logging as log:
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                   errors='replace', bufsize=1,
                                   env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8',
                                        'USE_TF': '0', 'USE_FLAX': '0'})
        try:
            for line in process.stdout:
                if echo:
                    print(line, end='', flush=True)
                if log is not None:
                    log.write(line)
                    log.flush()
                if on_line is not None:
                    on_line(line.rstrip('\n'))
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdout.close()
