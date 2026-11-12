"""Project assets and cache locations, shared by all pipeline components."""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get('UAV_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()
PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / 'configs'
MODELS_DIR = PROJECT_ROOT / 'models'
SAMPLES_DIR = PROJECT_ROOT / 'data' / 'samples'
CACHE_DIR = PROJECT_ROOT / '.cache'
OUTPUTS_DIR = PROJECT_ROOT / 'outputs'


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT/path


def source_signature():
    """Invalidate resumes whenever any executable pipeline module changes."""
    import hashlib
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_ROOT.rglob('*.py')):
        digest.update(path.relative_to(PACKAGE_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
