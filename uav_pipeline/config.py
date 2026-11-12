"""Local YAML inheritance with deep mapping merges and cycle detection."""
from copy import deepcopy
from pathlib import Path
import yaml


def load_config(path, seen=()):
    path = Path(path).resolve()
    if path in seen or len(seen) >= 8:
        raise ValueError(f'Configuration inheritance cycle/depth limit: {path}')
    data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Configuration must be a mapping: {path}')
    parent = data.pop('extends', None)
    if parent is None:
        return data
    base = load_config(path.parent / parent, (*seen, path))
    return merge_config(base, data)


def merge_config(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = (merge_config(result[key], value)
                       if isinstance(value, dict) and isinstance(result.get(key), dict)
                       else deepcopy(value))
    return result
