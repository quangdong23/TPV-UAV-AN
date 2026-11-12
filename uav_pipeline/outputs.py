"""Quản lý thư mục đầu ra tập trung và tự sinh tên không trùng lặp cho hệ thống UAV."""
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

# Thư mục gốc đầu ra hợp nhất duy nhất của hệ thống UAV: UAV/outputs
from uav_pipeline.paths import PROJECT_ROOT as UAV_ROOT
UNIFIED_OUTPUTS_DIR = UAV_ROOT / 'outputs'


def get_unified_outputs_root() -> Path:
    """Trả về đường dẫn thư mục outputs hợp nhất duy nhất."""
    UNIFIED_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return UNIFIED_OUTPUTS_DIR


def increment_path(path: Union[str, Path], exist_ok: bool = False, sep: str = '_') -> Path:
    """
    Tự động tăng chỉ số đuôi nếu thư mục đã tồn tại và không rỗng.
    Ví dụ: outputs/pipeline_2 -> outputs/pipeline_2_1 -> outputs/pipeline_2_2
    """
    path = Path(path)
    if not path.exists():
        return path
    if path.is_dir() and not any(path.iterdir()):
        return path
    if exist_ok:
        return path

    parent = path.parent
    stem = path.name

    # Chỉ coi là chỉ số đếm nếu số có tối đa 4 chữ số (để phân biệt với chuỗi ngày giờ YYYYMMDD / HHMMSS)
    match = re.search(rf'{re.escape(sep)}(\d{{1,4}})$', stem)
    if match:
        base = stem[:match.start()]
        idx = int(match.group(1)) + 1
    else:
        base = stem
        idx = 1

    while True:
        candidate = parent / f"{base}{sep}{idx}"
        if not candidate.exists() or (candidate.is_dir() and not any(candidate.iterdir())):
            return candidate
        idx += 1


def generate_unique_output_dir(
    output: Optional[Union[str, Path]] = None,
    task_name: str = 'run',
    source_name: Optional[str] = None,
    allow_existing: bool = False,
    use_timestamp: bool = True,
) -> Path:
    """
    Tạo hoặc phân giải đường dẫn đầu ra duy nhất dưới thư mục UAV/outputs:
    
    1. Nếu `output` là None (không truyền vào CLI):
       Tự động sinh tên thư mục mới dưới UAV/outputs/ với tiền tố task, tên nguồn và timestamp.
       Ví dụ: UAV/outputs/pipeline_2_20261104_144500
    2. Nếu `output` được chỉ định:
       - Đảm bảo trỏ đúng vị trí (nếu là đường dẫn tương đối đơn, đưa về UAV/outputs/).
       - Nếu thư mục đã tồn tại và không rỗng (và không bật resume/allow_existing),
         tự động tăng đuôi (_1, _2, ...) để không bao giờ ghi đè hoặc xung đột.
    """
    root = get_unified_outputs_root()

    if output is None or str(output).strip() == '':
        now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        parts = [task_name]
        if source_name:
            clean_source = Path(source_name).stem
            parts.append(clean_source)
        if use_timestamp:
            parts.append(now_str)
        dir_name = '_'.join(parts)
        candidate = root / dir_name
        target = increment_path(candidate, exist_ok=False)
        target.mkdir(parents=True, exist_ok=True)
        return target

    target = Path(output)
    # Tự động chuẩn hóa và gom mọi đường dẫn (kể cả runs/...) về duy nhất thư mục UAV/outputs
    if not target.is_absolute():
        if target.parts and target.parts[0] in ('runs', 'outputs'):
            target = root / Path(*target.parts[1:]) if len(target.parts) > 1 else root
        elif len(target.parts) == 1:
            target = root / target
        else:
            target = target.resolve()
    else:
        try:
            rel = target.relative_to(UAV_ROOT / 'runs')
            target = root / rel
        except ValueError:
            pass
        target = target.resolve()

    if not allow_existing and target.exists() and any(target.iterdir()):
        new_target = increment_path(target, exist_ok=False)
        if new_target != target:
            print(f"[*] Thư mục đầu ra '{target}' đã tồn tại và không rỗng. Tự động chuyển sang: '{new_target}'", flush=True)
            target = new_target

    target.mkdir(parents=True, exist_ok=True)
    return target
