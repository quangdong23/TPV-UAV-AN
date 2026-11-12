"""Structural contracts for the consolidated system."""
import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import os

from uav_pipeline.paths import PROJECT_ROOT, PACKAGE_ROOT, CONFIG_DIR, MODELS_DIR


def test_one_package_and_external_assets():
    from uav_pipeline.segmentation import PipelineConfig
    assert Path(PipelineConfig().cache_dir).is_absolute()
    assert not (PROJECT_ROOT/'lan_chiem').exists()
    assert not (PROJECT_ROOT/'road_sidewalk_seg').exists()
    assert (CONFIG_DIR/'detection.yaml').is_file()
    assert (MODELS_DIR/'yoloe-v8s-single.pt').is_file()
    for path in PACKAGE_ROOT.rglob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            names = [name.name for name in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
            assert not any(name.split('.')[0] in ('lan_chiem','road_sidewalk_seg') for name in names), path


def test_default_configuration_works_outside_project_cwd():
    env = {**os.environ, 'PYTHONPATH': str(PROJECT_ROOT)}
    source = 'from uav_pipeline.detection.detector import YOLOEDetector; from uav_pipeline.segmentation import PipelineConfig; from uav_pipeline.paths import CONFIG_DIR; d=YOLOEDetector(device="cpu"); assert len(d.vocabulary)==63; c=PipelineConfig.from_json(CONFIG_DIR/"segmentation/gpu_t4.json"); assert c.models'
    with tempfile.TemporaryDirectory() as folder:
        result = subprocess.run([sys.executable, '-c', source], cwd=folder, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
