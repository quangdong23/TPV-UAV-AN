"""Catalogue migration and prompt configuration checks."""
from pathlib import Path
import tempfile
import yaml
from lan_chiem.object_classes import ADVANCED_OBJECT_CLASSES, CATALOG_VERSION, GROUNDING_DINO_PROMPT
from lan_chiem.registry import ClassRegistry
from lan_chiem.detector import GroundingDINODetector


def config():
    return yaml.safe_load((Path(__file__).parents[1] / 'config.yaml').read_text(encoding='utf-8'))


def test_registry_ten_classes():
    assert ClassRegistry.all_ids() == list(range(1, 11))
    assert ClassRegistry.get(0) is None
    assert ClassRegistry.get(3).display_name == 'Xe ô tô'
    assert ClassRegistry.get(4).display_name == 'Bàn nhựa'
    assert ClassRegistry.get(5).display_name == 'Ghế nhựa'
    assert ClassRegistry.get(7).display_name == 'Biển quảng cáo'
    assert ClassRegistry.get(8).display_name == 'Cột điện'
    assert ClassRegistry.get(9).display_name == 'Thân cây'
    assert ClassRegistry.get(10).display_name == 'Hàng hóa'
    assert not ClassRegistry.action_classes()
    assert not ClassRegistry.region_classes()


def test_config_prompt_groups():
    data = config()
    assert data['catalog_version'] == CATALOG_VERSION
    for item in ADVANCED_OBJECT_CLASSES:
        entry = data['classes'][item['id']]
        assert entry['prompt'] == item['prompt']
        assert entry['display_name'] == item['name_vi']
    assert sorted({cid for g in data['prompt_groups'].values() if g.get('enabled', True) for cid in g['class_ids']}) == list(range(1, 11))
    assert 'traffic sign' not in data['classes'][7]['prompt']
    assert ' . .' not in GROUNDING_DINO_PROMPT
    assert GROUNDING_DINO_PROMPT.endswith(' .')


def test_config_inference_presets():
    inf = config()['inference']
    assert inf['tile_size'] == 768
    assert inf['overlap_ratio'] == .25
    assert inf['nms_iou'] == .5
    assert inf['mode'] == 'full'


def test_legacy_config_is_rejected_before_model_load():
    data = config()
    data.pop('catalog_version')
    with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1] / 'outputs') as folder:
        path = Path(folder) / 'legacy.yaml'
        path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding='utf-8')
        try:
            GroundingDINODetector(config_path=path)
        except ValueError as exc:
            assert 'catalog_version' in str(exc)
        else:
            raise AssertionError('Legacy mapping must not be silently reused')
