from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
import torch
from PIL import Image
from lan_chiem.embedding_cache import text_embeddings
from lan_chiem.yoloe_detector import YOLOEDetector


def test_embedding_cache_reuses_valid_data_and_invalidates_changed_weights():
    root=Path(__file__).parents[1]/'outputs'
    with tempfile.TemporaryDirectory(dir=root) as directory:
        engine=SimpleNamespace(model=SimpleNamespace(model=[SimpleNamespace(embed=3)]),get_text_pe=Mock(return_value=torch.ones(1,2,3)))
        args=(engine,['a','b'],'cpu','weights_a','vocab',{'torch':'test'})
        a,meta=text_embeddings(*args,directory=directory)
        b,hit=text_embeddings(*args,directory=directory)
        assert hit['status']=='hit' and engine.get_text_pe.call_count==1
        assert torch.equal(a,b)
        text_embeddings(engine,['a','b'],'cpu','weights_b','vocab',{'torch':'test'},directory=directory)
        assert engine.get_text_pe.call_count==2
        Path(meta['path']).write_bytes(b'broken cache')
        text_embeddings(*args,directory=directory)
        assert engine.get_text_pe.call_count==3


def test_fixed_checkpoint_never_loads_text_encoder_or_sets_new_classes():
    d=YOLOEDetector(weights='fake-fixed.pt',device='cpu')
    model=torch.nn.Module()
    head=torch.nn.Module();head.is_fused=True
    model.model=torch.nn.ModuleList([head])
    model.names={i:v[0] for i,v in enumerate(d.vocabulary)}
    model.single_vocab_sha256=d.vocabulary_sha256
    engine=SimpleNamespace(model=model,to=Mock(),get_text_pe=Mock(side_effect=AssertionError('text encoder loaded')),
                           set_classes=Mock(side_effect=AssertionError('fixed classes replaced')),ckpt_path=None,task='detect')
    with patch.dict('sys.modules',{'ultralytics':SimpleNamespace(YOLOE=Mock(return_value=engine))}):d.load_model()
    assert d.provenance()['detector_count']==1
    assert d.provenance()['text_embeddings']['status']=='fixed_checkpoint'
    assert d.provenance()['reparameterized']
    assert d.trunk_refiner is None


def test_legacy_hybrid_profile_is_disabled_for_single_yoloe():
    import warnings
    from copy import deepcopy
    cfg=deepcopy(YOLOEDetector(device='cpu').config)
    cfg['model']['trunk_refiner']={'enabled':True}
    with warnings.catch_warnings(record=True) as seen,patch.object(YOLOEDetector,'_load_config',return_value=cfg):
        d=YOLOEDetector(device='cpu')
    assert seen and not d.config['model']['trunk_refiner']['enabled']


def test_video_roi_budget_keeps_rotating_coverage_and_limits_batch_size():
    from lan_chiem.roi import grid_regions
    d=YOLOEDetector(device='cpu',imgsz=960,roi_max_patches=12)
    d.config['model']['roi_strategy']='grid'
    empty=lambda:SimpleNamespace(boxes=SimpleNamespace(data=torch.empty(0,6)))
    d.engine=Mock()
    d.engine.predict.side_effect=lambda **kwargs:[empty() for _ in kwargs['source']] if isinstance(kwargs['source'],list) else [empty()]
    result=d.predict_video_frame(Image.new('RGB',(1920,1040)),roi_patch_budget=2,roi_offset=2)
    assert result['model_calls']==2 and result['images_evaluated']==3
    assert result['roi_regions']==[list(r) for r in grid_regions(1920,1040,640)[2:4]]
    result=d.predict_video_frame(Image.new('RGB',(1920,1040)),roi_patch_budget=0)
    assert result['model_calls']==1 and not result['roi_regions']


def test_yoloe_cli_does_not_silently_switch_to_dino_for_prompt():
    import sys,io
    from lan_chiem.predict import main
    args=['predict','--backend','yoloe','--input',str(__file__),'--output','lan_chiem/outputs/unused','--prompt','tree trunk']
    with patch.object(sys,'argv',args),patch.object(sys,'stderr',io.StringIO()) as error:
        assert main()==2
        assert 'does not switch to DINO' in error.getvalue()
