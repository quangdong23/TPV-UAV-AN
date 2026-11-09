from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
import torch
from PIL import Image
from lan_chiem.demo_hybrid import DemoHybridDetector,LocalTrunkRefiner
from lan_chiem.postprocess import Detection


def test_sparse_demo_never_reuses_old_boxes_and_preserves_primary_trunks():
    base={'object_detections':[{'id':'primary9','class_id':9,'bbox_xyxy':[10,10,30,100]}],
          'review_candidates':[],'observed_class_ids':list(range(1,11)),
          'model_calls':1,'images_evaluated':1,'inference_time_seconds':.01,
          'model_stage_seconds':{},'conflict_reports':[]}
    trunk=Detection('fresh9',9,'trunk','trunk','tree trunk','trunk_refiner','trunk_refiner',.7,[11,10,31,100])
    helper=SimpleNamespace(predict=Mock(return_value=[trunk]),provenance=lambda:{'model_revision':'test'},last_output={})
    with patch('lan_chiem.demo_hybrid.LocalTrunkRefiner',return_value=helper):d=DemoHybridDetector(device='cpu')
    with patch('lan_chiem.yoloe_detector.YOLOEDetector.predict',side_effect=lambda *a,**k:deepcopy(base)):
        a=d.predict(Image.new('RGB',(200,200)),_video_frame=True)
        assert not a['trunk_refiner_ran'] and 9 not in a['observed_class_ids']
        assert a['object_detections'][0]['id']=='primary9' and helper.predict.call_count==0
        b=d.predict(Image.new('RGB',(200,200)),refine_trunks=True)
        assert b['object_detections'][0]['id']=='fresh9' and b['model_calls']==2
        assert b['detector_count']==2 and 9 in b['observed_class_ids']
        c=d.predict(Image.new('RGB',(200,200)),refine_trunks=False)
        assert c['object_detections'][0]['id']=='primary9' and helper.predict.call_count==1


def test_dino_grounded_scope_and_counters_are_preserved():
    class Inputs(dict):
        input_ids=None
        def to(self,device):return self
    class Processor:
        def __call__(self,**kwargs):return Inputs()
        def post_process_grounded_object_detection(self,*a,**k):
            return [{'boxes':torch.tensor([[10,0,40,100],[20,50,30,100],[float('nan'),0,10,20],[100,0,120,100]]),
                     'scores':torch.tensor([.8,.6,.9,.95]),'text_labels':['tree trunk','tree trunk','tree trunk','flag']}]
    d=LocalTrunkRefiner.__new__(LocalTrunkRefiner)
    d.device='cpu';d.precision='fp32';d.model=lambda **kwargs:None;d.processor=Processor()
    out=d.predict(Image.new('RGB',(200,200)))
    assert len(out)==1 and out[0].class_id==9 and out[0].evidence['alternative_trunk_scopes']


def test_ungrounded_assistance_is_skipped_without_guessing_class_or_crashing():
    class Inputs(dict):
        input_ids=None
        def to(self,device):return self
    class Processor:
        def __call__(self,**kwargs):return Inputs()
        def post_process_grounded_object_detection(self,*a,**k):
            return [{'boxes':torch.tensor([[10,10,30,100]]),'scores':torch.tensor([.7]),'text_labels':[None]}]
    d=LocalTrunkRefiner.__new__(LocalTrunkRefiner)
    d.device='cpu';d.precision='fp32';d.model=lambda **kwargs:None;d.processor=Processor()
    assert d.predict(Image.new('RGB',(200,200)))==[]
    assert d.last_output['invalid_semantic_output']


def test_image_path_uses_primary_preprocessed_rgb_image_for_assistant():
    image=Image.new('RGB',(200,200))
    base={'_image_pil':image,'object_detections':[],'review_candidates':[],
          'observed_class_ids':list(range(1,11)),'model_calls':1,'images_evaluated':1,
          'inference_time_seconds':.01,'model_stage_seconds':{},'conflict_reports':[]}
    helper=SimpleNamespace(predict=Mock(return_value=[]),provenance=lambda:{},last_output={})
    with patch('lan_chiem.demo_hybrid.LocalTrunkRefiner',return_value=helper):d=DemoHybridDetector(device='cpu')
    with patch('lan_chiem.yoloe_detector.YOLOEDetector.predict',return_value=base):
        d.predict('example.png',refine_trunks=True)
    assert helper.predict.call_args.args[0] is image
