import numpy as np
from uav_pipeline.tracking.objects import ObjectTracker


def motion(local=None):
    return {'scene_cut':False,'reliable':True,'matrix':np.array([[1.,0,0],[0,1.,0]]),'local_boxes':local or {}}


def detection(box,cid=2,source='full'):
    return {'class_id':cid,'bbox_xyxy':box,'score':.8,'source':source}


def test_detection_anchors_exact_position_even_after_covariance_collapses():
    tracker=ObjectTracker(min_hits=1)
    box=np.array([100.,100,150,180])
    tracker.update(0,[detection(box)],motion())
    for i in range(1,101):
        tracker.update(i/30,[detection(box)],motion({1:box}))
    moved=box+[15,35,15,35]
    out=tracker.update(101/30,[detection(moved)],motion())
    assert out[0]['track_id']==1
    np.testing.assert_allclose(out[0]['bbox_xyxy'],moved,atol=1e-8)


def test_flow_and_camera_motion_do_not_accumulate_lag_on_fast_motorcycle():
    tracker=ObjectTracker(min_hits=1)
    base=np.array([100.,100,160,220])
    tracker.update(0,[detection(base)],motion())
    for index in range(1,40):
        box=base+[index*2,index*12,index*2,index*12]
        out=tracker.update(index/30,[detection(box)] if index%6==0 else None,motion({1:box}))
        assert len(out)==1 and out[0]['track_id']==1
        np.testing.assert_allclose(out[0]['bbox_xyxy'],box,atol=1e-8)


def test_sparse_trunk_observations_confirm_without_global_negative_resets():
    tracker=ObjectTracker(min_hits=2,max_static_prediction_seconds=1.2,max_static_lost_seconds=2.5)
    d=detection([10,20,30,100],9,'trunk_refiner')
    tracker.update(0,[d],motion(),observed_classes=[9])
    for timestamp in (.2,.4,.6,.8):
        tracker.update(timestamp,[],motion(),observed_classes=[1,2,3,4,5,6,7,8,10])
        assert tracker.tracks[0]['hits']==1
    out=tracker.update(1.0,[d],motion(),observed_classes=[9])
    assert out[0]['track_id']==1
    assert not tracker.update(2.3,None,motion())
    assert not tracker.update(3.6,[],motion(),observed_classes=[9])


def test_sparse_refiner_does_not_remove_existing_yolo_trunk_positive():
    import torch
    from unittest.mock import Mock
    from types import SimpleNamespace
    from PIL import Image
    from uav_pipeline.detection.detector import YOLOEDetector
    detector=YOLOEDetector(device='cpu',roi_focus=False)
    index=next(i for i,(_,cid,_) in enumerate(detector.vocabulary) if cid==9)
    detector.engine=Mock()
    detector.engine.predict.return_value=[SimpleNamespace(boxes=SimpleNamespace(data=torch.tensor([[10.,10,20,60,.8,index]])))]
    detector.trunk_refiner=SimpleNamespace(provenance=lambda:{},last_output=None)
    result=detector.predict_video_frame(Image.new('RGB',(64,64)),refine_trunks=False)
    assert result['object_detections'][0]['class_id']==9
