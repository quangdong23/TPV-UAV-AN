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


def test_supported_gaps_keep_id_during_camera_pan_then_expire():
    tracker=ObjectTracker(min_hits=2,motion_supported_persistence=True,
                          max_static_prediction_seconds=1.5,max_static_lost_seconds=5)
    box=np.array([100.,100,150,180])
    tracker.update(0,[detection(box,11)],motion())
    assert tracker.update(.1,[detection(box,11)],motion())[0]['track_id']==1
    pan=motion();pan['matrix'][0,2]=10
    for index in (1,2):
        rows=tracker.update(.1+index*.1,[],pan)
        assert rows[0]['track_id']==1
        np.testing.assert_allclose(rows[0]['bbox_xyxy'],box+[10*index,0,10*index,0])
        assert rows[0]['position_source']=='camera_motion'
        assert tracker.tracks[0]['hits']==2
        assert tracker.tracks[0]['last_detection']==.1
    assert tracker.update(.4,[],pan)==[]
    rows=tracker.update(.5,[detection(box+[30,0,30,0],11)],motion())
    assert rows[0]['track_id']==1
    assert tracker.update(6,None,motion())==[]


def test_dynamic_gap_requires_foreground_flow_not_camera_alone():
    for supported in (False,True):
        tracker=ObjectTracker(min_hits=2,motion_supported_persistence=True)
        box=[100,100,150,180]
        tracker.update(0,[detection(box,1)],motion())
        tracker.update(.1,[detection(box,1)],motion())
        rows=tracker.update(.2,[],motion({1:box}) if supported else motion())
        assert bool(rows)==supported
        if rows:assert rows[0]['position_source']=='optical_flow'


def test_supported_gap_never_confirms_tentative_or_outlives_age_limit():
    tracker=ObjectTracker(min_hits=2,motion_supported_persistence=True,max_prediction_seconds=.2)
    box=[100,100,150,180]
    tracker.update(0,[detection(box)],motion())
    assert tracker.update(.1,[],motion({1:box}))==[]
    assert tracker.confirmed_ids==set()
    tracker=ObjectTracker(min_hits=2,motion_supported_persistence=True,max_prediction_seconds=.2)
    tracker.update(0,[detection(box)],motion())
    tracker.update(.1,[detection(box)],motion())
    assert tracker.update(.31,[],motion({1:box}))==[]


def test_active_identity_wins_over_better_overlapping_lost_identity():
    tracker=ObjectTracker(min_hits=2,motion_supported_persistence=True)
    a=[100,100,140,160];b=[110,100,150,160]
    tracker.update(0,[detection(a),detection(b)],motion())
    tracker.update(.1,[detection(a),detection(b)],motion())
    tracker.tracks[0]['missed_detection']=True
    rows=tracker.update(.2,[detection(a)],motion())
    assert [r['track_id'] for r in rows]==[2]
    assert tracker.associations[0]['track_id']==2


def test_supported_tracking_rejects_unreliable_motion_and_scene_cuts():
    tracker=ObjectTracker(min_hits=2,motion_supported_persistence=True)
    box=[100,100,150,180]
    tracker.update(0,[detection(box,12)],motion())
    tracker.update(.1,[detection(box,12)],motion())
    bad=motion();bad['reliable']=False
    assert tracker.update(.2,[],bad)==[]
    cut=motion();cut['scene_cut']=True
    assert tracker.update(.3,[detection(box,12)],cut)==[]
    assert tracker.tracks[0]['id']==2


def test_supported_tracking_resume_preserves_miss_budget():
    import json
    tracker=ObjectTracker(min_hits=2,motion_supported_persistence=True)
    box=[100,100,150,180]
    tracker.update(0,[detection(box,11)],motion())
    tracker.update(.1,[detection(box,11)],motion())
    tracker.update(.2,[],motion())
    restored=ObjectTracker(min_hits=2,motion_supported_persistence=True)
    restored.load_state_dict(json.loads(json.dumps(tracker.state_dict())))
    for ts in (.3,.4,.5):
        assert tracker.update(ts,[],motion())==restored.update(ts,[],motion())


def test_lost_recovery_accepts_serialized_list_boxes_with_same_id():
    tracker=ObjectTracker(min_hits=2,motion_supported_persistence=True)
    box=[100,100,150,180]
    tracker.update(0,[detection(box)],motion())
    tracker.update(.1,[detection(box)],motion())
    assert tracker.update(.2,[],motion())==[]
    rows=tracker.update(.3,[detection(box)],motion())
    assert rows[0]['track_id']==1


def test_display_hold_is_bounded_and_does_not_invent_tracks_or_contact_evidence():
    from uav_pipeline.tracking.objects import SidewalkTrackVisibility
    display=SidewalkTrackVisibility(.75)
    row={'track_id':1,'class_id':5,'position_source':'detection','bbox_xyxy':[10,10,30,30]}
    inside={**row,'is_encroaching':True}
    assert display.update(0,[row],[inside],motion())[0]['sidewalk_visibility']=='current_mask'
    kept=display.update(.5,[row],[],motion())
    assert len(kept)==1 and kept[0]['is_encroaching'] is False
    assert kept[0]['sidewalk_visibility']=='recent_confirmed_track'
    assert kept[0]['sidewalk_last_confirmed_seconds']==0
    assert display.update(.76,[row],[],motion())==[]
    assert display.update(.8,[row],[],motion())==[]
    display.update(1,[row],[inside],motion())
    assert display.update(1.1,[],[],motion())==[]


def test_display_hold_rejects_unreliable_predictions_and_resets_on_cut():
    from uav_pipeline.tracking.objects import SidewalkTrackVisibility
    display=SidewalkTrackVisibility(.75)
    row={'track_id':1,'class_id':1,'position_source':'prediction'}
    display.update(0,[row],[row],motion())
    assert display.update(.1,[row],[],motion())==[]
    measured={**row,'position_source':'optical_flow'}
    assert display.update(.2,[measured],[],motion())
    cut=motion();cut['scene_cut']=True
    assert display.update(.3,[measured],[],cut)==[]
    assert display.update(.4,[measured],[],motion())==[]


def test_display_hysteresis_never_enters_event_evidence():
    import io,json,tempfile
    from contextlib import redirect_stdout
    from pathlib import Path
    from unittest.mock import patch
    from tests.test_combined_video import FakeSidewalk,make_video
    from tests.test_video import FakeDetector
    from uav_pipeline.video.pipeline import VideoOptions,process_video
    from uav_pipeline.segmentation.session import SidewalkSession
    from uav_pipeline.segmentation.media.video import VideoPreviewOptions
    from uav_pipeline.events import ViolationMonitor
    class Detector(FakeDetector):
        def predict(self,image,**kwargs):
            return {'object_detections':[detection([20,20,50,50],4)],'model_calls':1}
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);source=root/'source.avi';make_video(source)
        session=SidewalkSession(FakeSidewalk(('left','left','empty','left')),
                                VideoPreviewOptions(keyframe_interval=1),sample_step=0)
        options=VideoOptions(min_hits=2,keyframe_interval=1,min_detection_interval=1,
                             motion_supported_persistence=True,sidewalk_display_hold_seconds=.75)
        monitor=ViolationMonitor()
        with redirect_stdout(io.StringIO()),patch.object(monitor,'update',wraps=monitor.update) as update:
            process_video(source,root/'out',Detector(),options,sidewalk=session,events=monitor)
        rows=[json.loads(x) for x in (root/'out/tracks.jsonl').read_text().splitlines()]
        assert rows[2]['tracks'][0]['track_id']==rows[1]['tracks'][0]['track_id']
        assert rows[2]['tracks'][0]['sidewalk_visibility']=='recent_confirmed_track'
        assert not rows[2]['tracks'][0]['is_encroaching']
        assert update.call_args_list[2].args[2]==[]
        assert rows[2]['rule_alerts']==[]
