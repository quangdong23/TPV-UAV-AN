from uav_pipeline.detection.postprocess import Detection,reconcile_roi_fragments


def det(name,box,source,score=.8,cid=1):
    return Detection(name,cid,'person','person','person','yoloe_vocabulary',source,score,box)


def test_real_image_five_partial_person_keeps_complete_lower_score_box():
    a=det('part',[891.5,502.5,949.5,617.5],'roi_388_0',.9072)
    b=det('whole',[891.75,502.25,970.25,680.0],'roi_776_364',.6929)
    kept=reconcile_roi_fragments([a,b],{'roi_388_0':(388,0,1028,640),'roi_776_364':(776,364,1416,1004)},(1804,1004))
    assert kept==[b] and b.score==.6929
    assert b.evidence['roi_fragments'][0]['id']=='part'


def test_nested_boxes_without_crop_boundary_evidence_are_preserved():
    a=det('a',[20,20,40,60],'roi_0_0')
    b=det('b',[19,19,41,85],'full')
    assert len(reconcile_roi_fragments([a,b],{'roi_0_0':(0,0,100,100)},(200,200)))==2


def test_rider_motorcycle_and_aggregate_crowd_are_not_merged():
    a=det('a',[20,65,40,98],'roi')
    b=det('b',[19,60,41,140],'full',cid=2)
    regions={'roi':(0,0,100,100)}
    assert len(reconcile_roi_fragments([a,b],regions,(200,200)))==2
    b.class_id=1;b.bbox_xyxy=[10,60,90,140]
    c=det('c',[60,65,80,98],'roi')
    assert len(reconcile_roi_fragments([a,b,c],regions,(200,200)))==3


def test_image_border_is_not_an_artificial_crop_boundary():
    a=det('a',[10,5,30,50],'roi')
    b=det('b',[10,0,30,80],'full')
    assert len(reconcile_roi_fragments([a,b],{'roi':(0,0,100,100)},(200,200)))==2


def test_decoder_receives_regions_and_preserves_original_confidence():
    import numpy as np
    from uav_pipeline.detection.detector import YOLOEDetector
    d=YOLOEDetector(device='cpu')
    rows=np.array([[891.5,502.5,949.5,617.5,.9072,0],
                   [891.75,502.25,970.25,680.,.6929,0]])
    pred,review,_=d.decode(rows,img_w=1804,img_h=1004,
        sources=['roi_388_0','roi_776_364'],roi_regions={'roi_388_0':(388,0,1028,640)})
    assert len(pred)==1 and not review and pred[0].score==.6929


def test_global_clone_of_corroborated_fragment_does_not_survive_as_duplicate():
    a=det('part',[891.5,502.5,949.5,617.5],'roi_388_0',.9072)
    b=det('whole',[891.75,502.25,970.25,680.0],'roi_776_364',.6929)
    c=det('global',[890.72,501.74,949.92,617.31],'full',.8604)
    kept=reconcile_roi_fragments([c,a,b],{'roi_388_0':(388,0,1028,640)},(1804,1004))
    assert kept==[b] and len(b.evidence['roi_fragments'])==2
