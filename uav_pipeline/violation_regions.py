"""Gom vùng vi phạm; tư thế bbox chỉ là heuristic, cần người kiểm duyệt."""
import cv2
import numpy as np

SURFACE = 'lấn chiếm vỉa hè/lòng lề đường'
NAMES = {
    'furniture': f'bày bán bàn ghế {SURFACE}',
    'dining': f'kinh doanh dịch vụ ăn uống {SURFACE}',
    'goods': f'bày bán hàng hoá {SURFACE}',
    'materials': f'tập kết vật liệu {SURFACE}',
    'machinery': f'đặt máy móc, sửa chữa thiết bị {SURFACE}',
    'sign': 'đặt biển quảng cáo sai quy định',
    'cart': f'bày bán hàng rong {SURFACE}',
}
FAMILIES = {4:'furniture',5:'furniture',6:'cart',7:'sign',10:'goods',11:'materials',12:'machinery'}


def union_box(rows):
    boxes=np.asarray([r['bbox_xyxy'] for r in rows],float)
    return np.r_[boxes[:,:2].min(axis=0),boxes[:,2:].max(axis=0)]


def close(a,b,ratio=.3):
    a=np.asarray(a,float);b=np.asarray(b,float)
    scale=max(1.,min(np.linalg.norm(a[2:]-a[:2]),np.linalg.norm(b[2:]-b[:2])))
    gap=np.maximum(0,np.maximum(a[:2]-b[2:],b[:2]-a[2:]))
    return np.linalg.norm(gap)<=ratio*scale


def violation_mask(mask, road_edge_pixels=20, road_probability=None):
    # Chỉ mở rộng sang mặt đường kề vỉa hè, không coi cả lòng đường là lề đường.
    result=mask.copy()
    if np.any(mask==2):
        distance=cv2.distanceTransform((mask!=2).astype(np.uint8),cv2.DIST_L2,3)
        road=(mask==1) if road_probability is None else (np.asarray(road_probability)>=.55)
        result[road&(distance<=road_edge_pixels)]=2
    return result


def grouped_regions(objects, people):
    remaining=list(objects);regions=[]
    while remaining:
        members=[remaining.pop(0)]
        family=FAMILIES[members[0]['class_id']]
        if family not in ('sign','cart'):
            changed=True
            while changed:
                changed=False
                for row in remaining[:]:
                    if FAMILIES[row['class_id']]==family and any(close(row['bbox_xyxy'],m['bbox_xyxy'],.6 if family=='furniture' else .3) for m in members):
                        members.append(row);remaining.remove(row);changed=True
        related=[]
        if family in ('furniture','goods','machinery'):
            for person in people:
                p=np.asarray(person['bbox_xyxy'],float);height=max(1.,p[3]-p[1])
                nearby=any(close(p,m['bbox_xyxy'],.2) and
                    abs(p[3]-m['bbox_xyxy'][3])<=.6*height for m in members)
                # Ngồi/cúi: bbox thấp, rộng; không suy ra động tác sửa máy từ việc đứng cạnh.
                posture=(p[2]-p[0])/height>=.65 or 'seated' in person.get('raw_label','').lower()
                if nearby and (family=='goods' or posture):related.append(person)
        label='dining' if family=='furniture' and related else family
        regions.append({'family':family,'behavior_name':NAMES[label],
            'objects':members,'members':members+related,
            'bbox_xyxy':union_box(members+related).tolist(),
            'association_rule':'proximity_and_bbox_posture_requires_review'})
    return regions


def crowd_regions(people,min_people=5,gap_ratio=.9):
    """Cụm người đứng gần nhau; không xác minh mục đích hay tính hợp pháp."""
    standing=[]
    for person in {p['track_id']:p for p in people}.values():
        box=np.asarray(person['bbox_xyxy'],float)
        height=box[3]-box[1]
        if height<=0 or (box[2]-box[0])/height>=.65 or 'seated' in person.get('raw_label','').lower():
            continue
        standing.append(person)
    def distance(a,b):
        a=np.asarray(a['bbox_xyxy'],float);b=np.asarray(b['bbox_xyxy'],float)
        feet_a=np.array([(a[0]+a[2])/2,a[3]])
        feet_b=np.array([(b[0]+b[2])/2,b[3]])
        return np.linalg.norm(feet_a-feet_b)/max(1.,min(a[3]-a[1],b[3]-b[1]))
    remaining=sorted(standing,key=lambda p:p['track_id']);regions=[]
    while remaining:
        members=[remaining.pop(0)];changed=True
        while changed:
            changed=False
            for person in remaining[:]:
                distances=[distance(person,other) for other in members]
                # Tránh nối một hàng người dài thành một cụm qua chuỗi hàng xóm.
                if min(distances)<=gap_ratio and max(distances)<=2*gap_ratio:
                    members.append(person);remaining.remove(person);changed=True
        if len(members)>=min_people:
            regions.append({'family':'crowd','kind':'crowd_gathering_suspected',
                'behavior_name':'tụ tập trái phép','objects':members,'members':members,
                'bbox_xyxy':union_box(members).tolist(),
                'association_rule':'standing_bbox_and_proximity_only_requires_review'})
    return regions
