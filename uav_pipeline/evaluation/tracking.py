"""Human-labelled clip evaluation through the official TrackEval metric classes.

Evaluation is offline and optional; never changes the production tracker or model.
"""
import argparse
from contextlib import redirect_stdout
import io
import json
import math
import hashlib
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from uav_pipeline.evaluation.helpers import write_json
from uav_pipeline.evaluation.regression import records
from uav_pipeline.evaluation.helpers import sidewalk_raster


def load_annotations(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema_version') != 1 or data.get('coordinate_space') != 'source_pixels':
        raise ValueError('GT requires schema_version=1 and source_pixels coordinates')
    if data.get('human_verified') is not True:
        raise ValueError('Ground truth must be explicitly human_verified; pseudo-labels are not accuracy evidence')
    size = data.get('source_size', [])
    if len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError('GT source_size must be [positive width, positive height]')
    if not isinstance(data.get('source_sha256'),str) or len(data['source_sha256']) != 64:
        raise ValueError('GT needs the source video SHA256')
    clips = data.get('clips', [])
    if not clips:
        raise ValueError('No annotated clips')
    seen, names = set(), set()
    for clip in clips:
        name = clip.get('name')
        if not isinstance(name,str) or not name or name in names:
            raise ValueError('Clip names must be nonempty and unique')
        names.add(name)
        frames = clip.get('frames', [])
        if not frames:
            raise ValueError('Every clip needs frames, including explicitly empty frames')
        previous = None
        classes = {}
        for frame in frames:
            index = frame['frame_index']
            if type(index) is not int or index < 0 or index in seen or (previous is not None and index != previous+1):
                raise ValueError('GT clips need contiguous, nonoverlapping source frame indices')
            seen.add(index)
            previous = index
            validate_tracks(frame['tracks'], size)
            for track in frame['tracks']:
                tid, cid = track['track_id'], track['class_id']
                if tid in classes and classes[tid] != cid:
                    raise ValueError('A GT ID cannot change class inside a clip')
                classes[tid] = cid
    return data


def validate_tracks(tracks, size):
    seen = set()
    if not isinstance(tracks, list):
        raise ValueError('tracks must be a list')
    for track in tracks:
        tid, cid = track['track_id'], track['class_id']
        if type(tid) is not int or tid < 0 or tid in seen:
            raise ValueError('Track IDs must be unique nonnegative integers in each frame')
        if type(cid) is not int or not 1 <= cid <= 10:
            raise ValueError('Class IDs must be 1..10')
        seen.add(tid)
        box = np.asarray(track['bbox_xyxy'], dtype=float)
        if (box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2])
                or np.any(box[:2] < 0) or np.any(box[2:] > np.asarray(size))):
            raise ValueError('Boxes must be positive, finite xyxy inside the source image')
        if track.get('ignore'):
            raise ValueError('Ignore regions are not supported; annotate fully evaluated clips instead')


def box_similarities(gt, prediction):
    a = np.asarray([row['bbox_xyxy'] for row in gt],float).reshape(-1,4)
    b = np.asarray([row['bbox_xyxy'] for row in prediction],float).reshape(-1,4)
    intersection = np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-
                              np.maximum(a[:,None,:2],b[None,:,:2])).prod(-1)
    union = (a[:,2:]-a[:,:2]).prod(-1)[:,None]+(b[:,2:]-b[:,:2]).prod(-1)[None,:]-intersection
    return intersection/np.maximum(union,1e-12)


def sequence_data(frames, predictions, class_id, max_pairs):
    gt = [[t for t in row['tracks'] if t['class_id']==class_id] for row in frames]
    pred = [[t for t in predictions[row['frame_index']]['tracks'] if t['class_id']==class_id] for row in frames]
    gt_map = {key:i for i,key in enumerate(sorted({t['track_id'] for rows in gt for t in rows}))}
    pred_map = {key:i for i,key in enumerate(sorted({t['track_id'] for rows in pred for t in rows}))}
    if len(gt_map)*len(pred_map) > max_pairs:
        raise ValueError('Too many identity pairs for evaluation RAM budget; split into shorter labelled clips')
    return {'num_timesteps':len(frames),'num_gt_ids':len(gt_map),'num_tracker_ids':len(pred_map),
            'num_gt_dets':sum(map(len,gt)),'num_tracker_dets':sum(map(len,pred)),
            'gt_ids':[np.asarray([gt_map[t['track_id']] for t in rows],dtype=int) for rows in gt],
            'tracker_ids':[np.asarray([pred_map[t['track_id']] for t in rows],dtype=int) for rows in pred],
            'similarity_scores':[box_similarities(a,b) for a,b in zip(gt,pred)]}


def metric_summary(results):
    hota, identity, clear = results['HOTA'],results['Identity'],results['CLEAR']
    return {'HOTA':float(np.mean(hota['HOTA'])),'DetA':float(np.mean(hota['DetA'])),
            'AssA':float(np.mean(hota['AssA'])),'IDF1':float(identity['IDF1']),
            'IDTP':int(identity['IDTP']),'IDFN':int(identity['IDFN']),'IDFP':int(identity['IDFP']),
            'MOTA':float(clear['MOTA']),'ID_switches':int(clear['IDSW']),
            'fragmentations':int(clear['Frag']),'false_positives':int(clear['CLR_FP']),
            'false_negatives':int(clear['CLR_FN']),'true_positives':int(clear['CLR_TP'])}


def evaluate_events(folder, gt, predictions):
    """Custom event protocol: kind, confirmation window and all target boxes.

    IDs need not agree between annotations and tracker. Match targets geometrically
    at confirmation, then assign events one-to-one. Unlabelled times are excluded.
    """
    if 'events' not in gt:
        return {'accuracy_measured':False,'reason':'No explicit human event annotations'}
    truth = gt['events']
    if not isinstance(truth,list):raise ValueError('GT events must be a list, including [] for verified negatives')
    clips = {clip['name']:clip for clip in gt['clips']}
    frames = {frame['frame_index']:frame for clip in gt['clips'] for frame in clip['frames']}
    kinds = {'sidewalk_encroachment','sign_attachment_suspected','motorcycle_repair_suspected'}
    for event in truth:
        clip = clips.get(event['clip'])
        if clip is None or event['kind'] not in kinds:raise ValueError('Invalid labelled event clip/kind')
        lo,hi=clip['frames'][0]['frame_index'],clip['frames'][-1]['frame_index']
        if not lo<=event['onset_frame']<=event['end_frame']<=hi:raise ValueError('Labelled events must stay inside one annotated clip')
        ids=event['track_ids']
        if not ids or len(set(ids))!=len(ids) or any(type(v) is not int for v in ids):raise ValueError('Invalid labelled event targets')
        available={t['track_id'] for t in frames[event['onset_frame']]['tracks']}
        if not set(ids)<=available:raise ValueError('Event targets are missing at onset')
    report=json.loads((Path(folder)/'events.json').read_text(encoding='utf-8'))
    all_predicted=report['events']
    predicted=[event for event in all_predicted if event['confirmation_frame'] in frames]
    costs=np.full((len(truth),len(predicted)),1e9)
    delays={}
    for i,event in enumerate(truth):
        for j,candidate in enumerate(predicted):
            index=candidate['confirmation_frame']
            if candidate['kind']!=event['kind'] or not event['onset_frame']<=index<=event['end_frame']:continue
            targets=[t for t in frames[index]['tracks'] if t['track_id'] in event['track_ids']]
            boxes=candidate.get('evidence',{}).get('confirmation_boxes_source_pixels',[])
            if len(targets)!=len(event['track_ids']) or len(boxes)!=len(targets):continue
            validate_tracks(boxes,gt['source_size'])
            similarity=box_similarities(targets,boxes)
            for a,target in enumerate(targets):
                for b,box in enumerate(boxes):
                    if target['class_id']!=box['class_id']:similarity[a,b]=0
            a,b=linear_sum_assignment(1-similarity)
            if not np.all(similarity[a,b]>=.5):continue
            onset=predictions[event['onset_frame']].get('timestamp_seconds')
            confirmed=predictions[index].get('timestamp_seconds')
            if onset is None or confirmed is None or not math.isfinite(onset) or not math.isfinite(confirmed) or confirmed<onset:
                raise ValueError('Event delay requires valid source timestamps in tracks.jsonl')
            delays[i,j]=confirmed-onset
            costs[i,j]=(index-event['onset_frame'])/max(1,event['end_frame']-event['onset_frame'])
    a,b=linear_sum_assignment(costs)
    matches=[(int(i),int(j)) for i,j in zip(a,b) if costs[i,j]<1e8]
    tp=len(matches);fp=len(predicted)-tp;fn=len(truth)-tp
    delay=[delays[pair] for pair in matches]
    return {'accuracy_measured':True,'true_positives':tp,'false_positives':fp,'false_negatives':fn,
            'precision':tp/len(predicted) if predicted else 1.,'recall':tp/len(truth) if truth else 1.,
            'F1':2*tp/(2*tp+fp+fn) if tp+fp+fn else 1.,
            'mean_confirmation_delay_seconds':float(np.mean(delay)) if delay else None,
            'max_confirmation_delay_seconds':max(delay) if delay else None,
            'excluded_predictions_outside_labelled_clips':len(all_predicted)-len(predicted),
            'protocol':'One-to-one kind + inclusive confirmation window + all target boxes at IoU >=0.5; delay uses source PTS; not a legal attachment verdict'}


def sidewalk_step(folder, annotation_root, labelled, predicted, result):
    filename=labelled.get('sidewalk_mask')
    if not filename:return
    import cv2
    path=(annotation_root/filename).resolve()
    if not path.is_relative_to(annotation_root):raise ValueError('GT mask must stay inside annotation directory')
    truth=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    w,h=predicted['source_size']
    if truth is None or truth.ndim!=2 or truth.shape!=(h,w):raise ValueError('GT sidewalk mask must be source-sized grayscale labels')
    native_name = predicted.get('sidewalk', {}).get('mask_path')
    native = (Path(folder)/native_name).resolve() if native_name else (Path(folder)/f"sidewalk_masks/{labelled['frame_index']:08d}.png").resolve()
    if not native.is_relative_to(Path(folder).resolve()):
        raise ValueError('Prediction mask must stay inside candidate directory')
    if native.is_file():
        pixels=cv2.imread(str(native),cv2.IMREAD_UNCHANGED)
        if pixels is None or pixels.shape!=truth.shape:raise ValueError('Invalid exported native sidewalk mask')
        candidate=pixels==2
        representation='native_mask'
    else:
        candidate=sidewalk_raster(predicted,size=(w,h))
        representation='exported_polygons'
    valid=truth!=255
    if not np.any(valid):raise ValueError('GT mask has no evaluated pixels')
    a=(truth==2)&valid;b=candidate&valid
    intersection=int(np.count_nonzero(a&b));union=int(np.count_nonzero(a|b))
    iou=intersection/union if union else 1.
    result['frames']+=1;result['intersection_pixels']+=intersection;result['union_pixels']+=union
    result['gt_positive_pixels']+=int(np.count_nonzero(a));result['predicted_positive_pixels']+=int(np.count_nonzero(b))
    result['minimum_frame_iou']=min(result['minimum_frame_iou'],iou)
    result['representations'].add(representation)


def evaluate_tracking(folder, annotations, threshold=.5, max_frames=5000, max_pairs=1_000_000, rule_events=True):
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('IoU threshold must be in (0,1]')
    gt = load_annotations(annotations)
    folder = Path(folder)
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    if summary['source_sha256'] != gt['source_sha256']:
        raise ValueError('Annotation video hash does not match candidate')
    wanted = {frame['frame_index'] for clip in gt['clips'] for frame in clip['frames']}
    if len(wanted) > max_frames or max_frames < 1 or max_pairs < 1:
        raise ValueError('Annotation set exceeds the configured evaluation budget')
    selected = {}
    labelled={frame['frame_index']:frame for clip in gt['clips'] for frame in clip['frames']}
    sidewalk={'frames':0,'intersection_pixels':0,'union_pixels':0,'gt_positive_pixels':0,
              'predicted_positive_pixels':0,'minimum_frame_iou':1.,'representations':set()}
    previous = -1
    last_wanted = max(wanted)
    for row in records(folder/'tracks.jsonl'):
        index = row['frame_index']
        if type(index) is not int or index <= previous:
            raise ValueError('Prediction frames must be strictly increasing')
        previous = index
        if index in wanted:
            if row.get('source_size') != gt['source_size'] or row.get('coordinate_space') != 'source_pixels':
                raise ValueError('Prediction coordinate space/size does not match annotations')
            validate_tracks(row['tracks'],gt['source_size'])
            sidewalk_step(folder,Path(annotations).resolve().parent,labelled[index],row,sidewalk)
            selected[index] = {'tracks':row['tracks'],'timestamp_seconds':row.get('timestamp_seconds')}
        if index >= last_wanted:break
    if set(selected) != wanted:
        raise ValueError('Predictions do not cover all annotated frames')
    sidewalk['accuracy_measured']=bool(sidewalk['frames'])
    sidewalk['IoU']=(sidewalk['intersection_pixels']/sidewalk['union_pixels'] if sidewalk['union_pixels'] else 1.) if sidewalk['frames'] else None
    sidewalk['representations']=sorted(sidewalk['representations'])
    warnings = io.StringIO()
    try:
        with redirect_stdout(warnings):
            import trackeval
            from trackeval.metrics import HOTA, Identity, CLEAR
            metrics = [HOTA(),Identity({'THRESHOLD':threshold,'PRINT_CONFIG':False}),
                       CLEAR({'THRESHOLD':threshold,'PRINT_CONFIG':False})]
    except ImportError as error:
        raise RuntimeError('Install requirements/evaluation.txt in a separate evaluation environment') from error
    if trackeval.__version__ != '1.3.0':
        raise RuntimeError('Tracking adapter is validated against TrackEval 1.3.0')
    combined = {metric.get_name():{} for metric in metrics}
    by_class = {cid:{metric.get_name():{} for metric in metrics} for cid in range(1,11)}
    clips = {}
    for clip in gt['clips']:
        for cid in range(1,11):
            data = sequence_data(clip['frames'],selected,cid,max_pairs)
            if data['num_gt_dets']+data['num_tracker_dets']==0:continue
            raw = {metric.get_name():metric.eval_sequence(data) for metric in metrics}
            name = f"{clip['name']}/class_{cid}"
            clips[name] = metric_summary(raw)
            for metric in metrics:
                key = metric.get_name()
                combined[key][name] = raw[key]
                by_class[cid][key][name] = raw[key]
    empty_case=not any(combined.values())
    if empty_case:
        data=sequence_data(gt['clips'][0]['frames'],selected,1,max_pairs)
        for metric in metrics:combined[metric.get_name()]['empty_sequence']=metric.eval_sequence(data)
    overall = {metric.get_name():metric.combine_sequences(combined[metric.get_name()]) for metric in metrics}
    classes = {}
    for cid,raw in by_class.items():
        if raw['HOTA']:
            classes[str(cid)] = metric_summary({metric.get_name():metric.combine_sequences(raw[metric.get_name()]) for metric in metrics})
    return {'accuracy_measured':True,'human_verification_declared':True,'backend':'TrackEval '+trackeval.__version__,
            'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'annotations_sha256':hashlib.sha256(Path(annotations).read_bytes()).hexdigest(),
            'source_sha256':gt['source_sha256'],'annotated_frames':len(wanted),'annotated_clips':len(gt['clips']),
            'all_source_frames_processed':bool(summary.get('all_source_frames_processed')),
            'IoU_threshold_CLEAR_Identity':threshold,'HOTA_thresholds':[float(v) for v in metrics[0].array_labels],
            'metric_scale':'fractions 0..1; MOTA may be negative; counts are integers',
            'overall':metric_summary(overall),'by_class':classes,'clips':clips,
            'no_object_case':empty_case,
            'event_metrics':evaluate_events(folder,gt,selected) if rule_events else {'accuracy_measured':False,'reason':'Event evaluation explicitly disabled'},
            'sidewalk_metrics':sidewalk,
            'backend_warnings':warnings.getvalue().strip(),
            'scope':'Public exported tracks on explicitly labelled contiguous clips; no ignore-region preprocessing; not full-video accuracy unless all frames are labelled'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--ground-truth',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iou',type=float,default=.5)
    p.add_argument('--max-frames',type=int,default=5000)
    p.add_argument('--max-pairs',type=int,default=1_000_000)
    p.add_argument('--skip-events',action='store_true')
    args = p.parse_args(argv)
    result = evaluate_tracking(args.candidate,args.ground_truth,args.iou,args.max_frames,args.max_pairs,not args.skip_events)
    write_json(args.output,result)
    print(json.dumps(result['overall'],indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
