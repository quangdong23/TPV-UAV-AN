"""Upload video, ảnh và metadata; họ tên/CCCD giả luôn được đánh dấu DEMO."""
import argparse
import importlib.util
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import math
import mimetypes
import re
import os
from pathlib import Path
import time
import requests
from uav_pipeline.video.io import atomic_json


BEHAVIORS = {
    'crowd_gathering_suspected': 'tụ tập trái phép',
    'sidewalk_encroachment': 'Lấn chiếm vỉa hè',
    'sign_attachment_suspected': 'đặt biển quảng cáo sai quy định',
    'motorcycle_repair_suspected': 'sửa xe máy lấn chiếm vỉa hè/lòng lề đường',
    'vehicle_repair_suspected': 'sửa xe lấn chiếm vỉa hè/lòng lề đường',
    'vehicle_washing_suspected': 'rửa xe lấn chiếm vỉa hè/lòng lề đường',
    'street_vending_suspected': 'Bán hàng rong',
}
CLASS_BEHAVIORS = {
    4: 'bày bán bàn ghế lấn chiếm vỉa hè/lòng lề đường', 5: 'bày bán bàn ghế lấn chiếm vỉa hè/lòng lề đường',
    6: 'Đặt xe bán hàng trên vỉa hè', 7: 'đặt biển quảng cáo sai quy định',
    10: 'bày bán hàng hoá lấn chiếm vỉa hè/lòng lề đường',
    11: 'tập kết vật liệu lấn chiếm vỉa hè/lòng lề đường',
    12: 'đặt máy móc, sửa chữa thiết bị lấn chiếm vỉa hè/lòng lề đường',
}


def behavior_name(event):
    if event.get('grouped_behavior_name'):
        return event['grouped_behavior_name']
    if event.get('evidence',{}).get('behavior_name'):
        return event['evidence']['behavior_name']
    if event['kind'] == 'sidewalk_encroachment':
        classes = set(event.get('class_ids', []))
        if len(classes) == 1:
            label = CLASS_BEHAVIORS.get(next(iter(classes)))
            if label:return label
    if event['kind'] not in BEHAVIORS:
        raise ValueError('No configured behavior name for event kind')
    return BEHAVIORS[event['kind']]


class UploadError(RuntimeError):pass


class APIError(UploadError):
    def __init__(self,status,detail=''):
        self.status=status
        super().__init__('API HTTP '+str(status)+(': '+detail if detail else ''))


class EvidenceUploader:
    def __init__(self,api_url,api_key,location='Hà Nội',http=None,sleep=time.sleep):
        if not api_url.startswith('https://'):
            raise ValueError('Upload API must use HTTPS')
        if not api_key:raise ValueError('Missing upload API key')
        self.api_url=api_url.rstrip('/')
        self.api_key=api_key
        self.location=location
        self.http=http if http is not None else requests.Session()
        self.sleep=sleep
        client_path=Path(__file__).with_name('be_multipart_client.py')
        spec=importlib.util.spec_from_file_location('_tpv_be_client',client_path)
        self._client=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._client)
        # Each uploader has its own module/transport; concurrent clients do not
        # overwrite global requests state or leak API credentials between sessions.
        self._client.requests=SimpleNamespace(request=self.http.request,put=self.http.put)
        self._client.time=SimpleNamespace(sleep=self.sleep)
        import shutil
        tools=Path(__file__).resolve().parent.parent/'.tools/ffmpeg/bin'
        def find_tool(name):
            installed=shutil.which(name)
            bundled=tools/name
            return installed or (str(bundled) if bundled.is_file() else None)
        self._client.shutil=SimpleNamespace(which=find_tool)


    @classmethod
    def from_config(cls,path):
        config=json.loads(Path(path).read_text())
        key=os.environ.get('TPV_UPLOAD_API_KEY') or os.environ.get('UPLOAD_API_KEY') or config.get('api_key')
        return cls(config['api_url'],key,config.get('location','Hà Nội'))

    def upload_one(self,output,event):
        output=Path(output).resolve()
        clip=event.get('clip') or {}
        if clip.get('status')!='saved':raise UploadError('Evidence clip is not saved')
        video=(output/clip['path']).resolve()
        if not video.is_relative_to(output) or not video.is_file() or video.stat().st_size<=0:
            raise UploadError('Evidence video missing or outside session directory')
        images=[];image_paths=[]
        image_items=event.get('images',[]) if event.get('evidence_status') in (None,'ready') and not event.get('evidence_image_error') else []
        for item in image_items:
            path=(output/item['path']).resolve()
            if not path.is_relative_to(output) or not path.is_file() or not 0<path.stat().st_size<=25*1024*1024:
                raise UploadError('Invalid evidence image')
            content_type=mimetypes.guess_type(path.name)[0]
            if content_type not in ('image/jpeg','image/png','image/webp'):
                raise UploadError('Unsupported evidence image type')
            image_paths.append(path)
            images.append({'filename':path.name,'file_size':path.stat().st_size,'content_type':content_type})
        if len(images)>50:raise UploadError('Too many evidence images')
        moment=event.get('detected_at')
        if not moment or datetime.fromisoformat(moment).tzinfo is None:
            raise UploadError('Event occurrence time with timezone is required; upload time is never substituted')
        # Convert the same instant, never merely relabel a UTC clock as +07.
        moment=datetime.fromisoformat(moment).astimezone(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat(timespec='milliseconds')
        location=event.get('location',self.location)
        if not isinstance(location,str) or not re.fullmatch(r'\s*[+-]?\d+(?:\.\d+)?\s*,\s*[+-]?\d+(?:\.\d+)?\s*',location):
            raise UploadError('Explicit GPS location required: latitude, longitude')
        latitude,longitude=map(float,location.split(','))
        if not (-90<=latitude<=90 and -180<=longitude<=180):
            raise UploadError('GPS coordinates out of range')
        metadata={'video':clip['path'],'detected_at':moment,'location':location,
                  'behavior_name':behavior_name(event)}
        if images:metadata['images']=images
        from uav_pipeline.demo_people import api_demo_identity
        demo_identity=({'violators':event['violators']} if 'violators' in event else api_demo_identity(event))
        people=demo_identity['violators']
        if people == []:
            people=None
        if people is not None:
            if not isinstance(people,list) or len(people)>100:
                raise UploadError('violators must contain at most 100 people or null')
            for person in people:
                if (not isinstance(person,dict) or not isinstance(person.get('full_name'),str) or
                        not person['full_name'].strip() or
                        (person.get('citizen_id') is not None and
                         (not isinstance(person['citizen_id'],str) or not person['citizen_id'].strip()))):
                    raise UploadError('Invalid violator: full_name required, citizen_id string or null')
            people=[{'full_name':p['full_name'],'citizen_id':p.get('citizen_id')} for p in people]
        demo_identity={'violators':people}
        metadata.update(demo_identity)
        from uav_pipeline.demo_confidence import api_demo_confidence
        confidence=api_demo_confidence(event)
        if event.get('model_version'):metadata['model_version']=event['model_version']
        identity=hashlib.sha256(json.dumps([event['event_id'],metadata],sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:24]
        queue=output/'uploads';queue.mkdir(exist_ok=True)
        record=queue/(identity+'.json')
        state=json.loads(record.read_text()) if record.exists() else {'metadata':metadata,'state':'pending'}
        if state['metadata']!=metadata:raise UploadError('Upload metadata changed')
        if state['state']=='ready':return 'already_ready'
        if state['state'] in ('initiating','initiate_uncertain','completing','complete_uncertain','aborting'):
            # Without a READ_API_KEY an ambiguous remote commit cannot be safely
            # reconciled. Never create a duplicate upload by guessing.
            raise UploadError('Upload outcome uncertain; administrator reconciliation required')
        confidence=state.setdefault('demo_confidence',confidence)
        backend=self._client
        from uav_pipeline.upload_audit import session_transport
        original_transport=backend.requests
        backend.requests,audit=session_transport(self,output,original_transport)
        original_request=backend.api_request
        attempted=False

        def safe_error(error):
            if isinstance(error,UploadError):
                return str(error)
            response=getattr(error,'response',None)
            if response is None:
                return type(error).__name__
            try:
                detail=json.dumps(response.json(),ensure_ascii=False)
            except Exception:
                detail=getattr(response,'text','') or ''
            detail=detail.replace(self.api_key,'[REDACTED]')
            detail=re.sub(r'https?://[^\s<>"\']+','[URL]',detail)
            detail=re.sub(r'(?i)("[^"]*(?:key|token|secret|password|authorization|cookie|signature)[^"]*"\s*:\s*)"[^"]*"',r'\1"[REDACTED]"',detail)
            return 'API HTTP '+str(response.status_code)+': '+' '.join(detail.split())[:1500]

        def observed_request(method,url,key,**kwargs):
            nonlocal attempted
            endpoint=url.removeprefix(self.api_url)
            if endpoint.endswith('/initiate'):
                attempted=True
                state['state']='initiating'
            elif endpoint.endswith('/complete'):
                state['state']='completing'
                print('[upload] Completing '+state.get('violation_id',''),flush=True)
            elif endpoint.endswith('/abort'):
                state['state']='aborting'
            atomic_json(record,state)
            try:
                result=original_request(method,url,key,**kwargs)
            except Exception as error:
                raise UploadError(method+' '+endpoint+'; '+safe_error(error)) from None
            if endpoint.endswith('/initiate'):
                state.update(state='uploading',violation_id=result['violation_id'])
            elif endpoint.endswith('/abort'):
                state['state']='aborted'
            atomic_json(record,state)
            return result

        backend.api_request=observed_request
        try:
            if state.get('violation_id') and state['state']!='aborted':
                # A legacy incomplete multipart session must be closed before
                # the BE client starts a replacement; never create it silently.
                observed_request('POST',self.api_url+'/api/uploads/'+state['violation_id']+'/abort',self.api_key)
            state.pop('violation_id',None)
            state['state']='pending'
            args=argparse.Namespace(api_url=self.api_url,api_key=self.api_key,
                video=str(video),image=[str(p) for p in image_paths],
                detected_at=moment,location=location,behavior=metadata['behavior_name'],
                camera_id=event.get('camera_id'),confidence=confidence.get('confidence'),
                model_version=event.get('model_version'),
                violators_json=json.dumps(demo_identity['violators'],ensure_ascii=False)
                    if demo_identity['violators'] else None,workers=2)
            result=backend.upload_video(args)
            if (result.get('success') is not True or result.get('status')!='ready' or
                    result.get('violation_id')!=state.get('violation_id')):
                state['state']='complete_uncertain'
                raise UploadError('BE client did not confirm matching ready violation')
            state.update(state='ready',result=result)
            state.pop('error',None)
            atomic_json(record,state)
            print('[upload] Ready '+result['violation_id'],flush=True)
            return 'ready'
        except Exception as error:
            import traceback
            audit('upload_exception',exception=type(error).__name__,
                  traceback=''.join(traceback.format_exception(error)))
            if state['state']=='aborted' or not attempted and not state.get('violation_id'):
                state['state']='retry_pending'
                state.pop('violation_id',None)
            else:
                state['state']=('complete_uncertain' if state['state']=='completing' else
                                'aborting' if state['state']=='aborting' else 'initiate_uncertain')
            state['error']=safe_error(error)
            atomic_json(record,state)
            raise UploadError(state['error']) from None
        finally:
            backend.api_request=original_request
            backend.requests=original_transport

    def upload_events(self,output,events):
        events=list(events)
        counts={'ready':0,'already_ready':0,'failed':0}
        images_by_clip={}
        for row in events:
            path=(row.get('clip') or {}).get('path')
            images_by_clip.setdefault(path,{})
            for image in row.get('images',[]):
                images_by_clip[path][image['path']]=image
        from uav_pipeline.demo_people import merged_demo_people
        people_by_clip={}
        for row in events:
            key=(row.get('clip') or {}).get('path')
            people_by_clip.setdefault(key,[]).append(row)
        seen=set()
        for event in events:
            clip=event.get('clip') or {}
            key=clip.get('path')
            if clip.get('alerts') and key and key in seen:
                continue
            if key and clip.get('alerts'):seen.add(key)
            if clip.get('alerts'):
                event={**event,'event_id':clip['clip_id'],
                       'grouped_behavior_name':clip['description'],
                       'images':list(images_by_clip.get(key,{}).values())}
                people=merged_demo_people(people_by_clip.get(key,[]))
                if people:event['demo_people']=people
            try:
                result=self.upload_one(output,event);counts[result]+=1
                print('[upload] Evidence '+result,flush=True)
            except Exception as error:
                counts['failed']+=1
                print('[upload] Evidence retained locally; '+(str(error) if isinstance(error,UploadError) else type(error).__name__),flush=True)
        atomic_json(Path(output)/'upload_summary.json',counts)
        return counts


def main(argv=None):
    from uav_pipeline.paths import PROJECT_ROOT
    parser=argparse.ArgumentParser(description='Retry saved evidence uploads without rerunning AI')
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/upload_api.json')
    args=parser.parse_args(argv)
    report=json.loads((args.session/'events.json').read_text())
    if report.get('events_file') == 'closed_events.jsonl':
        def archived_events():
            with (args.session/'closed_events.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    yield json.loads(line)
        events=archived_events()
    else:
        events=report['events']
    result=EvidenceUploader.from_config(args.config).upload_events(args.session,events)
    return 1 if result['failed'] else 0


if __name__=='__main__':raise SystemExit(main())
