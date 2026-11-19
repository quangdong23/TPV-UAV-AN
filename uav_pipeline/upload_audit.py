"""Sanitized per-session HTTP evidence for diagnosing multipart uploads."""
from datetime import datetime
import json
from pathlib import Path
import re
import threading
import traceback
from types import SimpleNamespace
from urllib.parse import urlsplit
import hashlib
from zoneinfo import ZoneInfo


def sanitized(value,key):
    if isinstance(value,dict):
        return {name:('[REDACTED]' if any(word in str(name).lower() for word in
                ('key','token','secret','password','authorization','cookie','signature','url'))
                else sanitized(item,key)) for name,item in value.items()}
    if isinstance(value,list):return [sanitized(item,key) for item in value]
    if isinstance(value,str):
        value=value.replace(key,'[REDACTED]')
        value=re.sub(r'https?://[^\s<>"\']+','[URL]',value)
        value=re.sub(r'(?i)((?:api[_-]?key|token|password|authorization|signature)\s*[=:]\s*)[^\s,;]+',r'\1[REDACTED]',value)
        return value
    return value


def session_transport(client,output,transport):
    lock=threading.Lock()
    targets={}
    context={}
    logfile=Path(output)/'upload_http.jsonl'

    def log(event,**fields):
        entry=sanitized({'time':datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat(),
                         'event':event,**context,**fields},client.api_key)
        with lock:
            with logfile.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(entry,ensure_ascii=False)+'\n')
            print('[upload-http] '+json.dumps(entry,ensure_ascii=False),flush=True)

    def request(method,url,**kwargs):
        endpoint=url.removeprefix(client.api_url)
        if endpoint.endswith('/initiate'):
            payload=kwargs.get('json') or {}
            log('initiate',filename=payload.get('filename'),file_size=payload.get('file_size'),
                image_count=len(payload.get('images',[])),images=payload.get('images',[]))
        try:
            response=transport.request(method,url,**kwargs)
        except Exception as error:
            log('request_exception',method=method,endpoint=endpoint,
                exception=type(error).__name__,traceback=''.join(traceback.format_exception(error)))
            raise
        try:body=response.json()
        except Exception:body=getattr(response,'text','')[:8000]
        log('api_response',method=method,endpoint=endpoint,status=response.status_code,
            **({'body':body} if endpoint.endswith(('/complete','/abort')) or response.status_code>=400 else {}))
        if 200<=response.status_code<300 and isinstance(body,dict):
            if endpoint.endswith('/initiate'):
                context['violation_id']=body.get('violation_id')
                context['total_parts']=body.get('total_parts')
                declared=(kwargs.get('json') or {}).get('images',[])
                for index,target in enumerate(body.get('images',[]),1):
                    image=declared[index-1] if index<=len(declared) else {}
                    targets[target['upload_url']]={'kind':'image','index':index,
                        'total_images':len(declared),'filename':image.get('filename'),
                        'file_size':image.get('file_size')}
            elif endpoint.endswith('/part-urls'):
                for part in body.get('parts',[]):
                    targets[part['url']]={'kind':'video_part','part_number':part['part_number']}
        if endpoint.endswith(('/complete','/abort')) and response.status_code!=200 and 200<=response.status_code<300:
            raise RuntimeError('Expected HTTP 200 for '+endpoint)
        return response

    def put(url,**kwargs):
        target=targets.get(url,{'kind':'unknown'})
        data=kwargs.get('data')
        digest=hashlib.md5()
        if hasattr(data,'read'):
            position=data.tell()
            for chunk in iter(lambda:data.read(1024*1024),b''):digest.update(chunk)
            sent_size=data.tell()-position
            data.seek(position)
        else:
            digest.update(data or b'')
            sent_size=len(data or b'')
        try:
            response=transport.put(url,**kwargs)
        except Exception as error:
            log('put_exception',**target,exception=type(error).__name__,
                traceback=''.join(traceback.format_exception(error)))
            raise
        parsed=urlsplit(url)
        request=getattr(response,'request',None)
        final_method=getattr(request,'method',None)
        content_length=getattr(request,'headers',{}).get('Content-Length')
        etag=response.headers.get('ETag','').strip('"')
        log('put_response',**target,status=response.status_code,
            storage_host=parsed.hostname,object_path=parsed.path,
            sent_bytes=sent_size,sent_content_length=content_length,
            final_method=final_method,
            response_content_type=response.headers.get('Content-Type'),
            storage_request_id=response.headers.get('x-amz-request-id'),
            server=response.headers.get('Server'),
            redirects=[{'status':r.status_code,'method':getattr(getattr(r,'request',None),'method',None)}
                       for r in getattr(response,'history',[])],
            etag_matches_uploaded_bytes=etag==digest.hexdigest(),
            content_type=kwargs.get('headers',{}).get('Content-Type'),
            etag_present=bool(response.headers.get('ETag')),
            **({'body':getattr(response,'text','')[:8000]} if response.status_code>=400 else {}))
        if final_method and final_method.upper()!='PUT':
            raise RuntimeError('Storage redirected PUT to '+final_method+'; upload not confirmed')
        return response

    return SimpleNamespace(request=request,put=put),log
