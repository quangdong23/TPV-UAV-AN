"""Gắn hồ sơ GIẢ cho demo; không nhận diện hay xác minh danh tính từ video."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import secrets

_SALT=secrets.token_bytes(16)


@lru_cache(maxsize=1)
def profiles():
    path=Path(__file__).resolve().parent.parent/'configs/demo_people.json'
    return json.loads(path.read_text(encoding='utf-8'))['people']


def attach_demo_people(event,tracks):
    people={t['track_id'] for t in tracks if t['class_id']==1}
    if not people:
        event.pop('demo_people',None)
        return
    old={p['person_track_id']:p for p in event.get('demo_people',[])}
    pool=profiles()
    result=[]
    for tid in sorted(people):
        # Cùng track giữ cùng hồ sơ trong phiên; retry giữ dữ liệu đã lưu.
        number=int.from_bytes(hashlib.blake2b(str(tid).encode(),key=_SALT,digest_size=8).digest(),'big')
        result.append(old.get(tid) or {**pool[number%len(pool)],'person_track_id':tid,
            'is_demo':True,'identity_verified':False,'identity_source':'synthetic'})
    event['demo_people']=result


def merged_demo_people(events):
    return list({p['person_track_id']:p for event in events
                 for p in event.get('demo_people',[])}.values())


def api_demo_identity(event):
    # Synthetic profiles remain marked internally; map all members to BE schema.
    people=[]
    for person in event.get('demo_people',[]):
        if person.get('is_demo') is not True:
            continue
        citizen_id=person.get('citizen_id',person.get('cccd'))
        if citizen_id is not None and not isinstance(citizen_id,str):
            raise ValueError('Synthetic citizen_id must be a string or null')
        name=person['full_name'].removesuffix(' (DEMO)').strip()
        people.append({'full_name':name,'citizen_id':citizen_id})
    return {'violators':people or None}
