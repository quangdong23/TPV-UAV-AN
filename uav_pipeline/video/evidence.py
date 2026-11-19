"""Disk-backed live evidence and bounded recording retention."""
from collections import Counter
import json
import math
from pathlib import Path
import threading
import time
import shutil
from types import SimpleNamespace

from uav_pipeline.video.io import atomic_json, consolidate_recording, open_writer, write_frame


class LiveEvidence:
    def __init__(self, output, monitor, fps, uploader=None, retention_seconds=30,vfr=False,upload_on_confirmation=False,cfr=False,
                 async_clips=False,upload_interval_seconds=30.,encoder="opencv"):
        self.async_clips=async_clips
        self.encoder=encoder
        self.upload_interval_seconds=float(upload_interval_seconds)
        if self.upload_interval_seconds<=0:raise ValueError('Upload interval must be positive')
        self.report_lock=threading.RLock()
        self.output=Path(output)
        self.monitor=monitor
        self.fps=fps
        self.vfr=vfr
        self.cfr=cfr
        self.upload_on_confirmation=upload_on_confirmation and uploader is not None
        self.retention_seconds=retention_seconds
        self.segments=[]
        self.buffer_writer=None
        self.buffer_count=0
        self.buffer_frames=0
        self.buffer_part=0
        self.buffer_timestamp=0.
        (self.output/'evidence_buffer').mkdir(exist_ok=True)
        self.counts=Counter()
        self.upload_counts=Counter()
        self.uploader=uploader
        self.draining=threading.Event()
        self.stop=threading.Event()
        self.wake=threading.Event()
        self.pending=self.output/'pending_uploads'
        self.pending.mkdir(exist_ok=True)
        self.clip_jobs=self.output/'clip_jobs'
        self.clip_jobs.mkdir(exist_ok=True)
        self.clip_wake=threading.Event()
        self.clip_draining=threading.Event()
        self.clip_worker=None
        if self.async_clips:
            self.clip_worker=threading.Thread(target=self._clip_jobs,daemon=True,name='live-evidence-cut')
            self.clip_worker.start()
        self.worker=None
        if uploader is not None:
            self.worker=threading.Thread(target=self._uploads,daemon=True,name='live-evidence-upload')
            self.worker.start()

    def write(self, frame, timestamp):
        """Small disk chunks used only for evidence; main recordings stay open."""
        if self.monitor is None:return
        if self.buffer_writer is None:
            path=self.output/f'evidence_buffer/part_{self.buffer_part:06d}.mp4'
            self.buffer_start_timestamp=timestamp
            if self.cfr:
                from uav_pipeline.video.cfr import open_cfr_writer
                self.buffer_writer=open_cfr_writer(path,self.fps,frame.shape,self.encoder)
            elif self.vfr:
                from uav_pipeline.video.vfr import VFRWriter
                self.buffer_writer=VFRWriter(path,self.fps,frame.shape,self.encoder)
            else:self.buffer_writer=open_writer(path,self.fps,frame.shape)
        if self.vfr and not self.cfr:self.buffer_writer.write_at(frame,timestamp)
        else:write_frame(self.buffer_writer,frame)
        self.buffer_count+=1
        self.buffer_frames+=1
        self.buffer_timestamp=timestamp
        # Close the short evidence buffer at the first clip deadline, not only
        # on its 2-second rotation or when the whole analysis finishes.
        if self.ready(self.buffer_frames) or (timestamp-self.buffer_start_timestamp>=2 if self.vfr else self.buffer_count>=max(1,math.ceil(self.fps*2))):
            self._commit_buffer()

    def _commit_buffer(self):
        if self.buffer_writer is None:return
        self.buffer_writer.release()
        self.buffer_writer=None
        self.segments.append({'video':f'evidence_buffer/part_{self.buffer_part:06d}.mp4',
            'frames':self.buffer_count,'start_frame':self.buffer_frames-self.buffer_count,
            'part':self.buffer_part,'start_timestamp':self.buffer_start_timestamp,
            'end_timestamp':self.buffer_timestamp+1/self.fps})
        self.buffer_part+=1
        self.buffer_count=0
        self.flush(self.segments,self.buffer_frames,self.buffer_timestamp)

    def finish(self):
        self._commit_buffer()
        self.flush(self.segments,self.buffer_frames,self.buffer_timestamp,final=True)
        self._drain_clips()
        return self._save_report()

    def _uploads(self):
        deadline=time.monotonic()+self.upload_interval_seconds
        while not self.stop.is_set():
            if not self.draining.is_set():
                self.wake.wait(max(0.,deadline-time.monotonic()))
                self.wake.clear()
                if not self.draining.is_set() and time.monotonic()<deadline:continue
            final_pass=self.draining.is_set()
            # Snapshot this batch; new arrivals wait until the next interval.
            for path in sorted(self.pending.glob('*.json')):
                if self.stop.is_set():break
                try:
                    event=json.loads(path.read_text())
                    print('[upload] Sending '+event['event_id'],flush=True)
                    result=self.uploader.upload_events(self.output,[event])
                    self.upload_counts.update(result)
                    if not result['failed']:path.unlink()
                except Exception:
                    self.upload_counts['failed']+=1
            if final_pass:break
            deadline=time.monotonic()+self.upload_interval_seconds

    def _clip_jobs(self):
        while True:
            final_pass=self.clip_draining.is_set()
            for path in sorted(self.clip_jobs.glob('*/job.json')):
                try:
                    job=json.loads(path.read_text())
                    monitor=SimpleNamespace(events=job['events'],active_events=[],
                        options=self.monitor.options,presence={},pairs={},repairs={},model_activities={},
                        _write_clip=self.monitor._write_clip)
                    self._flush_sync(job['segments'],job['frames'],job['timestamp'],final=True,
                                     monitor_override=monitor)
                    shutil.rmtree(path.parent)
                except Exception as error:
                    (path.parent/'last_error.txt').write_text(type(error).__name__+': '+str(error))
                    print('[evidence] Clip job failed; retained on disk: '+path.parent.name,flush=True)
            if final_pass:break
            self.clip_wake.wait(2.)
            self.clip_wake.clear()

    def _drain_clips(self):
        if self.clip_worker is not None:
            self.clip_draining.set()
            self.clip_wake.set()
            self.clip_worker.join()
            self.clip_worker=None

    def flush(self, segments, frames, timestamp, final=False):
        if not self.async_clips:
            return self._flush_sync(segments,frames,timestamp,final)
        monitor=self.monitor
        scale=1. if self.vfr else self.fps
        extent=timestamp+1/self.fps if self.vfr else frames
        ready=[]
        if monitor is not None:
            for event in monitor.events:
                anchor=(event.get('first_timestamp_seconds',event['first_frame']/self.fps)
                        if self.vfr else event['first_frame'])
                first=max(0,anchor-monitor.options.pre_seconds*scale)
                end=min(anchor+monitor.options.post_seconds*scale,
                        first+monitor.options.max_clip_seconds*scale)
                if final or self.upload_on_confirmation or extent+1e-9>=end:ready.append(event)
        if ready:
            # Hard links pin source chunks while the AI continues and retention rotates.
            jobdir=self.clip_jobs/ready[0]['event_id'];jobdir.mkdir(exist_ok=True)
            pinned=[]
            import os
            for segment in segments:
                item=dict(segment)
                source=self.output/segment['video'];link=jobdir/source.name
                if not link.exists():os.link(source,link)
                item['video']=str(link.relative_to(self.output))
                pinned.append(item)
            atomic_json(jobdir/'job.json',{'events':ready,'segments':pinned,
                                          'frames':frames,'timestamp':timestamp})
            ids={e['event_id'] for e in ready}
            monitor.events[:]=[e for e in monitor.events if e['event_id'] not in ids]
            monitor.active_events[:]=[e for e in monitor.active_events
                                     if e['event']['event_id'] not in ids]
            self.clip_wake.set()
        keep=max(0,extent-self.retention_seconds*scale)
        if monitor is not None:
            protected=[(e.get('first_timestamp_seconds',e['first_frame']/self.fps)
                        if self.vfr else e['first_frame']) for e in monitor.events]
            for mapping in (monitor.presence,monitor.pairs,monitor.repairs,monitor.model_activities):
                protected.extend(s['first_timestamp'] if self.vfr else s['first_frame']
                                 for s in mapping.values() if not s['alerted'])
            if protected:keep=min(keep,max(0,min(protected)-monitor.options.pre_seconds*scale))
        while len(segments)>1:
            old=segments[0]
            end=old['end_timestamp'] if self.vfr else old['start_frame']+old['frames']
            if end>keep:break
            segments.pop(0)
            (self.output/old['video']).unlink(missing_ok=True)
        return self._save_report()

    def ready(self, frames):
        """Finalize ready evidence without waiting for a long recording segment."""
        if self.monitor is None:return False
        if self.upload_on_confirmation:
            return bool(self.monitor.events)
        options=self.monitor.options
        if self.vfr:
            return any(self.buffer_timestamp+1/self.fps>=min(
                e['first_timestamp_seconds']+options.post_seconds,
                max(0,e['first_timestamp_seconds']-options.pre_seconds)+options.max_clip_seconds)
                for e in self.monitor.events)
        limit=max(1,int(min(options.max_clip_seconds,
                            options.pre_seconds+options.post_seconds)*self.fps))
        for event in self.monitor.events:
            first=max(0,math.floor(event['first_frame']-options.pre_seconds*self.fps))
            end=min(math.ceil(event['first_frame']+options.post_seconds*self.fps),first+limit)
            if frames>=end:return True
        return False

    def _flush_sync(self, segments, frames, timestamp, final=False, monitor_override=None):
        monitor=monitor_override if monitor_override is not None else self.monitor
        scale=1. if self.vfr else self.fps
        extent=timestamp+1/self.fps if self.vfr else frames
        def start_of(segment):
            return segment['start_timestamp'] if self.vfr else segment['start_frame']
        def end_of(segment):
            return segment['end_timestamp'] if self.vfr else segment['start_frame']+segment['frames']
        if monitor is not None:
            from uav_pipeline.uploads import behavior_name
            groups=[]
            limit=monitor.options.max_clip_seconds*scale
            for event in sorted(monitor.events,key=lambda e:e['first_frame']):
                anchor=event.get('first_timestamp_seconds',event['first_frame']/self.fps)*scale if self.vfr else event['first_frame']
                first=max(0,anchor-monitor.options.pre_seconds*scale)
                end=min(anchor+monitor.options.post_seconds*scale,first+limit)
                if self.upload_on_confirmation:
                    # Only frames already recorded are needed for early upload.
                    # Later confirmations cannot postpone an existing ready job.
                    end=min(end,extent)
                # Pending future post-roll must not postpone a clip that is
                # already complete. Only group events whose own clips are ready.
                if not final and extent+1e-9<end:
                    continue
                group=next((g for g in reversed(groups) if first<g['last'] and
                            max(end,g['last'])-g['first']<=limit),None)
                if group is None:
                    group={'first':first,'last':end,'events':[]};groups.append(group)
                group['last']=max(group['last'],end);group['events'].append(event)
            for group in groups:
                first,end=group['first'],group['last'];members=group['events']
                if not final and extent<end:continue
                last=min(extent,end);event=members[0]
                chosen=[s for s in segments if start_of(s)<last and end_of(s)>first]
                alerts=[{'event_id':e['event_id'],'kind':e['kind'],'track_ids':e['track_ids'],
                         'description':behavior_name(e),
                         **({'demo_people':e['demo_people']} if e.get('demo_people') else {})} for e in members]
                clip={'status':'error','error':'Recording window unavailable'}
                if chosen and start_of(chosen[0])<=first and last>first:
                    staging=self.output/'clip_staging';staging.mkdir(exist_ok=True)
                    (staging/'segments').mkdir(exist_ok=True)
                    import os
                    for segment in chosen:
                        link=staging/'segments'/Path(segment['video']).name
                        if not link.exists():os.link(self.output/segment['video'],link)
                    try:
                        consolidate_recording(staging,['segments/'+Path(s['video']).name for s in chosen],
                                              'annotated.mp4',sum(s['frames'] for s in chosen),
                                              start_timestamps=[s['start_timestamp'] for s in chosen] if self.vfr else None)
                        target=self.output/'event_clips'/f"{event['event_id']}.mp4"
                        target.parent.mkdir(exist_ok=True)
                        if self.cfr:
                            from uav_pipeline.video.cfr import cut_cfr
                            cut_cfr(staging/'annotated.mp4',target,first-start_of(chosen[0]),last-start_of(chosen[0]),self.fps)
                        elif self.vfr:
                            from uav_pipeline.video.vfr import cut_vfr
                            cut_vfr(staging/'annotated.mp4',target,first-start_of(chosen[0]),last-start_of(chosen[0]))
                        else:monitor._write_clip(staging/'annotated.mp4',target,
                            int(first-start_of(chosen[0])),int(last-start_of(chosen[0])),self.fps)
                        clip={'status':'saved','path':str(target.relative_to(self.output)),
                              'start_output_seconds':first/scale,'end_output_seconds':last/scale,
                              'timebase':'source_timestamps' if self.vfr else 'CFR_frames',
                              'truncated_end':last<end,'truncated_start':first==0}
                    except (OSError,RuntimeError) as error:
                        clip={'status':'error','error':str(error)}
                    finally:
                        for path in (staging/'segments').iterdir():path.unlink()
                        (staging/'annotated.mp4').unlink(missing_ok=True)
                        (staging/'segments').rmdir();staging.rmdir()
                if monitor_override is not None and clip['status']!='saved':
                    raise RuntimeError(clip.get('error','Unable to cut evidence'))
                clip.update(upload_trigger='confirmation' if self.upload_on_confirmation else 'post_roll',
                            post_roll_omitted=self.upload_on_confirmation,
                            clip_id=event['event_id'],alerts=alerts,shared=len(members)>1,
                            description='; '.join(dict.fromkeys(a['description'] for a in alerts)))
                from uav_pipeline.demo_confidence import demo_confidence
                clip.update(demo_confidence((clip['clip_id'],min(e.get('detected_at') or '' for e in members))))
                images={image['path']:image for e in members for image in e.get('images',[])}
                from uav_pipeline.demo_people import merged_demo_people
                upload={**event,'clip':clip,'images':list(images.values())}
                people=merged_demo_people(members)
                if people:upload['demo_people']=people
                for member in members:
                    member['clip']=clip
                    with (self.output/'closed_events.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(member,ensure_ascii=False,allow_nan=False)+'\n')
                    self.counts[member['kind']]+=1
                    monitor.events.remove(member)
                    monitor.active_events=[e for e in monitor.active_events if e['event'] is not member]
                if self.uploader is not None and clip['status']=='saved':
                    # Publish the report before waking the uploader, which may
                    # read it immediately, and before cutting subsequent clips.
                    self._save_report()
                    atomic_json(self.pending/(event['event_id']+'.json'),upload)
                    self.wake.set()
                    print(f'[upload] Queued {event["event_id"]} at source_time={timestamp:.3f}s '
                          f'trigger={clip["upload_trigger"]}; analysis continues',flush=True)
                self.counts['clip_errors' if clip['status']!='saved' else 'saved_clips']+=1
        keep=max(0,extent-self.retention_seconds*scale)
        if monitor is not None:
            # Protect pre-roll even before an event reaches confirmation.
            protected=[e.get('first_timestamp_seconds',e['first_frame']/self.fps)*scale if self.vfr else e['first_frame'] for e in monitor.events]
            for mapping in (monitor.presence,monitor.pairs,monitor.repairs,monitor.model_activities):
                protected.extend(s['first_timestamp']*scale if self.vfr else s['first_frame'] for s in mapping.values() if not s['alerted'])
            if protected:keep=min(keep,max(0,min(protected)-monitor.options.pre_seconds*scale))
        while len(segments)>1 and end_of(segments[0])<=keep:
            old=segments.pop(0)
            for key in ('video','records','original_video'):
                if old.get(key):(self.output/old[key]).unlink(missing_ok=True)

        report=self._save_report()
        self.wake.set()
        return report

    def _save_report(self):
        with self.report_lock:
            return self._save_report_locked()

    def _save_report_locked(self):
        report={'alert_count':sum(v for k,v in self.counts.items() if k not in ('saved_clips','clip_errors')),
                'counts':dict(self.counts),'events_file':'closed_events.jsonl',
                'pending_events':len(self.monitor.events) if self.monitor else 0,
                'recording_retention_seconds':self.retention_seconds,
                'pending_clip_jobs':sum(1 for _ in self.clip_jobs.glob('*/job.json')),
                'upload_interval_seconds':self.upload_interval_seconds}
        atomic_json(self.output/'events.json',report)
        return report

    def close(self):
        self._drain_clips()
        if self.buffer_writer is not None:
            self.buffer_writer.release()
            self.buffer_writer=None
        if self.worker:
            print('[upload] Waiting for pending uploads to finish...',flush=True)
            self.draining.set()
            self.wake.set()
            self.worker.join()
            pending=sum(1 for _ in self.pending.glob('*.json'))
            atomic_json(self.output/'upload_summary.json',
                        {**dict(self.upload_counts),'pending':pending})
            print(f'[upload] Finished: ready={self.upload_counts["ready"]}, '
                  f'already_ready={self.upload_counts["already_ready"]}, pending={pending}',flush=True)
        self.stop.set()
        self.wake.set()
