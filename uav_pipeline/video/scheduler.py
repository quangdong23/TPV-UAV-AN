"""Bound adaptive detector calls without allowing a weak track to cause a storm."""


def detection_reason(index, timestamp, tracks, motion, last_index, last_timestamp, options):
    if last_index < 0:
        return 'initial'
    if motion['scene_cut']:
        return 'scene_cut'
    gap = (index - last_index) / options.frame_step
    if gap >= options.keyframe_interval:
        return 'scheduled'
    # Last *global detector run*, refreshing active tracks when prediction limit approaches.
    active = [t for t in tracks if not t.get('missed_detection', False)]
    if active and timestamp - last_timestamp >= options.max_prediction_seconds and gap >= options.min_detection_interval:
        return 'refresh_age'
    if gap < min(options.min_detection_interval, options.keyframe_interval):
        return None
    if not motion['reliable']:
        return 'camera_flow_unreliable'
    active = [t for t in tracks if not t.get('missed_detection', False)
              and timestamp - t['last_detection'] <= options.max_prediction_seconds]
    if active:
        missing = sum(t['id'] not in motion['local_boxes'] for t in active)
        if missing / len(active) >= options.weak_track_fraction:
            return 'many_weak_tracks'
    return None
