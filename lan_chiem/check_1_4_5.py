import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

for num in ['1', '4', '5']:
    path = f'lan_chiem/outputs/{num}_result.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    dets = data.get('object_detections', [])
    print(f"\n=================== IMAGE {num}.png ({len(dets)} detections) ===================")
    by_c = {}
    for d in dets:
        by_c.setdefault(f"{d['class_id']}: {d['class_name_vi']}", []).append(d)
    for cname, dlist in sorted(by_c.items()):
        print(f"  {cname} ({len(dlist)}):")
        for d in sorted(dlist, key=lambda x: -x['score']):
            b = [round(x, 1) for x in d['bbox_xyxy']]
            bw = round(b[2] - b[0], 1)
            bh = round(b[3] - b[1], 1)
            print(f"    - [{d['id']}] score={d['score']:.2f}, bbox={b}, size={bw}x{bh}, raw='{d['raw_label']}'")
