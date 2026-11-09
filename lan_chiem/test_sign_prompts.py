import torch
from PIL import Image
import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from lan_chiem.detector import GroundingDINODetector

d = GroundingDINODetector(device='cuda')
im3 = Image.open('lan_chiem/sample_data/3.png')

# Let's test sign prompts on 3.png around the NHÀ NGHỈ sign: [790, 454, 850, 550]
prompts = [
    "shop sign . advertising signboard . signboard . hanging sign . advertising board",
    "shop sign . advertising signboard . signboard . motel sign . hanging signboard on tree",
    "signboard . shop sign . motel sign . advertising board",
    "shop sign . advertising signboard . motel sign . hanging sign",
]

for p in prompts:
    res = d.predict_single_prompt(im3, p, box_threshold=0.15)
    print(f"\nPrompt: {p}")
    for det in res['detections']:
        b = det['bbox_xyxy']
        if 750 < b[0] < 830 and 420 < b[1] < 490:
            print(f"  FOUND SIGN: score={det['score']}, box={b}, raw='{det['raw_label']}'")
