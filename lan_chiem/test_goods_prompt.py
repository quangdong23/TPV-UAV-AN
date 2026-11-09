import torch
from PIL import Image
import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from lan_chiem.detector import GroundingDINODetector

d = GroundingDINODetector(device='cuda')

# Test on Image 5 (bush at [1468, 348, 1670, 513])
im5 = Image.open('lan_chiem/sample_data/5.png')
res5 = d.predict_single_prompt(im5, "fruit crate . vegetable box . fruit display . packaged products", box_threshold=0.20)
print("=== NEW GOODS PROMPT ON 5.png ===")
for det in res5['detections']:
    print(f"score={det['score']}, box={det['bbox_xyxy']}, raw='{det['raw_label']}'")

# Test on Image 7 (fruit crates at [946, 576] and [1024, 568])
im7 = Image.open('lan_chiem/sample_data/7.png')
res7 = d.predict_single_prompt(im7, "fruit crate . vegetable box . fruit display . packaged products", box_threshold=0.20)
print("=== NEW GOODS PROMPT ON 7.png ===")
for det in res7['detections']:
    print(f"score={det['score']}, box={det['bbox_xyxy']}, raw='{det['raw_label']}'")
