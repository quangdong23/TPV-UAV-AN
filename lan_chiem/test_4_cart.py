import torch
from PIL import Image
import json
import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from lan_chiem.detector import GroundingDINODetector

d = GroundingDINODetector(device='cuda')
im4 = Image.open('lan_chiem/sample_data/4.png')

# Let's test querying class 6 on 4.png
res6 = d.predict_single_prompt(im4, "street food cart . wheeled food vending cart . mobile food vending cart . fruit food stand . food cart", box_threshold=0.15)
print("=== CLASS 6 PROMPTS ON 4.png ===")
for det in res6['detections']:
    print(f"score={det['score']}, box={det['bbox_xyxy']}, raw='{det['raw_label']}'")

# Let's test querying class 10 on 4.png
res10 = d.predict_single_prompt(im4, "basket of fruit . crate of vegetables . fruit basket . packaged products", box_threshold=0.15)
print("=== CLASS 10 PROMPTS ON 4.png ===")
for det in res10['detections']:
    print(f"score={det['score']}, box={det['bbox_xyxy']}, raw='{det['raw_label']}'")
