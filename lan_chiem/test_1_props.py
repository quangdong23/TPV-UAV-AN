import torch
from PIL import Image
import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from lan_chiem.detector import GroundingDINODetector

d = GroundingDINODetector(device='cuda')
im1 = Image.open('lan_chiem/sample_data/1.png')

# Stroller and pillar check on 1.png
res_stroller = d.predict_single_prompt(im1, "baby stroller . stroller . pushchair . building column . concrete wall", box_threshold=0.15)
print("=== STROLLER / WALL ON 1.png ===")
for det in res_stroller['detections']:
    print(f"score={det['score']}, box={det['bbox_xyxy']}, raw='{det['raw_label']}'")

# Vending cart prompts on 1.png
res_cart = d.predict_single_prompt(im1, "street food cart . wheeled food vending cart . mobile food vending cart . fruit food stand", box_threshold=0.15)
print("=== VENDING CART ON 1.png ===")
for det in res_cart['detections']:
    print(f"score={det['score']}, box={det['bbox_xyxy']}, raw='{det['raw_label']}'")
