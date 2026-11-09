import torch
from PIL import Image
import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from lan_chiem.detector import GroundingDINODetector

d = GroundingDINODetector(device='cuda')
im5 = Image.open('lan_chiem/sample_data/5.png')

# Let's check the plant in image 5: box around [1468, 348, 1670, 513]
# If we run goods prompt vs plant prompt:
res_plant = d.predict_single_prompt(im5, "potted plant . green shrub . small tree . bush", box_threshold=0.15)
print("=== PLANT PROMPTS ON 5.png ===")
for det in res_plant['detections']:
    print(f"score={det['score']}, box={det['bbox_xyxy']}, raw='{det['raw_label']}'")

# Let's check canopy/awning in image 5: box around [1245, 112, 1712, 309]
res_awning = d.predict_single_prompt(im5, "storefront canopy . fabric awning . canopy . awning", box_threshold=0.15)
print("=== AWNING PROMPTS ON 5.png ===")
for det in res_awning['detections']:
    print(f"score={det['score']}, box={det['bbox_xyxy']}, raw='{det['raw_label']}'")
