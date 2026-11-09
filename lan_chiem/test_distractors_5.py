import torch
from PIL import Image
import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

from lan_chiem.detector import GroundingDINODetector

d = GroundingDINODetector(device='cuda')
im5 = Image.open('lan_chiem/sample_data/5.png')

# Goods prompt WITH distractors:
prompt = "basket of fruit . crate of vegetables . fruit basket . packaged products . potted plant . green shrub . bush . storefront canopy . fabric awning ."
res = d.predict_single_prompt(im5, prompt, box_threshold=0.20)
print("=== GOODS + DISTRACTORS ON 5.png ===")
for det in res['detections']:
    print(f"score={det['score']}, box={det['bbox_xyxy']}, raw='{det['raw_label']}'")
