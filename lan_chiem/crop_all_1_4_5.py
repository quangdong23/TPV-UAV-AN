from PIL import Image, ImageDraw
import json
import os

crop_dir = "lan_chiem/outputs/inspect_1_4_5"
os.makedirs(crop_dir, exist_ok=True)

def inspect_image(num):
    im_path = f"lan_chiem/sample_data/{num}.png"
    json_path = f"lan_chiem/outputs/{num}_result.json"
    with open(json_path, "r", encoding="utf-8") as f:
        dets = json.load(f)["object_detections"]
    im = Image.open(im_path).convert("RGB")
    w, h = im.size
    
    for d in dets:
        b = d["bbox_xyxy"]
        pad = 20
        cb = [max(0, b[0] - pad), max(0, b[1] - pad), min(w, b[2] + pad), min(h, b[3] + pad)]
        c_im = im.crop(cb)
        draw = ImageDraw.Draw(c_im)
        draw.rectangle([b[0]-cb[0], b[1]-cb[1], b[2]-cb[0], b[3]-cb[1]], outline="red", width=2)
        cname = d["class_name_vi"].replace("/", "_").replace(" ", "_")
        c_im.save(f"{crop_dir}/{num}_{d['id']}_{d['class_id']}_{cname}.jpg")

inspect_image("1")
inspect_image("4")
inspect_image("5")
print("Cropped all detections for 1, 4, 5")
