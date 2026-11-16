"""Phân loại crop người–xe bằng MobileNetV3 Small đã tinh chỉnh."""
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import models, transforms


def activity_crop_box(box, shape):
    """Tọa độ crop thật, gồm lề 10% và được chặn trong ảnh."""
    h, w = shape[:2]
    box = np.asarray(box, float)
    padding = .1 * (box[2:] - box[:2])
    start = np.maximum(0, np.floor(box[:2] - padding)).astype(int)
    end = np.minimum([w, h], np.ceil(box[2:] + padding)).astype(int)
    return np.r_[start, end].tolist()


class ActivityClassifier:
    def __init__(self, weights, device='cpu'):
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        self.classes = checkpoint['classes']
        if set(self.classes) != {'other_activity', 'repairing_vehicle', 'street_vending', 'washing_vehicle'}:
            raise ValueError('Checkpoint không đúng danh mục hành động')
        self.input_size = checkpoint.get('config', {}).get('preprocessing', {}).get('size', 224)
        if type(self.input_size) is not int or self.input_size < 64:
            raise ValueError('Kích thước đầu vào checkpoint phải là số nguyên >= 64')
        self.device = device
        self.model = models.mobilenet_v3_small(weights=None)
        self.model.classifier[3] = torch.nn.Linear(self.model.classifier[3].in_features, len(self.classes))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(device).eval()
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
        ])

    @torch.inference_mode()
    def predict(self, frame_bgr, box):
        coords = activity_crop_box(box, frame_bgr.shape)
        start, end = np.array(coords[:2]), np.array(coords[2:])
        if np.any(end - start < 32):
            return None
        crop = Image.fromarray(frame_bgr[start[1]:end[1], start[0]:end[0], ::-1])
        crop = ImageOps.contain(crop, (224, 224), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (224, 224), (114, 114, 114))
        canvas.paste(crop, ((224-crop.width)//2, (224-crop.height)//2))
        scores = self.model(self.transform(canvas)[None].to(self.device)).softmax(1)[0].cpu().tolist()
        order = np.argsort(scores)
        index = int(order[-1])
        return {'label': self.classes[index], 'confidence': scores[index],
                'margin': scores[index]-scores[int(order[-2])],
                'scores': dict(zip(self.classes, scores)),
                'crop_bbox_xyxy': [*start.tolist(), *end.tolist()]}
