"""Small offline helpers; no torch, transformers or production model imports."""
import json
from pathlib import Path
import cv2
import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def sidewalk_raster(record,size=(480,260)):
    width,height=record['source_size']
    collection=record.get('sidewalk',{}).get('features')
    if collection is None:raise ValueError('Missing sidewalk geometry; cannot verify the combined pipeline')
    canvas=np.zeros((size[1],size[0]),np.uint8)
    for feature in collection.get('features',[]):
        if feature.get('properties',{}).get('class_id')!=2:continue
        geometry=feature['geometry']
        polygons=[geometry['coordinates']] if geometry['type']=='Polygon' else geometry['coordinates'] if geometry['type']=='MultiPolygon' else []
        for polygon in polygons:
            if not polygon:continue
            layer=np.zeros_like(canvas)
            for index,ring in enumerate(polygon):
                points=np.rint(np.asarray(ring)*[size[0]/width,size[1]/height]).astype(np.int32)
                if len(points)>=3:cv2.fillPoly(layer,[points],1 if index==0 else 0)
            canvas|=layer
    return canvas.astype(bool)

