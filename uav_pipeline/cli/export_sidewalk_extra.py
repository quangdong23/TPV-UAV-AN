"""Build the fixed-vocabulary supplementary checkpoint for configured supplemental classes and counter prompts.

Run: python -m uav_pipeline.cli.export_sidewalk_extra
Keeps the original fine-tuned ten-class checkpoint intact.
"""
import json
from pathlib import Path
import torch
from ultralytics import YOLOE
from uav_pipeline.paths import PROJECT_ROOT
from uav_pipeline.config import load_config
from uav_pipeline.detection.embeddings import _compute_text_embeddings
from uav_pipeline.detection.classes import CATALOG_VERSION
from uav_pipeline.video.io import file_hash


def main():
    root=PROJECT_ROOT
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=root/'configs/detection.yaml')
    args=parser.parse_args()
    cfg=load_config(args.config)
    extension=cfg['model']['supplementary_detector']
    from uav_pipeline.detection.detector import YOLOEDetector
    vocabulary=YOLOEDetector.build_vocabulary(cfg['model']['target_phrases'],cfg['model']['counter_phrases'])
    names=[phrase for phrase,cid,_ in vocabulary
           if cid in extension['class_ids'] or phrase in extension.get('counter_phrases',[])]
    source=root/'models/yoloe-26x-seg.pt'
    target=root/extension['weights']
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    engine=YOLOE(str(source));engine.to(device)
    pe=_compute_text_embeddings(engine,names,device)
    engine.set_classes(names,pe)
    head=engine.model.model[-1];head.fuse(pe)
    for module in head.modules():
        for name,parameter in list(module._parameters.items()):
            if parameter is not None and torch.is_inference(parameter):
                module._parameters[name]=torch.nn.Parameter(parameter.clone(),requires_grad=False)
        for name,buffer in list(module._buffers.items()):
            if buffer is not None and torch.is_inference(buffer):
                module._buffers[name]=buffer.clone()
    engine.model.eval();engine.model.fuse(verbose=False)
    if hasattr(engine.model,'clip_model'):
        del engine.model.clip_model
    # Publish a complete checkpoint before replacing the previous file.
    temporary=target.with_name(target.stem+'.building.pt')
    engine.save(temporary)
    temporary.replace(target)
    digest=file_hash(target)
    target.with_suffix('.sha256').write_text(digest+'  '+target.name+'\n')
    target.with_suffix('.json').write_text(json.dumps({
        'catalog_version':CATALOG_VERSION,'sha256':digest,
        'source':str(source),'source_sha256':file_hash(source),
        'classes':extension['class_ids'],'names':names,
        'fine_tuned':False,'quality_verified':False},indent=2))
    print('Exported',target,'prompts=',len(names),flush=True)


if __name__=='__main__':
    main()
