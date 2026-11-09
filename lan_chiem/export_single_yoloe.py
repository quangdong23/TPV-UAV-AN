"""Export one fixed-vocabulary YOLOE checkpoint; no DINO or text encoder at runtime."""
import argparse
import json
from pathlib import Path
import torch
from lan_chiem.yoloe_detector import YOLOEDetector
from lan_chiem.video_io import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-weights',required=True,help='Explicit path to compatible source YOLOE weights')
    parser.add_argument('--config',default='lan_chiem/configs/object_yoloe.yaml')
    parser.add_argument('--output',type=Path,default=Path('lan_chiem/models/yoloe-v8s-single.pt'))
    args=parser.parse_args()
    target=args.output.resolve()
    if not target.is_relative_to(Path(__file__).resolve().parent):
        parser.error('Output must stay inside lan_chiem')
    torch.set_num_threads(2)
    d=YOLOEDetector(weights=args.source_weights,device='cuda' if torch.cuda.is_available() else 'cpu',config_path=args.config)
    if Path(d.weights).resolve()==target:
        parser.error('Output must differ from source checkpoint')
    d.config['model']['reparameterize']=True
    d.load_model()
    d.engine.model.fuse(verbose=False)
    d.engine.model.single_vocab_sha256=d.vocabulary_sha256
    d.engine.model.fixed_vocabulary_export={'source_weights_sha256':d.revision,'catalog_version':d.provenance()['catalog_version'],
        'vocabulary_sha256':d.vocabulary_sha256,'fine_tuned':False,'detector_count':1}
    target.parent.mkdir(parents=True,exist_ok=True)
    d.engine.save(target)
    report={'checkpoint':str(target),'sha256':file_hash(target),'source':d.provenance(),'fine_tuned':False,
            'detector_count':1,'text_encoder_in_checkpoint':any('clip' in type(m).__name__.lower() for m in d.engine.model.modules()),
            'vocabulary':d.vocabulary}
    target.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('vocabulary','source')}),flush=True)


if __name__=='__main__':main()
