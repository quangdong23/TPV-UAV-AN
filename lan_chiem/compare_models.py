"""Compare compatible open-vocabulary YOLO models without changing production defaults."""
import argparse,json,os,subprocess,sys,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CANDIDATES=('yoloe-11s-seg','yoloe-11m-seg','yoloe-v8m-seg')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-only',action='store_true')
    parser.add_argument('--baseline-only',action='store_true')
    parser.add_argument('--include-baseline',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/model_comparison')
    parser.add_argument('--models',nargs='+',choices=('baseline',*CANDIDATES),default=list(CANDIDATES))
    args=parser.parse_args()
    models=ROOT/'models/comparison';models.mkdir(exist_ok=True)
    output=args.output.resolve()
    if not output.is_relative_to(ROOT):parser.error('Output must stay inside lan_chiem')
    output.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'PYTHONIOENCODING':'utf-8','YOLO_CONFIG_DIR':str(ROOT/'outputs/ultralytics_config')}
    def run(name,cmd):
        print('START',name,flush=True)
        with (output/f'{name}.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u',*cmd],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('DONE',name,flush=True)
    names=['baseline'] if args.baseline_only else (['baseline'] if args.include_baseline else [])+args.models
    for name in dict.fromkeys(names):
        source=models/f'{name}.pt';fixed=models/f'{name}-fixed.pt'
        if name=='baseline':source=fixed=ROOT/'models/yoloe-v8s-single.pt'
        if name=='baseline' and not fixed.is_file():
            parser.error('Baseline checkpoint missing; run python -m lan_chiem.export_single_yoloe first')
        if not source.exists() and (args.download_only or not fixed.exists()):
            url=f'https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}.pt'
            print('DOWNLOAD',url,flush=True)
            temporary=source.with_suffix('.download')
            try:
                with urllib.request.urlopen(url,timeout=45) as response,temporary.open('wb') as file:
                    import shutil
                    shutil.copyfileobj(response,file)
                temporary.replace(source)
            finally:
                temporary.unlink(missing_ok=True)
        if args.download_only:continue
        if not fixed.exists():
            run(name+'_export',['-m','lan_chiem.export_single_yoloe','--source-weights',str(source),'--output',str(fixed)])
        run(name+'_images',['-m','lan_chiem.benchmark_yoloe','--config',str(ROOT/'yoloe_single.yaml'),
            '--weights',str(fixed),'--output',str(output/name/'images'),'--sizes','1280','--repeats','3'])
        for i,path in enumerate(sorted((ROOT/'sample_data').glob('*.mp4')),1):
            run(name+f'_video_{i}',['-m','lan_chiem.video_predict','--input',str(path),
                '--output',str(output/name/f'video_{i}'),'--weights',str(fixed),'--device','cuda'])
    return 0


if __name__=='__main__':raise SystemExit(main())
