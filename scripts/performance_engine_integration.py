"""Real local CPU model + OCR through the worker; never allocates GPU weights.

Use an isolated data root and synthetic input. This is an execution check,
not a GPU throughput benchmark or an independent translation evaluation.
"""
import argparse,copy,io,json,os,subprocess,sys
from pathlib import Path
import pymupdf as fitz
from PIL import Image,ImageDraw,ImageFont
from papertranslator.jobs import create_job
from papertranslator.storage import atomic_json,read_json,root,sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--exe',type=Path);args=parser.parse_args()
    project=Path(__file__).resolve().parents[1];work=project/'work/performance-engine';work.mkdir(parents=True,exist_ok=True)
    registry=copy.deepcopy(read_json(root()/'model_registry.json'));model=registry.get('models',{}).get('hy7b') or registry.get('model')
    if not model:raise ValueError('Install a local Hy-MT2 model before this optional integration check.')
    model['options']={**model.get('options',{}),'gpu_layers':0,'threads':4,'context':4096,'parallel':1}
    registry.update(models=dict(hy7b=model),model=model,default_key='hy7b')
    home=work/'home';home.mkdir(exist_ok=True);atomic_json(home/'model_registry.json',registry);os.environ['PAPERTRANSLATOR_HOME']=str(home)
    source=work/'Synthetic CPU performance study.pdf'
    with fitz.open() as doc:
        page=doc.new_page(width=600,height=800)
        page.insert_text((50,150),'Research methods',fontsize=14)
        page.insert_text((50,250),'The study included 24 participants.',fontsize=14)
        image=Image.new('RGB',(1200,1600),'white');draw=ImageDraw.Draw(image);font=ImageFont.truetype('arial.ttf',36)
        draw.text((100,300),'Research methods',font=font,fill='black')
        draw.text((100,500),'The study included 24 participants.',font=font,fill='black')
        stream=io.BytesIO();image.save(stream,format='PNG');page=doc.new_page(width=600,height=800);page.insert_image(page.rect,stream=stream.getvalue());doc.save(source)
    results=[];targets=[]
    for workers in [1,2]:
        folder=create_job(source,project/'test_vault',title='CPU 실행 파일 성능 경로 합성 검사',model_key='hy7b',quality_mode='contextual',inference_parallel=workers)
        command=([str(args.exe.resolve())] if args.exe else [sys.executable,'-m','papertranslator'])+['--worker',str(folder)]
        subprocess.run(command,check=True,timeout=240)
        s=read_json(folder/'state.json');assert s['status']=='completed_with_warnings',s
        q=read_json(Path(s['output'])/'Quality.json');assert q['failed_paragraphs']==0 and q['ocr_pages']==1 and q['layout']['validation']['status']=='passed'
        assert sha256(source)==sha256(Path(s['output'])/'Original.pdf')
        assert s['inference_slots']==workers
        assert (folder/'checkpoint.sqlite3').exists()
        times={f.name:f.stat().st_mtime_ns for f in (folder/'responses').glob('*.json')}
        subprocess.run(command,check=True,timeout=90)
        assert times=={f.name:f.stat().st_mtime_ns for f in (folder/'responses').glob('*.json')}
        assert read_json(folder/'state.json')['cache_hits']==len(q['records'])
        assert read_json(folder/'state.json')['analysis_cache_hit']
        targets.append([r['target'] for r in q['records']])
        results.append(dict(slots=workers,failed=0,ocr_pages=1,output_pages=q['layout']['validation']['all_output_pages_checked'],duplicate_hits=sum(bool(r.get('translation_cache')) for r in q['records']),replay_without_generation=True))
    result=dict(status='passed',frozen=bool(args.exe),gpu_layers=0,actual_local_model=True,runs=results,exact_target_match_between_slot_counts=targets[0]==targets[1],scope='CPU execution only; GPU throughput and semantic quality not certified',human_approved=False)
    atomic_json(work/('frozen-result.json' if args.exe else 'source-result.json'),result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
