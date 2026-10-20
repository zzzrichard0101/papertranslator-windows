"""GPU comparison of production edit/label/alignment session lifetimes.

Synthetic texts only; no user job/model/settings edits. Includes model hashing,
startup, inference and teardown. This is not a document-wide quality benchmark.
"""
import argparse
import copy
import json
from pathlib import Path
import time
import uuid

from benchmark_acceleration import gpu_idle
from papertranslator.acceleration import apply
from papertranslator.alignment import run as align
from papertranslator.backend import LlamaBackend
from papertranslator.label_fit import run as labels,fits
from papertranslator.models import resolve_model
from papertranslator.offline import install_loopback_guard
from papertranslator.refinement import run as refine
from papertranslator.sessions import Sessions
from papertranslator.storage import atomic_json,read_json,root,worker_lock
from papertranslator.translation import blocking
from papertranslator.quality import check


def fixtures():
    source=('The study enrolled 24 participants. The observed change was -2.4. '
            'This association does not establish causation, and the authors do not claim that treatment is effective for every person.')
    draft='연구에는 24명이 참여했다. 관측된 변화량은 2.4였다. 이 연관성은 인과관계를 입증하며, 저자들은 치료가 모든 사람에게 효과적이라고 주장한다.'
    def record(key,source,target,kind='paragraph',fragments=None):
        return dict(id=key,source=source,target=target,status='translated',kind=kind,source_page_index=0,
            source_page_indices=[0],bbox=[40,100,290,140],source_fragments=fragments or [],issues=[],context={})
    main=record('main',source,draft)
    parts=[dict(source_page_index=0,bbox=[40,200,290,230],source='The effect was observed during'),
           dict(source_page_index=1,bbox=[40,100,290,130],source='treatment, and measurements continued.')]
    boundary=record('boundary','The effect was observed during treatment, and measurements continued.',
                    '치료 중 효과가 관찰되었으며 측정은 계속됐다.',fragments=parts)
    boundary['source_page_indices']=[0,1]
    label=record('label','Gaussian position','가우시안의 위치를 나타내는 도식상의 표시','diagram_label',
                 [dict(source_page_index=0,bbox=[40,300,120,320],source='Gaussian position')])
    return main,boundary,label


def compare(folder,model,reuse):
    folder.mkdir(parents=True);metrics=dict(startups=[],requests=[])
    class Metered:
        def __init__(self,*a,**kw):self.backend=LlamaBackend(*a,**kw)
        def __getattr__(self,key):return getattr(self.backend,key)
        def __enter__(self):self.backend.__enter__();return self
        def __exit__(self,*a):return self.backend.__exit__(*a)
        def start(self,*a):
            began=time.perf_counter();self.backend.start(*a);metrics['startups'].append(time.perf_counter()-began)
        def translate(self,*a,**kw):
            raw=self.backend.translate(*a,**kw)
            metrics['requests'].append(dict(seconds=raw['seconds'],usage=raw.get('usage',{})))
            return raw
    pool=Sessions(Metered) if reuse else None;factory=pool or Metered
    main,boundary,label=fixtures();records={r['id']:r for r in [main,boundary]}
    ir=dict(blocks=copy.deepcopy(list(records.values())))
    settings=dict(mode='ensemble',version='session-comparison',editor=model,auxiliary=None)
    began=time.perf_counter()
    try:
        refine(records,ir,settings,folder,'general',factory,lambda:None,lambda _:None,lambda:False)
        records['label']=label
        labels(records,model,folder,factory,lambda:None,lambda _:None,lambda:False)
        align(records,model,folder,factory,lambda:None,lambda _:None,lambda:False)
    finally:
        if pool:pool.close()
    metrics['wall_seconds']=time.perf_counter()-began
    result=dict(mode='reused' if reuse else 'separate',**metrics,
        loads=len(metrics['startups']),checks=dict(
            main_guards=not blocking(check(main['source'],main['target'])),
            main_negative_value=bool(__import__('re').search(r'[-−]2\.4',main['target'])),
            main_negation=any(w in main['target'] for w in ['않','아니','없']),
            label_fits=fits(label),label_terms=all(w in label['target'] for w in ['가우시안','위치']),
            aligned=boundary.get('page_alignment',{}).get('status')=='aligned'),
        records=list(records.values()))
    atomic_json(folder/'result.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rounds',type=int,choices=[1,2],default=2)
    args=parser.parse_args();project=Path(__file__).resolve().parents[1]
    install_loopback_guard()
    folder=project/'work'/('efficiency-gpu-'+uuid.uuid4().hex[:8]);folder.mkdir()
    registry=read_json(root()/'model_registry.json');model=apply(resolve_model('qwen38'),registry,'mtp')
    model['options']={**model.get('options',{}),'context':8192,'temperature':0.,'parallel':1}
    results=[]
    with worker_lock():
        gpu_idle()
        for n in range(args.rounds):
            for reuse in ([False,True] if n==0 else [True,False]):
                result=compare(folder/f'round-{n}-{reuse}',model,reuse);results.append(result)
                print(json.dumps({k:result[k] for k in ['mode','loads','wall_seconds','checks']}),flush=True)
    output=dict(scope='synthetic production edit/label/alignment, including startup and teardown',
        sampling='greedy',rounds=args.rounds,model=model['id'],results=results,
        all_checks_passed=all(all(r['checks'].values()) for r in results),human_approved=False)
    atomic_json(folder/'result.json',output)
    print(json.dumps(dict(output=str(folder/'result.json'),all_checks_passed=output['all_checks_passed'])),flush=True)
    if not output['all_checks_passed']:raise SystemExit(1)


if __name__=='__main__':main()
