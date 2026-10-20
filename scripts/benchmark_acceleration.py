"""Opt-in local benchmark. GPU runs refuse to overlap another inference job.

Synthetic text only. Reports warm inference wall time separately from startup.
No settings, models, runtimes or user jobs are modified; no automatic promotion.
"""
import argparse
from contextlib import nullcontext
import copy
import json
import os
import re
from pathlib import Path
import threading
import time
import uuid

import psutil

from papertranslator.acceleration import apply,capable
from papertranslator.backend import LlamaBackend
from papertranslator.inference import completed
from papertranslator.models import resolve_model
from papertranslator.refinement import edit_record
from papertranslator.storage import atomic_json,read_json,root,worker_lock
from papertranslator.translation import translate_record

SOURCES=[
    'Research evidence does not replace clinical judgement.',
    'An association does not establish causation.',
    'The study included 24 participants. The observed change was -2.4.',
    'Treatment consisted of six weekly sessions, not six sessions per week.',
    'The participants may benefit, but the study does not prove effectiveness.',
    'The therapist asked the participant to describe the drawing without interpreting it.',
    'The authors reported an effect size of 0.42 and a confidence interval from 0.10 to 0.74.',
    'Qualitative interviews explored how participants understood the treatment process.',
]
PREFIX='The observational study describes the materials and procedures without adding causal claims. '
KOREAN_PREFIX='이 관찰 연구는 인과적 주장을 추가하지 않고 자료와 절차를 설명한다. '
EDITOR_CASES=[
    ('Research evidence does not replace clinical judgement.','연구 근거는 임상 판단을 대체한다.'),
    ('An association does not establish causation.','연관성은 인과관계를 입증한다.'),
    ('The observed change was -2.4.','관측된 변화는 2.4였다.'),
    ('Treatment consisted of six weekly sessions.','치료는 주 6회 진행되었다.'),
]


def gpu_idle():
    for process in psutil.process_iter(['name']):
        try:
            if (process.info['name'] or '').lower()=='llama-server.exe':
                raise RuntimeError('다른 추론 엔진이 실행 중입니다. 현재 작업을 유지하고 GPU 비교는 실행하지 않습니다.')
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue


def block(index,source):
    return dict(id=f'synthetic-{index}',source=source,kind='paragraph',source_page_index=0,
        source_page_indices=[0],source_fragments=[],bbox=[50,100,550,300])


def speculation_metrics(backend):
    response=backend.client.get('/metrics',timeout=10);response.raise_for_status()
    names=['spec_decode_num_draft_tokens_total','spec_decode_num_accepted_tokens_total','spec_decode_num_drafts_total']
    return {name:float(re.search(r'^llamacpp:'+name+r'\s+(\S+)',response.text,re.M)[1]) for name in names}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','gpu'],required=True)
    parser.add_argument('--stage',choices=['generation','editor'],default='generation')
    parser.add_argument('--model',default='qwen38');parser.add_argument('--limit',type=int,default=8)
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    if not 1<=args.limit<=8 or not 1<=args.repeats<=3:raise ValueError('비교 입력/반복 범위 오류')
    registry=read_json(root()/'model_registry.json');model=resolve_model(args.model,registry)
    project=Path(__file__).resolve().parents[1]
    work=project/'work/acceleration-benchmark'/uuid.uuid4().hex;work.mkdir(parents=True)
    # The GPU lock is acquired against the production data root before entering
    # an isolated root, so the app cannot start a second worker during the run.
    with worker_lock() if args.device=='gpu' else nullcontext():
        if args.device=='gpu':gpu_idle()
        home=work/'home';home.mkdir();atomic_json(home/'model_registry.json',registry)
        os.environ['PAPERTRANSLATOR_HOME']=str(home)
        variants=([('ordinary-1','none',1,False),('mtp-1','mtp',1,False),
                   ('ordinary-2','none',2,False),('mtp-2','mtp',2,False),
                   ('ordinary-1-repeat','none',1,False)] if args.stage=='generation' else
                  [('full','none',1,False),('patch','none',1,True),('mtp-patch','mtp',1,True)])
        runs=[];baseline=None
        for name,mode,workers,compact in variants:
            if args.device=='gpu':gpu_idle()
            folder=work/name;folder.mkdir()
            run=dict(name=name,requested_speculation=mode,requested_slots=workers)
            if mode=='mtp' and not capable(model,registry):
                runs.append({**run,'status':'unavailable','reason':'model/runtime metadata not qualified'});continue
            configured=apply(model,registry,mode)
            configured['options']={**configured.get('options',{}),'context':8192 if args.stage=='editor' else 4096,
                'parallel':workers,'gpu_layers':0 if args.device=='cpu' else 99,
                'threads':4 if args.device=='cpu' else 12,'temperature':0.,'seed':42,'benchmark_metrics':True}
            try:
                with LlamaBackend(folder,model=configured) as backend:
                    began=time.perf_counter();backend.start();load=time.perf_counter()-began
                    run.update(actual_slots=backend.options['parallel'],actual_speculation=backend.options.get('speculation','none'),startup_seconds=load)
                    if run['actual_slots']!=workers or run['actual_speculation']!=mode:
                        runs.append({**run,'status':'unavailable','reason':'engine fell back to ordinary or sequential mode'});continue
                    backend.translate('Translate into Korean. Output only the translation: Research methods.',max_tokens=100)
                    metrics_before=speculation_metrics(backend)
                    raw=[];lock=threading.Lock();original=backend.translate
                    def metered(*a,**kw):
                        result=original(*a,**kw)
                        with lock:raw.append(result)
                        return result
                    backend.translate=metered
                    items=list(range(min(args.limit,len(SOURCES) if args.stage=='generation' else len(EDITOR_CASES))))*args.repeats
                    indexed=list(enumerate(items));began=time.perf_counter()
                    def work_item(item):
                        unique,index=item
                        if args.stage=='generation':return translate_record(backend,block(unique,SOURCES[index]),'general',model.get('adapter','qwen'),folder)
                        source,draft=EDITOR_CASES[index];source=PREFIX*7+source;draft=KOREAN_PREFIX*7+draft
                        b=block(unique,source);record={**b,'status':'translated','target':draft,'candidate':draft,'issues':[]}
                        settings=dict(editor=configured)
                        if compact:settings['editor_output']='patch-v1'
                        return edit_record(backend,record,b,settings,name,folder,lambda:False)
                    results=[(item[0],value) for item,value in completed(indexed,work_item,workers,abort=backend.abort)]
                    wall=time.perf_counter()-began;results.sort()
                    metrics_after=speculation_metrics(backend)
                    speculation={k:metrics_after[k]-metrics_before[k] for k in metrics_after}
                    records=[r for _,r in results];atomic_json(folder/'records.json',records)
                    targets=[r['target'] for r in records]
                    if baseline is None:baseline=targets
                    from papertranslator.quality import check
                    from papertranslator.translation import blocking
                    failed=sum(r['status']=='failed' or blocking(check(r['source'],r['target'])) for r in records)
                    uncertain=sum(r.get('refinement',{}).get('decision')=='uncertain' for r in records)
                    run.update(status='completed',warm_wall_seconds=wall,calls=len(raw),failed=failed,uncertain=uncertain,
                        completion_tokens=sum((r.get('usage') or {}).get('completion_tokens') or 0 for r in raw),
                        speculation_metrics=speculation,mtp_observed=mode=='mtp' and speculation['spec_decode_num_draft_tokens_total']>0,
                        exact_targets_equal_baseline=targets==baseline,
                        compact_accepted=sum(r.get('refinement',{}).get('compact',{}).get('status')=='accepted' for r in records),
                        compact_fallbacks=sum(r.get('refinement',{}).get('compact',{}).get('status')=='fallback' for r in records))
                    atomic_json(folder/'raw-responses.json',raw)
                    if runs and runs[0].get('warm_wall_seconds'):run['wall_speedup_vs_first']=runs[0]['warm_wall_seconds']/wall
                    runs.append(run)
            except Exception as exc:runs.append({**run,'status':'error','reason':str(exc)[:300]})
            atomic_json(work/'result.json',dict(device=args.device,stage=args.stage,temperature=0.,runs=runs,
                human_approved=False,scope='Synthetic greedy decoding probe; no production-sampling or semantic quality certification; no automatic promotion'))
            print(json.dumps(runs[-1],ensure_ascii=False),flush=True)
    print('Result:',work/'result.json')


if __name__=='__main__':main()
