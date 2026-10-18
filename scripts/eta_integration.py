"""CPU timing simulation through real jobs/checkpoint/export; no GPU required.

The backend is a deterministic test double. This checks ETA integration, not
model speed, OCR accuracy or translation quality. Only synthetic PDFs are used.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pymupdf as fitz

from papertranslator import jobs
from papertranslator.eta import work_units
from papertranslator.storage import atomic_json, read_json, sha256


def main():
    project=Path(__file__).resolve().parents[1]
    work=project/'work/eta/integration';work.mkdir(parents=True,exist_ok=True)
    home=work/'home';home.mkdir(exist_ok=True)
    vault=project/'test_vault';vault.mkdir(exist_ok=True)
    os.environ['PAPERTRANSLATOR_HOME']=str(home)
    model=dict(id='synthetic-timing-backend',key='synthetic',adapter='hy',path='unused',sha256='synthetic')
    atomic_json(home/'model_registry.json',dict(model=model,runtime_release='synthetic-clock'))
    source=work/'Synthetic ETA Study.pdf'
    sentence='An association was observed in the study. '
    with fitz.open() as doc:
        for repeat in [1,2,4,6,10,12,8,3]:
            page=doc.new_page()
            page.insert_textbox(fitz.Rect(45,65,550,750),sentence*repeat,fontsize=12)
        doc.save(source)
    clock=[0.];calls=[0];starts=[0];snapshots=[]
    class TimingBackend:
        options=dict(context=4096)
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def start(self,progress):starts[0]+=1;progress('합성 시험 모델 준비')
        def translate(self,prompt,**kwargs):
            text=prompt.rsplit('\n\n',1)[-1];calls[0]+=1
            seconds=work_units(text)*.008
            if calls[0]==4:
                clock[0]+=60
                raise RuntimeError('Synthetic one-off delay for bounded retry')
            clock[0]+=seconds
            return dict(text='연구에서 연관성이 관찰되었다. '*text.count('An association'),seconds=seconds)
    original_save=jobs.atomic_json
    def capture(path,value):
        original_save(path,value)
        if Path(path).name=='state.json':snapshots.append(json.loads(json.dumps(value)))
    jobs.time=SimpleNamespace(monotonic=lambda:clock[0])
    jobs.LlamaBackend=TimingBackend;jobs.atomic_json=capture
    folder=jobs.create_job(source,vault,title='ETA 합성 통합 시험',ocr_mode='off')
    jobs.run_job(folder)
    state=read_json(folder/'state.json')
    assert state['status']=='completed_with_warnings',state.get('error')
    records=list(read_json(folder/'checkpoint.json')['records'].values())
    assert len(records)==8 and all(r['status']=='translated' for r in records)
    retry=next(r for r in records if r['attempts']==2)
    assert retry['wall_seconds']>=60 and retry['wall_seconds']>retry['seconds']
    ready=[s for s in snapshots if (s.get('eta') or {}).get('state')=='ready']
    assert ready and any((s.get('eta') or {}).get('state')=='calibrating' for s in snapshots)
    assert all(s['eta']['lower_seconds']<=s['eta']['seconds']<=s['eta']['upper_seconds'] for s in ready)
    assert all(s['eta']['at_elapsed_seconds']<=s['elapsed_seconds']+.01 for s in ready)
    assert sha256(Path(state['output'])/'Original.pdf')==sha256(source)
    calls_before=calls[0];first_output=state['output']
    jobs.run_job(folder)
    resumed=read_json(folder/'state.json')
    assert resumed['cache_hits']==8 and calls[0]==calls_before and starts[0]==1
    assert resumed['output']!=first_output
    result=dict(status='passed',backend='deterministic CPU test double',paragraphs=8,
                ready_snapshots=len(ready),retry_wall_time_captured=True,cache_reuse=True,source_hash_equal=True)
    atomic_json(work/'result.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
