"""Synthetic worker: exact literal reuse, durable replay, versioned old jobs."""
import os,json,re
from pathlib import Path
import pymupdf as fitz
from papertranslator import jobs
from papertranslator.storage import atomic_json,read_json,sha256


def main():
    project=Path(__file__).resolve().parents[1];work=project/'work/efficiency-integration';work.mkdir(exist_ok=True)
    home=work/'home';home.mkdir(exist_ok=True);os.environ['PAPERTRANSLATOR_HOME']=str(home)
    model=dict(id='synthetic',adapter='qwen',path='unused',sha256='synthetic',options=dict(context=8192,temperature=.1))
    atomic_json(home/'model_registry.json',dict(runtime_release='synthetic',model=model))
    source=work/'Synthetic repeated captions.pdf'
    with fitz.open() as doc:
        for n in range(3):
            page=doc.new_page();page.insert_text((40,120),'Research findings [Miller et al. 2020].',fontsize=12)
            page.insert_text((40,180),'L = A + B',fontsize=12)
            page.insert_text((40,220),'Ours-30K',fontsize=12)
        doc.save(source)
    calls=[]
    class Backend:
        def __init__(self,*a,model=None,**kw):self.model=model;self.options=dict(context=8192,parallel=1)
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def start(self,*a):pass
        def translate(self,prompt,**kw):
            calls.append(prompt)
            match=re.search(r'\[PX[A-F0-9]+X0\]',prompt)
            return dict(text='연구 결과 '+match[0] if match else '연구 방법',seconds=.01,usage={})
    jobs.LlamaBackend=Backend
    folder=jobs.create_job(source,project/'test_vault',ocr_mode='off',bilingual_layout='reflow')
    jobs.run_job(folder);state=read_json(folder/'state.json')
    assert state['status']=='completed_with_warnings',state
    q=read_json(Path(state['output'])/'Quality.json')
    assert q['failed_paragraphs']==0 and len(calls)==1
    assert any(r.get('translation_cache') for r in q['records'])
    assert q['efficiency_policy']['validation']=='typed-v2'
    assert sha256(source)==sha256(Path(state['output'])/'Original.pdf')
    before=len(calls);stamp=state['fingerprint'];jobs.run_job(folder)
    assert len(calls)==before and read_json(folder/'state.json')['fingerprint']==stamp
    assert read_json(folder/'state.json')['analysis_cache_hit']
    # Replaying an old snapshot does not silently enable the new policy.
    old_source=work/'Synthetic legacy paper.pdf'
    with fitz.open() as doc:
        doc.new_page().insert_text((40,120),'Research methods',fontsize=12);doc.save(old_source)
    old=jobs.create_job(old_source,project/'test_vault',ocr_mode='off',bilingual_layout='reflow')
    saved=read_json(old/'state.json');saved.pop('efficiency_policy');saved['quality_policy'].pop('validation')
    atomic_json(old/'state.json',saved)
    jobs.run_job(old);first=read_json(old/'state.json');assert first['status']=='completed_with_warnings'
    old_calls=len(calls);jobs.run_job(old);second=read_json(old/'state.json')
    assert first['fingerprint']==second['fingerprint'] and len(calls)==old_calls
    assert 'efficiency_policy' not in second and second['analysis_cache_hit']
    assert all('validation_policy' not in b for b in read_json(old/'ir.json')['blocks'])
    result=dict(status='passed',backend='deterministic CPU double',model_calls=before,
        exact_context_cache=True,literals_preserved=True,durable_replay=True,old_policy_not_enabled=True)
    atomic_json(work/'result.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
