"""Synthetic CPU pipeline comparison; not a GPU speed or semantic accuracy claim."""
import argparse,json,os,re,threading,time
from pathlib import Path
import pymupdf as fitz
from papertranslator import jobs
from papertranslator.models import options_for
from papertranslator.checkpoint import read_checkpoint
from papertranslator.storage import atomic_json,read_json,sha256


def main():
    project=Path(__file__).resolve().parents[1];work=project/'work/performance-integration';work.mkdir(parents=True,exist_ok=True)
    home=work/'home';home.mkdir(exist_ok=True);os.environ['PAPERTRANSLATOR_HOME']=str(home)
    atomic_json(home/'model_registry.json',dict(runtime_release='synthetic',default_key='qwen38',models={
        'hy7b':dict(id='synthetic-auxiliary',adapter='hy',sha256='synthetic'),
        'qwen38':dict(id='synthetic-main-editor',adapter='qwen',sha256='synthetic')}))
    source=work/'Synthetic performance study.pdf'
    with fitz.open() as doc:
        for page_index in range(2):
            page=doc.new_page(width=600,height=800)
            for i in range(2):
                number=(page_index+1)*10+i+1
                text=f'The synthetic study number {number} describes how research evidence informs clinical judgement. It does not establish a causal relationship.'
                page.insert_textbox(fitz.Rect(50,120+i*130,550,220+i*130),text,fontsize=11)
            for y in [450,600]:page.insert_text((50,y),'Box 1.1 Synthetic observation',fontsize=11)
        doc.save(source)
    class SyntheticBackend:
        lock=threading.Lock();active=0;maximum=0;calls=0
        def __init__(self,folder,cancel,model=None,**kwargs):self.model=model;self.cancel=cancel;self.options=options_for(model)
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def start(self,progress):pass
        def check(self):
            if self.cancel():raise jobs.Cancelled()
        def translate(self,prompt,**kwargs):
            self.check()
            with self.lock:type(self).active+=1;type(self).maximum=max(type(self).maximum,type(self).active);type(self).calls+=1
            try:
                time.sleep(.025);self.check()
                if kwargs.get('json_schema'):
                    text=json.dumps(dict(decision='keep',source_evidence='',draft_evidence='',explanation_ko='합성 원문과 일치',revision=''),ensure_ascii=False)
                elif 'Box 1.1' in prompt.rsplit('\n\n',1)[-1]:text='상자 1.1 합성 관찰'
                else:
                    match=re.search(r'number (\d+)',prompt.rsplit('\n\n',1)[-1]);assert match
                    text=f'합성 연구 {match[1]}은 연구 근거가 임상적 판단에 도움이 되는 과정을 설명한다. 인과관계를 입증하지는 않는다.'
                return dict(text=text,seconds=.025)
            finally:
                with self.lock:type(self).active-=1
    jobs.LlamaBackend=SyntheticBackend
    outcomes=[];summary=[]
    for workers in [1,2]:
        SyntheticBackend.maximum=0;SyntheticBackend.calls=0
        folder=jobs.create_job(source,project/'test_vault',title='합성 성능 검사',quality_mode='ensemble',ocr_mode='off',inference_parallel=workers)
        jobs.run_job(folder);state=read_json(folder/'state.json');assert state['status']=='completed_with_warnings',state
        output=Path(state['output']);q=read_json(output/'Quality.json');assert q['failed_paragraphs']==0 and q['layout']['validation']['status']=='passed'
        assert q['refinement_processed']==4 and q['refinement_accepted']==0
        assert sha256(source)==sha256(output/'Original.pdf')
        outcomes.append([(r['id'],r['target'],r['status']) for r in q['records']])
        hits=sum(bool(r.get('translation_cache')) for r in q['records']);assert hits==3
        assert len(read_checkpoint(folder)['records'])==8
        with fitz.open(source) as src,fitz.open(output/'Bilingual.pdf') as dual:
            for i in range(2):assert src[i].get_pixmap().samples==dual[i].get_pixmap(clip=src[i].rect).samples
        first_calls=SyntheticBackend.calls;maximum=SyntheticBackend.maximum
        times={f.name:f.stat().st_mtime_ns for f in (folder/'responses').glob('*.json')}
        jobs.run_job(folder);assert SyntheticBackend.calls==first_calls
        assert read_json(folder/'state.json')['cache_hits']==8
        assert read_json(folder/'state.json')['analysis_cache_hit']
        assert times=={f.name:f.stat().st_mtime_ns for f in (folder/'responses').glob('*.json')}
        assert maximum==workers
        summary.append(dict(requested_slots=workers,max_simultaneous_requests=maximum,model_calls=first_calls,duplicate_hits=hits,replay_without_calls=True))
    assert outcomes[0]==outcomes[1]
    result=dict(status='passed',backend='deterministic CPU test doubles',identical_targets_and_statuses=True,source_pixels_equal=True,layout='passed',runs=summary,human_approved=False)
    atomic_json(work/'result.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
