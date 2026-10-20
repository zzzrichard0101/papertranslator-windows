"""Synthetic full worker comparison of compact edits, fallback and replay."""
import json
import os
from pathlib import Path
import re

import pymupdf as fitz

from papertranslator import jobs
from papertranslator.acceleration import MTP_MODEL_SHA256
from papertranslator.models import options_for
from papertranslator.storage import atomic_json,read_json,sha256

PREFIX='The observational study describes the materials and procedures without adding causal claims. '
KOREAN_PREFIX='이 관찰 연구는 인과적 주장을 추가하지 않고 자료와 절차를 설명한다. '


def main():
    project=Path(__file__).resolve().parents[1];work=project/'work/acceleration-integration';work.mkdir(exist_ok=True)
    home=work/'home';home.mkdir(exist_ok=True);os.environ['PAPERTRANSLATOR_HOME']=str(home)
    atomic_json(home/'model_registry.json',dict(runtime_release='b11718',default_key='hy7b',models={
        'hy7b':dict(id='synthetic-translator',adapter='hy',path='unused',sha256='synthetic'),
        'qwen38':dict(id='synthetic-editor',adapter='qwen',path='unused',sha256=MTP_MODEL_SHA256)}))
    source=work/'Synthetic compact review.pdf'
    with fitz.open() as doc:
        for number in [24,25,26]:
            first='Research evidence informs clinical judgement.' if number==26 else 'Research evidence does not replace clinical judgement.'
            page=doc.new_page(width=600,height=800)
            page.insert_textbox(fitz.Rect(50,150,550,700),first+' '+PREFIX*7+f'The cohort included {number} participants.',fontsize=11)
        doc.save(source)
    calls=[]
    class Backend:
        def __init__(self,*args,model=None,**kwargs):self.model=model;self.options=options_for(model)
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def start(self,*args):
            # Simulate a reported MTP startup fallback, not acceleration.
            if self.options.get('speculation')=='mtp':self.options['speculation']='none'
        def translate(self,prompt,**kwargs):
            calls.append(self.model['id'])
            supplied=json.loads(prompt.split('\n')[-1])['SOURCE'] if kwargs.get('json_schema') else prompt.rsplit('\n\n',1)[-1]
            number=int(re.search(r'included (24|25|26)',supplied)[1])
            first='연구 근거는 임상 판단에 도움이 된다.' if number==26 else '연구 근거는 임상 판단을 대체한다.'
            draft=first+' '+KOREAN_PREFIX*7+f'참가자는 {number}명이다.'
            if not kwargs.get('json_schema'):return dict(text=draft,seconds=.01)
            if 'patches' in kwargs['json_schema']['properties']:
                if number==26:value=dict(decision='keep',patches=[],explanation_ko='')
                else:value=dict(decision='patch',patches=[dict(source_evidence='does not replace',before='없는 구절' if number==25 else '대체한다',after='대체하지 않는다')],explanation_ko='부정 복원')
            else:
                value=dict(decision='keep',source_evidence='',draft_evidence='',explanation_ko='',revision='') if number==26 else dict(
                    decision='revise',source_evidence='does not replace',draft_evidence='대체한다',explanation_ko='부정 복원',revision=draft.replace('대체한다','대체하지 않는다'))
            return dict(text=json.dumps(value,ensure_ascii=False),seconds=.01)
    jobs.LlamaBackend=Backend;targets=[];summaries=[]
    for compact,workers in [(False,1),(True,1),(True,2)]:
        folder=jobs.create_job(source,project/'test_vault',title='가속 설정·부분 교정 합성 검사',quality_mode='ensemble',
            ocr_mode='off',compact_editor=compact,inference_parallel=workers,inference_acceleration='mtp' if compact else 'none')
        jobs.run_job(folder);state=read_json(folder/'state.json')
        assert state['status']=='completed_with_warnings',state
        out=Path(state['output']);q=read_json(out/'Quality.json')
        assert q['failed_paragraphs']==0 and q['refinement_processed']==3 and q['refinement_accepted']==2
        assert q['layout']['validation']['status']=='passed' and sha256(source)==sha256(out/'Original.pdf')
        targets.append([r['target'] for r in q['records']])
        if compact:
            assert q['compact_edit_attempted_paragraphs']==3 and q['compact_edit_accepted_paragraphs']==2 and q['compact_edit_fallback_paragraphs']==1
            assert all(r['refinement']['inference_speculation']=='none' for r in q['records'])
        with fitz.open(source) as src,fitz.open(out/'Bilingual.pdf') as dual:
            assert len(dual)==3
            for a,b in zip(src,dual):assert a.get_pixmap().samples==b.get_pixmap(clip=a.rect).samples
        before=len(calls);files={p.name:p.stat().st_mtime_ns for p in (folder/'responses').glob('*.json')}
        jobs.run_job(folder);assert len(calls)==before
        assert files=={p.name:p.stat().st_mtime_ns for p in (folder/'responses').glob('*.json')}
        summaries.append(dict(compact=compact,slots=workers,all_three_paragraphs_reviewed=True,replay_without_calls=True))
    assert targets[0]==targets[1]==targets[2]
    result=dict(status='passed',backend='deterministic CPU test doubles',exact_final_targets_equal=True,
        source_pixels_equal=True,layout='passed',compact_fallback_validated=True,mtp_fallback_reported=True,
        runs=summaries,human_approved=False)
    atomic_json(work/'result.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
