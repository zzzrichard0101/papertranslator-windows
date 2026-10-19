"""CPU test doubles through real jobs/refinement/checkpoint/PDF export.

This verifies stage integration and guards, not model or translation quality.
Only synthetic input and an isolated data root are used.
"""
import json
import os
from pathlib import Path

import pymupdf as fitz

from papertranslator import jobs
from papertranslator.storage import atomic_json,read_json,sha256


def main():
    project=Path(__file__).resolve().parents[1]
    work=project/'work/quality/integration';work.mkdir(parents=True,exist_ok=True)
    home=work/'home';home.mkdir(exist_ok=True)
    os.environ['PAPERTRANSLATOR_HOME']=str(home)
    vault=project/'test_vault';vault.mkdir(exist_ok=True)
    atomic_json(home/'model_registry.json',dict(runtime_release='synthetic',default_key='hy7b',models={
        'hy7b':dict(id='synthetic-translator',adapter='hy',path='unused',sha256='synthetic'),
        'qwen38':dict(id='synthetic-editor',adapter='qwen',path='unused',sha256='synthetic')}))
    source=work/'Synthetic Quality Study.pdf'
    text=('Research evidence does not replace clinical judgement. '
          'The source contains no claim that an association establishes causation.')
    with fitz.open() as doc:
        page=doc.new_page()
        page.insert_textbox(fitz.Rect(45,65,550,300),text,fontsize=12)
        doc.save(source)
    calls=[];stages=[]
    draft='연구 근거는 임상 판단을 대체한다. 원문에는 연관성이 인과관계를 확립한다는 주장이 없다.'
    target='연구 근거는 임상 판단을 대체하지 않는다. 원문에는 연관성이 인과관계를 확립한다는 주장이 없다.'
    class SyntheticBackend:
        options=dict(context=8192)
        def __init__(self,*args,model=None,**kwargs):self.model=model
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def start(self,progress):pass
        def translate(self,prompt,**kwargs):
            calls.append(self.model['id'])
            if self.model['id']=='synthetic-editor':
                return dict(text=json.dumps(dict(decision='revise',source_evidence='does not replace',
                    draft_evidence='대체한다',explanation_ko='부정 복원',revision=target),ensure_ascii=False),seconds=1)
            return dict(text=draft,seconds=1)
    original_save=jobs.atomic_json
    def capture(path,value):
        original_save(path,value)
        if Path(path).name=='state.json':stages.append(value['status'])
    jobs.LlamaBackend=SyntheticBackend;jobs.atomic_json=capture
    folder=jobs.create_job(source,vault,title='다중 모델 합성 통합 시험',quality_mode='ensemble',ocr_mode='off')
    jobs.run_job(folder)
    state=read_json(folder/'state.json')
    assert state['status']=='completed_with_warnings',state.get('error')
    assert 'refining' in stages and 'typesetting' in stages
    output=Path(state['output']);quality=read_json(output/'Quality.json')
    assert quality['refinement_processed']==1 and quality['refinement_accepted']==1
    assert quality['human_approved'] is False and quality['semantic_review']=='not_performed'
    assert quality['records'][0]['target']==target
    assert quality['records'][0]['refinement']['draft']==draft
    assert sha256(source)==sha256(output/'Original.pdf')
    assert (folder/'responses'/('edit-'+quality['records'][0]['id']+'.json')).is_file()
    with fitz.open(output/'Korean.pdf') as doc:
        assert '대체하지' in ''.join(p.get_text() for p in doc)
    before=len(calls);jobs.run_job(folder)
    assert len(calls)==before and read_json(folder/'state.json')['cache_hits']==1
    result=dict(status='passed',backend='deterministic CPU test doubles',refinement_accepted=1,
        resume_without_model_calls=True,human_approved=False,source_hash_equal=True)
    atomic_json(work/'result.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
