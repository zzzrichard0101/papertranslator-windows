"""Synthetic worker regression: complete captions, references, footnotes, replay."""
import json
import os
from pathlib import Path
import pymupdf as fitz
from papertranslator import jobs
from papertranslator.storage import atomic_json,read_json,sha256


def main():
    project=Path(__file__).resolve().parents[1]
    work=project/'work/source-integrity-integration';work.mkdir(exist_ok=True)
    home=work/'home';home.mkdir(exist_ok=True);os.environ['PAPERTRANSLATOR_HOME']=str(home)
    atomic_json(home/'model_registry.json',dict(runtime_release='synthetic',model=dict(
        id='synthetic',adapter='qwen',path='unused',sha256='synthetic')))
    source=work/'Synthetic source integrity.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        p.insert_text((50,120),'Fig. 6. At 30K itera-\ntions, the artifacts are reduced.',fontsize=9)
        p.insert_text((50,200),'REFERENCES\nAlice Smith. 2020. Neural study.\nBob Jones. 2021. Another study.\nCarol King. 2022. Final study.',fontsize=10)
        p.insert_text((50,350),'APPENDIX A',fontsize=17)
        p.insert_text((50,390),'The experiment uses three samples.',fontsize=10)
        p.insert_text((50,710),'2',fontsize=6)
        p.insert_text((53.5,712),'We trained for 12 hours.',fontsize=8)
        doc.save(source)
    calls=[]
    class Backend:
        options=dict(context=8192,parallel=1)
        def __init__(self,*a,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def start(self,*a):pass
        def translate(self,prompt,**kw):
            calls.append(prompt)
            text=prompt.split('\n\n')[-1]
            if text.startswith('Fig. 6.'):
                assert '30K iterations' in text and text.endswith('reduced.')
                target='그림 6. 30,000회의 반복에서 인공적인 흔적이 줄어든다.'
            else:target={'REFERENCES':'참고문헌','APPENDIX A':'부록 A',
                'The experiment uses three samples.':'실험은 세 개의 표본을 사용한다.',
                'Research methods':'연구 방법',
                'We trained for 12 hours.':'12시간 동안 학습했다.'}[text]
            return dict(text=target,seconds=.01)
    jobs.LlamaBackend=Backend
    folder=jobs.create_job(source,project/'test_vault',ocr_mode='off',title='원문 입력 무결성 합성 검사')
    jobs.run_job(folder);state=read_json(folder/'state.json')
    assert state['status']=='completed_with_warnings',state
    q=read_json(Path(state['output'])/'Quality.json');ir=read_json(folder/'ir.json')
    assert q['failed_paragraphs']==0 and q['source_integrity']['status']=='passed'
    assert q['layout']['validation']['status']=='passed'
    assert any(r['source']=='2' and r['status']=='preserved' for r in q['records'])
    assert all('Alice Smith' not in prompt for prompt in calls)
    assert any(r['source']=='We trained for 12 hours.' and r['status']=='translated' for r in q['records'])
    assert sha256(source)==sha256(Path(state['output'])/'Original.pdf')
    before=len(calls);stamp=state['fingerprint'];jobs.run_job(folder)
    assert len(calls)==before and read_json(folder/'state.json')['fingerprint']==stamp
    # A pre-policy job must keep its old IR and checkpoint, without migration.
    legacy_source=work/'Synthetic legacy source.pdf'
    with fitz.open() as doc:
        doc.new_page().insert_text((50,120),'Research methods');doc.save(legacy_source)
    legacy=jobs.create_job(legacy_source,project/'test_vault',ocr_mode='off')
    saved=read_json(legacy/'state.json');saved.pop('source_integrity_policy');atomic_json(legacy/'state.json',saved)
    jobs.run_job(legacy);first=read_json(legacy/'state.json');legacy_calls=len(calls)
    assert first['status']=='completed_with_warnings' and 'source_integrity' not in read_json(legacy/'ir.json')
    jobs.run_job(legacy);second=read_json(legacy/'state.json')
    assert len(calls)==legacy_calls and first['fingerprint']==second['fingerprint']
    old=jobs.analyze(source,[0],ocr_mode='off',academic=True)
    assert 'source_integrity' not in old
    assert len([b for b in old['blocks'] if b['kind']=='caption'])==2
    result=dict(status='passed',backend='deterministic CPU double',complete_caption=True,
        bibliography_not_sent_to_model=True,footnote_body_translated=True,footnote_marker_retained=True,
        layout='passed',source_unchanged=True,replay_without_calls=True,old_job_policy_retained=True,model_calls=before)
    atomic_json(work/'result.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
