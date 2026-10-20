"""CPU worker regression: body references, footnotes and column/page boundaries."""
import json,os,re
from pathlib import Path
import pymupdf as fitz
from papertranslator import jobs
from papertranslator.storage import read_json,atomic_json,sha256


def main():
    project=Path(__file__).resolve().parents[1];work=project/'work/continuity-integration';work.mkdir(exist_ok=True)
    home=work/'home';home.mkdir(exist_ok=True);os.environ['PAPERTRANSLATOR_HOME']=str(home)
    atomic_json(home/'model_registry.json',dict(runtime_release='synthetic',model=dict(id='synthetic',adapter='qwen',path='unused',sha256='synthetic')))
    source=work/'Synthetic body boundaries.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        for x,side in [(40,'Left'),(330,'Right')]:
            for n in range(9):p.insert_text((x,160+n*17),f'{side} research sentence explains the evidence {n}.',fontsize=9)
        p.insert_text((40,630),'The results are reported in',fontsize=9)
        p.insert_text((40,642),'Table 7 using a common scale. Examples can',fontsize=9)
        p.insert_text((330,130),'be seen in a separate experiment.',fontsize=9)
        p.insert_text((330,650),'The observations continued for',fontsize=9)
        p.insert_text((40,700),'2',fontsize=6)
        p.insert_text((43.5,702),'We used 12 hours for calibration.',fontsize=8)
        p.insert_text((40,714),'The calibration was performed locally.',fontsize=8)
        p=doc.new_page(width=600,height=800)
        p.insert_text((40,100),'twelve months, including winter.',fontsize=10)
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
            if 'KOREAN_SENTENCES' in prompt:
                return dict(text=json.dumps(dict(boundaries=[dict(last_before=0,first_after=0,decision='matched')])),seconds=.01)
            text=prompt.split('\n\n')[-1]
            if text.startswith('The results'):
                assert 'in Table 7' in text and 'Examples can be seen' in text
                target='결과는 공통 척도를 사용해 표 7에 보고되며, 예시는 별도의 실험에서 볼 수 있다.'
            elif text.startswith('The observations'):
                assert text.endswith('twelve months, including winter.')
                target='관찰은 겨울을 포함해 12개월간 계속됐다.'
            elif text.startswith('We used'):
                assert text.endswith('performed locally.')
                target='보정을 위해 12시간을 사용했으며 보정은 로컬에서 수행했다.'
            elif 'research sentence' in text:
                side='왼쪽' if text.startswith('Left') else '오른쪽'
                target=' '.join(f'{side} 연구 문장은 근거를 설명한다 {n}.' for n in re.findall(r'\d+',text))
            else:raise AssertionError(text)
            return dict(text=target,seconds=.01)
    jobs.LlamaBackend=Backend
    folder=jobs.create_job(source,project/'test_vault',title='본문 참조와 각주 문맥 연결 합성 검사',ocr_mode='off')
    jobs.run_job(folder);state=read_json(folder/'state.json');assert state['status']=='completed_with_warnings',state
    q=read_json(Path(state['output'])/'Quality.json')
    assert not q['failed_paragraphs'] and q['layout']['validation']['status']=='passed'
    body=next(r for r in q['records'] if r['source'].startswith('The results'))
    across=next(r for r in q['records'] if r['source'].startswith('The observations'))
    note=next(r for r in q['records'] if r['kind']=='footnote')
    assert len(body['source_fragments'])==2 and body['source_page_indices']==[0]
    assert across['source_page_indices']==[0,1] and 'calibration' not in across['source']
    assert note['status']=='translated' and note['source'].endswith('performed locally.')
    assert any(r['kind']=='footnote_marker' and r['source']=='2' and r['status']=='preserved' for r in q['records'])
    assert len(body['page_alignment']['parts'])==len(across['page_alignment']['parts'])==2
    assert sha256(source)==sha256(Path(state['output'])/'Original.pdf')
    before=len(calls);stamp=state['fingerprint'];jobs.run_job(folder)
    assert len(calls)==before and read_json(folder/'state.json')['fingerprint']==stamp
    result=dict(status='passed',backend='deterministic CPU double',body_reference_in_sentence=True,
        footnote_separately_translated=True,column_boundary_join=True,page_boundary_join=True,
        source_preserved=True,layout='passed',replay_without_model_calls=True,model_calls=before)
    atomic_json(work/'result.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
