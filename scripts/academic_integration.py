"""Synthetic academic worker: mixed columns, same-page continuation, replay."""
import json
import os
from pathlib import Path
import re

import pymupdf as fitz

from papertranslator import jobs
from papertranslator.storage import atomic_json,read_json,sha256


def main():
    project=Path(__file__).resolve().parents[1];work=project/'work/academic-integration';work.mkdir(exist_ok=True)
    home=work/'home';home.mkdir(exist_ok=True);os.environ['PAPERTRANSLATOR_HOME']=str(home)
    atomic_json(home/'model_registry.json',dict(runtime_release='synthetic',model=dict(id='synthetic',adapter='qwen',path='unused',sha256='synthetic')))
    source=work/'Synthetic academic columns.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        p.insert_text((65,65),'Full width scientific research title',fontsize=16)
        p.insert_textbox(fitz.Rect(40,95,560,145),'Fig. 1. This full width figure caption describes the research methods and the reported experimental results.',fontsize=10)
        p.draw_rect(fitz.Rect(40,150,560,180),fill=(.8,.9,.95))
        for row in range(9):
            for x,side in ((40,'Left'),(330,'Right')):
                p.insert_text((x,220+row*17),f'{side} research sentence explains the evidence {row}.',fontsize=9)
        p.insert_text((40,650),'The effect was observed during',fontsize=9)
        p.insert_text((330,200),'treatment, and the measurements continued.',fontsize=9)
        p.insert_text((40,500),'def update(total):\n    return total + 1',fontsize=9,fontname='cour')
        doc.save(source)
    calls=[]
    class Backend:
        options=dict(context=4096)
        def __init__(self,*a,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def start(self,*a):pass
        def translate(self,prompt,**kw):
            calls.append(prompt)
            if 'KOREAN_SENTENCES' in prompt:
                return dict(text=json.dumps(dict(boundaries=[dict(last_before=0,first_after=0,decision='matched')])),seconds=.01)
            text=prompt.split('\n\n')[-1]
            if 'The effect was observed' in text:target='치료 중 효과가 관찰됐으며 측정은 계속됐다.'
            elif 'research sentence' in text:
                side='왼쪽' if 'Left' in text else '오른쪽'
                target=' '.join(f'{side} 연구 문장은 근거를 설명한다 {n}.' for n in re.findall(r'\d+',text))
            elif text.startswith('Fig.'):target='그림 1. 이 그림은 연구 방법과 보고된 실험 결과를 설명한다.'
            elif 'scientific research title' in text:target='전체 폭 과학 연구 제목'
            else:raise AssertionError(text)
            return dict(text=target,seconds=.01)
    jobs.LlamaBackend=Backend
    folder=jobs.create_job(source,project/'test_vault',title='학술 2단 자동 인식 합성 검사',ocr_mode='off')
    jobs.run_job(folder);s=read_json(folder/'state.json');assert s['status']=='completed_with_warnings',s
    q=read_json(Path(s['output'])/'Quality.json');ir=read_json(folder/'ir.json')
    assert q['failed_paragraphs']==0 and q['layout']['validation']['status']=='passed'
    assert q['reading_column_pages']==1 and any(b.get('column_continuation') for b in ir['blocks'])
    joined=next(r for r in q['records'] if 'The effect was observed during treatment' in r['source'])
    assert len(joined['page_alignment']['parts'])==2 and all(p['shared_boundary'] for p in joined['page_alignment']['parts'])
    assert sha256(source)==sha256(Path(s['output'])/'Original.pdf')
    with fitz.open(source) as original,fitz.open(Path(s['output'])/'Bilingual.pdf') as dual:
        assert original[0].get_pixmap().samples==dual[0].get_pixmap(clip=original[0].rect).samples
        assert dual[0].search_for('왼쪽')[0].x0<900<dual[0].search_for('오른쪽')[0].x0
    before=len(calls);jobs.run_job(folder);assert len(calls)==before and read_json(folder/'state.json')['analysis_cache_hit']
    # Existing revision 2 jobs keep the old parser policy and checkpoint key.
    old=jobs.create_job(source,project/'test_vault',ocr_mode='off');state=read_json(old/'state.json')
    state['processing_revision']=2;atomic_json(old/'state.json',state);jobs.run_job(old)
    old_ir=read_json(old/'ir.json');first=read_json(old/'state.json')
    assert not any(b.get('column_continuation') for b in old_ir['blocks'])
    assert not any(p.get('reading_columns',{}).get('version','').startswith('academic') for p in old_ir['pages'])
    before=len(calls);jobs.run_job(old)
    assert len(calls)==before and read_json(old/'state.json')['fingerprint']==first['fingerprint']
    result=dict(status='passed',backend='deterministic CPU test double',automatic_columns=2,
        same_page_sentence_join=True,column_alignment_parts=2,source_pixels_equal=True,
        layout='passed',replay_without_model_calls=True,revision_2_policy_preserved=True,human_approved=False)
    atomic_json(work/'result.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
