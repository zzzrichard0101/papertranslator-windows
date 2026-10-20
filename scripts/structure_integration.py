"""Full synthetic worker: prose table, two columns, local repair and replay."""
import json
import os
import re
from pathlib import Path

import pymupdf as fitz

from papertranslator import jobs
from papertranslator.checkpoint import read_checkpoint
from papertranslator.storage import atomic_json,read_json,sha256

SOURCE=('Participants completed drawing tasks (Example, 2021). '
        'The observed value was -2.4 (Sample, 2022). '
        'The team discussed the result without claiming causation (Writer, 2023).')
DRAFT=('참가자들은 그림 과제를 마쳤다 (Example, 2021). '
       '관측값은 2.4였다 (Sample, 2022). '
       '연구진은 인과관계를 주장하지 않고 결과를 논의했다 (Writer, 2023).')


def main():
    project=Path(__file__).resolve().parents[1];work=project/'work/structure-integration';work.mkdir(exist_ok=True)
    home=work/'home';home.mkdir(exist_ok=True);os.environ['PAPERTRANSLATOR_HOME']=str(home)
    atomic_json(home/'model_registry.json',dict(runtime_release='synthetic',model=dict(
        id='synthetic',adapter='qwen',path='unused',sha256='synthetic')))
    source=work/'Synthetic structure study.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        for point,text in [((50,80),'Table 1. Study strategies'),((50,110),'STRATEGY'),((280,110),'EXAMPLE OF OUTCOME')]:
            p.insert_text(point,text,fontsize=11)
        for y,left in [(150,'Treatment A'),(250,'Treatment B'),(350,'Treatment intensity')]:
            p.insert_text((50,y),left,fontsize=11);p.insert_text((56,y+16),'and timing',fontsize=11)
            p.insert_text((280,y),'Does treatment improve outcomes?',fontsize=11)
            p.insert_text((286,y+16),'The same question continues here.',fontsize=11)
        p.draw_line((48,410),(555,410));p.insert_text((50,460),'Body text outside the table stays separate.')
        p=doc.new_page(width=600,height=800)
        for row in range(8):
            p.insert_text((40,150+row*15),f'Left column sentence number {row}.',fontsize=10)
            p.insert_text((330,150+row*15),f'Right column sentence number {row}.',fontsize=10)
        p=doc.new_page(width=600,height=800);p.insert_textbox(fitz.Rect(50,150,550,300),SOURCE,fontsize=11)
        doc.save(source)
    calls=[]
    class Backend:
        options=dict(context=4096)
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def start(self,*args):pass
        def translate(self,prompt,**kwargs):
            calls.append(prompt);text=prompt.split('\n\n')[-1]
            if prompt.startswith('Translate only this source sentence'):
                target='관측값은 -2.4였다 (Sample, 2022).'
            elif 'Participants completed' in text:target=DRAFT
            elif 'column sentence' in text:
                word='왼쪽' if 'Left' in text else '오른쪽'
                target=' '.join(f'{word} 열의 연구 문장 {n}.' for n in re.findall(r'\d+',text))
            elif text.startswith('Table'):target='표 1. 연구 전략'
            elif 'Treatment' in text:target='치료 전략과 시기.'
            elif 'Does treatment' in text:target='치료는 결과를 개선하는가? 같은 질문이 여기에서 이어진다.'
            elif 'outside' in text:target='표 밖의 본문은 별도로 유지된다.'
            elif text=='STRATEGY':target='전략'
            elif text=='EXAMPLE OF OUTCOME':target='결과 사례'
            else:raise AssertionError(text)
            literals=re.findall(r'\([A-Z][a-z]+, 20\d{2}\)',target)
            tokens=list(dict.fromkeys(re.findall(r'\[PX[A-F0-9]{8}X\d+\]',prompt)))
            for literal,token in zip(literals,tokens):target=target.replace(literal,token)
            return dict(text=target,seconds=.01)
    jobs.LlamaBackend=Backend
    folder=jobs.create_job(source,project/'test_vault',title='표·2단·부분 교정 합성 검사',ocr_mode='off')
    jobs.run_job(folder);state=read_json(folder/'state.json')
    assert state['status']=='completed_with_warnings',state
    out=Path(state['output']);q=read_json(out/'Quality.json');ir=read_json(folder/'ir.json')
    assert q['failed_paragraphs']==0 and q['layout']['validation']['status']=='passed'
    assert ir['pages'][0]['table_regions'][0]['method']=='captioned_prose_columns'
    assert ir['pages'][1]['reading_columns']['columns']==2
    repaired=[r for r in q['records'] if r.get('sentence_repair',{}).get('status')=='accepted']
    assert len(repaired)==1 and repaired[0]['target']==DRAFT.replace('관측값은 2.4였다','관측값은 -2.4였다')
    assert sha256(source)==sha256(out/'Original.pdf')
    with fitz.open(source) as original,fitz.open(out/'Bilingual.pdf') as dual:
        assert len(dual)==3
        for a,b in zip(original,dual):assert a.get_pixmap().samples==b.get_pixmap(clip=a.rect).samples
        left=dual[1].search_for('왼쪽')[0];right=dual[1].search_for('오른쪽')[0]
        assert left.x0<900<right.x0
    count=len(calls);jobs.run_job(folder);assert len(calls)==count
    assert read_json(folder/'state.json')['analysis_cache_hit']
    # Revision 1 must continue with the 0.5.3 analysis identity and no repair flag.
    old=jobs.create_job(source,project/'test_vault',ocr_mode='off');s=read_json(old/'state.json')
    s['processing_revision']=1;atomic_json(old/'state.json',s)
    jobs.run_job(old);old_ir=read_json(old/'ir.json');first=read_json(old/'state.json')
    assert not old_ir['pages'][0]['table_regions'] and 'reading_columns' not in old_ir['pages'][1]
    assert not any(b.get('selective_repair') for b in old_ir['blocks'])
    before=len(calls);jobs.run_job(old)
    assert len(calls)==before and read_json(old/'state.json')['fingerprint']==first['fingerprint']
    result=dict(status='passed',backend='deterministic CPU test double',table_cells=8,
        reading_columns=2,source_pixels_equal=True,layout='passed',local_repairs=1,
        replay_without_model_calls=True,revision_1_policy_preserved=True,human_approved=False)
    atomic_json(work/'result.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
