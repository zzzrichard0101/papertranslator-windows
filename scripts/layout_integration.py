"""Synthetic full-job test of source-position pages, cells and alignment replay."""
import json,os
from pathlib import Path
import pymupdf as fitz
from papertranslator import jobs
from papertranslator.storage import atomic_json,read_json,sha256


def main():
    project=Path(__file__).resolve().parents[1];work=project/'work/layout-integration';work.mkdir(exist_ok=True)
    home=work/'home';home.mkdir(exist_ok=True);os.environ['PAPERTRANSLATOR_HOME']=str(home)
    atomic_json(home/'model_registry.json',dict(runtime_release='synthetic',model=dict(id='synthetic',adapter='hy',path='unused',sha256='synthetic')))
    source=work/'Synthetic positioned study.pdf'
    with fitz.open() as doc:
        page=doc.new_page(width=600,height=800);page.insert_text((50,100),'Alpha is the first paragraph.')
        page.insert_text((50,710),'The study included 24 participants and')
        page=doc.new_page(width=600,height=800);page.insert_text((50,90),'the treatment lasted two years.')
        for x in [50,250,450]:page.draw_line((x,200),(x,350))
        for y in [200,250,300,350]:page.draw_line((50,y),(450,y))
        for x,y,text in [(70,225,'Group'),(270,225,'Value'),(70,275,'Active'),(270,275,'3'),(70,325,'Control'),(270,325,'4')]:page.insert_text((x,y),text)
        doc.save(source)
    target='연구에는 24명이 참여했다. 치료는 2년간 지속되었다.';calls=[]
    translations={'The study included 24 participants and the treatment lasted two years.':target,
        'Alpha is the first paragraph.':'알파는 첫 번째 문단이다.','Group':'집단','Value':'값','Active':'실험군','Control':'대조군'}
    class Backend:
        options=dict(context=8192)
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def start(self,*args):pass
        def translate(self,prompt,**kwargs):
            if kwargs.get('json_schema'):
                calls.append('alignment');value=dict(boundaries=[dict(last_before=0,first_after=1,before_quote='24 participants',after_quote='two years',decision='matched')])
                return dict(text=json.dumps(value,ensure_ascii=False),seconds=.01)
            calls.append('translation')
            text=next(v for k,v in translations.items() if k in prompt)
            return dict(text=text,seconds=.01)
    jobs.LlamaBackend=Backend
    folder=jobs.create_job(source,project/'test_vault',title='원문 위치 대응 합성 통합 검사',ocr_mode='off')
    jobs.run_job(folder);state=read_json(folder/'state.json');assert state['status']=='completed_with_warnings',state.get('error')
    out=Path(state['output']);q=read_json(out/'Quality.json');assert q['failed_paragraphs']==0
    assert q['output_layout']=='source_position' and q['layout']['table_regions']==1
    assert not q['layout']['unresolved_page_alignments'] and calls.count('alignment')==1
    with fitz.open(out/'Bilingual.pdf') as doc:
        assert len(doc)==2
        a=doc[0].get_text(clip=fitz.Rect(600,0,1200,800));b=doc[1].get_text(clip=fitz.Rect(600,0,1200,800))
        assert '24명' in a and '2년' not in a and '2년' in b and '24명' not in b
        assert '실험군' in b and '대조군' in b
    before=len(calls);jobs.run_job(folder);assert len(calls)==before
    assert sha256(source)==sha256(Path(read_json(folder/'state.json')['output'])/'Original.pdf')
    result=dict(status='passed',backend='CPU test doubles',source_page_correspondence=True,ruled_table_cells=True,
                target_unchanged=True,alignment_replay_without_model_calls=True,human_approved=False)
    atomic_json(work/'result.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
