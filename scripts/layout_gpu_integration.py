"""Actual worker: source regions, scanned cells, sideways labels and page links.

Synthetic input only; separate data root, no accuracy percentage or approval.
"""
import argparse,io,json,os,subprocess,sys
from pathlib import Path
import pymupdf as fitz
from PIL import Image,ImageDraw,ImageFont
from papertranslator.jobs import create_job
from papertranslator.storage import atomic_json,read_json,root,sha256


def sideways_diagram():
    image=Image.new('RGB',(1800,1200),'white');draw=ImageDraw.Draw(image);font=ImageFont.truetype('arial.ttf',32)
    labels=['Clinical\njudgement','Clinical\npractice','Patient\ncare',
            'Research\nevidence','Case series\nevaluation','Training\neducation']
    for i,label in enumerate(labels):
        x=[100,650,1200][i%3];y=150 if i<3 else 650
        draw.rectangle((x,y,x+300,y+130),outline='black',width=4)
        draw.multiline_text((x+20,y+20),label,font=font,fill='black')
    draw.line((400,215,650,215),fill='black',width=4)
    draw.polygon([(650,215),(632,205),(632,225)],fill='black')
    draw.line((950,215,1200,215),fill='black',width=4)
    return image.rotate(90,expand=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--exe',type=Path);args=parser.parse_args()
    project=Path(__file__).resolve().parents[1];work=project/'work/layout-gpu';work.mkdir(parents=True,exist_ok=True)
    registry=read_json(root()/'model_registry.json');home=work/'home';home.mkdir(exist_ok=True)
    atomic_json(home/'model_registry.json',registry);os.environ['PAPERTRANSLATOR_HOME']=str(home)
    source=work/'Synthetic positioned OCR study.pdf'
    with fitz.open() as doc:
        page=doc.new_page(width=600,height=800)
        page.insert_text((50,100),'The first paragraph describes the research.',fontsize=12)
        page.insert_text((50,710),'The study included 24 participants and',fontsize=12)
        page=doc.new_page(width=600,height=800)
        page.insert_text((50,90),'the treatment lasted two years.',fontsize=12)
        image=Image.new('RGB',(1200,1600),'white');draw=ImageDraw.Draw(image);font=ImageFont.truetype('arial.ttf',28)
        for x in [100,600,1100]:draw.line((x,400,x,850),fill='black',width=3)
        for y in [400,550,700,850]:draw.line((100,y,1100,y),fill='black',width=3)
        for x,y,text in [(140,455,'Group'),(640,455,'Count'),(140,605,'Treatment'),(640,605,'24'),(140,755,'Control'),(640,755,'24')]:
            draw.text((x,y),text,font=font,fill='black')
        stream=io.BytesIO();image.save(stream,format='PNG');page=doc.new_page(width=600,height=800)
        page.insert_image(page.rect,stream=stream.getvalue())
        stream=io.BytesIO();sideways_diagram().save(stream,format='PNG')
        page=doc.new_page(width=600,height=900);page.insert_image(page.rect,stream=stream.getvalue())
        doc.save(source)
    folder=create_job(source,project/'test_vault',title='위치 대응 OCR GPU 합성 검사',model_key='hy7b',ocr_mode='auto',quality_mode='ensemble',profile='art_therapy')
    command=([str(args.exe)] if args.exe else [sys.executable,'-m','papertranslator'])+['--worker',str(folder)]
    subprocess.run(command,check=True,timeout=300)
    state=read_json(folder/'state.json');assert state['status']=='completed_with_warnings',state.get('error')
    out=Path(state['output']);q=read_json(out/'Quality.json');layout=q['layout']
    assert q['failed_paragraphs']==0 and q['connected_paragraphs']==1
    assert layout['table_regions']==1 and not layout['unresolved_page_alignments']
    assert layout['validation']['status']=='passed',layout['validation']
    labels=[r for r in q['records'] if r['kind']=='diagram_label']
    assert len(labels)==6 and all(r['target'] and r['status']=='translated' for r in labels)
    assert {'Clinical judgement','Clinical practice','Patient care','Research evidence','Case series evaluation','Training education'}=={r['source'] for r in labels}
    assert not any(r['overflow'] for r in layout['regions'] if set(r['ids'])&{label['id'] for label in labels})
    connected=next(r for r in q['records'] if len(r['source_page_indices'])>1)
    alignment=connected['page_alignment'];assert alignment['status'] in {'aligned','proposed'}
    assert all(p['target']==connected['target'][p['start']:p['end']] for p in alignment['parts'])
    cells=[r for r in q['records'] if r['kind']=='table_cell'];assert len(cells)==6
    assert sum(bool(r.get('refinement')) for r in cells)==4
    assert all(r['target'] for r in cells) and sha256(source)==sha256(out/'Original.pdf')
    with fitz.open(source) as src,fitz.open(out/'Bilingual.pdf') as dual:
        for page,index in zip(dual,layout['source_indices']):
            assert page.get_pixmap(clip=src[index].rect).samples==src[index].get_pixmap().samples
        numeric=[r for r in cells if r['status']=='preserved' and r['source']=='24']
        assert len(numeric)==2
        page=dual[layout['source_indices'].index(2)]
        for r in numeric:
            box=fitz.Rect(r['source_fragments'][0]['cell_bbox'])+fitz.Rect(1,1,-1,-1)
            assert src[2].get_pixmap(clip=box).samples==page.get_pixmap(clip=box+fitz.Rect(600,0,600,0)).samples
    responses={p.name:p.stat().st_mtime_ns for p in (folder/'responses').glob('*.json')}
    subprocess.run(command,check=True,timeout=90)
    replay=read_json(folder/'state.json');assert replay['cache_hits']==len(q['records'])
    assert responses=={p.name:p.stat().st_mtime_ns for p in (folder/'responses').glob('*.json')}
    result=dict(status='passed',frozen=bool(args.exe),source_position=True,scanned_table_cells=len(cells),sideways_diagram_labels=len(labels),
                all_output_pages_checked=layout['validation']['all_output_pages_checked'],
                page_alignment=alignment['status'],shared_boundary=any(p['shared_boundary'] for p in alignment['parts']),
                target_unchanged=True,left_pixels_equal=True,source_hash_equal=True,human_approved=False,
                job=str(folder),output=str(out))
    atomic_json(work/('frozen-result.json' if args.exe else 'source-result.json'),result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
