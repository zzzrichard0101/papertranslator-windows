from collections import Counter
import copy
import pymupdf as fitz

from papertranslator.column_layout import compose,Fonts,normalized,scientific_regions,label_color
from papertranslator.document import analyze


def make_source(path):
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        for x,side in [(40,'Left'),(330,'Right')]:
            for row in range(10):
                p.insert_text((x,150+row*17),f'{side} research sentence explains the evidence {row}.',fontsize=9)
        doc.save(path)


def test_two_column_faces_keep_all_text_in_source_column_on_continuations(tmp_path):
    source=tmp_path/'source.pdf';make_source(source)
    ir=analyze(source,[0],ocr_mode='off',academic=True)
    assert ir['pages'][0]['reading_columns']['columns']==2
    records={b['id']:{**b,'target':('왼쪽 번역문을 보존합니다. ' if 'Left' in b['source'] else '오른쪽 번역문을 보존합니다. ')*160,
                     'status':'preserved' if b['preserve'] else 'translated','issues':[]} for b in ir['blocks']}
    before=copy.deepcopy(records)
    layout=compose(ir,records,source,tmp_path/'dual.pdf',tmp_path/'ko.pdf')
    assert records==before and layout['validation']['status']=='passed'
    assert layout['overflow_pages']>0 and len(set(layout['source_indices']))==1
    with fitz.open(source) as src,fitz.open(tmp_path/'dual.pdf') as dual,fitz.open(tmp_path/'ko.pdf') as ko:
        for n in range(len(ko)):
            assert src[0].get_pixmap().samples==dual[n].get_pixmap(clip=src[0].rect).samples
            assert normalized(ko[n].get_text())==normalized(dual[n].get_text(clip=fitz.Rect(600,0,1200,822)))
            for word in ko[n].get_text('words'):
                if '왼쪽' in word[4]:assert word[2]<300
                if '오른쪽' in word[4]:assert word[0]>300
        actual=Counter(normalized(''.join(p.get_text() for p in ko)))
        wanted=Counter(normalized(''.join(r['target'] for r in records.values() if r['status']=='translated')))
        assert not wanted-actual


def test_math_unicode_survives_pdf_search_and_copy(tmp_path):
    output=tmp_path/'math.pdf';fonts=Fonts();expected={0:[]}
    with fitz.open() as doc:
        p=doc.new_page();fonts.draw(p,50,100,'회전 𝑞와 위치 𝑥, 불투명도 𝛼 − 1',10,(0,0,0),expected)
        fonts.finish(doc);doc.subset_fonts();doc.save(output)
    with fitz.open(output) as doc:
        assert '𝑞' in doc[0].get_text() and '𝛼' in doc[0].get_text()
        assert doc[0].search_for('𝑥')
    assert not fonts.missing


def test_algorithm_without_function_and_numeric_table_are_indivisible(tmp_path):
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        p.draw_line((40,100),(280,100));p.draw_line((40,120),(280,120));p.draw_line((40,230),(280,230))
        p.insert_text((40,110),'Algorithm 1 Optimization',fontsize=9)
        p.insert_text((50,145),'while not converged do\n    UpdateGaussian()\nend while',fontsize=9)
        table='Metric  SceneA  SceneB  SceneC\n'+'\n'.join(f'Ours {n}.123  {n}.456  {n}.789' for n in range(5))
        p.insert_text((330,160),table,fontsize=9)
        regions=scientific_regions(p)
        assert any(label=='알고리즘' and box.y1>=230 for box,label in regions)
        assert any(label=='숫자 표' for _,label in regions)


def test_label_contrast_distinguishes_white_panels_from_dark_photos():
    with fitz.open() as doc:
        p=doc.new_page();p.draw_rect(fitz.Rect(30,30,200,80),fill=(0,0,0))
        assert label_color(p,fitz.Rect(40,40,190,70))==(1,1,1)
        assert label_color(p,fitz.Rect(40,100,190,130))!=(1,1,1)


def test_short_label_never_overlaps_surviving_native_word(tmp_path):
    source=tmp_path/'labels.pdf';make_source(source)
    with fitz.open(source) as doc:
        p=doc[0];p.insert_text((40,450),'Optimization',fontsize=7)
        p.insert_text((40,458),'Continues',fontsize=7);doc.saveIncr()
    ir=analyze(source,[0],ocr_mode='off',academic=True)
    # The parser may group both lines. Give the renderer two independent
    # labels, as recovered diagram labels arrive in real mixed PDFs.
    combined=next(b for b in ir['blocks'] if 'Optimization' in b['source'])
    ir['blocks'].remove(combined)
    for n,line in enumerate(combined['layout_lines']):
        value=['Optimization','Continues'][n];b=copy.deepcopy(combined)
        b.update(id=combined['id']+str(n),source=value,raw_source=value,bbox=line['bbox'],kind='diagram_label',preserve=n==1,
                 layout_lines=[dict(start=0,end=len(value),bbox=line['bbox'])])
        fragment=b['source_fragments'][0]
        fragment.update(source=value,raw_source=value,bbox=line['bbox'],layout_lines=b['layout_lines'])
        ir['blocks'].append(b)
    records={b['id']:{**b,'target':'최적화를 계속합니다.' if 'Optimization' in b['source'] else '연구 문장의 번역.',
                     'status':'preserved' if b['preserve'] or 'Continues' in b['source'] else 'translated',
                     'issues':[]} for b in ir['blocks']}
    result=compose(ir,records,source,tmp_path/'dual.pdf',tmp_path/'ko.pdf')
    assert result['validation']['status']=='passed'
    with fitz.open(tmp_path/'ko.pdf') as doc:
        assert '최적화를계속합니다.' in normalized(''.join(p.get_text() for p in doc))


def test_mixed_single_column_scientific_page_does_not_drop_right_side_translation(tmp_path):
    source=tmp_path/'mixed.pdf';make_source(source)
    with fitz.open(source) as doc:
        p=doc.new_page(width=600,height=800)
        p.draw_line((330,100),(560,100));p.draw_line((330,120),(560,120));p.draw_line((330,230),(560,230))
        p.insert_text((330,110),'Algorithm 1 Optimization',fontsize=9)
        p.insert_text((340,145),'while not converged do\n    UpdateGaussian()\nend while',fontsize=9)
        doc.saveIncr()
    ir=analyze(source,[0,1],ocr_mode='off',academic=True)
    ir['pages'][1].pop('reading_columns',None)
    records={b['id']:{**b,'target':'단일열 오른쪽 알고리즘 번역.' if b['source_page_index']==1 else '연구 문장의 번역.',
                     'status':'translated','issues':[]} for b in ir['blocks']}
    result=compose(ir,records,source,tmp_path/'dual.pdf',tmp_path/'ko.pdf')
    assert result['validation']['status']=='passed'
    with fitz.open(tmp_path/'ko.pdf') as doc:
        assert '단일열오른쪽알고리즘번역.' in normalized(''.join(p.get_text() for p in doc))


def test_compact_heading_stays_above_preserved_bibliography(tmp_path):
    source=tmp_path/'bibliography.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        p.insert_text((40,90),'REFERENCES',fontsize=10)
        for x in [40,330]:
            for row in range(15):
                p.insert_text((x,104+row*12),f'Alice Smith. 2020. Academic study number {row}.',fontsize=8)
        doc.save(source)
    ir=analyze(source,[0],ocr_mode='off',academic=True,source_integrity=True)
    assert ir['pages'][0]['reading_columns']
    records={b['id']:{**b,'target':'참고문헌' if b['source']=='REFERENCES' else b['source'],
        'status':'preserved' if b['preserve'] else 'translated','issues':[]} for b in ir['blocks']}
    result=compose(ir,records,source,tmp_path/'dual.pdf',tmp_path/'ko.pdf')
    assert result['validation']['status']=='passed'
    with fitz.open(tmp_path/'ko.pdf') as doc:
        heading=doc[0].search_for('참고문헌')[0]
        first=doc[0].search_for('Alice Smith')[0]
        assert heading.y1<=first.y0 and heading.y0<100
