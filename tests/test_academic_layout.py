from collections import Counter

import pymupdf as fitz
import pytest
from PIL import Image,ImageDraw,ImageFont
import io
import re

from papertranslator.alignment import page_sources,validate_boundaries
from papertranslator.document import analyze
from papertranslator.layout import bilingual_pdf,record_entries


def fixture(path,scan=False):
    with fitz.open() as doc:
        page=doc.new_page(width=600,height=800)
        page.insert_text((60,70),'A full width scientific research title',fontsize=16)
        page.insert_textbox(fitz.Rect(40,100,560,150),
            'Fig. 1. This full width figure caption explains the research findings and the experimental procedures. '
            'Its final short line must remain with its original caption.',fontsize=10)
        for row in range(10):
            for x,side in ((40,'Left'),(330,'Right')):
                page.insert_text((x,220+row*17),f'{side} research sentence explains the evidence {row}.',fontsize=9)
        page.insert_text((40,430),'def update(total):\n    return total + 1',fontname='cour',fontsize=9)
        page.insert_text((330,430),'Additional prose remains in the right column.',fontsize=9)
        if scan:
            image=page.get_pixmap(matrix=fitz.Matrix(2,2)).tobytes('png')
            doc.delete_page(0);page=doc.new_page(width=600,height=800);page.insert_image(page.rect,stream=image)
        doc.save(path)


@pytest.mark.parametrize('scan',[False,True])
def test_caption_and_code_do_not_disable_column_reading_or_lose_words(tmp_path,scan):
    source=tmp_path/'columns.pdf';fixture(source,scan)
    mode='auto' if scan else 'off'
    old=analyze(source,[0],ocr_mode=mode,structured=True)
    new=analyze(source,[0],ocr_mode=mode,structured=True,academic=True)
    assert new['pages'][0]['reading_columns']['columns']==2
    sources=[b['source'] for b in new['blocks']]
    left=[i for i,s in enumerate(sources) if 'Left research' in s]
    right=[i for i,s in enumerate(sources) if 'Right research' in s]
    assert left and right and max(left)<min(right)
    tokens=lambda ir:Counter(w for b in ir['blocks'] for w in b.get('raw_source',b['source']).split())
    assert tokens(old)==tokens(new)
    captions=[b for b in new['blocks'] if b['kind']=='caption']
    assert 'final short line' in ' '.join(b['source'] for b in captions)
    if not scan:
        code=next(b for b in new['blocks'] if 'def update' in b['source'])
        assert code['preserve'] and 'return total + 1' in code['source']


def test_neighbor_evidence_supports_unequal_columns_but_not_a_full_width_page(tmp_path):
    source=tmp_path/'mixed.pdf'
    with fitz.open() as doc:
        for index in range(3):
            p=doc.new_page(width=600,height=800)
            if index==2:
                for row in range(12):p.insert_text((40,160+row*17),'This full width sentence discusses the research evidence and the experimental procedures in detail.',fontsize=10)
                continue
            for row in range(12):
                for x,side in ((40,'Left'),(330,'Right')):
                    if index==1 and side=='Left' and row>1:continue
                    p.insert_text((x,160+row*17),f'{side} research sentence explains the evidence {row}.',fontsize=9)
        doc.save(source)
    ir=analyze(source,[0,1,2],ocr_mode='off',academic=True)
    assert ir['pages'][1]['reading_columns']['evidence']['document_supported']
    assert 'reading_columns' not in ir['pages'][2]
    skipped=analyze(source,[0,2],ocr_mode='off',academic=True)
    assert 'reading_columns' not in skipped['pages'][1]


def test_short_labels_and_single_column_code_do_not_become_prose_columns(tmp_path):
    source=tmp_path/'labels.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        for row in range(12):
            p.insert_text((40,140+row*20),f'Group {row}',fontsize=10)
            p.insert_text((280,140+row*20),'The clinical research explanation continues here.',fontsize=10)
        doc.save(source)
    assert 'reading_columns' not in analyze(source,[0],ocr_mode='off',academic=True)['pages'][0]


def test_unfinished_column_sentence_is_one_translation_with_separate_layout_regions(tmp_path):
    source=tmp_path/'continuation.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        for row in range(9):
            p.insert_text((40,140+row*17),f'Left research sentence explains the evidence {row}.',fontsize=9)
            p.insert_text((330,180+row*17),f'Right research sentence explains the evidence {row}.',fontsize=9)
        p.insert_text((40,650),'The effect was observed during',fontsize=9)
        p.insert_text((330,140),'treatment, and the measurements continued.',fontsize=9)
        doc.save(source)
    ir=analyze(source,[0],ocr_mode='off',academic=True)
    joined=next(b for b in ir['blocks'] if b.get('column_continuation'))
    assert joined['source']=='The effect was observed during treatment, and the measurements continued.'
    assert len(page_sources(joined))==2 and joined['source_page_indices']==[0]
    record={**joined,'target':'치료 중 효과가 관찰됐으며 측정은 계속됐다.','status':'translated','issues':[]}
    parts,status=validate_boundaries(dict(boundaries=[dict(last_before=0,first_after=0,
        before_quote='during',after_quote='treatment',decision='matched')]),record)
    record['page_alignment']=dict(parts=parts,status=status)
    entries,uncertain=record_entries(dict(pages=ir['pages'],blocks=[joined]),{joined['id']:record})
    assert not uncertain and len(entries[0])==2
    assert entries[0][0]['bbox'].x1<300<entries[0][1]['bbox'].x0
    assert all(e['shared_boundary'] for e in entries[0])


def test_translated_positions_cannot_expand_across_the_gutter(tmp_path):
    source=tmp_path/'layout.pdf';fixture(source)
    ir=analyze(source,[0],ocr_mode='off',academic=True)
    records={b['id']:{**b,'target':b['source'] if b['preserve'] else '연구 문장의 한국어 번역.',
                     'status':'preserved' if b['preserve'] else 'translated','issues':[]} for b in ir['blocks']}
    output=tmp_path/'bilingual.pdf';layout=bilingual_pdf(ir,records,source,output)
    assert layout['validation']['status']=='passed'
    with fitz.open(source) as original,fitz.open(output) as dual:
        assert original[0].get_pixmap().samples==dual[0].get_pixmap(clip=original[0].rect).samples
    for region in layout['regions']:
        # The geometry report stores each placement in source coordinates.
        for bid in region['ids']:
            block=next(b for b in ir['blocks'] if b['id']==bid)
            if block.get('column_bbox'):
                box=region['target_bbox'];column=block['column_bbox']
                assert column[0]<=box[0]<=box[2]<=column[2]


def test_original_cmyk_transparency_group_and_pixels_survive_bilingual_copy(tmp_path):
    source=tmp_path/'cmyk.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        p.draw_rect(fitz.Rect(40,100,270,180),fill=(.2,.7,.3))
        p.insert_text((40,220),'The research description remains independent of the figure.')
        doc.xref_set_key(p.xref,'Group','<< /S /Transparency /CS /DeviceCMYK /I false /K false >>')
        doc.save(source)
    ir=analyze(source,[0],ocr_mode='off',academic=True)
    records={b['id']:{**b,'target':'연구에 대한 설명.', 'status':'translated','issues':[]} for b in ir['blocks']}
    output=tmp_path/'dual.pdf';layout=bilingual_pdf(ir,records,source,output)
    assert layout['validation']['status']=='passed'
    with fitz.open(source) as a,fitz.open(output) as b:
        assert a[0].get_pixmap().samples==b[0].get_pixmap(clip=a[0].rect).samples


def test_native_text_stays_primary_and_embedded_image_label_is_recovered(tmp_path):
    source=tmp_path/'native-and-image.pdf'
    image=Image.new('RGB',(600,110),'white');draw=ImageDraw.Draw(image)
    draw.text((20,25),'Embedded research image label.',font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',28),fill='black')
    stream=io.BytesIO();image.save(stream,format='PNG')
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800);p.insert_image(fitz.Rect(40,100,340,155),stream=stream.getvalue())
        for row in range(10):
            for x,side in ((40,'Left'),(330,'Right')):
                p.insert_text((x,250+row*17),f'{side} research sentence explains the evidence {row}.',fontsize=9)
        doc.save(source)
    ir=analyze(source,[0],ocr_mode='auto',recovery=True,academic=True)
    assert ir['pages'][0]['ocr']['native_text_primary']
    assert sum(b['source'].count('Left research') for b in ir['blocks'])==10
    assert sum(b['source'].count('Right research') for b in ir['blocks'])==10
    assert any('Embedded research image label' in b['source'] for b in ir['blocks'])


def test_wide_numeric_float_is_read_before_both_body_columns(tmp_path):
    source=tmp_path/'wide-table.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        p.insert_text((40,100),'Table 1. A full width comparison of research measurements across the study groups.',fontsize=10)
        p.draw_line((40,125),(560,125))
        for row in range(4):
            for col,x in enumerate((90,240,400,540)):p.insert_text((x,145+row*15),str(10+row*4+col),fontsize=9)
        for row in range(10):
            for x,side in ((40,'Left'),(330,'Right')):p.insert_text((x,300+row*17),f'{side} research sentence explains the evidence {row}.',fontsize=9)
        doc.save(source)
    ir=analyze(source,[0],ocr_mode='off',academic=True)
    assert ir['pages'][0]['reading_columns']['full_width_numeric_tables']==1
    positions=[i for i,b in enumerate(ir['blocks']) if 'research sentence' in b['source']]
    numeric=[(i,b) for i,b in enumerate(ir['blocks']) if re.fullmatch(r'\d+(?:\s+\d+)*',b['source'].strip())]
    assert sum(len(b['source'].split()) for _,b in numeric)==16
    assert max(i for i,_ in numeric)<min(positions)


def test_tight_caption_collision_reflows_without_clipping_or_missing_text(tmp_path):
    source=tmp_path/'tight.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        p.insert_text((50,100),'Fig. 1. Alpha',fontsize=6)
        p.insert_text((50,108),'Fig. 2. Beta',fontsize=6);doc.save(source)
    ir=analyze(source,[0],ocr_mode='off',academic=True)
    records={b['id']:{**b,'target':'첫째문장' if 'Alpha' in b['source'] else '둘째문장','status':'translated','issues':[]} for b in ir['blocks']}
    output=tmp_path/'dual.pdf';layout=bilingual_pdf(ir,records,source,output)
    assert layout['automatic_collision_reflows']>=1 and layout['validation']['status']=='passed'
    with fitz.open(output) as doc:
        text=''.join(p.get_text(clip=fitz.Rect(600,0,1200,800)) for p in doc)
        assert text.count('첫째문장')==text.count('둘째문장')==1


def test_algorithm_with_serif_pseudocode_is_preserved_as_one_region(tmp_path):
    source=tmp_path/'algorithm.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        for y,value in [(100,'Algorithm 1. Calculate research weights'),(120,'function Calculate(weights)'),
                        (140,'for all values in weights do'),(160,'return weights'),(180,'end function')]:
            p.insert_text((40,y),value,fontsize=10,fontname='tiro')
        p.insert_text((40,250),'The ordinary discussion follows below the algorithm.',fontsize=10);doc.save(source)
    ir=analyze(source,[0],ocr_mode='off',academic=True)
    algorithm=next(b for b in ir['blocks'] if b.get('preserve_reason')=='algorithm')
    assert algorithm['preserve'] and 'return weights' in algorithm['source'] and 'end function' in algorithm['source']
    assert any(not b['preserve'] and 'ordinary discussion' in b['source'] for b in ir['blocks'])


def test_translation_cannot_cover_preserved_equation_or_image(tmp_path):
    source=tmp_path/'protected.pdf'
    image=Image.new('RGB',(240,160),(140,190,230));stream=io.BytesIO();image.save(stream,format='PNG')
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        p.insert_text((40,110),'A short source paragraph.',fontsize=9)
        p.insert_text((40,145),'x = 3',fontsize=10,fontname='cour')
        p.insert_image(fitz.Rect(40,180,280,340),stream=stream.getvalue());doc.save(source)
    ir=analyze(source,[0],ocr_mode='off',academic=True)
    records={b['id']:{**b,'target':b['source'] if b['preserve'] else '연구의 의미와 결과를 설명하는 문장이다. '*20,
                     'status':'preserved' if b['preserve'] else 'translated','issues':[]} for b in ir['blocks']}
    output=tmp_path/'dual.pdf';layout=bilingual_pdf(ir,records,source,output)
    assert layout['validation']['status']=='passed' and layout['overflow_pages']>0
    with fitz.open(source) as a,fitz.open(output) as b:
        assert a[0].get_pixmap(clip=fitz.Rect(40,180,280,340)).samples==b[0].get_pixmap(clip=fitz.Rect(640,180,880,340)).samples
        assert 'x = 3' in b[0].get_text(clip=fitz.Rect(600,0,1200,800))
