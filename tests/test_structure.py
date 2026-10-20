from collections import Counter
import io

import pymupdf as fitz
import pytest
from PIL import Image, ImageDraw

from papertranslator.document import analyze
from papertranslator.structure import prose_tables, two_columns


def prose_fixture(path,bottom=True,scan=False):
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        p.insert_text((50,80),'Table 1. Study strategies',fontsize=11)
        p.insert_text((50,110),'STRATEGY',fontsize=11)
        p.insert_text((280,110),'EXAMPLE OF OUTCOME',fontsize=11)
        for y,left,right in [(150,'Treatment A','Does treatment improve anxiety?'),
                             (250,'Treatment B','Does treatment improve function?'),
                             (350,'Treatment intensity','Does additional treatment help?')]:
            p.insert_text((50,y),left,fontsize=11);p.insert_text((280,y),right,fontsize=11)
            p.insert_text((56,y+16),'and timing',fontsize=11)
            p.insert_text((286,y+16),'The same question continues here.',fontsize=11)
        if bottom:p.draw_line((48,410),(555,410))
        p.insert_text((50,460),'Body text outside the table stays separate.',fontsize=11)
        if scan:
            pix=p.get_pixmap(matrix=fitz.Matrix(2,2));image=pix.tobytes('png')
            doc.delete_page(0);p=doc.new_page(width=600,height=800);p.insert_image(p.rect,stream=image)
        doc.save(path)


@pytest.mark.parametrize('scan',[False,True])
def test_prose_table_keeps_multiline_labels_and_questions_in_separate_rows(tmp_path,scan):
    source=tmp_path/'prose.pdf';prose_fixture(source,scan=scan)
    ir=analyze(source,[0],ocr_mode='auto' if scan else 'off',structured=True)
    tables=ir['pages'][0]['table_regions']
    assert len(tables)==1 and tables[0]['method']=='captioned_prose_columns'
    assert len(tables[0]['cells'])==8  # header plus three logical rows, not physical lines
    cells=[b for b in ir['blocks'] if b['kind']=='table_cell']
    for row in range(1,4):
        left=' '.join(b['source'] for b in cells if b['cell_index']==row*2)
        right=' '.join(b['source'] for b in cells if b['cell_index']==row*2+1)
        assert 'Treatment' in left and 'and timing' in left and 'Does' not in left
        assert 'Does' in right and 'continues here' in right and 'and timing' not in right
    assert any('outside the table' in b['source'] and b['kind']!='table_cell' for b in ir['blocks'])
    old=analyze(source,[0],ocr_mode='auto' if scan else 'off')
    assert not old['pages'][0]['table_regions']
    # Every recognised source token remains, even when its column is recovered.
    tokens=lambda ir:Counter(w for b in ir['blocks'] for w in b['source'].split())
    assert tokens(old)==tokens(ir)


def test_caption_without_a_bottom_rule_does_not_guess_a_table(tmp_path):
    source=tmp_path/'ambiguous.pdf';prose_fixture(source,bottom=False)
    assert not analyze(source,[0],ocr_mode='off',structured=True)['pages'][0]['table_regions']


def columns_fixture(path):
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        p.insert_text((120,80),'Full width research section heading',fontsize=16)
        for row in range(8):
            p.insert_text((40,150+row*15),f'Left column sentence number {row}.',fontsize=10)
            p.insert_text((330,150+row*15),f'Right column sentence number {row}.',fontsize=10)
        p.insert_text((120,380),'Another full width section heading',fontsize=16)
        for row in range(8):
            p.insert_text((40,450+row*15),f'Later left column sentence {row}.',fontsize=10)
            p.insert_text((330,450+row*15),f'Later right column sentence {row}.',fontsize=10)
        doc.save(path)


def test_two_column_reading_order_survives_final_ir_sort_and_keeps_all_words(tmp_path):
    source=tmp_path/'columns.pdf';columns_fixture(source)
    old=analyze(source,[0],ocr_mode='off')
    new=analyze(source,[0],ocr_mode='off',structured=True)
    assert new['pages'][0]['reading_columns']['columns']==2
    sources=[b['source'] for b in new['blocks']]
    positions=lambda phrase:[i for i,text in enumerate(sources) if phrase in text]
    assert max(positions('Left column'))<min(positions('Right column'))
    assert max(positions('Right column'))<min(positions('Another full'))
    assert max(positions('Later left'))<min(positions('Later right'))
    for b in new['blocks']:
        assert not ('Left column' in b['source'] and 'Right column' in b['source'])
        assert not ('Later left' in b['source'] and 'Later right' in b['source'])
    tokens=lambda ir:Counter(w for b in ir['blocks'] for w in b['source'].split())
    assert tokens(old)==tokens(new)


def test_single_column_and_monospace_code_are_not_reordered(tmp_path):
    source=tmp_path/'single.pdf'
    with fitz.open() as doc:
        p=doc.new_page()
        for i in range(12):p.insert_text((50,150+i*15),'An ordinary full width paragraph describes academic research methods.')
        p.insert_text((50,400),'def update(total):\n    return total + 1',fontname='cour');doc.save(source)
    old=analyze(source,[0],ocr_mode='off');new=analyze(source,[0],ocr_mode='off',structured=True)
    assert 'reading_columns' not in new['pages'][0]
    assert [b['source'] for b in old['blocks']]==[b['source'] for b in new['blocks']]


def test_short_labels_beside_long_explanations_are_not_treated_as_prose_columns(tmp_path):
    source=tmp_path/'labels.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        for row in range(12):
            p.insert_text((50,150+row*20),f'Group {row}',fontsize=10)
            p.insert_text((280,150+row*20),'An extended explanation of the clinical outcome.',fontsize=10)
        doc.save(source)
    ir=analyze(source,[0],ocr_mode='off',structured=True)
    assert 'reading_columns' not in ir['pages'][0]
