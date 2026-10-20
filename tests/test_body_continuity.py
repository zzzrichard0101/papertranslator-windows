import pymupdf as fitz
import pytest
from copy import deepcopy

from papertranslator.document import analyze
from papertranslator.source_integrity import caption_start
from papertranslator.refinement import attach_context


@pytest.mark.parametrize('text',[
    'Table 7 using a common scale. Examples can',
    'Figure 4 shows the measurements.', 'Box 3 provides an example.',
    'Table 2 is described below.', 'Figure 3 and the appendix show the result.'
])
def test_prose_references_are_not_caption_titles(text):
    assert not caption_start(text)


@pytest.mark.parametrize('text',['Table 7. Quantitative results','Figure 4: Experimental evidence',
    'Fig. 2. Main findings','Box 1 Research methods','Table 2'])
def test_explicit_titles_are_retained(text):
    assert caption_start(text)


def footnote(page):
    page.insert_text((40,700),'2',fontsize=6)
    page.insert_text((43.5,702),'We used 12 hours for calibration.',fontsize=8)
    page.insert_text((40,714),'The calibration was performed locally.',fontsize=8)


def test_inline_table_reference_and_footnote_do_not_sever_column_sentence(tmp_path):
    source=tmp_path/'two columns.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        for x,side in [(40,'Left'),(330,'Right')]:
            for n in range(10):p.insert_text((x,160+n*17),f'{side} research sentence explains the evidence {n}.',fontsize=9)
        p.insert_text((40,650),'The results are reported in',fontsize=9)
        p.insert_text((40,662),'Table 7 using a common scale. Examples can',fontsize=9)
        p.insert_text((330,130),'be seen in a separate experiment.',fontsize=9)
        footnote(p);doc.save(source)
    old=analyze(source,[0],ocr_mode='off',academic=True,source_integrity='source-integrity-1')
    new=analyze(source,[0],ocr_mode='off',academic=True,source_integrity=True)
    block=next(b for b in new['blocks'] if b['source'].startswith('The results'))
    assert block['source']=='The results are reported in Table 7 using a common scale. Examples can be seen in a separate experiment.'
    assert block['column_continuation'] and len(block['source_fragments'])==2
    assert not any('Examples can be seen' in b['source'] for b in old['blocks'])
    note=next(b for b in new['blocks'] if 'We used 12 hours' in b['source'])
    assert note['kind']=='footnote' and note['source'].endswith('performed locally.') and not note['preserve']
    assert any(b['kind']=='footnote_marker' and b['source']=='2' and b['preserve'] for b in new['blocks'])
    assert new['source_integrity']['version']=='source-integrity-2'
    assert old['source_integrity']['version']=='source-integrity-1'


def test_footnote_does_not_become_part_of_cross_page_body(tmp_path):
    source=tmp_path/'two pages.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        p.insert_text((40,650),'The study continued for',fontsize=10);footnote(p)
        p=doc.new_page(width=600,height=800)
        p.insert_text((40,100),'twelve months, including winter.',fontsize=10)
        doc.save(source)
    ir=analyze(source,[0,1],ocr_mode='off',academic=True,source_integrity=True)
    body=next(b for b in ir['blocks'] if b['source'].startswith('The study'))
    assert body['source']=='The study continued for twelve months, including winter.'
    assert body['source_page_indices']==[0,1]
    assert 'calibration' not in body['source']
    assert any(b['kind']=='footnote' and '12 hours' in b['source'] for b in ir['blocks'])


def test_body_context_skips_footnotes_without_changing_previous_policy():
    def block(key,text,kind='paragraph'):
        return dict(id=key,source=text*4,kind=kind,preserve=False,source_page_index=0)
    blocks=[block('a','The main discussion describes the first research question. '),
            block('note','The calibration used a separate device for 12 hours. ','footnote'),
            block('b','The main discussion continues with the second research question. ')]
    old=dict(blocks=deepcopy(blocks),source_integrity=dict(version='source-integrity-1'))
    new=dict(blocks=deepcopy(blocks),source_integrity=dict(version='source-integrity-2'))
    attach_context(old,'Synthetic study');attach_context(new,'Synthetic study')
    assert 'calibration' in old['blocks'][0]['context']['after']
    assert 'second research' in new['blocks'][0]['context']['after']
    assert 'first research' in new['blocks'][2]['context']['before']
    assert new['blocks'][1]['context']['before']==new['blocks'][1]['context']['after']==''
    assert new['blocks'][0]['context']['terminology_usage']=='meaning_hints'
    assert 'terminology_usage' not in old['blocks'][0]['context']


def test_recognized_short_footnotes_are_not_discarded_as_repeated_footers(tmp_path):
    source=tmp_path/'bottom notes.pdf'
    with fitz.open() as doc:
        for n in [1,2]:
            p=doc.new_page(width=600,height=800)
            p.insert_text((40,120),'Research results were recorded.',fontsize=10)
            p.insert_text((40,768),str(n),fontsize=6)
            p.insert_text((43.5,770),'Calibration.',fontsize=8)
        doc.save(source)
    ir=analyze(source,[0,1],ocr_mode='off',academic=True,source_integrity=True)
    notes=[b for b in ir['blocks'] if b['source']=='Calibration.']
    assert len(notes)==2 and all(b['kind']=='footnote' and not b['preserve'] for b in notes)


def test_uniform_random_distribution_hint_is_versioned_and_source_grounded():
    from papertranslator.terminology import build,for_source
    source='We initialize uniformly random points inside the volume.'
    block=dict(id='sampling',source=source,kind='paragraph',preserve=False,source_page_index=0)
    old=build(dict(blocks=[block],source_integrity=dict(version='source-integrity-1')),'general')
    new=build(dict(blocks=[block],source_integrity=dict(version='source-integrity-2')),'general')
    assert ('uniformly random','균일하게 무작위로') not in for_source(source,old)
    assert ('uniformly random','균일하게 무작위로') in for_source(source,new)
    assert not for_source('The random method was uniformly improved.',new)


@pytest.mark.parametrize('label',['Table 7.','Figure 4:','Box 3.'])
def test_wrapped_reference_punctuation_does_not_override_body_grammar(tmp_path,label):
    source=tmp_path/'wrapped reference.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        p.insert_text((40,140),'The values are reported in',fontsize=10)
        p.insert_text((40,154),label+' The analysis then continues.',fontsize=10)
        p.insert_text((40,168),'The results remain consistent.',fontsize=10)
        doc.save(source)
    ir=analyze(source,[0],ocr_mode='off',academic=True,source_integrity=True)
    block=next(b for b in ir['blocks'] if b['source'].startswith('The values'))
    assert block['kind']=='paragraph' and 'in '+label in block['source']
    assert 'The results remain consistent.' in block['source']
