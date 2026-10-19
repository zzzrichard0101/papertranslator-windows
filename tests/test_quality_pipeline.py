import copy
import json
from pathlib import Path

import pymupdf as fitz
import pytest

from papertranslator.document import analyze,join_lines
from papertranslator.quality import check,make_prompt
from papertranslator.refinement import validate_edit,policy,run
from papertranslator.translation import blocking


@pytest.mark.parametrize('source,target',[
    ('75 per cent agreed.','75%가 동의했다.'),
    ('75 per cent agreed.','75퍼센트가 동의했다.'),
    ('The twentieth century.','20세기.'),
    ('A report until1952 describes RCTs.','1952년까지의 보고서는 무작위 대조시험을 설명한다.'),
])
def test_legitimate_academic_notation(source,target):
    assert not blocking(check(source,target))


def test_actual_value_and_economic_context_errors_still_block():
    assert blocking(check('75 per cent agreed.','70%가 동의했다.'))
    assert blocking(check('The report describes economic depression.','보고서는 경제적 우울증을 설명한다.'))
    assert not blocking(check('The patient has depression.','환자는 우울증이 있다.'))
    assert blocking(check('A case series was reported.','무작위 대조시험이 보고되었다.'))


def test_known_layout_hyphens_only():
    assert join_lines('The para-','digm changed.')[0]=='The paradigm changed.'
    assert join_lines('evidence-','based practice')[0]=='evidence-based practice'
    assert join_lines('non-','randomized design')[0]=='non-randomized design'


def test_running_headers_do_not_break_page_continuation(tmp_path):
    source=tmp_path/'synthetic.pdf'
    with fitz.open() as document:
        for index in range(3):
            page=document.new_page()
            page.insert_text((40,35),'Synthetic Research Methods',fontsize=10)
            if index==0:page.insert_text((40,710),'The new para-')
            elif index==1:page.insert_text((40,80),'digm improves the research methods.')
            else:page.insert_text((40,80),'A different sentence ends here.')
        document.save(source)
    ir=analyze(source,[0,1,2],ocr_mode='off')
    assert sum(b['kind']=='running_header' for b in ir['blocks'])==3
    joined=next(b for b in ir['blocks'] if len(b['source_page_indices'])==2)
    assert joined['source']=='The new paradigm improves the research methods.'
    assert any(f['raw_source'].endswith('para-') for f in joined['source_fragments'])


def test_ocr_cache_reuses_same_raster_and_engine_signature(tmp_path):
    source=tmp_path/'scan.pdf'
    with fitz.open() as document:document.new_page();document.save(source)
    calls=[]
    def recognize(pixels):
        calls.append(True)
        return [[[[10,10],[200,10],[200,30],[10,30]],'A synthetic scanned sentence.',.9]]
    for _ in range(2):analyze(source,[0],ocr_mode='force',ocr_engine=recognize,cache_dir=tmp_path/'cache')
    assert len(calls)==1


def test_ocr_margin_specks_do_not_block_cross_page_sentence(tmp_path):
    source=tmp_path/'scans.pdf'
    with fitz.open() as document:
        document.new_page(width=600,height=800);document.new_page(width=600,height=800);document.save(source)
    calls=[]
    def recognize(pixels):
        scale=pixels.shape[0]/800
        def line(text,x,y,w):
            return [[[x*scale,y*scale],[(x+w)*scale,y*scale],[(x+w)*scale,(y+12)*scale],[x*scale,(y+12)*scale]],text,None]
        calls.append(True)
        if len(calls)==1:return [line('The study continued with',50,700,400),line('— a',40,750,400)]
        return [line('pe',30,25,300),line('the same research question.',50,80,400)]
    ir=analyze(source,[0,1],ocr_mode='force',ocr_engine=recognize)
    joined=[b for b in ir['blocks'] if len(b['source_page_indices'])>1]
    assert len(joined)==1 and joined[0]['source']=='The study continued with the same research question.'
    assert sum(b.get('preserve_reason')=='ocr_margin_fragment' for b in ir['blocks'])==2


def test_unscored_ocr_overlapping_font_boxes_keep_logical_line_order(tmp_path):
    source=tmp_path/'scan.pdf'
    with fitz.open() as document:document.new_page(width=600,height=800);document.save(source)
    def recognize(pixels):
        scale=pixels.shape[0]/800
        def line(text,x,y,w,h):
            return [[[x*scale,y*scale],[(x+w)*scale,y*scale],[(x+w)*scale,(y+h)*scale],[x*scale,(y+h)*scale]],text,None]
        return [line('Systematic observations can increase confidence.',50,100,450,12),
                line('The absence',400,99,100,26),
                line('of observations must lead to caution.',50,99,450,40)]
    ir=analyze(source,[0],ocr_mode='force',ocr_engine=recognize)
    assert ir['blocks'][0]['source']==('Systematic observations can increase confidence. '
                                      'The absence of observations must lead to caution.')
    assert ir['pages'][0]['ocr']['order_adjusted_lines']==2


def test_contents_references_entry_does_not_preserve_the_rest_of_the_book(tmp_path):
    source=tmp_path/'contents.pdf'
    with fitz.open() as document:
        page=document.new_page();page.insert_text((40,60),'Contents')
        page.insert_text((40,600),'References');page.insert_text((40,630),'Index')
        page=document.new_page();page.insert_text((40,70),'This biography introduces the researcher and the study.')
        page=document.new_page();page.insert_text((40,70),'References')
        for offset,year in enumerate([2001,2002,2003]):page.insert_text((40,120+offset*30),f'Author, A. ({year}) Research Methods. Publisher.')
        document.save(source)
    ir=analyze(source,[0,1,2],ocr_mode='off')
    assert not next(b for b in ir['blocks'] if b['source'].startswith('This biography'))['preserve']
    assert all(b['preserve'] for b in ir['blocks'] if b['source'].startswith('Author,'))


def revision(**kwargs):
    value=dict(decision='revise',source_evidence='does not replace',draft_evidence='대체한다',
               explanation_ko='부정 표현 복원',revision='연구 근거는 임상 판단을 대체하지 않는다.')
    return {**value,**kwargs}


def test_edit_requires_real_source_and_draft_quotes():
    source='Research evidence does not replace clinical judgement.';draft='연구 근거는 임상 판단을 대체한다.'
    assert validate_edit(revision(),source,draft)['decision']=='revise'
    with pytest.raises(ValueError):validate_edit(revision(source_evidence='invented quote'),source,draft)
    with pytest.raises(ValueError):validate_edit(revision(draft_evidence='없는 표현'),source,draft)
    with pytest.raises(ValueError):validate_edit(revision(revision='연구 근거は 임상 판단을 대체하지 않는다.'),source,draft)
    with pytest.raises(ValueError):validate_edit(revision(decision='keep'),source,draft)


def test_edit_cannot_drop_urls_or_change_quantities():
    source='Research does not replace 24 observations. See https://example.org/study.'
    draft='연구는 24건의 관찰을 대체한다. https://example.org/study'
    with pytest.raises(ValueError):validate_edit(revision(source_evidence='does not replace',revision='연구는 24건의 관찰을 대체하지 않는다.'),source,draft)
    with pytest.raises(ValueError):validate_edit(revision(source_evidence='does not replace',revision='연구는 42건의 관찰을 대체하지 않는다. https://example.org/study'),source,draft)


def test_specialist_contract_does_not_receive_neighbour_context():
    context=dict(before='CONTEXT_ONLY_MARKER')
    assert 'CONTEXT_ONLY_MARKER' in make_prompt('TARGET SENTENCE',{},'general','hy',context)
    assert 'CONTEXT_ONLY_MARKER' not in make_prompt('TARGET SENTENCE',{},'general','translategemma',context)


def test_model_roles_are_snapshots_and_require_editor():
    registry=dict(models={'hy7b':dict(id='translator',adapter='hy'),
                           'qwen38':dict(id='editor',adapter='qwen',options=dict(context=4096))})
    before=copy.deepcopy(registry)
    result=policy('ensemble',registry['models']['hy7b'],registry)
    assert result['editor']['options']['context']==8192 and result['auxiliary'] is None
    assert registry==before
    with pytest.raises(ValueError):policy('ensemble',dict(adapter='hy'),dict(models={'hy7b':dict(adapter='hy')}))


def test_refinement_checkpoint_prevents_repeated_edits(tmp_path):
    source='Research evidence does not replace clinical judgement. '*3
    draft='연구 근거는 임상 판단을 대체한다.'
    record=dict(id='synthetic',source=source,target=draft,status='translated',issues=[])
    records={'synthetic':record};ir=dict(blocks=[dict(id='synthetic',source=source)])
    settings=dict(mode='ensemble',version='test',editor={'id':'synthetic-editor'})
    calls=[];saves=[]
    class Backend:
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def start(self,progress):pass
        def translate(self,*args,**kwargs):calls.append(True);return dict(text=json.dumps(revision()),seconds=1)
    for _ in range(2):run(records,ir,settings,tmp_path,'general',Backend,lambda:saves.append(True),lambda _:None,lambda:False)
    assert len(calls)==1 and record['refinement']['accepted']
    assert record['target'].endswith('대체하지 않는다.') and saves


def test_invalid_edit_is_retained_and_does_not_replace_draft(tmp_path):
    source='Research evidence does not replace clinical judgement. '*3
    draft='연구 근거는 임상 판단을 대체한다.'
    record=dict(id='bad-edit',source=source,target=draft,status='translated',issues=[])
    class Backend:
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def start(self,progress):pass
        def translate(self,*args,**kwargs):return dict(text=json.dumps(revision(source_evidence='invented quote')),seconds=1)
    run({'bad-edit':record},dict(blocks=[dict(id='bad-edit')]),
        dict(mode='ensemble',editor={'id':'synthetic-editor'}),tmp_path,'general',Backend,lambda:None,lambda _:None,lambda:False)
    assert record['target']==draft and not record['refinement']['accepted']
    assert record['refinement']['decision']=='uncertain'
    assert (tmp_path/'responses/edit-bad-edit.json').is_file()
