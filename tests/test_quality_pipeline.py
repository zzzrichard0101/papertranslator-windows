import copy
import json
from pathlib import Path

import pymupdf as fitz
import pytest

from papertranslator.document import analyze,join_lines
from papertranslator.quality import check,make_prompt,protect,restore
from papertranslator.refinement import validate_edit,policy,run,attach_context
from papertranslator.translation import blocking


@pytest.mark.parametrize('source,target',[
    ('75 per cent agreed.','75%가 동의했다.'),
    ('75 per cent agreed.','75퍼센트가 동의했다.'),
    ('The twentieth century.','20세기.'),
    ('A report until1952 describes RCTs.','1952년까지의 보고서는 무작위 대조시험을 설명한다.'),
    ('Meta-analyses of RCTS.','무작위 대조시험의 메타분석.'),
    ('June 2006','2006년 6월'),
    ('After 100 million years, it returned.','1억 년 후 돌아왔다.'),
    ('One day, after reading, I left.','어느 날, 독서 후 나는 떠났다.'),
    ('Example (2010,2015) reported these findings.','Example(2010, 2015)는 이 결과를 보고했다.'),
    ('Primary research and secondary evidence inform practice.','1차 연구와 2차 근거는 실천에 정보를 제공한다.'),
    ('The interviews were conducted individually.','면담은 1대1로 진행되었다.'),
    ('The project used a year-long intervention.','이 프로젝트는 1년간의 중재를 사용했다.'),
    ('The control group attended weekly sessions.','대조군은 주 1회 세션에 참여했다.'),
    ('Firsthand sources were used.','1차 자료가 사용되었다.'),
    ('The studies (Example, 1991,1995,1997,2000) agreed.','연구들(Example, 1991, 1995, 1997, 2000)은 일치했다.'),
    ('The studies (Example, 1991,1995, 1997, 2000; Demo, forthcoming) agreed.','연구들(Example, 1991, 1995, 1997, 2000; Demo, 출간 예정)은 일치했다.'),
])
def test_legitimate_academic_notation(source,target):
    assert not blocking(check(source,target))


def test_actual_value_and_economic_context_errors_still_block():
    assert blocking(check('75 per cent agreed.','70%가 동의했다.'))
    assert blocking(check('The report describes economic depression.','보고서는 경제적 우울증을 설명한다.'))
    assert not blocking(check('The patient has depression.','환자는 우울증이 있다.'))
    assert blocking(check('A case series was reported.','무작위 대조시험이 보고되었다.'))
    assert blocking(check('After 100 million years, it returned.','5천만 년 후 돌아왔다.'))
    assert blocking(check('It took one day, after the meeting.','회의 후 어느 날이 걸렸다.'))
    assert blocking(check('One day, not two days, was sufficient.','어느 날이면 충분했다.'))
    assert blocking(check('OCR speck.', '주변 문맥을 반복해서 임의로 추가한 내용입니다. '*20))
    assert blocking(check('Example (2010,2015) reported these findings.','Example(2010, 2016)는 이 결과를 보고했다.'))
    assert blocking(check('The value was 1991,1995.','값은 1991, 1995였다.'))
    assert blocking(check('Primary research informs practice.','3차 연구가 실천에 정보를 제공한다.'))
    assert blocking(check('Primary research informs practice.','연구자 1명이 실천에 정보를 제공한다.'))
    assert blocking(check('The study measured individual differences.','연구는 1대1 차이를 측정했다.'))
    assert blocking(check('The control group attended weekly sessions.','대조군은 주 2회 세션에 참여했다.'))
    assert blocking(check('The project used a year-long intervention.','이 프로젝트는 2년간의 중재를 사용했다.'))
    assert blocking(check('Primary research was published in 1995.','1차 연구가 발표되었다.'))


def test_author_year_citations_are_restored_verbatim():
    source='The study (Example and Writer, 2015) follows Demo (2010b: 42) and (Sample, 2001,2005).'
    protected,mapping=protect(source)
    assert set(mapping.values())=={'(Example and Writer, 2015)','(2010b: 42)','(Sample, 2001,2005)'}
    assert all(value not in protected for value in mapping.values())
    assert restore(protected,mapping)==source


def test_narrative_parentheses_with_a_year_are_still_translation_text():
    source='The account (the study began in 1995 with 24 people) describes the method.'
    protected,mapping=protect(source)
    assert protected==source and not mapping


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


def test_tall_ocr_header_is_excluded_from_continuing_body(tmp_path):
    source=tmp_path/'tall-header.pdf'
    with fitz.open() as doc:
        for _ in range(3):doc.new_page(width=600,height=800)
        doc.save(source)
    calls=[]
    def recognize(pixels):
        scale=pixels.shape[0]/800;calls.append(True)
        def line(text,y,height):
            return [[[50*scale,y*scale],[550*scale,y*scale],[550*scale,(y+height)*scale],[50*scale,(y+height)*scale]],text,None]
        header=line('Synthetic Treatment Studies',30,24)
        if len(calls)==1:return [header,line('The children used themes and',700,14)]
        if len(calls)==2:
            fragment=[[[200*scale,38*scale],[280*scale,38*scale],[280*scale,54*scale],[200*scale,54*scale]],'en betw',None]
            return [header,line('2',35,10),fragment,line('games to express their feelings.',95,14)]
        return [header,line('Another paragraph ends here.',95,14)]
    ir=analyze(source,[0,1,2],ocr_mode='force',ocr_engine=recognize)
    assert sum(b['kind']=='running_header' for b in ir['blocks'])==3,[(b['source'],b['kind']) for b in ir['blocks']]
    joined=next(b for b in ir['blocks'] if len(b['source_page_indices'])>1)
    assert joined['source']=='The children used themes and games to express their feelings.'
    assert next(b for b in ir['blocks'] if b['source']=='en betw')['preserve']


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


def test_alphabetic_scan_speck_above_running_header_is_not_body(tmp_path):
    source=tmp_path/'scans.pdf'
    with fitz.open() as document:
        for _ in range(3):document.new_page(width=600,height=800)
        document.save(source)
    calls=[]
    def recognize(pixels):
        scale=pixels.shape[0]/800
        def line(text,x,y,w):
            return [[[x*scale,y*scale],[(x+w)*scale,y*scale],[(x+w)*scale,(y+10)*scale],[x*scale,(y+10)*scale]],text,None]
        calls.append(True)
        rows=[line('Synthetic Research Book',150,35,300)]
        if len(calls)==1:return rows+[line('The methods and processes of',50,700,450)]
        if len(calls)==2:return [line('EERE',250,1,60)]+rows+[line('research and evidence inform practice.',50,80,450)]
        return rows+[line('A different paragraph ends here.',50,80,450)]
    ir=analyze(source,[0,1,2],ocr_mode='force',ocr_engine=recognize)
    joined=next(b for b in ir['blocks'] if len(b['source_page_indices'])>1)
    assert joined['source']=='The methods and processes of research and evidence inform practice.'
    assert next(b for b in ir['blocks'] if b['source']=='EERE')['preserve']


def test_wide_scan_rule_read_as_letters_is_not_a_page_continuation(tmp_path):
    source=tmp_path/'edge.pdf'
    with fitz.open() as document:
        document.new_page(width=600,height=800);document.new_page(width=600,height=800);document.save(source)
    calls=[]
    def recognize(pixels):
        scale=pixels.shape[0]/800
        def line(text,x,y,w,h):
            return [[[x*scale,y*scale],[(x+w)*scale,y*scale],[(x+w)*scale,(y+h)*scale],[x*scale,(y+h)*scale]],text,None]
        calls.append(True)
        if len(calls)==1:return [line('The methods and processes of',50,700,450,12)]
        return [line('EERE',250,1,160,7),line('research and evidence inform practice.',50,80,450,12)]
    ir=analyze(source,[0,1],ocr_mode='force',ocr_engine=recognize)
    joined=next(b for b in ir['blocks'] if len(b['source_page_indices'])>1)
    assert joined['source']=='The methods and processes of research and evidence inform practice.'
    assert next(b for b in ir['blocks'] if b['source']=='EERE')['preserve_reason']=='stretched_ocr_margin'


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


def test_reference_word_in_a_body_heading_does_not_hide_body(tmp_path):
    source=tmp_path/'body.pdf'
    with fitz.open() as document:
        page=document.new_page();page.insert_text((40,60),'References to previous studies')
        page.insert_text((40,100),'Author (2001), Author (2002), Author (2003) reported different results.')
        document.save(source)
    ir=analyze(source,[0],ocr_mode='off')
    assert all(not b['preserve'] for b in ir['blocks'])


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
    with pytest.raises(ValueError):validate_edit(revision(explanation_ko='긴 설명 '*100),source,draft)


def test_edit_cannot_drop_urls_or_change_quantities():
    source='Research does not replace 24 observations. See https://example.org/study.'
    draft='연구는 24건의 관찰을 대체한다. https://example.org/study'
    with pytest.raises(ValueError):validate_edit(revision(source_evidence='does not replace',revision='연구는 24건의 관찰을 대체하지 않는다.'),source,draft)
    with pytest.raises(ValueError):validate_edit(revision(source_evidence='does not replace',revision='연구는 42건의 관찰을 대체하지 않는다. https://example.org/study'),source,draft)


def test_edit_preserves_author_year_citation_spelling():
    source='Research evidence does not replace clinical judgement (Example and Writer, 2015).'
    draft='연구 근거는 임상 판단을 대체한다 (Example and Writer, 2015).'
    with pytest.raises(ValueError):validate_edit(revision(revision='연구 근거는 임상 판단을 대체하지 않는다 (예시와 저자, 2015).'),source,draft)


def test_specialist_contract_does_not_receive_neighbour_context():
    context=dict(before='CONTEXT_ONLY_MARKER')
    assert 'CONTEXT_ONLY_MARKER' in make_prompt('TARGET SENTENCE',{},'general','hy',context)
    assert 'CONTEXT_ONLY_MARKER' not in make_prompt('TARGET SENTENCE',{},'general','translategemma',context)


def test_short_labels_do_not_receive_long_neighbour_context():
    ir=dict(blocks=[dict(id='1',source='A long research paragraph. '*8,preserve=False,source_page_index=0),
                    dict(id='2',source='This native paragraph must appear only once.',preserve=False,source_page_index=0),
                    dict(id='3',source='The image reports two groups and 12 observations.',preserve=False,source_page_index=0)])
    attach_context(ir,'Synthetic document title')
    assert ir['blocks'][0]['context']['after']
    assert ir['blocks'][1]['context']=={} and ir['blocks'][2]['context']=={}


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
