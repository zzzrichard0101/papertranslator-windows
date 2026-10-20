from copy import deepcopy
import pytest
import pymupdf as fitz

from papertranslator.document import analyze
from papertranslator.source_integrity import reference_units,validate_lines,validate_segments,split_footnote_markers
from papertranslator.quality import check,protect,restore
from papertranslator.refinement import validate_edit,edit_record
import json


def test_caption_is_complete_and_next_caption_stays_separate(tmp_path):
    source=tmp_path/'caption.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        p.insert_text((50,120),'Fig. 6. At 30K itera-\ntions, the artifacts are reduced.',fontsize=9)
        p.insert_text((50,170),'Table 2. We start with 100K randomly\ninitialized points.',fontsize=9)
        doc.save(source)
    old=analyze(source,[0],ocr_mode='off',academic=True)
    new=analyze(source,[0],ocr_mode='off',academic=True,source_integrity=True)
    assert len(old['blocks'])==4 and len(new['blocks'])==2
    assert new['blocks'][0]['source']=='Fig. 6. At 30K iterations, the artifacts are reduced.'
    assert new['blocks'][1]['source']=='Table 2. We start with 100K randomly initialized points.'
    assert new['pages'][0]['source_accounting']['removed_line_hyphens']==1
    assert all(b['kind']=='caption' for b in new['blocks'])


def test_bare_year_references_start_after_heading_and_end_at_appendix(tmp_path):
    source=tmp_path/'references.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        p.insert_text((50,120),'REFERENCES\nAlice Smith. 2020. Neural study.\nBob Jones. 2021. Another study.\nCarol King. 2022. Final study.',fontsize=10)
        p.insert_text((50,250),'APPENDIX A',fontsize=17)
        p.insert_text((50,280),'The experiment uses three samples.',fontsize=10)
        doc.save(source)
    ir=analyze(source,[0],ocr_mode='off',academic=True,source_integrity=True)
    assert ir['blocks'][0]['source']=='REFERENCES' and not ir['blocks'][0]['preserve']
    refs=[b for b in ir['blocks'] if b['kind']=='reference']
    assert refs and all(b['preserve'] for b in refs)
    body=next(b for b in ir['blocks'] if b['source'].startswith('The experiment'))
    assert body['kind']=='paragraph' and not body['preserve']


def test_reference_continuation_needs_density_and_closes_at_uppercase_heading():
    unit=lambda source,kind='paragraph':dict(source=source,kind=kind,preserve=False)
    units=[unit('Ann Able. 2020. A. Bob Baker. 2021. B. Carol Clark. 2022. C.'),
           unit('DETAILS OF GRADIENT COMPUTATION','heading'),unit('We use three values.')]
    assert not reference_units(units,previous=True)
    assert units[0]['preserve'] and not units[-1]['preserve']
    sparse=[unit('In 2020. We studied an approach.')]
    assert not reference_units(sparse,previous=True) and not sparse[0]['preserve']


def test_raised_native_footnote_marker_is_preserved_separately(tmp_path):
    source=tmp_path/'footnote.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        p.insert_text((50,710),'2',fontsize=6)
        p.insert_text((53.5,712),'We trained for 12 hours.',fontsize=8)
        doc.save(source)
    ir=analyze(source,[0],ocr_mode='off',academic=True,source_integrity=True)
    marker=next(b for b in ir['blocks'] if b['source']=='2')
    body=next(b for b in ir['blocks'] if 'trained' in b['source'])
    assert marker['preserve'] and marker['preserve_reason']=='footnote_marker'
    assert body['source']=='We trained for 12 hours.' and not body['preserve']


@pytest.mark.parametrize('changed',['normal_size','ocr','inline','not_raised'])
def test_ordinary_leading_numbers_and_ocr_are_not_footnotes(changed):
    marker=dict(text='2',size=6,bbox=[50,710,53,716])
    body=dict(text='We train.',size=8,bbox=[53,711,95,719])
    block=dict(type=0,bbox=[50,710,95,719],lines=[dict(bbox=[50,710,95,719],spans=[marker,body])])
    if changed=='normal_size':marker['size']=8
    if changed=='ocr':marker['extraction']='ocr'
    if changed=='inline':block['lines'][0]['bbox'][1]=200
    if changed=='not_raised':marker['bbox'][3]=719
    before=deepcopy(block)
    assert split_footnote_markers([block],800)==[before]


def test_source_accounting_rejects_omission_duplication_and_changed_order():
    block=dict(lines=[dict(spans=[dict(text='Alpha beta.')])])
    assert validate_lines([block],[dict(source='Alpha beta.')])['status']=='passed'
    for source in ['Alpha.','Alpha beta. beta.']:
        with pytest.raises(RuntimeError):validate_lines([block],[dict(source=source)])
    validate_segments('Alpha beta.', ['Alpha',' beta.'])
    with pytest.raises(RuntimeError):validate_segments('Alpha beta.', ['beta.','Alpha'])


@pytest.mark.parametrize('target',['100,000개의 점','10만 개의 점','100,000개의 무작위로 초기화된 점'])
def test_modified_scientific_counts_are_unit_checked_only_in_new_policy(target):
    source='100K randomly initialized points'
    assert not check(source,target,typed='typed-v2')
    assert any(i['code']=='numbers' for i in check(source,target,typed=True))


@pytest.mark.parametrize('target',['10,000개의 점','100,000개의 가우시안','-100,000개의 점','100,000명의 참가자','100,000개'])
def test_modified_counts_do_not_excuse_wrong_value_unit_sign_or_absent_unit(target):
    assert check('100K randomly initialized points',target,typed='typed-v2')


def test_dimensions_are_restored_exactly_and_missing_repeated_dimensions_fail():
    source='Use 16x16 pixel tiles and each overlapping 16×16 tile.'
    text,mapping=protect(source,typed='typed-v2')
    assert len(mapping)==2 and '16' not in text and restore(text,mapping)==source
    assert any(i['code']=='literal' for i in check(source,'16×16 크기의 타일을 사용한다.',typed='typed-v2'))
    assert not check(source,'16x16 픽셀 타일과 겹치는 각 16×16 타일을 사용한다.',typed='typed-v2')


def test_editor_cannot_keep_a_known_failed_draft_in_v2(tmp_path):
    source='The sorting method uses radix sort. '
    draft='정렬 방법은 라디ックス 정렬을 사용한다.'
    keep=dict(decision='keep',source_evidence='',draft_evidence='',explanation_ko='',revision='')
    assert validate_edit(keep,source,draft,typed=True)['decision']=='keep'
    with pytest.raises(ValueError,match='foreign_script'):
        validate_edit(keep,source,draft,typed='typed-v2')
    class Backend:
        def __init__(self):self.prompts=[]
        def translate(self,prompt,**kw):
            self.prompts.append(prompt)
            value=keep if len(self.prompts)==1 else dict(decision='revise',source_evidence='radix sort',
                draft_evidence='라디ックス',explanation_ko='외국 문자 교정',revision='정렬 방법은 기수 정렬을 사용한다.')
            return dict(text=json.dumps(value,ensure_ascii=False),seconds=.01)
    backend=Backend();record=dict(id='script',source=source,target=source,candidate=draft,status='failed',issues=[])
    result=edit_record(backend,record,{},dict(validation='typed-v2',editor=dict(id='synthetic')),
                       'test',tmp_path,lambda:False)
    assert len(backend.prompts)==2 and 'foreign_script' in backend.prompts[-1]
    assert result['status']=='translated' and not result['issues'] and result['target']=='정렬 방법은 기수 정렬을 사용한다.'


def test_pixel_tile_dimensions_cannot_be_translated_as_tile_counts():
    source='We split the screen into 16x16 pixel tiles.'
    assert any(i['code']=='dimension_role' for i in check(source,'화면을 16x16개의 픽셀 타일로 분할한다.',typed='typed-v2'))
    assert not check(source,'화면을 16x16 픽셀 크기의 타일로 분할한다.',typed='typed-v2')
    assert not any(i['code']=='dimension_role' for i in check('We create 16x16 tiles.','16x16개의 타일을 만든다.',typed='typed-v2'))


def test_retries_retain_all_original_failures_instead_of_oscillating(tmp_path):
    source='We use radix sort and 16x16 pixel tiles.'
    draft='라디ックス 정렬과 16x16개의 픽셀 타일을 사용한다.'
    class Backend:
        def __init__(self):self.prompts=[]
        def translate(self,prompt,**kw):
            self.prompts.append(prompt)
            revision='라디ックス 정렬과 16x16 픽셀 타일을 사용한다.' if len(self.prompts)==1 else '기수 정렬과 16x16 픽셀 타일을 사용한다.'
            return dict(text=json.dumps(dict(decision='revise',source_evidence='radix sort',
                draft_evidence='라디ックス',explanation_ko='원문 대조',revision=revision),ensure_ascii=False),seconds=.01)
    backend=Backend();record=dict(id='oscillation',source=source,target=source,candidate=draft,status='failed',issues=[])
    result=edit_record(backend,record,{},dict(validation='typed-v2',editor=dict(id='synthetic')),
                       'test',tmp_path,lambda:False)
    assert len(backend.prompts)==2 and result['status']=='translated'
    assert all('foreign_script' in prompt and 'dimension_role' in prompt for prompt in backend.prompts)
    assert not result['issues']


@pytest.mark.parametrize('source,target',[
    ('6–10K meaningful Gaussians','6,000~10,000개의 유의미한 가우시안'),
    ('6–10K Gaussians','6~10K개의 가우시안'),
    ('200–500K Gaussians','20만~50만 개의 가우시안'),
    ('200–500K Gaussians','20~50만 개의 가우시안'),
    ('6-10K points','6천~1만 개의 점'),
    ('1K–2M points','1천~200만 개의 점'),
    ('-6–-2K Gaussians','-6천~-2천 개의 가우시안'),
    ('6–10K iterations','6천~1만 회의 반복'),
])
def test_both_range_endpoints_units_and_scales_match(source,target):
    assert not check(source,target,typed='typed-v3')


@pytest.mark.parametrize('source,target',[
    ('6–10K Gaussians','6~10,000개의 가우시안'),
    ('6–10K Gaussians','6,000~11,000개의 가우시안'),
    ('6–10K Gaussians','6,000~10,000개의 점'),
    ('6–10K Gaussians','6,000~10,000개'),
    ('6–10K Gaussians','-6,000~10,000개의 가우시안'),
    ('+6–10K Gaussians','6,000~10,000개의 가우시안'),
    ('200–500K Gaussians','20만~50개의 가우시안'),
    ('6–10K points and 6–10K points','6천~1만 개의 점'),
])
def test_bad_ranges_cannot_hide_behind_a_scalar_upper_endpoint(source,target):
    assert any(i['code']=='quantity' for i in check(source,target,typed='typed-v3'))


def test_range_equivalence_does_not_change_older_validation_policy():
    for mode in [False,True,'typed-v2']:
        assert check('200–500K Gaussians','20만~50만 개의 가우시안',typed=mode)
