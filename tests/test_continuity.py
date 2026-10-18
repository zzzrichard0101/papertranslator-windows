import pymupdf as fitz
import pytest
from papertranslator.document import analyze,flow_pdf
from papertranslator.quality import check


@pytest.mark.parametrize('source,target',[
    ('The project lasted two decades.','사업은 20년간 이어졌다.'),
    ('They worked for 1-1/2 years.','그들은 1년 6개월 동안 일했다.'),
    ('They worked for 1.5 years.','그들은 18개월 동안 일했다.'),
    ('They worked for one and a half years.','그들은 1년 반 동안 일했다.'),
    ('The COVID-19 period ended in 2023.','코로나19 시기는 2023년에 끝났다.'),
    ('She showed two drawings and discussed the third hand.','그녀는 2장의 그림을 보여주고 제3의 손을 논의했다.'),
    ('One year, the group met after World War I and before WWII.','한 해, 모임은 제1차 세계대전 후와 제2차 세계대전 전에 열렸다.'),
])
def test_exact_localized_numbers_do_not_discard_translation(source,target):
    assert not any(i['code'] in {'numbers','quantity'} for i in check(source,target))


@pytest.mark.parametrize('source,target',[
    ('They worked for two years.','그들은 2개월 동안 일했다.'),
    ('They worked for 1-1/2 years.','그들은 1년 5개월 동안 일했다.'),
    ('The change after one year was -1.25.','1년 후 변화량은 1.25였다.'),
    ('The project lasted two decades.','사업은 20년간 이어졌고 20명이 참가했다.'),
    ('COVID-19 affected 24 people.','코로나19는 25명에게 영향을 주었다.'),
    ('I painted.','나는 1번 그렸다.'),
])
def test_localization_does_not_excuse_real_unit_sign_or_number_errors(source,target):
    assert any(i['code'] in {'numbers','quantity'} for i in check(source,target))


def test_return_in_ordinary_prose_is_translated_but_code_is_preserved(tmp_path):
    source=tmp_path/'ordinary.pdf'
    with fitz.open() as doc:
        p=doc.new_page();p.insert_text((50,90),'The painter decided to return home and return to teaching.')
        p.insert_text((50,160),'def update(total):\n    return total + 1',fontname='cour')
        doc.save(source)
    blocks=analyze(source,[0])['blocks']
    assert not next(b for b in blocks if 'painter' in b['source'])['preserve']
    assert next(b for b in blocks if 'def update' in b['source'])['preserve']


def fixture(path,ending='The research continued for',opening='six months, including winter.',heading=False):
    with fitz.open() as doc:
        a=doc.new_page();a.insert_text((50,700),ending);a.insert_text((290,810),'1')
        b=doc.new_page();b.insert_text((50,65),opening,fontsize=18 if heading else 11)
        b.insert_text((50,180),'A separate paragraph describes a different experiment.')
        b.insert_text((290,810),'2')
        c=doc.new_page();c.insert_text((50,65),'another unselected page fragment')
        doc.save(path)


def test_cross_page_sentence_is_one_request_and_translation_has_traceable_anchor(tmp_path):
    source=tmp_path/'continuation.pdf';fixture(source)
    ir=analyze(source,[0,1]);connected=[b for b in ir['blocks'] if len(b['source_page_indices'])>1]
    assert len(connected)==1
    block=connected[0]
    assert block['source']=='The research continued for six months, including winter.'
    assert block['source_page_indices']==[0,1]
    assert [f['source_page_index'] for f in block['source_fragments']]==[0,1]
    assert ir['pages'][1]['continuations'][0]['id']==block['id']
    assert all(b['id']!=block['id'] for b in ir['blocks'] if b is not block)
    records={b['id']:{**b,'status':'preserved' if b['preserve'] else 'translated','target':b['source'] if b['preserve'] else '연구는 겨울을 포함하여 6개월간 계속되었다.'} for b in ir['blocks']}
    output=tmp_path/'result.pdf';flow_pdf(ir,records,output,'합성 연결 시험')
    with fitz.open(output) as doc:
        assert '연결 문단' in doc[0].get_text() and '연결 안내' in doc[1].get_text()
        assert '[원문 보존] 1' not in doc[0].get_text()


@pytest.mark.parametrize('ending,opening,heading',[
    ('The research ended.','Another study started.',False),
    ('The research continued for','NEW CHAPTER',True),
    ('The research ended. (Keep this image visible)','At the next meeting we started again.',False),
])
def test_independent_sections_are_not_joined(tmp_path,ending,opening,heading):
    source=tmp_path/'boundary.pdf';fixture(source,ending,opening,heading)
    assert not any(len(b['source_page_indices'])>1 for b in analyze(source,[0,1])['blocks'])


def test_sparse_selection_does_not_join_across_an_omitted_page(tmp_path):
    source=tmp_path/'sparse.pdf';fixture(source)
    assert not any(len(b['source_page_indices'])>1 for b in analyze(source,[0,2])['blocks'])


def test_wrapped_age_is_not_a_numbered_list_boundary(tmp_path):
    path=tmp_path/'wrapped-value.pdf'
    with fitz.open() as doc:
        p=doc.new_page();p.insert_text((50,700),'The visitor finished the course at age\n31. The visitor then moved to the')
        p=doc.new_page();p.insert_text((50,65),'coast and started working.')
        doc.save(path)
    blocks=analyze(path,[0,1])['blocks']
    assert len(blocks)==1 and 'age 31.' in blocks[0]['source']
    assert blocks[0]['source_page_indices']==[0,1]


def test_glossary_preserves_core_concept_and_unknown_family_side():
    from papertranslator.quality import make_prompt,relevant_terms
    assert 'art-as-therapy translates to 치료로서의 미술' in make_prompt('An art-as-therapy approach.',{},'general')
    assert ('self-taught','독학한') in relevant_terms('A self-taught artist made this.')
    assert ('aunt','고모 또는 이모') in relevant_terms('Her aunt attended.')
    assert ('aunt','고모 또는 이모') not in relevant_terms('Her maternal aunt attended.')
    assert not any(i['code']=='kinship_side' for i in check('Her aunt attended.','고모 또는 이모가 참석했다.'))
