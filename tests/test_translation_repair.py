import pytest
from papertranslator.backend import Cancelled
from papertranslator.translation import translate_record


class Engine:
    options={'context':4096}
    def __init__(self,replies):self.replies=iter(replies);self.prompts=[]
    def translate(self,prompt,**kwargs):
        self.prompts.append(prompt);reply=next(self.replies)
        if isinstance(reply,Exception):raise reply
        return dict(text=reply,seconds=.1)


def block(source):
    return dict(id='p1-test',source_page_index=0,source_page_indices=[0],source_fragments=[],bbox=[0,0,1,1],kind='paragraph',source=source)


@pytest.mark.parametrize('bad',['그들은 우정을结었다.','그들은 친구가 되었다. 在那里, 그들은 만났다.'])
def test_embedded_han_gets_one_bounded_repair_and_both_responses_are_retained(tmp_path,bad):
    engine=Engine([bad,'그들은 친구가 되었다.'])
    result=translate_record(engine,block('They became friends.'),'general','qwen',tmp_path)
    assert result['status']=='translated' and result['target']=='그들은 친구가 되었다.'
    assert result['attempts']==2 and len(engine.prompts)==2
    assert len(list((tmp_path/'responses').glob('*.json')))==2
    assert 'previous translation failed' in engine.prompts[1]


def test_persistent_sign_error_is_not_silently_accepted(tmp_path):
    engine=Engine(['값은 2.4였다.','값은 2.4였다.'])
    source='The value was -2.4.'
    result=translate_record(engine,block(source),'general','qwen',tmp_path)
    assert len(engine.prompts)==2 and result['status']=='failed'
    assert result['target']==source and result['candidate']=='값은 2.4였다.'


def test_cancellation_is_not_retried(tmp_path):
    engine=Engine([Cancelled()])
    with pytest.raises(Cancelled):translate_record(engine,block('Stop here.'),'general','qwen',tmp_path)
    assert len(engine.prompts)==1


def test_specialist_does_not_receive_generic_repair_instruction(tmp_path):
    engine=Engine(['값은 2.4였다.'])
    result=translate_record(engine,block('The value was -2.4.'),'general','translategemma',tmp_path)
    assert result['status']=='failed' and len(engine.prompts)==1


def test_failed_candidate_is_visible_with_warning_and_not_silently_approved(tmp_path):
    import pymupdf as fitz
    from papertranslator.document import flow_pdf
    record={**block('The value was -2.4.'),'status':'failed','target':'The value was -2.4.','candidate':'값은 2.4였다.'}
    ir=dict(pages=[dict(source_page_index=0,warnings=[],block_ids=[record['id']])])
    path=tmp_path/'candidate.pdf';flow_pdf(ir,{record['id']:record},path,'합성 검사 실패 후보')
    with fitz.open(path) as doc:
        text=doc[0].get_text()
        assert '검토용 번역' in text and '값은 2.4였다.' in text
    assert record['status']=='failed' and record['target']=='The value was -2.4.'


def test_total_weekly_sessions_cannot_become_sessions_per_week(tmp_path):
    engine=Engine(['치료는 주 8회 진행되었다.','치료는 주 1회씩 총 8회 진행되었다.'])
    result=translate_record(engine,block('Treatment consisted of eight weekly sessions.'),'general','hy',tmp_path)
    assert result['attempts']==2 and result['status']=='translated'
    assert '총 8회' in result['target']


def test_severe_omission_requires_retranslation(tmp_path):
    source='This synthetic paragraph describes the purpose, material and procedures of an observation. '*7
    engine=Engine(['관찰이다.','이 합성 문단은 관찰의 목적과 자료 및 절차를 설명한다. '*7])
    result=translate_record(engine,block(source),'general','hy',tmp_path)
    assert result['attempts']==2 and result['status']=='translated'


@pytest.mark.parametrize('source,bad,good',[
    ('Attention Deficit Hyperactivity Disorder (ADHD)','주의력결핍공격성장애(ADHD)','주의력결핍 과잉행동장애(ADHD)'),
    ('Research allows meaningful comparison.','연구는 상당한 ~을 제공한다.','연구는 의미 있는 비교를 가능하게 한다.')])
def test_known_term_and_unfinished_placeholder_require_repair(tmp_path,source,bad,good):
    result=translate_record(Engine([bad,good]),block(source),'general','hy',tmp_path)
    assert result['attempts']==2 and result['status']=='translated' and result['target']==good
