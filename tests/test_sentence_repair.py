import re
import pytest
from papertranslator.backend import Cancelled
from papertranslator.quality import check
from papertranslator.sentence_repair import plan,splice
from papertranslator.translation import translate_record

SOURCE=('Participants completed drawing tasks (Example, 2021). '
        'The observed value was -2.4 (Sample, 2022). '
        'The team discussed the result without claiming causation (Writer, 2023).')
DRAFT=('참가자들은 그림 과제를 마쳤다 (Example, 2021). '
       '관측값은 2.4였다 (Sample, 2022). '
       '연구진은 인과관계를 주장하지 않고 결과를 논의했다 (Writer, 2023).')
FIX='관측값은 -2.4였다 (Sample, 2022).'


def block():
    return dict(id='synthetic',source_page_index=0,source_page_indices=[0],source_fragments=[],bbox=[0,0,100,50],kind='paragraph',source=SOURCE,selective_repair=True)


def test_unique_literal_citation_localises_one_error_and_preserves_every_other_character():
    window=plan(SOURCE,DRAFT,check(SOURCE,DRAFT));assert window
    assert window['source']=='The observed value was -2.4 (Sample, 2022).'
    fixed=splice(SOURCE,DRAFT,window,FIX)
    assert fixed==DRAFT.replace('관측값은 2.4였다','관측값은 -2.4였다')
    assert fixed[:window['target_start']]==DRAFT[:window['target_start']]
    with pytest.raises(ValueError):splice(SOURCE,DRAFT,window,'관측값은 -2.4였다.')


def test_missing_anchor_ambiguous_repetition_and_multiple_errors_force_full_retry():
    assert plan(SOURCE,DRAFT.replace('(Sample, 2022)',''),check(SOURCE,DRAFT)) is None
    source=SOURCE+' Another account (Sample, 2022).'
    target=DRAFT+' 다른 설명 (Sample, 2022).'
    assert plan(source,target,check(source,target)) is None
    assert plan(SOURCE,DRAFT.replace('2021','2020'),check(SOURCE,DRAFT.replace('2021','2020'))) is None
    assert plan(SOURCE,DRAFT,[dict(code='undertranslation',severity='major')]) is None


class Backend:
    options=dict(context=4096)
    def __init__(self,replies):self.replies=iter(replies);self.prompts=[]
    def translate(self,prompt,**kwargs):
        self.prompts.append(prompt);reply=next(self.replies)
        if isinstance(reply,Exception):raise reply
        literals=re.findall(r'\([A-Z][a-z]+, 20\d{2}\)',reply)
        tokens=list(dict.fromkeys(re.findall(r'\[PX[A-F0-9]{8}X\d+\]',prompt)))
        for literal,token in zip(literals,tokens):reply=reply.replace(literal,token)
        return dict(text=reply,seconds=.1)


def test_worker_uses_short_window_and_still_checks_the_whole_paragraph(tmp_path):
    engine=Backend([DRAFT,FIX]);result=translate_record(engine,block(),'general','qwen',tmp_path)
    assert result['status']=='translated' and result['attempts']==2
    assert result['sentence_repair']['status']=='accepted'
    assert result['target']==DRAFT.replace('관측값은 2.4였다','관측값은 -2.4였다')
    assert len(list((tmp_path/'responses').glob('*.json')))==2
    assert 'Translate only this source sentence' in engine.prompts[1]
    assert SOURCE not in engine.prompts[1]


def test_rejected_window_falls_back_to_original_full_retry_with_a_bounded_extra_call(tmp_path):
    engine=Backend([DRAFT,'관측값은 2.4였다 (Sample, 2022).',DRAFT.replace('관측값은 2.4였다','관측값은 -2.4였다')])
    result=translate_record(engine,block(),'general','qwen',tmp_path)
    assert result['status']=='translated' and result['attempts']==3
    assert result['sentence_repair']['status']=='rejected'
    assert 'previous translation failed automatic validation' in engine.prompts[-1]


def test_cancel_during_local_repair_never_starts_a_full_retry(tmp_path):
    engine=Backend([DRAFT,Cancelled()])
    with pytest.raises(Cancelled):translate_record(engine,block(),'general','qwen',tmp_path)
    assert len(engine.prompts)==2
