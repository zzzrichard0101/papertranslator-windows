import pytest
from papertranslator.reviewer import validate_review


def test_review_must_quote_actual_source_and_target():
    value=dict(verdict='fail',severity='critical',source_evidence='not',target_evidence='입증한다',explanation_ko='부정 누락',suggested_revision='입증하지 않는다')
    assert validate_review(value,'does not establish','입증한다')['evidence_validated']
    with pytest.raises(ValueError,match='원문 근거'):validate_review({**value,'source_evidence':'never'},'does not establish','입증한다')
    with pytest.raises(ValueError,match='누락'):validate_review({**value,'source_evidence':''},'does not establish','입증한다')


def test_uncertain_verdict_does_not_require_invented_evidence():
    value=dict(verdict='uncertain',severity='none',source_evidence='',target_evidence='',explanation_ko='추가 확인 필요',suggested_revision='')
    assert validate_review(value,'source','번역')['verdict']=='uncertain'
