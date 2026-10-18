import math

import pytest

from papertranslator.eta import TranslationETA, remaining_label, work_units


def calibrated(rate=.02):
    estimator=TranslationETA()
    for _ in range(10):estimator.observe(800,800*rate)
    return estimator


def ready_state(**values):
    eta=dict(state='ready',seconds=900,lower_seconds=720,upper_seconds=1080,
             current_budget_seconds=90,at_elapsed_seconds=100)
    eta.update(values)
    return dict(status='translating',eta=eta)


def test_calibration_requires_enough_normal_observations():
    estimator=TranslationETA()
    for _ in range(2):estimator.observe(1000,20)
    assert estimator.estimate(10000)['state']=='calibrating'
    estimator.observe(1000,20)
    assert estimator.estimate(10000)['state']=='ready'
    assert estimator.estimate(0)['state']=='finishing'
    assert TranslationETA().estimate(0)['state']=='finishing'


def test_alternating_short_and_long_paragraphs_do_not_swing_eta():
    estimator=TranslationETA();values=[]
    for chars in [20,1600]*20:
        units=work_units('x'*chars)
        estimator.observe(units,units*.02)
        eta=estimator.estimate(100000)
        if eta['state']=='ready':values.append(eta['seconds'])
    assert len(values)>20 and max(values)/min(values)<1.01


def test_one_slow_paragraph_and_one_retry_do_not_redefine_all_future_work():
    estimator=calibrated();before=estimator.estimate(100000)['seconds']
    estimator.observe(800,180,attempts=2)
    after=estimator.estimate(100000)['seconds']
    assert before<=after<before*1.15
    estimator.observe(800,180)
    assert estimator.estimate(100000)['seconds']<before*1.2


@pytest.mark.parametrize('new_rate',[.08,.004])
def test_sustained_speed_change_recalibrates_then_adapts(new_rate):
    estimator=calibrated()
    for _ in range(3):estimator.observe(800,800*new_rate)
    assert estimator.estimate(100000)['state']=='recalibrating'
    for _ in range(3):estimator.observe(800,800*new_rate)
    eta=estimator.estimate(100000)
    assert eta['state']=='ready'
    assert eta['seconds']==pytest.approx(100000*new_rate)


def test_failures_and_invalid_times_do_not_become_normal_speed():
    estimator=TranslationETA()
    for time in [0,-1,math.nan,math.inf]:estimator.observe(800,time)
    estimator.observe(800,180,attempts=2,success=False)
    assert estimator.estimate(100000)['state']=='calibrating'
    for _ in range(3):estimator.observe(800,16)
    eta=estimator.estimate(100000)
    assert eta['state']=='ready' and 2000<=eta['seconds']<=2600


def test_countdown_uses_estimate_clock_and_does_not_expire_after_30_seconds():
    state=ready_state();state['updated_at']='unrelated-state-write'
    assert remaining_label(state,100,True)=='약 10~20분'
    assert remaining_label(state,145,True)=='약 10~20분'
    # A delayed paragraph gets a clear explanation, not an invented one-hour ETA.
    assert '현재 문단 지연' in remaining_label(state,195,True)
    assert remaining_label(state,145,False)=='—'


def test_stage_and_uncertainty_labels():
    assert '속도 측정 중' in remaining_label(dict(status='translating'),0,True)
    assert remaining_label(dict(status='translating',eta={'state':'loading'}),0,True)=='모델 준비 중'
    assert remaining_label(dict(status='analyzing'),0,True)=='PDF 분석·OCR 중'
    assert remaining_label(dict(status='reviewing'),0,True)=='추가 검수 중'
    assert remaining_label(dict(status='typesetting'),0,True)=='PDF 생성 중'
    assert remaining_label(ready_state(lower_seconds=10,upper_seconds=50),100,True)=='약 1분 이내'
    assert '재측정' in remaining_label(ready_state(lower_seconds=0,upper_seconds=0),100,True)
