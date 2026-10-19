"""Length-aware translation estimates; ranges are heuristics, not guarantees."""
import math
from statistics import median

VERSION = 1


def work_units(text):
    # Account for fixed request/prompt work as well as the variable paragraph.
    # This is a size proxy calibrated on this run, not an exact token count.
    return 40 + len(text)


class TranslationETA:
    def __init__(self):
        self.samples = []
        self.rate = None
        self.settling = 0

    def observe(self, units, seconds, attempts=1, success=True):
        if units <= 0 or not math.isfinite(seconds) or seconds <= 0:
            return
        sample = dict(units=units, seconds=seconds, attempts=max(1, attempts), success=success)
        self.samples.append(sample)
        self.samples = self.samples[-40:]
        clean = [s for s in self.samples if s['success'] and s['attempts'] == 1]
        if not clean:
            return
        rates = [s['seconds']/s['units'] for s in clean]
        # A sustained speed change invalidates the old estimate. A single retry
        # or slow paragraph must not imply that every remaining paragraph is slow.
        if success and attempts == 1:
            if self.settling:
                self.settling -= 1
            elif self.rate and len(rates) >= 6:
                recent = rates[-3:]
                if all(r > self.rate*1.8 for r in recent) or all(r < self.rate/1.8 for r in recent):
                    self.samples = clean[-3:]
                    rates = recent
                    self.rate = median(recent)
                    self.settling = 3
        target = median(rates)
        self.rate = target if self.rate is None else math.exp(.8*math.log(self.rate)+.2*math.log(target))

    def estimate(self, remaining_units, current_units=0):
        if remaining_units <= 0:
            return dict(state='finishing', seconds=None)
        clean = [s for s in self.samples if s['success'] and s['attempts'] == 1]
        if self.settling:
            return dict(state='recalibrating', seconds=None)
        if len(clean) < 3 or sum(s['units'] for s in clean) < 1200 or self.rate is None:
            return dict(state='calibrating', seconds=None)
        logs = [math.log(s['seconds']/s['units']) for s in clean]
        middle = median(logs)
        deviation = median(abs(value-middle) for value in logs)*1.4826
        spread = max(.3 if len(clean) >= 8 else .5, min(.9, deviation*1.5))
        # Amortize observed extra attempts; do not learn timeout/retry latency as
        # the normal speed of every future first attempt.
        retry_ratio = sum(s['attempts']-1 for s in self.samples)/len(self.samples)
        multiplier = 1+min(1., retry_ratio)
        seconds = remaining_units*self.rate*multiplier
        return dict(state='ready', seconds=round(seconds, 2),
                    lower_seconds=round(seconds*math.exp(-spread), 2),
                    upper_seconds=round(seconds*math.exp(spread), 2),
                    current_budget_seconds=max(30., current_units*self.rate*math.exp(spread)*2),
                    samples=len(clean), version=VERSION)


def _minutes(seconds, lower=False):
    minutes = max(1, math.ceil(seconds/60))
    # Avoid apparent precision in long jobs and minute-by-minute oscillation.
    step = 1 if minutes < 10 else (5 if minutes < 60 else 10)
    if lower:
        return max(1, int(math.floor(seconds/60/step)*step))
    return int(math.ceil(minutes/step)*step)


def remaining_label(state, elapsed_seconds, active):
    """UI countdown is anchored to the estimate, not unrelated state writes."""
    if not active:
        return '—'
    if state.get('status') != 'translating':
        return {'queued':'대기 중', 'preparing':'원본 확인 중', 'analyzing':'PDF 분석·OCR 중',
                'reviewing':'추가 검수 중','refining':'다중 모델 교정 중','cloud_reviewing':'무료 API 검토 중', 'typesetting':'PDF 생성 중', 'saving':'저장 중'}.get(state.get('status'), '준비 중')
    eta = state.get('eta') or {}
    phase = eta.get('state')
    if phase == 'loading':
        return '모델 준비 중'
    if phase == 'finishing':
        return '번역 마무리 중'
    if phase == 'recalibrating':
        return '처리 속도 재측정 중'
    if phase != 'ready':
        return '번역 속도 측정 중'
    age = max(0., elapsed_seconds-eta.get('at_elapsed_seconds', elapsed_seconds))
    if age > eta.get('current_budget_seconds', 30):
        return '현재 문단 지연 · 처리 속도 재측정 중'
    lower = max(0., eta['lower_seconds']-age)
    upper = max(0., eta['upper_seconds']-age)
    if upper <= 0:
        return '처리 속도 재측정 중'
    if upper < 60:
        return '약 1분 이내'
    a, b = _minutes(lower,lower=True), _minutes(upper)
    return f'약 {a}~{b}분' if a != b else f'약 {b}분'
