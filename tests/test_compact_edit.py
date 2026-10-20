import copy
import json
import pytest
from papertranslator.backend import Cancelled
from papertranslator.compact_edit import validate
from papertranslator.refinement import edit_record

SOURCE='Research evidence does not replace clinical judgement. '+'The study describes the materials and procedures. '*8
DRAFT=('연구 근거는 임상 판단을 대체한다. '+'연구는 인과적 주장을 추가하지 않고 관찰에 사용된 자료와 절차를 설명한다. '*8).strip()
TARGET=DRAFT.replace('대체한다','대체하지 않는다')


def patch():return dict(decision='patch',patches=[dict(source_evidence='does not replace',before='대체한다',after='대체하지 않는다')],explanation_ko='부정 복원')


def test_exact_local_patch_keeps_all_other_characters_and_uses_full_validation():
    value=validate(patch(),SOURCE,DRAFT)
    assert value['revision']==TARGET and value['decision']=='revise'
    assert value['output_format']=='source-grounded-patches-1'
    assert validate(dict(decision='keep',patches=[],explanation_ko=''),SOURCE,TARGET)['revision']==''


@pytest.mark.parametrize('change',['source','missing','duplicate','overlap','number','citation','broad','full','extra','keep_patches'])
def test_bad_patch_candidates_are_rejected(change):
    value=patch();source=SOURCE;draft=DRAFT
    if change=='source':value['patches'][0]['source_evidence']='not in source'
    if change=='missing':value['patches'][0]['before']='not in draft'
    if change=='duplicate':draft+=' 대체한다'
    if change=='overlap':value['patches'].append(dict(source_evidence='clinical judgement',before='판단을 대체한다',after='판단을 대체하지 않는다'))
    if change=='number':value['patches'][0]['after']='대체하지 않는다 777'
    if change=='citation':
        source+=' (Example, 2021)';draft+=' (Example, 2021)'
        value['patches'][0].update(before='(Example, 2021)',after='')
    if change=='broad':draft='대체한다'
    if change=='full':value.update(decision='full',patches=[])
    if change=='extra':value['unexpected']='value'
    if change=='keep_patches':value['decision']='keep'
    with pytest.raises(ValueError):validate(value,source,draft)


def record():return dict(id='synthetic',source=SOURCE,target=DRAFT,status='translated',issues=[])
def settings():return dict(editor=dict(id='synthetic'),editor_output='patch-v1')


class Backend:
    def __init__(self,replies):self.replies=iter(replies);self.calls=[]
    def translate(self,prompt,**kwargs):
        self.calls.append((prompt,kwargs));reply=next(self.replies)
        if isinstance(reply,Exception):raise reply
        return dict(text=json.dumps(reply,ensure_ascii=False),seconds=.01)


def test_long_paragraph_review_saves_patch_provenance_without_mutating_input(tmp_path):
    backend=Backend([patch()]);original=record();before=copy.deepcopy(original)
    result=edit_record(backend,original,{},settings(),'stamp',tmp_path,lambda:False)
    assert result['target']==TARGET and result['refinement']['accepted']
    assert result['refinement']['compact']['status']=='accepted' and original==before
    assert (tmp_path/'responses/edit-synthetic-patch.json').exists()
    assert 'COMPLETE English SOURCE' in backend.calls[0][0] and SOURCE in backend.calls[0][0] and DRAFT in backend.calls[0][0]


def test_rejected_patch_uses_the_existing_complete_edit_fallback_and_saves_both(tmp_path):
    bad=patch();bad['patches'][0]['before']='not in draft'
    full=dict(decision='revise',source_evidence='does not replace',draft_evidence='대체한다',explanation_ko='부정 복원',revision=TARGET)
    backend=Backend([bad,full])
    result=edit_record(backend,record(),{},settings(),'stamp',tmp_path,lambda:False)
    assert result['target']==TARGET and result['refinement']['compact']['status']=='fallback'
    assert len(backend.calls)==2 and len(list((tmp_path/'responses').glob('*.json')))==2


def test_cancel_during_patch_does_not_fall_back(tmp_path):
    backend=Backend([Cancelled()])
    with pytest.raises(Cancelled):edit_record(backend,record(),{},settings(),'stamp',tmp_path,lambda:False)
    assert len(backend.calls)==1
