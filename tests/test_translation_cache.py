from copy import deepcopy
import pytest

from papertranslator.checkpoint import Checkpoint
from papertranslator.translation import translate_record
from papertranslator.storage import read_json


class Engine:
    options=dict(context=4096,temperature=.2,seed=42)
    model=dict(id='synthetic',sha256='synthetic',adapter='hy')
    registry=dict(runtime_release='synthetic')
    def __init__(self):self.calls=[]
    def translate(self,prompt,**kwargs):
        self.calls.append(prompt)
        return dict(text='연구 방법',seconds=1.,usage=dict(completion_tokens=4))


def block(key='first'):
    return dict(id=key,kind='heading',source='Research methods',source_page_index=0,
                source_page_indices=[0],bbox=[10,10,80,30],source_fragments=[])


def test_identical_input_reuses_validated_text_with_its_own_geometry(tmp_path):
    store=Checkpoint(tmp_path,'job');engine=Engine()
    first=translate_record(engine,block(),'general','hy',tmp_path,memory=store)
    other={**block('second'),'source_page_index':3,'source_page_indices':[3],'bbox':[50,200,200,240]}
    second=translate_record(engine,other,'general','hy',tmp_path,memory=store)
    assert len(engine.calls)==1 and second['target']==first['target']=='연구 방법'
    assert second['bbox']==other['bbox'] and second['source_page_indices']==[3]
    assert second['seconds']==0 and second['translation_cache']['origin_id']=='first'
    assert read_json(tmp_path/'responses/second.json')['cache']['original_usage']==dict(completion_tokens=4)
    store.close()


@pytest.mark.parametrize('change',['context','model','options','source'])
def test_changed_model_settings_context_or_unmasked_source_never_share_a_cache(tmp_path,change):
    store=Checkpoint(tmp_path,'job');engine=Engine();original=block()
    translate_record(engine,original,'general','hy',tmp_path,memory=store)
    other=block('second')
    if change=='context':other['context']=dict(before='A different neighbouring paragraph.')
    elif change=='model':engine.model={**engine.model,'sha256':'different weights'}
    elif change=='options':engine.options={**engine.options,'temperature':.3}
    else:other['source']='Research Methods'
    translate_record(engine,other,'general','hy',tmp_path,memory=store)
    assert len(engine.calls)==2
    store.close()


def test_cached_response_still_has_to_pass_current_quality_checks(tmp_path):
    store=Checkpoint(tmp_path,'job');engine=Engine()
    translate_record(engine,block(),'general','hy',tmp_path,memory=store)
    hit=translate_record(engine,block('second'),'general','hy',tmp_path,memory=store)
    key=hit['translation_cache']['key'];cached=store.get_response(key)
    cached['raw']['text']='';store.put_response(key,cached)
    final=translate_record(engine,block('third'),'general','hy',tmp_path,memory=store)
    assert len(engine.calls)==2 and final['status']=='translated'
    store.close()
