from copy import deepcopy
import re
import pytest

from papertranslator.backend import Cancelled
from papertranslator.checkpoint import Checkpoint
from papertranslator.quality import check,protect,restore
from papertranslator.sessions import Sessions
from papertranslator.translation import translate_record
from papertranslator.typed_literals import preserved_kind


@pytest.mark.parametrize('target',['30,000회의 반복','3만 회의 반복','30,000회'])
def test_equal_compact_counts_do_not_trigger_retry(target):
    assert not check('30K iterations',target,typed=True)
    assert any(i['code']=='numbers' for i in check('30K iterations',target))


@pytest.mark.parametrize('source,target',[
    ('30K iterations','3천 회 반복'),('30K iterations','3만 개'),
    ('-30K iterations','3만 회 반복'),('30K points','3만 개의 가우시안'),
    ('+30K iterations','3만 회 반복'),('30K iterations','+3만 회 반복'),
    ('30K iterations and 7K iterations','3만 회 반복'),
    ('n = 24, -1.25','n = 24, 1.25'),('Ours-30K','우리의 30,000개')])
def test_numeric_role_sign_count_and_identifier_errors_still_block(source,target):
    assert any(i['severity'] in {'major','critical'} for i in check(source,target,typed=True))


def test_citation_list_and_url_are_preserved_without_model_interpretation():
    source='See [Gortler et al. 1996; Levoy and Hanrahan 1996] and openreview.net/forum?id=abc.'
    masked,mapping=protect(source,typed=True)
    assert len(mapping)==2 and 'Levoy' not in masked and restore(masked,mapping)==source
    target='참조: [Gortler et al. 1996] 및 openreview.net/forum?id=abc.'
    assert any(i['code']=='literal' for i in check(source,target,typed=True))


def test_only_pure_scientific_objects_bypass_translation():
    for value in ['L = (1 −𝜆)L1 + 𝜆LD-SSIM','Ours-30K','openreview.net/forum?id=abc']:
        assert preserved_kind(value) and not check(value,value,typed=True)
    for value in ['Research evidence = clinical knowledge.','THE RESULT = FAIL','The loss = 1 for every participant.','Research methods']:
        assert preserved_kind(value) is None


@pytest.mark.parametrize('validation',['typed-v1','typed-v2','typed-v3'])
def test_resolved_cache_ignores_random_tokens_but_not_context(tmp_path,validation):
    class Engine:
        options=dict(context=4096,temperature=.2,seed=42)
        model=dict(id='synthetic',sha256='synthetic',adapter='hy')
        registry=dict(runtime_release='synthetic')
        def __init__(self):self.calls=[]
        def translate(self,prompt,**kwargs):
            self.calls.append(prompt)
            return dict(text='연구 결과 '+re.search(r'\[PX[A-F0-9]+X0\]',prompt)[0],seconds=1.,usage={})
    block=dict(id='a',source='Research findings [Miller et al. 2020].',kind='paragraph',
        source_page_index=0,source_page_indices=[0],source_fragments=[],bbox=[0,0,10,10],validation_policy=validation)
    engine=Engine();store=Checkpoint(tmp_path,'test')
    first=translate_record(engine,block,'general','hy',tmp_path,store)
    second=translate_record(engine,{**block,'id':'b','bbox':[30,30,40,40]},'general','hy',tmp_path,store)
    assert first['status']==second['status']=='translated' and first['target']==second['target']
    assert len(engine.calls)==1 and second['seconds']==0 and second['bbox']==[30,30,40,40]
    translate_record(engine,{**block,'id':'c','context':dict(before='Different context.')},'general','hy',tmp_path,store)
    assert len(engine.calls)==2
    key=second['translation_cache']['key'];cached=store.get_response(key)
    cached['raw']['text']='연구 결과';store.put_response(key,cached)
    translate_record(engine,{**block,'id':'d'},'general','hy',tmp_path,store)
    assert len(engine.calls)==3
    cached=store.get_response(key);cached['raw']['text']+=' [PXAABBCCDDX0]';store.put_response(key,cached)
    translate_record(engine,{**block,'id':'e'},'general','hy',tmp_path,store)
    assert len(engine.calls)==4
    store.close()


def test_session_reuse_does_not_change_options_and_closes_before_model_switch(tmp_path):
    events=[]
    class Engine:
        def __init__(self,*a,model=None,**kw):self.model=model;self.options=deepcopy(model['options'])
        def __enter__(self):events.append(('enter',self.model['id']));return self
        def __exit__(self,*a):events.append(('close',self.model['id']))
        def start(self,*a):events.append(('start',self.model['id']))
    pool=Sessions(Engine);model=dict(id='editor',options=dict(context=8192,temperature=.1))
    for name in ['edit','labels','alignment']:
        with pool(tmp_path,model=deepcopy(model),log_name=name) as backend:
            backend.start();assert backend.options==model['options']
    assert pool.metrics['loads']==1 and pool.metrics['reuses']==2
    other=dict(id='other',options=model['options'])
    with pool(tmp_path,model=other) as backend:backend.start()
    pool.close()
    assert events.index(('close','editor'))<events.index(('start','other'))
    assert events.count(('close','other'))==1


def test_session_cancel_and_start_failure_never_leave_an_owned_engine(tmp_path):
    closed=[]
    class Engine:
        def __init__(self,*a,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):closed.append(True)
        def start(self,*a):raise Cancelled()
    pool=Sessions(Engine)
    with pytest.raises(Cancelled):
        with pool(tmp_path,model={}) as backend:backend.start()
    assert pool.backend is None and len(closed)==1
    pool.close();assert len(closed)==1
