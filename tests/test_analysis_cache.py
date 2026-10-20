import pytest
from papertranslator.analysis_cache import get_analysis
from papertranslator.storage import read_json,atomic_json
from papertranslator.backend import Cancelled


def test_complete_analysis_reuse_and_changed_contract_invalidation(tmp_path):
    identity=dict(hash='source',pages=[0,1],parser='one',ocr='one');calls=[]
    def analyze():calls.append(True);return dict(blocks=[dict(id='one')],pages=[dict(source_page_index=0)])
    first,hit=get_analysis(tmp_path,identity,analyze,lambda:False);assert not hit
    second,hit=get_analysis(tmp_path,identity,analyze,lambda:False);assert hit and first==second and len(calls)==1
    for key,value in [('hash','changed'),('pages',[1]),('parser','two'),('ocr','two')]:
        _,hit=get_analysis(tmp_path,{**identity,key:value},analyze,lambda:False);assert not hit
    assert len(calls)==5


def test_corrupt_analysis_payload_is_recomputed(tmp_path):
    def analyze():return dict(blocks=[],pages=[])
    get_analysis(tmp_path,dict(hash='same'),analyze,lambda:False)
    cache=read_json(tmp_path/'analysis-cache.json');cache['ir']['blocks']=[dict(id='corrupted')]
    atomic_json(tmp_path/'analysis-cache.json',cache)
    ir,hit=get_analysis(tmp_path,dict(hash='same'),analyze,lambda:False)
    assert not hit and ir['blocks']==[]


def test_cancelled_analysis_is_never_saved_as_complete(tmp_path):
    cancelled=[]
    def analyze():cancelled.append(True);return dict(blocks=[],pages=[])
    with pytest.raises(Cancelled):get_analysis(tmp_path,{},analyze,lambda:bool(cancelled))
    assert not (tmp_path/'analysis-cache.json').exists()
