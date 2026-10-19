import json
import subprocess
import sys

import httpx
import pytest

from papertranslator.cloud import FreeReviewBackend
from papertranslator import cloud
from papertranslator.backend import Cancelled


def transport(calls,price='0',status=200):
    def handle(request):
        calls.append(request)
        if request.url.path.endswith('/models'):
            return httpx.Response(200,json={'data':[{'id':'example/model:free','pricing':{'prompt':price,'completion':'0'}}]})
        return httpx.Response(status,json={'model':'example/model','choices':[{'finish_reason':'stop','message':{'content':'{}'}}],
                                          'usage':{'cost':0}})
    return httpx.MockTransport(handle)


def test_free_review_enforces_zero_price_and_no_provider_fallback():
    calls=[]
    with FreeReviewBackend('example/model:free','synthetic-key',transport=transport(calls)) as backend:
        backend.translate('Synthetic source data')
    payload=json.loads(calls[-1].content)
    assert payload['provider']['max_price']=={'prompt':0,'completion':0,'request':0}
    assert payload['provider']['allow_fallbacks'] is False
    assert payload['provider']['data_collection']=='deny'
    assert calls[-1].url.host=='openrouter.ai'
    assert not any(field in payload for field in ['files','tools','source_path','api_key'])


def test_paid_or_unverified_models_are_never_sent_document_text():
    with pytest.raises(ValueError):FreeReviewBackend('example/paid','synthetic-key')
    calls=[]
    with pytest.raises(ValueError):
        with FreeReviewBackend('example/model:free','synthetic-key',transport=transport(calls,price='0.01')):pass
    assert all(request.method=='GET' for request in calls)


def test_rate_limit_does_not_trigger_retry_or_paid_fallback():
    calls=[]
    with FreeReviewBackend('example/model:free','synthetic-key',transport=transport(calls,status=429)) as backend:
        with pytest.raises(RuntimeError):backend.translate('Synthetic data')
    assert len([r for r in calls if r.method=='POST'])==1


def test_hybrid_network_guard_rejects_other_hosts_and_nonpublic_dns():
    code='''import socket,sys
from papertranslator.offline import install_loopback_guard
socket.getaddrinfo=lambda *args,**kwargs:[(socket.AF_INET,socket.SOCK_STREAM,6,'',('104.18.1.1',443))]
install_loopback_guard(allow_free_api=True)
socket.getaddrinfo('openrouter.ai',443)
sys.audit('socket.connect',socket.socket(),('104.18.1.1',443))
try:socket.getaddrinfo('example.org',443)
except PermissionError:pass
else:raise AssertionError('unrelated host allowed')
try:sys.audit('socket.connect',socket.socket(),('8.8.8.8',443))
except PermissionError:pass
else:raise AssertionError('unapproved address allowed')
'''
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_review_budget_and_resume_never_repeat_private_requests(tmp_path,monkeypatch):
    calls=[]
    class Backend:
        def __init__(self,*args):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
    def edit(backend,source,draft,**kwargs):
        calls.append((source,kwargs['context']))
        return dict(decision='keep',source_evidence='',draft_evidence='',explanation_ko='유지',revision=''),{}
    monkeypatch.setattr(cloud,'load_key',lambda:'synthetic-key')
    monkeypatch.setattr(cloud,'FreeReviewBackend',Backend)
    monkeypatch.setattr(cloud,'edit',edit)
    records={str(i):dict(id=str(i),status='translated',source='Synthetic source',target='합성 번역',issues=[dict(code='synthetic')]) for i in range(4)}
    ir=dict(blocks=[dict(id=str(i),context=dict(before='Previous sentence',after='Next sentence',document='PRIVATE TITLE')) for i in range(4)])
    config=dict(cloud=dict(model='example/model:free',max_requests=2,enabled=True))
    for _ in range(2):cloud.review_records(records,ir,config,tmp_path,lambda:None,lambda _:None,lambda:False)
    assert len(calls)==2 and all(set(context)=={'before','after'} for _,context in calls)
    assert sum(r['cloud_review']['status']=='completed' for r in records.values())==2
    assert sum(r['cloud_review']['status']=='deferred' for r in records.values())==2
    assert sum(r['cloud_review']['attempted'] for r in records.values())==2


def test_review_cancellation_is_not_reported_as_api_failure(tmp_path,monkeypatch):
    class Backend:
        def __init__(self,*args):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
    monkeypatch.setattr(cloud,'load_key',lambda:'synthetic-key')
    monkeypatch.setattr(cloud,'FreeReviewBackend',Backend)
    record=dict(id='1',status='failed',source='Synthetic source',target='원문',issues=[])
    with pytest.raises(Cancelled):
        cloud.review_records({'1':record},dict(blocks=[dict(id='1')]),
            dict(cloud=dict(model='example/model:free',max_requests=2)),tmp_path,lambda:None,lambda _:None,lambda:True)
    assert not any(i['code']=='cloud_unavailable' for i in record['issues'])
