import threading
import pytest

from papertranslator.backend import Cancelled,LlamaBackend
from papertranslator.inference import completed


def test_two_slots_overlap_without_submitting_an_unbounded_document():
    barrier=threading.Barrier(2,timeout=3);started=[];lock=threading.Lock()
    def work(value):
        with lock:started.append(value)
        barrier.wait()
        return value*10
    iterator=completed(range(6),work,2)
    first=next(iterator)
    assert len(started)==2
    values=[first,*iterator]
    assert dict(values)=={i:i*10 for i in range(6)}


def test_cancellation_aborts_only_inflight_work_and_never_queues_more():
    cancel=threading.Event();release=threading.Event();started=[]
    def work(value):
        started.append(value);cancel.set();release.wait(3);raise Cancelled()
    with pytest.raises(Cancelled):list(completed(range(100),work,2,cancel.is_set,release.set))
    assert release.is_set() and len(started)<=2


def test_parallel_start_failure_retries_with_the_same_per_request_context(monkeypatch,tmp_path):
    import papertranslator.backend as module
    monkeypatch.setattr(module,'read_json',lambda *args:dict(model=dict(id='synthetic')))
    backend=LlamaBackend(tmp_path,model=dict(id='synthetic',options=dict(context=8192,parallel=2)))
    tried=[]
    def start(progress):
        tried.append(dict(backend.options))
        if len(tried)==1:raise RuntimeError('synthetic allocation failure')
    monkeypatch.setattr(backend,'_start',start)
    backend.start()
    assert [x['parallel'] for x in tried]==[2,1]
    assert all(x['context']==8192 for x in tried)
    assert backend.log_name.endswith('-sequential.log')
    backend.close()


def test_cancelled_start_never_falls_back(monkeypatch,tmp_path):
    import papertranslator.backend as module
    monkeypatch.setattr(module,'read_json',lambda *args:dict(model=dict(id='synthetic')))
    backend=LlamaBackend(tmp_path,model=dict(id='synthetic',options=dict(parallel=2)))
    calls=[]
    def start(progress):calls.append(True);raise Cancelled()
    monkeypatch.setattr(backend,'_start',start)
    with pytest.raises(Cancelled):backend.start()
    assert len(calls)==1
    backend.close()
