"""Bounded independent requests; never share paragraph text between requests."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from .backend import Cancelled


def completed(items, function, workers=1, cancel=lambda:False, abort=lambda:None):
    if workers not in {1,2}:raise ValueError('지원하지 않는 동시 처리 수')
    if workers==1:
        for item in items:
            if cancel():raise Cancelled()
            yield item,function(item)
        return
    pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='translation')
    iterator=iter(items);pending={};finished=False
    def submit():
        try:item=next(iterator)
        except StopIteration:return
        pending[pool.submit(function,item)]=item
    try:
        for _ in range(workers):submit()
        while pending:
            if cancel():raise Cancelled()
            done,_=wait(pending,timeout=.1,return_when=FIRST_COMPLETED)
            for future in done:
                item=pending.pop(future)
                yield item,future.result()
                submit()
        finished=True
    finally:
        if not finished:
            abort()
            for future in pending:future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)


def parallel_model(model,workers):
    if workers==1:return model
    return {**model,'options':{**model.get('options',{}),'parallel':workers}}


def slots(backend):
    return (getattr(backend,'options',None) or {}).get('parallel',1)
