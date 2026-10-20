"""One worker-owned model session, reused only for an identical configuration."""
from contextlib import contextmanager
import time

from .storage import fingerprint


class Sessions:
    def __init__(self,factory):
        self.factory=factory;self.backend=None;self.key=None;self.started=False;self.active=False
        self.metrics=dict(loads=0,reuses=0,startup_seconds=0.,models=[])

    def close(self):
        backend=self.backend
        self.backend=None;self.key=None;self.started=False
        if backend is not None:backend.__exit__(None,None,None)

    @contextmanager
    def __call__(self,folder,cancel=lambda:False,model=None,log_name='engine.log'):
        if self.active:raise RuntimeError('추론 세션을 중첩해서 사용할 수 없습니다.')
        key=fingerprint(dict(folder=str(folder),model=model))
        if self.backend is not None and key!=self.key:self.close()
        if self.backend is None:
            backend=self.factory(folder,cancel,model=model,log_name=log_name)
            try:backend.__enter__()
            except BaseException:
                backend.__exit__(*__import__('sys').exc_info());raise
            self.backend=backend;self.key=key
        else:self.metrics['reuses']+=1
        self.active=True
        try:yield Lease(self)
        except BaseException:
            self.close();raise
        finally:self.active=False


class Lease:
    def __init__(self,pool):self.pool=pool
    def __getattr__(self,name):return getattr(self.pool.backend,name)
    def start(self,progress=lambda _:None):
        if self.pool.started:
            if hasattr(self.pool.backend,'check'):self.pool.backend.check()
            return
        began=time.monotonic()
        self.pool.backend.start(progress)
        self.pool.metrics['startup_seconds']+=time.monotonic()-began
        self.pool.metrics['loads']+=1
        self.pool.metrics['models'].append((getattr(self.pool.backend,'model',None) or {}).get('id'))
        self.pool.started=True
