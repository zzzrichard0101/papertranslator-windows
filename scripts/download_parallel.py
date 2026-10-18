"""Bounded HTTPS ranged download; final full hash is the promotion boundary."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.request

from bootstrap import ROOT, digest
from papertranslator.storage import atomic_json, read_json


def fetch(asset):
    target=ROOT/'models'/asset['name']; target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and digest(target)==asset['sha256']: return target
    work=target.with_suffix('.gguf.ranged'); meta=target.with_suffix('.gguf.ranges.json')
    prior=target.with_suffix(target.suffix+'.part')
    if work.exists():
        state=read_json(meta)
        if not state or state['sha256']!=asset['sha256']: raise RuntimeError('Ranged download metadata mismatch')
    else:
        prefix=prior.stat().st_size if prior.exists() else 0
        if shutil.disk_usage(ROOT).free < asset['size']-prefix+1024**3: raise RuntimeError('Insufficient disk')
        if prior.exists(): prior.replace(work)
        else: work.touch()
        with work.open('r+b') as f: f.truncate(asset['size'])
        state=dict(sha256=asset['sha256'],size=asset['size'],prefix=prefix,done={})
        atomic_json(meta,state)
    block=64*1024**2; lock=threading.Lock(); started=time.monotonic(); last=[started]
    spans=[(start,min(start+block,asset['size'])) for start in range(state['prefix'],asset['size'],block)]
    def task(span):
        start,end=span; key=str(start)
        if key in state['done']: return
        for attempt in range(4):
            try:
                request=urllib.request.Request(asset['url'],headers={'Range':f'bytes={start}-{end-1}'})
                with urllib.request.urlopen(request,timeout=45) as response, work.open('r+b',buffering=0) as file:
                    if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end-1}/{asset["size"]}':
                        raise RuntimeError('Range response mismatch')
                    file.seek(start); received=0; h=hashlib.sha256()
                    while data:=response.read(min(4*1024**2,end-start-received)):
                        file.write(data); h.update(data); received+=len(data)
                        if received==end-start: break
                    if received!=end-start: raise RuntimeError('Short chunk')
                    os.fsync(file.fileno())
                with lock:
                    state['done'][key]=dict(size=end-start,sha256=h.hexdigest())
                    atomic_json(meta,state)
                    if time.monotonic()-last[0]>4:
                        total=state['prefix']+sum(x['size'] for x in state['done'].values())
                        print(json.dumps(dict(file=asset['name'],bytes=total,total=asset['size'],connections=4)),flush=True)
                        last[0]=time.monotonic()
                return
            except Exception:
                if attempt==3: raise
                time.sleep(2**attempt)
    with ThreadPoolExecutor(max_workers=4) as executor:
        for future in as_completed([executor.submit(task,span) for span in spans]): future.result()
    if digest(work)!=asset['sha256']: raise RuntimeError('Complete SHA256 mismatch; model not registered')
    work.replace(target)
    print(json.dumps(dict(file=asset['name'],status='verified',seconds=round(time.monotonic()-started,1))),flush=True)
    return target
