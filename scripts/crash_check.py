"""Kill the owned worker, assert its Job Object cleans engine, then resume."""
from pathlib import Path
import subprocess
import time

import psutil

from papertranslator.jobs import create_job
from papertranslator.storage import atomic_json, read_json

project=Path(__file__).resolve().parents[1]
folder=create_job(project/'benchmarks'/'runs'/'Synthetic Windows Study.pdf',project/'test_vault',title='강제 종료 복구 시험')
command=[str(project/'dist'/'PaperTranslator'/'PaperTranslator.exe'),'--worker',str(folder)]
process=subprocess.Popen(command); started=time.monotonic(); engines=[]
while time.monotonic()-started<90:
    state=read_json(folder/'state.json',{})
    if state.get('completed',0)>=1:
        engines=[p.pid for p in psutil.Process(process.pid).children(recursive=True) if p.name()=='llama-server.exe']
        break
    if process.poll() is not None: raise RuntimeError(state)
    time.sleep(.03)
assert engines, 'No owned engine observed'
process.kill(); process.wait(timeout=15)
for _ in range(100):
    if not any(psutil.pid_exists(pid) for pid in engines): break
    time.sleep(.05)
assert not any(psutil.pid_exists(pid) for pid in engines), 'Owned engine leaked'
checkpoint=read_json(folder/'checkpoint.json')
subprocess.run(command,check=True,timeout=120)
state=read_json(folder/'state.json')
assert state['status']=='completed_with_warnings' and state['cache_hits']>=len(checkpoint['records'])
result=dict(forced_worker_exit='passed',owned_engine_cleanup='passed',resume='passed',cache_hits=state['cache_hits'],job=str(folder))
atomic_json(project/'exports'/'crash-recovery.json',result)
print(result)
