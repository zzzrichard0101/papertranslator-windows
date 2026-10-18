"""Synthetic real-GPU end-to-end run; no user papers or Vault access."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import pymupdf as fitz

from papertranslator.jobs import create_job
from papertranslator.storage import atomic_json, read_json, sha256


def main():
    project=Path(__file__).resolve().parents[1]
    bench=project/'benchmarks'/'runs'; bench.mkdir(parents=True,exist_ok=True)
    vault=project/'test_vault'; (vault/'.obsidian').mkdir(parents=True,exist_ok=True)
    source=bench/'Synthetic Windows Study.pdf'
    with fitz.open() as doc:
        page=doc.new_page()
        page.insert_text((48,65),'A Synthetic Art Therapy Study',fontsize=19)
        page.insert_textbox(fitz.Rect(48,105,545,200),
            'The study included 24 participants. They were assigned to two groups. '
            'The association was not statistically significant (p = 0.032). '
            'The findings do not establish a causal relationship.',fontsize=12)
        page.insert_textbox(fitz.Rect(48,235,545,310),
            'Anna became the mentee of Rosa. Her father and two aunts attended the exhibition. '
            'The occupational therapist used oil pastels in the group session.',fontsize=12)
        page=doc.new_page()
        page.insert_text((48,65),'Computing and Statistical Notes',fontsize=19)
        page.insert_textbox(fitz.Rect(48,105,545,205),
            'The algorithm takes O(n log n) time. A thread must not modify the shared counter without a lock. '
            'The value was 1.25, and the sample size was 24. See https://example.org/study [1].',fontsize=12)
        page.insert_textbox(fitz.Rect(48,245,545,300),'def update(counter):\n    return counter + 1',fontname='cour',fontsize=11)
        doc.save(source)
    job=create_job(source,vault,title='Windows 합성 연구')
    atomic_json(bench/'latest-job.json',{'folder':str(job),'source':str(source)})
    command=[sys.executable,'-m','papertranslator','--worker',str(job)]
    p=subprocess.Popen(command)
    telemetry=[]; last=0
    while p.poll() is None:
        if time.monotonic()-last>2:
            r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw','--format=csv,noheader'],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
            telemetry.append(dict(time=time.time(),gpu=r.stdout.strip())); last=time.monotonic()
        time.sleep(.5)
    state=read_json(job/'state.json')
    atomic_json(bench/'gpu-telemetry.json',telemetry)
    atomic_json(bench/'integration-result.json',state)
    print(json.dumps(state,ensure_ascii=False,indent=2))
    if state['status']!='completed_with_warnings': return 1
    assert sha256(source)==sha256(Path(state['output'])/'Original.pdf')
    return 0


if __name__=='__main__': raise SystemExit(main())
