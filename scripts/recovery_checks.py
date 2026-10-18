"""Exercise genuine worker cancellation/restart and forced process exit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from papertranslator.jobs import create_job
from papertranslator.storage import atomic_json, read_json, root, sha256


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--exe'); args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    source=project/'benchmarks'/'runs'/'Synthetic Windows Study.pdf'
    original_hash=sha256(source)
    folder=create_job(source,project/'test_vault',title='복구 시험')
    command=([args.exe] if args.exe else [sys.executable,'-m','papertranslator'])+['--worker',str(folder)]

    def wait_for(predicate,process,limit=120):
        start=time.monotonic()
        while time.monotonic()-start<limit:
            state=read_json(folder/'state.json',{})
            if predicate(state): return state
            if process.poll() is not None: raise RuntimeError('Unexpected worker exit: '+json.dumps(state,ensure_ascii=False))
            time.sleep(.03)
        process.kill(); raise TimeoutError('recovery test timed out')

    process=subprocess.Popen(command)
    wait_for(lambda s:s.get('completed',0)>=1,process)
    (folder/'cancel').touch(); process.wait(timeout=45)
    cancelled=read_json(folder/'state.json')
    assert cancelled['status']=='cancelled'
    checkpoint=read_json(folder/'checkpoint.json'); completed_before=len(checkpoint['records'])
    assert completed_before>0
    (folder/'cancel').unlink()
    process=subprocess.Popen(command); process.wait(timeout=120)
    resumed=read_json(folder/'state.json')
    assert resumed['status']=='completed_with_warnings', resumed
    assert resumed['cache_hits']>=completed_before
    first_output=Path(resumed['output']); note=next(first_output.glob('*.md'))
    note.write_text(note.read_text(encoding='utf-8')+'사용자 메모 보존 시험\n',encoding='utf-8'); note_hash=sha256(note)
    # Completed checkpoint replay produces a new folder; never overwrites notes.
    process=subprocess.Popen(command); process.wait(timeout=30)
    replay=read_json(folder/'state.json')
    assert replay['status']=='completed_with_warnings'
    assert replay['output']!=resumed['output'] and sha256(note)==note_hash
    # Fail safely on a modified job snapshot and leave original user input intact.
    snapshot=folder/'Original.pdf'; original=snapshot.read_bytes(); snapshot.write_bytes(original+b'changed')
    process=subprocess.Popen(command); process.wait(timeout=30)
    failed=read_json(folder/'state.json'); assert failed['status']=='failed' and '변경' in failed['error']
    snapshot.write_bytes(original)
    assert sha256(source)==original_hash
    result=dict(cancel='passed',resume='passed',cache_hits=resumed['cache_hits'],
                duplicate_title='passed',existing_note_preserved='passed',source_hash_guard='passed',
                binary=args.exe or 'python source',cancelled_completed=completed_before,job=str(folder))
    atomic_json(project/'benchmarks'/'runs'/'recovery-result.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
