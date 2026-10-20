"""Real sequential GPU models through jobs, editing, replay and PDF export.

Uses only synthetic text and a dedicated data root. Model files are referenced
from the existing installation, never copied. Not an accuracy benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pymupdf as fitz

from papertranslator.jobs import create_job
from papertranslator.checkpoint import read_checkpoint, Checkpoint
from papertranslator.storage import atomic_json,read_json,root,sha256
from papertranslator.quality import check


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--exe',type=Path);args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    work=project/'work/quality/gpu';work.mkdir(parents=True,exist_ok=True)
    registry=read_json(root()/'model_registry.json')
    if not registry:raise RuntimeError('Install the local models before this GPU test.')
    home=work/'home';home.mkdir(exist_ok=True);atomic_json(home/'model_registry.json',registry)
    os.environ['PAPERTRANSLATOR_HOME']=str(home)
    source=work/'Synthetic model editing study.pdf'
    paragraphs=[
        'Research evidence does not replace clinical judgement. Clinicians should consider evidence together with their training, experience and the needs of each client.',
        'Economic uncertainty and depression can affect access to services. This discussion concerns the economy rather than a diagnosis of a mental health condition.',
        'The study found an association. It did not show that the treatment caused the improvement. The results should inform practice without being treated as proof of efficacy.',
        'The study included 24 participants. Of the respondents, 75 per cent found the arts therapies helpful. The findings apply only to this sample and require further investigation (Example and Writer, 2015).']
    with fitz.open() as document:
        page=document.new_page()
        for index,text in enumerate(paragraphs):
            page.insert_textbox(fitz.Rect(45,65+index*150,550,170+index*150),text,fontsize=12)
        document.save(source)
    folder=create_job(source,project/'test_vault',title='다중 모델 GPU 합성 통합 시험',
                      model_key='hy7b',profile='art_therapy',quality_mode='ensemble',ocr_mode='off')
    command=([str(args.exe)] if args.exe else [sys.executable,'-m','papertranslator'])+['--worker',str(folder)]
    subprocess.run(command,check=True,timeout=300)
    state=read_json(folder/'state.json');assert state['status']=='completed_with_warnings',state.get('error')
    quality=read_json(Path(state['output'])/'Quality.json')
    assert quality['refinement_processed']==4 and quality['failed_paragraphs']==0
    assert all(not r['refinement'].get('error') for r in quality['records'])
    assert any('(Example and Writer, 2015)' in r['target'] for r in quality['records'])
    assert sha256(Path(state['output'])/'Original.pdf')==sha256(source)
    # A deliberate, documented test corruption forces the correction path.
    checkpoint=read_checkpoint(folder)
    record=next(r for r in checkpoint['records'].values() if r['source'].startswith('Research evidence'))
    bad='연구 근거는 임상 판단을 대체한다. 임상가는 자신의 교육과 경험, 각 내담자의 필요와 함께 근거를 고려해야 한다.'
    record.update(target=bad,candidate=bad,issues=check(record['source'],bad));record.pop('refinement',None)
    store=Checkpoint(folder,checkpoint['fingerprint']);store.put(record);store.snapshot();store.close()
    subprocess.run(command,check=True,timeout=180)
    replay=read_json(folder/'state.json');assert replay['status']=='completed_with_warnings',replay.get('error')
    assert replay['cache_hits']==4
    result_quality=read_json(Path(replay['output'])/'Quality.json')
    corrected=next(r for r in result_quality['records'] if r['id']==record['id'])
    assert corrected['refinement']['accepted'] and corrected['target']!=bad
    assert result_quality['human_approved'] is False
    result=dict(status='passed',frozen=bool(args.exe),primary_and_editor_processed=4,
                corrupted_negation_revised=True,primary_cache_hits=4,human_approved=False,
                source_hash_equal=True,corrected_target=corrected['target'])
    atomic_json(work/('frozen-result.json' if args.exe else 'source-result.json'),result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
