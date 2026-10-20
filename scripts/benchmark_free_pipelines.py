"""Compare installed, local-only translation pipelines on synthetic challenges.

Uses production prompts, sampling and validation. Every draft gets the same
source-grounded editor. No model download, API call, default change or job edit.
Timings include retries; shared auxiliary work is charged to each applicable
pipeline. A blind comparison file supports semantic assessment, which cannot
be replaced by the automatic format/quantity checks.
"""
import argparse
from contextlib import closing
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from benchmark_acceleration import gpu_idle
from papertranslator.acceleration import apply
from papertranslator.backend import LlamaBackend
from papertranslator.inference import completed, slots
from papertranslator.models import catalog, resolve_model
from papertranslator.quality import check, make_prompt
from papertranslator.refinement import edit_record
from papertranslator.storage import atomic_json, read_json, root, worker_lock
from papertranslator.translation import blocking, translate_record

# Original synthetic text, not passages from a user's book. Criteria stay out
# of the translation prompt. These cases are probes, not a held-out benchmark.
CASES = [
    ('negation', 'Research evidence does not replace clinical judgement. Although the findings may inform practice, the authors do not claim that the intervention is effective for every client. The absence of evidence is not evidence that the treatment has no effect.',
     'Neither negation reversed; possible usefulness does not become proven universal effectiveness.'),
    ('condition', 'Unless the participant gives written consent, the therapist must not show the drawing to the research team. If consent is later withdrawn, the image must be removed from the analysis, even when the interview transcript can still be retained.',
     'Disclosure forbidden without consent; withdrawal requires image removal; transcript retention remains conditional.'),
    ('frequency', 'The treatment consisted of six weekly sessions, not six sessions per week. Each session lasted 50 minutes. Two participants missed one session, but neither withdrew from the study. Attendance should therefore be distinguished from completion of every scheduled session.',
     'Six sessions total at weekly intervals; 50 minutes each; two missed one each; no withdrawals.'),
    ('causality', 'Participants who attended more sessions reported less anxiety, but this association does not establish causation. The observational design cannot exclude the possibility that people with lower anxiety were more likely to continue attending. No participants were randomly assigned to treatment.',
     'Association not causation; reverse direction possible; observational, no random allocation.'),
    ('roles', 'The therapist asked the researcher to interview the participant after the session. The participant, rather than the therapist, selected the drawing discussed in that interview. The researcher did not interpret the image on behalf of either person.',
     'Therapist asks researcher; researcher interviews participant; participant selects; researcher does not interpret.'),
    ('quantity', 'The study enrolled 24 participants. The mean change was -2.4, with a 95% confidence interval from -4.1 to -0.7. These values describe the measured change rather than the number of people who improved. No estimate of individual benefit was reported.',
     '24; signed -2.4, -4.1, -0.7; 95%; group change not individual responder count.'),
    ('scope', 'Only participants who had completed the baseline interview were invited to the follow-up discussion. This restriction did not apply to the therapy sessions themselves. Staff members could attend training, but they were not included in the participant sample.',
     'Baseline prerequisite only for follow-up; no therapy restriction; staff training separate from research sample.'),
    ('kinship', 'The participant brought her aunt to the interview. The report does not say whether the aunt was related through the mother or the father, and the translator should not infer that relationship. The participant later described the relative as a source of practical support.',
     'Aunt relationship unspecified; do not invent maternal/paternal side; participant describes practical support.'),
    ('clinical_terms', 'The clinical account distinguishes dissociation from distraction. The participant described a period of feeling detached from her own experience; the authors did not diagnose a dissociative disorder. A description of an experience should not be turned into a confirmed diagnosis.',
     'Clinical dissociation is 해리; distinguish distraction; no diagnosis of dissociative disorder.'),
    ('evidence_levels', 'A case series can describe clinical change, but it is not a randomized controlled trial. Expert opinion may help formulate a question without providing the same level of evidence as a systematic review. The hierarchy concerns evidence for the stated question, not the personal worth of the clinician.',
     'Case series distinct from RCT; expert opinion distinct from systematic review; evidence hierarchy not personal worth.'),
    ('quotation', 'Miller (2018: 42) reported that participants could refuse to discuss a drawing. The later review (Miller and Chen, 2020) did not find that refusal predicted a poor outcome. Both reports distinguish the right to decline from an inability to communicate.',
     'Citations preserved; right to refuse; no evidence refusal predicts poor outcome; refusal not inability.'),
    ('modality', 'The results suggest that the intervention might be useful in some settings. They neither demonstrate that it is necessary nor show that it should replace existing services. Further research would be needed before the authors could recommend routine use across the health service.',
     'Tentative usefulness; neither necessity nor replacement proven; routine use not recommended yet.'),
]


def blocks(limit):
    return [dict(id=f'challenge-{i:02d}-{name}', source=source, kind='paragraph',
                 source_page_index=i, source_page_indices=[i], source_fragments=[],
                 bbox=[40, 60, 550, 700], context=dict(domain='art therapy and clinical research'))
            for i, (name, source, _) in enumerate(CASES[:limit])]


def configured(model, registry, *, editor=False, auxiliary=False, workers=1):
    value = apply(model, registry, 'mtp')
    value['options'] = {**value.get('options', {}), 'parallel': workers,
                        'context': 8192 if editor else 4096}
    if editor:
        value['options']['temperature'] = .1
    elif not auxiliary and value.get('adapter') != 'translategemma':
        value['options']['temperature'] = .2
    return value


def stage(folder, model, items, operation):
    folder.mkdir(parents=True)
    gpu_idle()
    with LlamaBackend(folder, model=model) as backend:
        began = time.perf_counter()
        backend.start()
        startup = time.perf_counter() - began
        backend.translate(make_prompt('Research methods.', {}, 'art_therapy', model['adapter']), max_tokens=100)
        began = time.perf_counter()
        results = []
        with closing(completed(items, lambda item: operation(backend, item, folder),
                               slots(backend), abort=backend.abort)) as pending:
            for item, value in pending:
                results.append(value)
        elapsed = time.perf_counter() - began
        results.sort(key=lambda r: r['id'])
        atomic_json(folder / 'records.json', results)
        metric = dict(startup_seconds=startup, warm_seconds=elapsed,
                      actual_slots=slots(backend), actual_speculation=backend.options.get('speculation', 'none'),
                      failed=sum(r['status'] == 'failed' for r in results),
                      blocking=sum(blocking(check(r['source'], r['target'])) for r in results),
                      uncertain=sum(r.get('refinement', {}).get('decision') == 'uncertain' for r in results),
                      model_sha256=model['sha256'], options=backend.options)
    atomic_json(folder / 'metrics.json', metric)
    print(json.dumps(dict(stage=folder.name, **metric), ensure_ascii=False), flush=True)
    return results, metric


def compare(registry, work, model_keys, limit, workers):
    inputs = blocks(limit)
    editor = configured(resolve_model('qwen38', registry), registry, editor=True, workers=workers)
    drafts = {}; metrics = {}; report = dict(scope='Synthetic production-sampling probe; semantic quality is unvalidated',
                                            human_approved=False, automatic_promotion=False, pipelines=[])
    for key in model_keys:
        model = configured(resolve_model(key, registry), registry, workers=workers)
        drafts[key], metrics[key] = stage(work / ('draft-' + key), model, inputs,
            lambda backend, block, folder: translate_record(backend, block, 'art_therapy', model['adapter'], folder))
    auxiliary = []; auxiliary_metrics = None
    if any(resolve_model(key, registry).get('adapter') != 'hy' for key in model_keys):
        auxiliary_model = configured(resolve_model('hy7b', registry), registry, auxiliary=True, workers=workers)
        auxiliary, auxiliary_metrics = stage(work / 'auxiliary-hy7b', auxiliary_model, inputs,
            lambda backend, block, folder: translate_record(backend, block, 'art_therapy', 'hy', folder))
    by_id = {b['id']: b for b in inputs}
    blind = []
    for key in model_keys:
        records = copy.deepcopy(drafts[key])
        needs_auxiliary = resolve_model(key, registry).get('adapter') != 'hy'
        if needs_auxiliary:
            for record, alt in zip(records, auxiliary, strict=True):
                assert record['id'] == alt['id']
                record['alternative'] = alt
        settings = dict(editor=editor, editor_output='patch-v1')
        edited, metric = stage(work / ('edit-' + key), editor, records,
            lambda backend, record, folder: edit_record(backend, record, by_id[record['id']], settings,
                                                       'free-pipeline-probe', folder, lambda: False))
        stages = [metrics[key], metric] + ([auxiliary_metrics] if needs_auxiliary else [])
        result = dict(primary=key, editor='qwen38', auxiliary='hy7b' if needs_auxiliary else None,
                      warm_seconds=sum(s['warm_seconds'] for s in stages),
                      startup_seconds=sum(s['startup_seconds'] for s in stages),
                      final_failed=metric['failed'], final_blocking=metric['blocking'], final_uncertain=metric['uncertain'],
                      quality_scope='Structural guards only; inspect blind-comparison.json for meaning errors')
        report['pipelines'].append(result)
        for record in edited:
            # Stable scrambled identities; mapping is separate from the blind file.
            label = hashlib.sha256(key.encode()).hexdigest()[:10]
            blind.append(dict(id=record['id'], candidate=label, source=record['source'], target=record['target']))
        atomic_json(work / 'result.json', report)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    blind.sort(key=lambda r: (r['id'], r['candidate']))
    atomic_json(work / 'blind-comparison.json', blind)
    atomic_json(work / 'blind-key.json', {hashlib.sha256(key.encode()).hexdigest()[:10]: key for key in model_keys})
    atomic_json(work / 'assessment-criteria.json', [dict(id=b['id'], criteria=CASES[i][2]) for i,b in enumerate(inputs)])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', help='Installed model keys; defaults to installed comparison candidates')
    parser.add_argument('--limit', type=int, default=len(CASES))
    parser.add_argument('--workers', type=int, choices=[1, 2], default=1)
    parser.add_argument('--registry', type=Path, help='Optional experimental registry; never written back to the app')
    args = parser.parse_args()
    registry = read_json(args.registry or root() / 'model_registry.json')
    if not registry:
        parser.error('Model registry is missing or invalid')
    args.models = args.models or [key for key in ['qwen38', 'hy7b', 'qwen30b', 'translategemma27b', 'hy30b']
                                  if key in catalog(registry)]
    if not 1 <= args.limit <= len(CASES) or len(set(args.models)) != len(args.models):
        parser.error('Invalid challenge count or duplicate models')
    # Validate all models before acquiring GPU resources or changing this process's home.
    for key in [*args.models, 'qwen38', 'hy7b']:
        resolve_model(key, registry)
    work = Path(__file__).resolve().parents[1] / 'work/free-pipeline-benchmark' / uuid.uuid4().hex
    with worker_lock():
        gpu_idle()
        home = work / 'home'; home.mkdir(parents=True)
        atomic_json(home / 'model_registry.json', registry)
        previous = os.environ.get('PAPERTRANSLATOR_HOME')
        os.environ['PAPERTRANSLATOR_HOME'] = str(home)
        try:
            compare(registry, work, args.models, args.limit, args.workers)
        finally:
            if previous is None: os.environ.pop('PAPERTRANSLATOR_HOME', None)
            else: os.environ['PAPERTRANSLATOR_HOME'] = previous
    print('Result:', work / 'result.json', flush=True)


if __name__ == '__main__':
    main()
