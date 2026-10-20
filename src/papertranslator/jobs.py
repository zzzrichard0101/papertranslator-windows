import ctypes
import json
import os
from pathlib import Path
import shutil
import time
import uuid
from contextlib import closing

import pymupdf as fitz

from .backend import Cancelled, LlamaBackend, OPTIONS
from .models import resolve_model, options_for
from .checkpoint import Checkpoint
from .analysis_cache import get_analysis
from .inference import completed as infer_completed, slots
from .reviewer import review, REVIEW_VERSION
from .document import analyze, export, PARSER
from .quality import PROMPT_VERSION, VERSION
from .translation import translate_record
from .ocr import signature, LOW_CONFIDENCE
from .eta import TranslationETA, work_units
from .refinement import policy, attach_context, run as refine_records
from .storage import atomic_json, fingerprint, now, pages, read_json, root, safe_title, sha256, worker_lock

TERMINAL = {'review_required','completed_with_warnings', 'failed', 'cancelled', 'interrupted'}


def create_job(source, destination, page_spec='', profile='general', title='',model_key=None,independent_review=False,ocr_mode='auto',quality_mode='contextual',bilingual_layout='source_position',inference_parallel=1,inference_acceleration='none',compact_editor=False):
    if inference_parallel not in {1,2}:raise ValueError('지원하지 않는 동시 처리 수')
    if inference_acceleration not in {'none','mtp'} or type(compact_editor) is not bool:raise ValueError('지원하지 않는 가속/교정 출력 설정')
    signature(ocr_mode)
    if bilingual_layout not in {'source_position','reflow'}:raise ValueError('지원하지 않는 대조본 배치입니다.')
    source=Path(source).resolve(); destination=Path(destination).resolve()
    if source.suffix.lower()!='.pdf' or not source.is_file():
        raise ValueError('PDF 파일을 선택하세요.')
    if not destination.is_dir():
        raise ValueError('저장할 폴더가 없습니다. 폴더를 다시 선택하세요.')
    cloud=None
    if quality_mode=='hybrid':
        from .credentials import load_key
        cloud=read_json(root()/'settings.json',{}).get('cloud',{})
        if not cloud.get('enabled') or not cloud.get('model','').endswith(':free') or not load_key():
            raise ValueError('무료 API 설정에서 키·무료 모델·전송 범위를 확인하세요.')
    jid=uuid.uuid4().hex
    folder=root()/'jobs'/jid; folder.mkdir(parents=True)
    state=dict(id=jid, created_at=now(), status='queued', stage='대기', source=str(source), source_name=source.name,
               destination=str(destination), page_spec=page_spec, profile=profile, title=title or source.stem,
               completed=0, total=0, current_source_page=None, elapsed_seconds=0, eta_seconds=None,
               output=None, error=None, pid=None, cache_hits=0)
    # Snapshot before worker launch so later global changes cannot affect this job.
    state['model']=resolve_model(model_key)
    if quality_mode!='legacy' and state['model'].get('adapter')!='translategemma':
        state['model']['options']={**state['model'].get('options',{}),'temperature':.2}
    state['inference_parallel']=inference_parallel
    if inference_parallel==2:state['model']['options']={**state['model'].get('options',{}),'parallel':2}
    registry=read_json(root()/'model_registry.json',{})
    state['quality_policy']=policy(quality_mode,state['model'],registry)
    if inference_acceleration=='mtp':
        from .acceleration import apply,capable
        applicable=[state['model'],state['quality_policy'].get('editor',{})]
        if not any(capable(m,registry) for m in applicable):raise ValueError('설치된 모델/교정 모델에 확인된 MTP 모델이 없습니다.')
        state['model']=apply(state['model'],registry,'mtp')
        if state['quality_policy'].get('editor'):
            state['quality_policy']['editor']=apply(state['quality_policy']['editor'],registry,'mtp')
    if compact_editor:
        if quality_mode not in {'ensemble','hybrid'}:raise ValueError('수정 구절 출력은 정밀/혼합 교정에서 사용합니다.')
        state['quality_policy']['editor_output']='patch-v1'
        from .compact_edit import VERSION as COMPACT_VERSION
        state['quality_policy']['editor_output_version']=COMPACT_VERSION
    state['inference_acceleration']=inference_acceleration
    state['processing_revision']=3
    state['document_layout']='auto_columns_v1'
    state['efficiency_policy']=dict(version='local-efficiency-1',validation='typed-v3',response_cache='resolved-2',session_reuse=True)
    state['quality_policy']['validation']='typed-v3'
    from .source_integrity import VERSION as SOURCE_VERSION
    state['source_integrity_policy']=SOURCE_VERSION
    if cloud:state['quality_policy']['cloud']={k:cloud[k] for k in ['enabled','model','max_requests']}
    state['ocr_mode']=ocr_mode
    state['bilingual_layout']=bilingual_layout
    if independent_review:
        reviewer=read_json(root()/'model_registry.json',{}).get('reviewer')
        if not reviewer or not reviewer.get('qualified_on_small_controls'):
            raise ValueError('검수 모델의 연결 검증이 아직 완료되지 않았습니다.')
        if reviewer.get('adapter')==state['model'].get('adapter','hy'):
            raise ValueError('같은 계열의 자기 검수는 독립 검수로 표시하지 않습니다.')
        state['reviewer']=reviewer
    atomic_json(folder/'state.json',state)
    return folder


def run_job(folder, retry_failed=False):
    folder=Path(folder).resolve()
    if folder.parent != (root()/'jobs').resolve():
        raise ValueError('허용되지 않은 작업 경로')
    with worker_lock():
        _run(folder,retry_failed)


def _run(folder, retry_failed):
    state=read_json(folder/'state.json')
    if not state:
        raise ValueError('작업 기록이 없습니다.')
    began=time.monotonic(); prior=state.get('elapsed_seconds',0); journal=None;session_pool=None
    efficient=state.get('efficiency_policy',{}).get('version')=='local-efficiency-1'
    last_flush=0.;phase_times={};phase_started=time.monotonic();phase=state.get('status','queued')
    cancel=lambda: (folder/'cancel').exists()
    def save(status=None, stage=None, **values):
        nonlocal last_flush,phase_started,phase
        force=status is not None or (stage is not None and stage!=state.get('stage')) or 'output' in values
        if status is not None and status!=phase:
            phase_times[phase]=phase_times.get(phase,0.)+time.monotonic()-phase_started
            phase_started=time.monotonic();phase=status
        if status: state['status']=status
        if stage: state['stage']=stage
        state.update(values); state['updated_at']=now(); state['elapsed_seconds']=round(prior+time.monotonic()-began,2)
        if efficient:
            state['performance']=dict(phases_seconds={k:round(v,3) for k,v in phase_times.items()},
                sessions=dict(session_pool.metrics) if session_pool else {})
        if force or not efficient or time.monotonic()-last_flush>=.5:
            atomic_json(folder/'state.json',state);last_flush=time.monotonic()
    try:
        backend_factory=LlamaBackend
        if efficient:
            from .sessions import Sessions
            session_pool=Sessions(LlamaBackend);backend_factory=session_pool
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        save('preparing','원본 확인',pid=os.getpid(),error=None,eta_seconds=None,eta=None)
        original=folder/'Original.pdf'
        if original.exists():
            if state.get('source_sha256') and sha256(original)!=state['source_sha256']:
                raise RuntimeError('저장된 원본이 변경되어 재개를 차단했습니다.')
        else:
            src=Path(state['source'])
            if src.stat().st_size > 500*1024**2:
                raise ValueError('초기 버전은 500MB 이하 PDF를 지원합니다.')
            temp=folder/'Original.copying'
            shutil.copyfile(src,temp)
            if sha256(src)!=sha256(temp):
                raise RuntimeError('원본 복사 중 파일이 변경됐습니다. 다시 시도하세요.')
            temp.replace(original)
        if cancel(): raise Cancelled()
        with fitz.open(original) as doc:
            if doc.needs_pass: raise ValueError('암호화된 PDF는 지원하지 않습니다.')
            if len(doc)>2000: raise ValueError('초기 버전은 2,000쪽 이하 PDF를 지원합니다.')
            selected=pages(state['page_spec'],len(doc)); total_pages=len(doc)
        save('analyzing','PDF 구조 분석',source_sha256=sha256(original),total_pages=total_pages,selected=selected)
        upgraded=state.get('processing_revision',0)>=1
        structured=state.get('processing_revision',0)>=2
        academic=state.get('processing_revision',0)>=3
        from .source_integrity import SUPPORTED
        SOURCE_VERSION=state.get('source_integrity_policy')
        source_integrity=SOURCE_VERSION if SOURCE_VERSION in SUPPORTED else False
        analysis_identity=dict(source_sha256=state['source_sha256'],selected=selected,
            parser=PARSER,ocr=signature(state.get('ocr_mode','off'),recovery=upgraded))
        if structured:
            from .structure import VERSION as STRUCTURE_VERSION
            analysis_identity['structure']=STRUCTURE_VERSION
        if academic:
            from .academic_layout import VERSION as ACADEMIC_VERSION
            analysis_identity['academic']=ACADEMIC_VERSION
        if source_integrity:analysis_identity['source_integrity']=SOURCE_VERSION
        ir,analysis_hit=get_analysis(folder,analysis_identity,
            lambda:analyze(original,selected,lambda p:save(stage='PDF 분석 · 필요 시 로컬 OCR',current_source_page=p),
                ocr_mode=state.get('ocr_mode','off'),cancel=cancel,cache_dir=folder/'ocr-cache',recovery=upgraded,structured=structured,academic=academic,source_integrity=source_integrity),cancel)
        save(analysis_cache_hit=analysis_hit)
        if efficient:
            from .typed_literals import preserved_kind
            for block in ir['blocks']:
                block['validation_policy']=state['efficiency_policy'].get('validation','typed-v1')
                kind=preserved_kind(block['source'])
                if kind and not block['preserve']:
                    block.update(kind=kind,preserve=True,preservation_reason='typed_literal_v1')
        quality_policy=state.get('quality_policy',{'mode':'legacy'})
        if quality_policy['mode']!='legacy':attach_context(ir,state['title'],state['profile'],document_terms=upgraded)
        if ir.get('terminology'):atomic_json(folder/'Terminology.json',ir['terminology'])
        if structured:
            for block in ir['blocks']:block['selective_repair']=True
        atomic_json(folder/'ir.json',ir)
        registry=read_json(root()/'model_registry.json')
        if not registry: raise RuntimeError('모델 설치가 아직 완료되지 않았습니다.')
        model=state.get('model') or resolve_model()
        state['model']=model
        identity=dict(hash=state['source_sha256'], selected=selected, profile=state['profile'],
                               model=model, parser=PARSER, prompt=PROMPT_VERSION, quality=VERSION,
                               runtime=registry['runtime_release'], options=options_for(model),ocr=ir['ocr'],quality_policy=quality_policy)
        if efficient:identity['efficiency']=state['efficiency_policy']
        if source_integrity:identity['source_integrity']=SOURCE_VERSION
        if upgraded:
            from .terminology import VERSION as TERMS_VERSION
            identity['processing']=dict(revision=state['processing_revision'],terminology=TERMS_VERSION)
            if structured:
                from .sentence_repair import VERSION as REPAIR_VERSION
                identity['processing'].update(structure=STRUCTURE_VERSION,repair=REPAIR_VERSION)
            if academic:identity['processing']['academic']=ACADEMIC_VERSION
        stamp=fingerprint(identity)
        journal=Checkpoint(folder,stamp)
        records=journal.load()
        if retry_failed:
            journal.remove([key for key,r in records.items() if r['status']=='failed'])
            records=journal.load()
        state['cache_hits']=len(records)
        save('translating','모델 준비',total=len(ir['blocks']),completed=len(records),fingerprint=stamp,
             eta=dict(state='loading',seconds=None),eta_seconds=None)
        if not any(not b['preserve'] for b in ir['blocks']):
            raise ValueError('번역할 본문을 찾지 못했습니다. OCR 설정과 스캔 선명도를 확인하세요. 원본은 보존됐습니다.')
        pending=[b for b in ir['blocks'] if b['id'] not in records]
        estimator=TranslationETA()
        remaining_work=sum(work_units(b['source']) for b in pending if not b['preserve'])
        def progress_eta(current_units=0):
            estimate=estimator.estimate(remaining_work,current_units)
            estimate['at_elapsed_seconds']=round(prior+time.monotonic()-began,2)
            return dict(eta=estimate,eta_seconds=estimate['seconds'])
        with backend_factory(folder,cancel,model=model) as backend:
            if any(not b['preserve'] for b in pending):
                backend.start(lambda msg: save(stage=msg))
            actual_slots=slots(backend)
            save(inference_slots=actual_slots,inference_speculation=backend.options.get('speculation','none'))
            def translate_block(block):
                if cancel():raise Cancelled()
                measured_start=time.monotonic()
                record={k:block[k] for k in ['id','source_page_index','source_page_indices','source_fragments','bbox','kind','source']}
                record.update(target=block['source'],status='preserved',issues=[])
                if not block['preserve']:
                    record=translate_record(backend,block,state['profile'],model.get('adapter','hy'),folder,memory=journal)
                    record['wall_seconds']=round(time.monotonic()-measured_start,3)
                record.update(extraction=block['extraction'],ocr_confidence_min=block['ocr_confidence_min'])
                if efficient:record['validation_policy']=block['validation_policy']
                if block.get('preservation_reason'):record['preservation_reason']=block['preservation_reason']
                if block['ocr_confidence_min'] is not None and block['ocr_confidence_min']<LOW_CONFIDENCE:
                    record['issues'].append(dict(code='ocr_low_confidence',severity='major',
                                                 message='OCR 인식 원문 신뢰도가 낮습니다. 이미지 원문과 대조하세요.'))
                return record
            last_observed=time.monotonic();observed_units=0;observed_ok=True;observed_attempts=1
            with closing(infer_completed(pending,translate_block,actual_slots,cancel,getattr(backend,'abort',lambda:None))) as results:
                for block,record in results:
                    units=0 if block['preserve'] else work_units(block['source'])
                    if units:
                        remaining_work-=units
                        if actual_slots==1:
                            if not record.get('translation_cache'):
                                estimator.observe(units,record['wall_seconds'],record.get('attempts',1),record['status']=='translated')
                        else:
                            observed_units+=units
                            observed_ok &= record['status']=='translated'
                            observed_attempts=max(observed_attempts,record.get('attempts',1))
                            elapsed=time.monotonic()-last_observed
                            if elapsed>=2:
                                estimator.observe(observed_units,elapsed,observed_attempts,observed_ok)
                                last_observed=time.monotonic();observed_units=0;observed_ok=True;observed_attempts=1
                    records[block['id']]=record
                    journal.put(record)
                    save(stage='문단 번역',current_source_page=block['source_page_index']+1,completed=len(records),**progress_eta())
        journal.snapshot()
        if cancel(): raise Cancelled()
        if quality_policy['mode'] in {'ensemble','hybrid'}:
            save('refining','다중 모델 원문 대조 교정',eta_seconds=None,eta=None)
            refine_records(records,ir,quality_policy,folder,state['profile'],backend_factory,
                journal,
                lambda message:save(stage=message),cancel,workers=state.get('inference_parallel',1))
        if cancel():raise Cancelled()
        if quality_policy['mode']=='hybrid':
            from .cloud import review_records
            save('cloud_reviewing','무료 API 선택 문단 검토',eta_seconds=None,eta=None)
            review_records(records,ir,quality_policy,folder,
                journal,
                lambda message:save(stage=message),cancel)
        if cancel():raise Cancelled()
        if state.get('reviewer'):
            reviewer=state['reviewer']
            eligible=[r for r in records.values() if r['status']=='translated']
            save('reviewing','독립 의미 검수 모델 준비',eta_seconds=None)
            with backend_factory(folder,cancel,model=reviewer,log_name='reviewer-engine.log') as backend:
                backend.start(lambda msg:save(stage='독립 검수 · '+msg))
                for index,record in enumerate(eligible):
                    if cancel(): raise Cancelled()
                    review_stamp=fingerprint(dict(source=record['source'],target=record['target'],model=reviewer,version=REVIEW_VERSION))
                    if record.get('review_fingerprint')==review_stamp:continue
                    save(stage=f'독립 의미 검수 {index+1}/{len(eligible)}',current_source_page=record['source_page_index']+1)
                    try: record['review']=review(backend,record['source'],record['target'])
                    except Cancelled:raise
                    except Exception as exc:
                        record['review']=dict(verdict='uncertain',evidence_validated=False,explanation_ko=str(exc)[:500])
                    record['review_fingerprint']=review_stamp
                    journal.put(record)
            if cancel(): raise Cancelled()
        if any(r['kind']=='diagram_label' for r in records.values()):
            from .label_fit import run as fit_labels
            from .models import catalog
            label_model=quality_policy.get('editor') or catalog(registry).get('qwen38') or model
            label_model={**label_model,'options':{**label_model.get('options',{}),'context':8192,'temperature':.1}}
            save('fitting_labels','도식 문구와 상자 크기 맞춤',eta_seconds=None,eta=None)
            fit_labels(records,label_model,folder,backend_factory,
                journal,
                lambda message:save(stage=message),cancel)
        if state.get('bilingual_layout','reflow')=='source_position':
            from .alignment import run as align_records
            from .models import catalog
            alignment_model=quality_policy.get('editor') or catalog(registry).get('qwen38') or model
            alignment_model={**alignment_model,'options':{**alignment_model.get('options',{}),'context':8192,'temperature':.1}}
            save('aligning','원문 쪽과 번역 대응 정리',eta_seconds=None,eta=None)
            align_records(records,alignment_model,folder,backend_factory,
                journal,
                lambda message:save(stage=message),cancel)
        journal.snapshot()
        if session_pool:session_pool.close()
        save('typesetting','한국어·영한 대조 PDF 조판',eta_seconds=None,current_source_page=None)
        # One result revision is committed only after both PDFs and report validate.
        staging=folder/('revision-'+uuid.uuid4().hex)
        quality=export(ir,records,original,staging,state['title'],state,cancel)
        if cancel(): raise Cancelled()
        save('saving','결과 저장')
        papers=Path(state['destination'])/'Papers'; papers.mkdir(exist_ok=True)
        if len(str(papers))>130: raise ValueError('저장 경로가 너무 깁니다. 더 짧은 폴더를 선택하세요.')
        name=safe_title(state['title'])
        temp=papers/('.papertranslator-'+state['id']+'-'+uuid.uuid4().hex)
        try:
            shutil.copytree(staging,temp)
        except Exception:
            if temp.exists(): shutil.rmtree(temp)
            raise
        if cancel():
            shutil.rmtree(temp); raise Cancelled()
        try:
            number=1
            while True:
                dest=papers/(name if number==1 else f'{name} ({number})')
                note=temp/(name+'.md')
                text=note.read_text(encoding='utf-8')
                # Explicit Vault-relative links avoid ambiguous identical PDF names.
                for filename in ['Original.pdf','Korean.pdf','Bilingual.pdf','Review.html']:
                    text=text.replace('[['+filename, '[[Papers/'+dest.name+'/'+filename)
                note.write_text(text,encoding='utf-8')
                try: temp.rename(dest); break
                except FileExistsError:
                    # Rebuild links for the next collision-safe destination name.
                    for filename in ['Original.pdf','Korean.pdf','Bilingual.pdf','Review.html']:
                        text=text.replace('[[Papers/'+dest.name+'/'+filename,'[['+filename)
                    note.write_text(text,encoding='utf-8'); number+=1
        except Exception:
            shutil.rmtree(temp); raise
        save('review_required' if quality['delivery_status']=='review_required' else 'completed_with_warnings',
             '검토 필요 · 배치 또는 번역 검사 미통과' if quality['delivery_status']=='review_required' else '생성됨 · 기계 검사 통과 · 원문 대조 필요',output=str(dest),eta_seconds=None,pid=None,
             quality_summary=dict(failed=quality['failed_paragraphs'],warnings=quality['warnings_paragraphs'],
                                  translated=quality['translated_paragraphs'],semantic_review_coverage=quality['semantic_review_coverage'],
                                  ocr_pages=quality['ocr_pages'],ocr_low_confidence_lines=quality['ocr_low_confidence_lines'],
                                  ocr_unscored_lines=quality['ocr_unscored_lines'],
                                  refinement_processed=quality['refinement_processed'],refinement_accepted=quality['refinement_accepted'],output_layout=quality['output_layout'],
                                  layout_overflow=(quality.get('layout') or {}).get('overflow_pages',0),
                                  layout_proposed=len((quality.get('layout') or {}).get('proposed_page_alignments',[])),
                                  layout_unresolved=len((quality.get('layout') or {}).get('unresolved_page_alignments',[])),
                                  layout_validation=(quality.get('layout') or {}).get('validation')))
    except Cancelled:
        save('cancelled','취소됨 · 이어하기 가능',pid=None,eta_seconds=None)
    except Exception as exc:
        save('failed','작업 실패 · 기록 보존',error=str(exc)[:1000],pid=None,eta_seconds=None)
    finally:
        try:
            if session_pool:session_pool.close()
            if journal is not None:
                try:journal.snapshot()
                except OSError:pass # Compatibility snapshot only; every database commit is already durable.
                finally:journal.close()
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
