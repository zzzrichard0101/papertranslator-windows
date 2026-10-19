import ctypes
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import pymupdf as fitz

from .backend import Cancelled, LlamaBackend, OPTIONS
from .models import resolve_model, options_for
from .reviewer import review, REVIEW_VERSION
from .document import analyze, export, PARSER
from .quality import PROMPT_VERSION, VERSION
from .translation import translate_record
from .ocr import signature, LOW_CONFIDENCE
from .eta import TranslationETA, work_units
from .refinement import policy, attach_context, run as refine_records
from .storage import atomic_json, fingerprint, now, pages, read_json, root, safe_title, sha256, worker_lock

TERMINAL = {'completed_with_warnings', 'failed', 'cancelled', 'interrupted'}


def create_job(source, destination, page_spec='', profile='general', title='',model_key=None,independent_review=False,ocr_mode='auto',quality_mode='contextual'):
    signature(ocr_mode)
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
    state['quality_policy']=policy(quality_mode,state['model'],read_json(root()/'model_registry.json',{}))
    if cloud:state['quality_policy']['cloud']={k:cloud[k] for k in ['enabled','model','max_requests']}
    state['ocr_mode']=ocr_mode
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
    began=time.monotonic(); prior=state.get('elapsed_seconds',0)
    cancel=lambda: (folder/'cancel').exists()
    def save(status=None, stage=None, **values):
        if status: state['status']=status
        if stage: state['stage']=stage
        state.update(values); state['updated_at']=now(); state['elapsed_seconds']=round(prior+time.monotonic()-began,2)
        atomic_json(folder/'state.json',state)
    try:
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
        ir=analyze(original,selected,lambda p: save(stage='PDF 분석 · 필요 시 로컬 OCR',current_source_page=p),
                   ocr_mode=state.get('ocr_mode','off'),cancel=cancel,cache_dir=folder/'ocr-cache')
        quality_policy=state.get('quality_policy',{'mode':'legacy'})
        if quality_policy['mode']!='legacy':attach_context(ir,state['title'],state['profile'])
        atomic_json(folder/'ir.json',ir)
        registry=read_json(root()/'model_registry.json')
        if not registry: raise RuntimeError('모델 설치가 아직 완료되지 않았습니다.')
        model=state.get('model') or resolve_model()
        state['model']=model
        stamp=fingerprint(dict(hash=state['source_sha256'], selected=selected, profile=state['profile'],
                               model=model, parser=PARSER, prompt=PROMPT_VERSION, quality=VERSION,
                               runtime=registry['runtime_release'], options=options_for(model),ocr=ir['ocr'],quality_policy=quality_policy))
        checkpoint=read_json(folder/'checkpoint.json',{})
        records=checkpoint.get('records',{}) if checkpoint.get('fingerprint')==stamp else {}
        if retry_failed:
            records={k:v for k,v in records.items() if v['status']!='failed'}
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
        with LlamaBackend(folder,cancel,model=model) as backend:
            if any(not b['preserve'] for b in pending):
                backend.start(lambda msg: save(stage=msg))
            for block in ir['blocks']:
                if cancel(): raise Cancelled()
                if block['id'] in records: continue
                units=0 if block['preserve'] else work_units(block['source'])
                save(stage='문단 번역',current_source_page=block['source_page_index']+1,**progress_eta(units))
                record={k:block[k] for k in ['id','source_page_index','source_page_indices','source_fragments','bbox','kind','source']}
                record.update(target=block['source'],status='preserved',issues=[])
                if not block['preserve']:
                    measured_start=time.monotonic()
                    record=translate_record(backend,block,state['profile'],model.get('adapter','hy'),folder)
                    wall_seconds=time.monotonic()-measured_start
                    record['wall_seconds']=round(wall_seconds,3)
                    estimator.observe(units,wall_seconds,record.get('attempts',1),record['status']=='translated')
                    remaining_work-=units
                record.update(extraction=block['extraction'],ocr_confidence_min=block['ocr_confidence_min'])
                if block['ocr_confidence_min'] is not None and block['ocr_confidence_min']<LOW_CONFIDENCE:
                    record['issues'].append(dict(code='ocr_low_confidence',severity='major',
                                                 message='OCR 인식 원문 신뢰도가 낮습니다. 이미지 원문과 대조하세요.'))
                records[block['id']]=record
                atomic_json(folder/'checkpoint.json',dict(fingerprint=stamp,records=records))
                save(completed=len(records),**progress_eta())
        if cancel(): raise Cancelled()
        if quality_policy['mode'] in {'ensemble','hybrid'}:
            save('refining','다중 모델 원문 대조 교정',eta_seconds=None,eta=None)
            refine_records(records,ir,quality_policy,folder,state['profile'],LlamaBackend,
                lambda:atomic_json(folder/'checkpoint.json',dict(fingerprint=stamp,records=records)),
                lambda message:save(stage=message),cancel)
        if cancel():raise Cancelled()
        if quality_policy['mode']=='hybrid':
            from .cloud import review_records
            save('cloud_reviewing','무료 API 선택 문단 검토',eta_seconds=None,eta=None)
            review_records(records,ir,quality_policy,folder,
                lambda:atomic_json(folder/'checkpoint.json',dict(fingerprint=stamp,records=records)),
                lambda message:save(stage=message),cancel)
        if cancel():raise Cancelled()
        if state.get('reviewer'):
            reviewer=state['reviewer']
            eligible=[r for r in records.values() if r['status']=='translated']
            save('reviewing','독립 의미 검수 모델 준비',eta_seconds=None)
            with LlamaBackend(folder,cancel,model=reviewer,log_name='reviewer-engine.log') as backend:
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
                    atomic_json(folder/'checkpoint.json',dict(fingerprint=stamp,records=records))
            if cancel(): raise Cancelled()
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
        save('completed_with_warnings','완료 · 원문 대조 필요',output=str(dest),eta_seconds=None,pid=None,
             quality_summary=dict(failed=quality['failed_paragraphs'],warnings=quality['warnings_paragraphs'],
                                  translated=quality['translated_paragraphs'],semantic_review_coverage=quality['semantic_review_coverage'],
                                  ocr_pages=quality['ocr_pages'],ocr_low_confidence_lines=quality['ocr_low_confidence_lines'],
                                  ocr_unscored_lines=quality['ocr_unscored_lines'],
                                  refinement_processed=quality['refinement_processed'],refinement_accepted=quality['refinement_accepted']))
    except Cancelled:
        save('cancelled','취소됨 · 이어하기 가능',pid=None,eta_seconds=None)
    except Exception as exc:
        save('failed','작업 실패 · 기록 보존',error=str(exc)[:1000],pid=None,eta_seconds=None)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
