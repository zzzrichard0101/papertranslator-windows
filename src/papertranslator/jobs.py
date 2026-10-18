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
from .quality import check, make_prompt, protect, restore, PROMPT_VERSION, VERSION
from .storage import atomic_json, fingerprint, now, pages, read_json, root, safe_title, sha256, worker_lock

TERMINAL = {'completed_with_warnings', 'failed', 'cancelled', 'interrupted'}


def create_job(source, destination, page_spec='', profile='general', title='',model_key=None,independent_review=False):
    source=Path(source).resolve(); destination=Path(destination).resolve()
    if source.suffix.lower()!='.pdf' or not source.is_file():
        raise ValueError('PDF 파일을 선택하세요.')
    if not destination.is_dir():
        raise ValueError('저장할 폴더가 없습니다. 폴더를 다시 선택하세요.')
    jid=uuid.uuid4().hex
    folder=root()/'jobs'/jid; folder.mkdir(parents=True)
    state=dict(id=jid, created_at=now(), status='queued', stage='대기', source=str(source), source_name=source.name,
               destination=str(destination), page_spec=page_spec, profile=profile, title=title or source.stem,
               completed=0, total=0, current_source_page=None, elapsed_seconds=0, eta_seconds=None,
               output=None, error=None, pid=None, cache_hits=0)
    # Snapshot before worker launch so later global changes cannot affect this job.
    state['model']=resolve_model(model_key)
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
        save('preparing','원본 확인',pid=os.getpid(),error=None,eta_seconds=None)
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
        ir=analyze(original,selected,lambda p: save(current_source_page=p))
        atomic_json(folder/'ir.json',ir)
        registry=read_json(root()/'model_registry.json')
        if not registry: raise RuntimeError('모델 설치가 아직 완료되지 않았습니다.')
        model=state.get('model') or resolve_model()
        state['model']=model
        stamp=fingerprint(dict(hash=state['source_sha256'], selected=selected, profile=state['profile'],
                               model=model, parser=PARSER, prompt=PROMPT_VERSION, quality=VERSION,
                               runtime=registry['runtime_release'], options=options_for(model)))
        checkpoint=read_json(folder/'checkpoint.json',{})
        records=checkpoint.get('records',{}) if checkpoint.get('fingerprint')==stamp else {}
        if retry_failed:
            records={k:v for k,v in records.items() if v['status']!='failed'}
        state['cache_hits']=len(records)
        save('translating','모델 준비',total=len(ir['blocks']),completed=len(records),fingerprint=stamp)
        if not any(not b['preserve'] for b in ir['blocks']):
            raise ValueError('번역할 영어 본문을 찾지 못했습니다. 스캔 PDF의 OCR은 다음 개발 단계입니다. 원본은 보존됐습니다.')
        timings=[]
        pending=[b for b in ir['blocks'] if b['id'] not in records]
        with LlamaBackend(folder,cancel,model=model) as backend:
            if any(not b['preserve'] for b in pending):
                backend.start(lambda msg: save(stage=msg))
            for block in ir['blocks']:
                if cancel(): raise Cancelled()
                if block['id'] in records: continue
                save(stage='문단 번역',current_source_page=block['source_page_index']+1)
                record={k:block[k] for k in ['id','source_page_index','bbox','kind','source']}
                record.update(target=block['source'],status='preserved',issues=[])
                if not block['preserve']:
                    protected,mapping=protect(block['source'])
                    prompt=make_prompt(protected,mapping,state['profile'],model.get('adapter','hy'))
                    try:
                        # Respect the context/output budget, with no silent truncation.
                        result=backend.translate(prompt,max_tokens=1800 if backend.options['context']==4096 else 2200)
                        # Keep raw local response separately from document-free engine diagnostics.
                        atomic_json(folder/'responses'/(block['id']+'.json'),result)
                        target=restore(result['text'],mapping)
                        issues=check(block['source'],target)
                        record.update(target=target,issues=issues,status='translated',seconds=result['seconds'])
                        if any(i['code'] in {'numbers','empty','untranslated','new_markup'} or i['severity']=='critical' for i in issues):
                            record.update(candidate=target,target=block['source'],status='failed')
                        timings.append(result['seconds'])
                    except Cancelled:
                        raise
                    except Exception as exc:
                        record.update(status='failed',issues=[dict(code='generation',severity='major',message=str(exc)[:500])])
                records[block['id']]=record
                atomic_json(folder/'checkpoint.json',dict(fingerprint=stamp,records=records))
                remaining=sum(b['id'] not in records and not b['preserve'] for b in ir['blocks'])
                eta=round(sum(timings[-8:])/len(timings[-8:])*remaining) if timings and remaining else None
                save(completed=len(records),eta_seconds=eta)
        if cancel(): raise Cancelled()
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
                                  translated=quality['translated_paragraphs'],semantic_review_coverage=quality['semantic_review_coverage']))
    except Cancelled:
        save('cancelled','취소됨 · 이어하기 가능',pid=None,eta_seconds=None)
    except Exception as exc:
        save('failed','작업 실패 · 기록 보존',error=str(exc)[:1000],pid=None,eta_seconds=None)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
