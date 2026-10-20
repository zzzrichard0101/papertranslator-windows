"""One bounded retry for broken responses; retain both attempts locally."""
from .typed_literals import validation_mode
from .backend import Cancelled
from contextlib import nullcontext
from .quality import check,make_prompt,protect,restore,PROMPT_VERSION,VERSION
from .storage import atomic_json,fingerprint

BLOCKING={'numbers','quantity','dimension_role','session_frequency','undertranslation','placeholder','adhd_term','empty','untranslated','new_markup','embedded_han','foreign_script','economic_depression','expansion'}


def blocking(issues):
    return any(i['code'] in BLOCKING or i.get('severity')=='critical' for i in issues)


def generation(backend,block,prompt,mapping,profile,adapter,memory,max_tokens):
    model=getattr(backend,'model',None) or {}
    typed=validation_mode(block.get('validation_policy'))
    cache_prompt=prompt
    if typed:
        for n,token in enumerate(mapping):cache_prompt=cache_prompt.replace(token,f'[PT_LITERAL_{n}]')
    contract=dict(version='translation-response-resolved-2' if typed else 'translation-response-1',source=block['source'],kind=block['kind'],
        prompt=cache_prompt,mapping=list(mapping.values()) if typed else mapping,profile=profile,adapter=adapter,max_tokens=max_tokens,
        model={k:model.get(k) for k in ['id','sha256','revision','adapter']},options=backend.options,
        runtime=(getattr(backend,'registry',None) or {}).get('runtime_release'),
        prompt_version=PROMPT_VERSION,quality_version=VERSION)
    if typed=='typed-v2':contract['validation']=typed
    key=fingerprint(contract)
    with memory.response_lock(key) if memory else nullcontext():
        if hasattr(backend,'check'):backend.check()
        cached=memory.get_response(key) if memory else None
        if isinstance(cached,dict) and isinstance(cached.get('raw'),dict) and isinstance(cached['raw'].get('text'),str):
            raw=cached['raw']
            if typed and cached.get('format')!='resolved-2':cached=None
            if typed:
                try:target=restore(raw['text'],{})
                except ValueError:cached=None;target=''
            else:target=restore(raw['text'],mapping)
            issues=check(block['source'],target,typed=typed)
            if cached is not None and not blocking(issues):
                result={**raw,'seconds':0.,'first_token_seconds':0.,'usage':{},
                    'cache':dict(key=key,origin_id=cached.get('origin_id'),original_seconds=raw.get('seconds'),original_usage=raw.get('usage',{}))}
                return result,target,issues
        raw=backend.translate(prompt,max_tokens=max_tokens)
        target=restore(raw['text'],mapping);issues=check(block['source'],target,typed=typed)
        if memory and not blocking(issues):
            saved={**raw,'text':target} if typed else raw
            memory.put_response(key,dict(raw=saved,origin_id=block['id'],format='resolved-2' if typed else 'raw-1'))
        return raw,target,issues


def translate_record(backend,block,profile,adapter,folder,memory=None):
    record={k:block[k] for k in ['id','source_page_index','source_page_indices','source_fragments','bbox','kind','source']}
    record.update(target=block['source'],status='failed',issues=[],attempts=0,seconds=0.)
    typed=validation_mode(block.get('validation_policy'))
    if typed:record['validation_policy']=block['validation_policy']
    text,mapping=protect(block['source'],typed=typed);base=make_prompt(text,mapping,profile,adapter,context=block.get('context'))
    # Specialist templates receive only their documented translation contract.
    for attempt in range(1 if adapter=='translategemma' else 2):
        if attempt and block.get('selective_repair') and record.get('candidate'):
            from .sentence_repair import plan,splice,VERSION as REPAIR_VERSION
            window=plan(block['source'],record['candidate'],record['issues'])
            if window:
                repair=dict(version=REPAIR_VERSION,window=window,status='attempted',human_approved=False)
                record['sentence_repair']=repair;record['attempts']+=1
                try:
                    local={**block,'id':block['id']+'-sentence-repair','source':window['source']}
                    context={**block.get('context',{}),'before':block['source'][:window['source_start']][-220:],
                             'after':block['source'][window['source_end']:][:220]}
                    masked,protected=protect(local['source'],typed=typed)
                    prompt=make_prompt(masked,protected,profile,adapter,context=context)
                    prompt=('Translate only this source sentence again. Correct the validation failures: '+
                            '; '.join(i['code'] for i in record['issues'])+'. Do not translate the neighbouring context.\n\n'+prompt)
                    raw,candidate,local_issues=generation(backend,local,prompt,protected,profile,adapter,memory,
                                                        min(1800,max(512,len(local['source'])*2)))
                    atomic_json(folder/'responses'/(local['id']+'.json'),raw)
                    record['seconds']+=raw['seconds']
                    if blocking(local_issues):raise ValueError('부분 교정 문장 검사 미통과')
                    target=splice(block['source'],record['candidate'],window,candidate)
                    repair.update(status='accepted',previous_candidate=record['candidate'])
                    record.update(target=target,candidate=target,issues=check(block['source'],target,typed=typed),status='translated')
                    return record
                except Cancelled:raise
                except Exception as exc:
                    repair.update(status='rejected',reason=str(exc)[:300])
        prompt=base
        if attempt:
            prompt=('The previous translation failed automatic validation. Translate the source again faithfully. '
                    'Use Korean Hangul, without inserting Han/Chinese characters into Korean words. '
                    'Keep Latin personal names in their original spelling. Preserve all quantities and signs; '
                    'do not invent facts. Output only the translation. '
                    'Correct these validation failures: '+
                    '; '.join(i['code']+': '+i.get('message','')[:100] for i in record['issues'][:2])[:300]+'\n\n'+base)
        record['attempts']+=1
        try:
            result,target,issues=generation(backend,block,prompt,mapping,profile,adapter,memory,
                                            1800 if backend.options['context']==4096 else 2200)
            name=block['id']+('' if not attempt else '-attempt2')+'.json'
            atomic_json(folder/'responses'/name,result)
            record['seconds']+=result['seconds']
            record.update(candidate=target,issues=issues)
            if result.get('cache'):record['translation_cache']=result['cache']
            if not blocking(issues):
                record.update(target=target,status='translated');return record
        except Cancelled:raise
        except Exception as exc:
            record['issues']=[dict(code='generation',severity='major',message=str(exc)[:500])]
    return record
