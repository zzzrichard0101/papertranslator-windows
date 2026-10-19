"""One bounded retry for broken responses; retain both attempts locally."""
from .backend import Cancelled
from .quality import check,make_prompt,protect,restore
from .storage import atomic_json

BLOCKING={'numbers','quantity','empty','untranslated','new_markup','embedded_han','foreign_script','economic_depression'}


def blocking(issues):
    return any(i['code'] in BLOCKING or i.get('severity')=='critical' for i in issues)


def translate_record(backend,block,profile,adapter,folder):
    record={k:block[k] for k in ['id','source_page_index','source_page_indices','source_fragments','bbox','kind','source']}
    record.update(target=block['source'],status='failed',issues=[],attempts=0,seconds=0.)
    text,mapping=protect(block['source']);base=make_prompt(text,mapping,profile,adapter,context=block.get('context'))
    # Specialist templates receive only their documented translation contract.
    for attempt in range(1 if adapter=='translategemma' else 2):
        prompt=base
        if attempt:
            prompt=('The previous translation failed automatic validation. Translate the source again faithfully. '
                    'Use Korean Hangul, without inserting Han/Chinese characters into Korean words. '
                    'Keep Latin personal names in their original spelling. Preserve all quantities and signs; '
                    'do not invent facts. Output only the translation.\n\n'+base)
        record['attempts']=attempt+1
        try:
            result=backend.translate(prompt,max_tokens=1800 if backend.options['context']==4096 else 2200)
            name=block['id']+('' if not attempt else '-attempt2')+'.json'
            atomic_json(folder/'responses'/name,result)
            record['seconds']+=result['seconds']
            target=restore(result['text'],mapping);issues=check(block['source'],target)
            record.update(candidate=target,issues=issues)
            if not blocking(issues):
                record.update(target=target,status='translated');return record
        except Cancelled:raise
        except Exception as exc:
            record['issues']=[dict(code='generation',severity='major',message=str(exc)[:500])]
    return record
