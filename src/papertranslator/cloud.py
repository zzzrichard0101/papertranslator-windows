"""Opt-in, text-only, zero-price OpenRouter review. No paid fallback."""
import json
import time
from decimal import Decimal,InvalidOperation
import httpx

from .backend import Cancelled
from .credentials import load_key
from .refinement import edit
from .quality import check
from .storage import atomic_json,fingerprint

BASE='https://openrouter.ai/api/v1'


def free_models(client):
    response=client.get('/models');response.raise_for_status()
    if len(response.content)>8*1024**2:raise ValueError('API 모델 목록이 너무 큽니다.')
    result=[]
    for model in response.json().get('data',[]):
        try:
            price=model.get('pricing',{})
            free=Decimal(str(price.get('prompt','NaN')))==0 and Decimal(str(price.get('completion','NaN')))==0
            free=free and all(Decimal(str(price.get(field,'0')))==0 for field in ['request','internal_reasoning'])
        except InvalidOperation:free=False
        if free and model.get('id','').endswith(':free'):result.append(model['id'])
    return result


def available_models():
    with httpx.Client(base_url=BASE,trust_env=False,follow_redirects=False,timeout=15) as client:
        return free_models(client)


class FreeReviewBackend:
    def __init__(self,model,key,cancel=lambda:False,transport=None):
        if not model.endswith(':free'):raise ValueError('무료 모델만 사용할 수 있습니다.')
        self.model=model;self.key=key;self.cancel=cancel;self.transport=transport;self.client=None

    def __enter__(self):
        self.client=httpx.Client(base_url=BASE,headers={'Authorization':'Bearer '+self.key},trust_env=False,
                                 follow_redirects=False,timeout=httpx.Timeout(10,read=60),transport=self.transport)
        try:
            if self.model not in free_models(self.client):
                raise ValueError('현재 목록에서 입력/출력 0원 모델임을 확인하지 못했습니다.')
        except Exception:
            self.client.close();raise
        return self

    def __exit__(self,*args):
        if self.client:self.client.close()

    def translate(self,prompt,max_tokens=2200,json_schema=None):
        if self.cancel():raise Cancelled()
        if len(prompt)>24000:raise ValueError('API 검토 문단이 너무 큽니다.')
        payload=dict(model=self.model,messages=[dict(role='user',content=prompt)],temperature=.1,
          max_tokens=min(2200,max_tokens),provider=dict(allow_fallbacks=False,data_collection='deny',
          require_parameters=True,max_price=dict(prompt=0,completion=0,request=0)),stream=False)
        if json_schema:payload['response_format']=dict(type='json_schema',json_schema=dict(name='edit',strict=True,schema=json_schema))
        began=time.monotonic()
        try:response=self.client.post('/chat/completions',json=payload)
        except httpx.HTTPError:raise RuntimeError('무료 API 연결에 실패했습니다. 로컬 번역을 유지합니다.') from None
        if response.status_code!=200:
            raise RuntimeError(f'무료 API 응답 {response.status_code}: 한도·모델·비수집 경로를 확인하세요. 유료 전환하지 않았습니다.')
        if len(response.content)>2*1024**2:raise ValueError('API 응답 크기 초과')
        data=response.json()
        if data.get('error'):raise RuntimeError('무료 API가 오류를 반환했습니다.')
        if Decimal(str(data.get('usage',{}).get('cost',0)))!=0:raise RuntimeError('API 0원 사용 조건 확인 실패. 추가 요청을 중단하세요.')
        if str(data.get('model','')).split(':')[0]!=self.model.split(':')[0]:raise ValueError('요청한 API 모델과 응답 모델이 다릅니다.')
        choice=data['choices'][0]
        if choice.get('finish_reason')!='stop':raise ValueError('API 교정 출력이 완성되지 않았습니다.')
        if self.cancel():raise Cancelled()
        text=choice['message'].get('content')
        if not isinstance(text,str):raise ValueError('API 응답 형식 오류')
        return dict(text=text,seconds=time.monotonic()-began,usage=data.get('usage',{}),model=self.model,provider='OpenRouter')


def review_records(records,ir,settings,folder,save_checkpoint,progress,cancel):
    from .checkpoint import persist
    config=settings['cloud'];stamp=fingerprint(config);key=load_key()
    if not key:raise ValueError('무료 API 키가 없습니다. API 설정에서 등록하세요.')
    blocks={b['id']:b for b in ir['blocks']}
    # Only unresolved/risky paragraphs, bounded per job. Never upload the PDF.
    eligible=[r for r in records.values() if r['status']!='preserved' and
              (r['status']=='failed' or r.get('issues') or r.get('refinement',{}).get('decision')=='uncertain')]
    used=sum(bool(r.get('cloud_review',{}).get('attempted')) for r in records.values())
    pending=[r for r in eligible if r.get('cloud_review',{}).get('policy')!=stamp]
    if not pending:return
    budget=max(0,min(20,int(config.get('max_requests',20)))-used)
    if not budget:
        for record in pending:
            record['cloud_review']=dict(policy=stamp,attempted=False,status='deferred',reason='job_request_limit')
            persist(save_checkpoint,record)
        save_checkpoint();return
    try:
        with FreeReviewBackend(config['model'],key,cancel) as backend:
            for index,record in enumerate(pending[:budget]):
                if cancel():raise Cancelled()
                progress(f'무료 API 선택 문단 검토 {index+1}/{min(budget,len(pending))}')
                draft=(record.get('candidate') if record['status']=='failed' else record['target']) or record['target']
                context={k:v for k,v in blocks[record['id']].get('context',{}).items() if k in {'before','after'}}
                record['cloud_review']=dict(policy=stamp,attempted=True,status='requesting',provider='OpenRouter',model=config['model'])
                persist(save_checkpoint,record) # Commit the budget before making the request.
                value,raw=edit(backend,record['source'],draft,context=context,
                    response_sink=lambda raw:atomic_json(folder/'responses'/('cloud-'+record['id']+'.json'),raw),typed=record.get('validation_policy')=='typed-v1')
                record['cloud_review'].update(status='completed',decision=value['decision'],evidence=value,accepted=False)
                if value['decision']=='revise':
                    target=value['revision'].strip();warnings=[i for i in record.get('issues',[]) if i['code'].startswith('ocr_')]
                    record.update(target=target,candidate=target,status='translated',issues=check(record['source'],target,typed=record.get('validation_policy')=='typed-v1')+warnings)
                    record['cloud_review'].update(accepted=True,draft=draft)
                persist(save_checkpoint,record)
    except Cancelled:raise
    except Exception:
        # Keep exact request/response material local; never echo headers or keys.
        for record in pending:
            entry=record.get('cloud_review',{})
            if entry.get('status')=='completed':continue
            record['cloud_review']={**entry,'policy':stamp,'status':'unavailable','accepted':False}
            record['issues'].append(dict(code='cloud_unavailable',severity='major',message='무료 API 검토를 완료하지 못해 로컬 번역을 유지했습니다.'))
            persist(save_checkpoint,record)
        save_checkpoint()
        return
    for record in pending[budget:]:
        record['cloud_review']=dict(policy=stamp,attempted=False,status='deferred',reason='job_request_limit')
        persist(save_checkpoint,record)
    save_checkpoint()
