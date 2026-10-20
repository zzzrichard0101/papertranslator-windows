"""Job-scoped model snapshots. Changing the default must not change resumed jobs."""
from copy import deepcopy
from .storage import read_json, root

DEFAULT_OPTIONS=dict(context=8192,parallel=1,kv='f16',threads=12,gpu_layers=99,
                     temperature=.7,top_p=.6,top_k=20,min_p=0.0,repeat_penalty=1.05,seed=42,flash_attention='auto')


def catalog(registry=None):
    registry=registry or read_json(root()/'model_registry.json',{})
    result=deepcopy(registry.get('models',{}))
    if not result and registry.get('model'):
        result['hy7b']={**registry['model'],'key':'hy7b','adapter':'hy'}
    return result


def resolve_model(key=None,registry=None):
    registry=registry or read_json(root()/'model_registry.json',{})
    models=catalog(registry)
    key=key or registry.get('default_key') or ('hy7b' if 'hy7b' in models else next(iter(models),None))
    if key not in models: raise ValueError('선택한 로컬 모델이 설치되지 않았습니다.')
    return deepcopy(models[key])


def options_for(model):
    options={**DEFAULT_OPTIONS,**model.get('options',{})}
    if model.get('adapter')=='translategemma':options['chat_template']='model_metadata'
    if options['context'] not in {4096,8192,16384} or options['parallel'] not in {1,2}:
        raise ValueError('검증되지 않은 문맥/병렬 설정')
    if options['kv'] not in {'f16','q8_0'} or options['flash_attention'] not in {'on','off','auto'}:
        raise ValueError('지원하지 않는 KV/Flash Attention 설정')
    return options
