"""Select an installed model without changing any existing job snapshots."""
import argparse
from papertranslator.storage import read_json,atomic_json,root


def select(key):
    path=root()/'model_registry.json';registry=read_json(path,{})
    if not registry.get('model'):
        raise ValueError('Run scripts/bootstrap.py before selecting a model.')
    models=registry.setdefault('models',{})
    models.setdefault('hy7b',{**registry['model'],'key':'hy7b','adapter':'hy',
        'options':dict(context=4096,temperature=.7,top_p=.6,top_k=20,min_p=0.,repeat_penalty=1.05,seed=42)})
    if key not in models:raise ValueError('Download this model before selecting it: '+key)
    labels={'hy7b':'Hy-MT2 7B · 경량 모드','qwen38':'Qwen3.8 27B · 고품질 실험 모드',
            'qwen30b':'Qwen3 30B-A3B · 비교용','translategemma27b':'TranslateGemma 27B · 비교용'}
    for name,model in models.items():
        model['display_name']=labels.get(name,model['id'])
        model['status_ko']='공개 알파 · 원문 대조 필요'
    registry['schema']=2;registry['default_key']=key
    atomic_json(path,registry)
    settings=read_json(root()/'settings.json',{});settings['model_key']=key
    atomic_json(root()/'settings.json',settings)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('model',choices=['hy7b','qwen38','qwen30b','translategemma27b'])
    select(parser.parse_args().model)
    print('Selected installed model. Existing jobs keep their saved model.')
