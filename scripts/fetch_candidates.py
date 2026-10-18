"""Download two public official-library GGUFs, verify digests, retain baseline."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

from bootstrap import ROOT
from download_parallel import fetch
from papertranslator.storage import atomic_json, read_json

PROJECT=Path(__file__).resolve().parents[1]
SPECS=[
    dict(key='qwen30b',library='qwen3',tag='30b-a3b-instruct-2507-q4_K_M',filename='Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf',
         sha256='78b329e716e7e9775973d392cd132b1f1ff1c8287a992887caeb6fd6c56ba9cc',size=18556685856,
         adapter='qwen',license='Apache-2.0',options=dict(context=4096,temperature=.7,top_p=.8,top_k=20,repeat_penalty=1.0,seed=42)),
    dict(key='translategemma27b',library='translategemma',tag='27b',filename='TranslateGemma-27B-Q4_K_M.gguf',
         sha256='d87ec9a33af49e86461d8309e28e6bdb663f852c09dfca5ff581bd796cdaf583',size=17358640928,
         adapter='translategemma',license='Gemma Terms of Use',options=dict(context=4096,temperature=0.0,top_p=1.0,top_k=1,repeat_penalty=1.0,seed=42))]


def download(spec):
    base=f'https://registry.ollama.ai/v2/library/{spec["library"]}'
    manifest_bytes=urllib.request.urlopen(base+'/manifests/'+spec['tag'],timeout=30).read()
    manifest=json.loads(manifest_bytes)
    model=next(l for l in manifest['layers'] if l['mediaType']=='application/vnd.ollama.image.model')
    assert model['digest']=='sha256:'+spec['sha256'] and model['size']==spec['size'], 'Pinned manifest changed'
    meta=ROOT/'models'/spec['key'];meta.mkdir(exist_ok=True)
    (meta/'ollama-manifest.json').write_bytes(manifest_bytes)
    for layer in manifest['layers']:
        if layer is model: continue
        if layer['mediaType'] in {'application/vnd.ollama.image.projector','application/vnd.ollama.image.draft'}:continue
        data=urllib.request.urlopen(base+'/blobs/'+layer['digest'],timeout=30).read()
        assert hashlib.sha256(data).hexdigest()==layer['digest'].split(':')[1]
        (meta/(layer['mediaType'].rsplit('.',1)[1]+'.txt')).write_bytes(data)
    asset=dict(name=spec['filename'],kind='model',size=spec['size'],sha256=spec['sha256'],url=base+'/blobs/sha256:'+spec['sha256'])
    path=fetch(asset)
    result=dict(id=spec['library']+':'+spec['tag'],key=spec['key'],path=str(path),sha256=spec['sha256'],
                revision='ollama-manifest-sha256:'+hashlib.sha256(manifest_bytes).hexdigest(),
                source_url='https://ollama.com/library/'+spec['library']+':'+spec['tag'],
                license=spec['license'],adapter=spec['adapter'],options=spec['options'],
                role='comparison candidate; not promoted',asset=asset)
    atomic_json(meta/'verified-model.json',result)
    return result


if __name__=='__main__':
    print(json.dumps(dict(planned_bytes=sum(s['size'] for s in SPECS),models=[s['key'] for s in SPECS])),flush=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        models=list(executor.map(download,SPECS))
    from convert_gemma_legacy import convert
    models=[convert(model) if model['key']=='translategemma27b' else model for model in models]
    registry=read_json(ROOT/'model_registry.json')
    registry.setdefault('models',{})['hy7b']={**registry['model'], 'key':'hy7b','adapter':'hy', 'options':dict(context=4096,temperature=.7,top_p=.6,top_k=20,repeat_penalty=1.05,seed=42)}
    for model in models: registry['models'][model['key']]=model
    atomic_json(ROOT/'model_registry.json',registry)
    atomic_json(ROOT/'candidate-downloads.json',models)
    print(json.dumps(dict(status='verified',models=[m['id'] for m in models])),flush=True)
