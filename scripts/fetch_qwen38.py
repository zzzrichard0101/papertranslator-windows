"""One newer dense challenger after older candidates exposed domain regressions."""
from fetch_candidates import download,ROOT,PROJECT
from papertranslator.storage import atomic_json,read_json

if __name__=='__main__':
    spec=dict(key='qwen38',library='qwen3.8',tag='27b',filename='Qwen3.8-27B-Q4_K_M.gguf',
        sha256='f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d',size=16810714464,
        adapter='qwen',license='Apache-2.0',options=dict(context=4096,temperature=.7,top_p=.8,top_k=20,
        min_p=0.0,repeat_penalty=1.0,seed=42,enable_thinking=False))
    model=download(spec)
    registry=read_json(ROOT/'model_registry.json');registry.setdefault('models',{})['qwen38']=model
    atomic_json(ROOT/'model_registry.json',registry)
    atomic_json(ROOT/'candidate-downloads.json',list(registry['models'].values()))
    print('Qwen3.8 challenger verified',flush=True)
