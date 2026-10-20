"""Install the official, pinned Hy-MT2 30B-A3B comparison candidate.

The model remains experimental. Registration adds a choice without changing
the selected default or any existing job. No remote code is executed.
"""
from bootstrap import ROOT, fetch
from papertranslator.storage import atomic_json, read_json

REVISION = 'fd3dcbb6b31e9e03923ef4f9f42500f74c3851c3'
ASSET = dict(name='Hy-MT2-30B-A3B-Q4_K_M.gguf', kind='model', size=18236702880,
    url=f'https://huggingface.co/tencent/Hy-MT2-30B-A3B-GGUF/resolve/{REVISION}/Hy-MT2-30B-A3B-Q4_K_M.gguf',
    sha256='bb44b11bb0f7cd3d1321645b41e911cd3de2e473731227fc6bf37aa18b543f88')


def install():
    if not read_json(ROOT / 'model_registry.json', {}).get('model'):
        raise ValueError('기본 런타임을 먼저 설치하세요: scripts/bootstrap.py')
    path = fetch(ASSET)
    # Read again after a potentially long download to preserve newer settings.
    registry = read_json(ROOT / 'model_registry.json')
    model = dict(id='tencent/Hy-MT2-30B-A3B-GGUF:Q4_K_M', key='hy30b', adapter='hy',
        path=str(path), sha256=ASSET['sha256'], revision=REVISION, license='Apache-2.0',
        source_url='https://huggingface.co/tencent/Hy-MT2-30B-A3B-GGUF', asset=ASSET,
        display_name='Hy-MT2 30B-A3B · 번역 전용 실험 모드',
        status_ko='실험 후보 · 의미 정확도 승격 미완료', role='comparison candidate; not promoted',
        options=dict(context=4096, temperature=.7, top_p=1., top_k=-1,
                     min_p=0., repeat_penalty=1., seed=42))
    # Retain the legacy primary when upgrading an older single-model registry.
    models = registry.setdefault('models', {})
    if not models:
        models['hy7b'] = {**registry['model'], 'key': 'hy7b', 'adapter': 'hy'}
    models['hy30b'] = model
    atomic_json(ROOT / 'model_registry.json', registry)
    atomic_json(ROOT / 'models/hy30b/verified-model.json', model)
    return model


if __name__ == '__main__':
    install()
    print('Hy-MT2 30B-A3B verified and registered; default and existing jobs unchanged.', flush=True)
