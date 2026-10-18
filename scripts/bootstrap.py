"""User-scope, pinned, resumable official runtime/model acquisition. No remote code."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request
import zipfile

from papertranslator.storage import root
ROOT = root()
REV = 'ab8472660ac61fac25f1af43fac2599d52a8a775'
ASSETS = [
    dict(name='llama-b11718-bin-win-cuda-12.4-x64.zip', kind='runtime', size=267470642,
         url='https://github.com/ggml-org/llama.cpp/releases/download/b11718/llama-b11718-bin-win-cuda-12.4-x64.zip',
         sha256='c13fda0ef74f47cd2bccda1eea2a62980c12a5778d39b39bedadb525b4e81aff'),
    dict(name='cudart-llama-bin-win-cuda-12.4-x64.zip', kind='runtime', size=391443627,
         url='https://github.com/ggml-org/llama.cpp/releases/download/b11718/cudart-llama-bin-win-cuda-12.4-x64.zip',
         sha256='8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6'),
    dict(name='HY-MT2-7B-Q6_K.gguf', kind='model', size=6164482720,
         url=f'https://huggingface.co/tencent/Hy-MT2-7B-GGUF/resolve/{REV}/HY-MT2-7B-Q6_K.gguf',
         sha256='88ef0aba59952a4cfe4be36cb5baf797dbb370bc60e9dcbd7297036021e52831')]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def fetch(asset):
    target = ROOT / ('models' if asset['kind'] == 'model' else 'downloads') / asset['name']
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and digest(target) == asset['sha256']:
        return target
    partial = target.with_suffix(target.suffix + '.part')
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        required = max(0, asset['size'] - offset) + 1024 ** 3
        if shutil.disk_usage(ROOT).free < required:
            raise RuntimeError(f'Insufficient disk space: need {required} bytes')
        request = urllib.request.Request(asset['url'], headers={'Range': f'bytes={offset}-', 'User-Agent': 'PaperTranslator-Setup/0.1'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and response.status != 206:
                    offset = 0
                if response.status == 206:
                    content_range = response.headers.get('Content-Range', '')
                    if not content_range.startswith(f'bytes {offset}-') or not content_range.endswith(f'/{asset["size"]}'):
                        raise RuntimeError('Invalid Content-Range')
                started = time.monotonic(); last = started; received = 0
                with partial.open('ab' if offset else 'wb') as out:
                    while data := response.read(4 * 1024 * 1024):
                        out.write(data); received += len(data)
                        now = time.monotonic()
                        if now - last > 3:
                            speed = received / max(now - started, .001)
                            print(json.dumps(dict(file=asset['name'], bytes=offset+received, total=asset['size'], MB_s=round(speed/1e6, 2), eta_s=round((asset['size']-offset-received)/speed))), flush=True)
                            last = now
            if partial.stat().st_size != asset['size'] or digest(partial) != asset['sha256']:
                partial.unlink(missing_ok=True)
                raise RuntimeError('Size/SHA256 mismatch')
            partial.replace(target)
            return target
        except Exception:
            if attempt == 3:
                raise
            time.sleep(min(2 ** attempt, 8))


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    runtime = ROOT / 'runtimes' / 'llama-b11718-cuda12'
    runtime.mkdir(parents=True, exist_ok=True)
    for asset in ASSETS:
        path = fetch(asset)
        if asset['kind'] == 'runtime':
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    dest = (runtime / info.filename).resolve()
                    if not dest.is_relative_to(runtime.resolve()):
                        raise ValueError('Unsafe archive path')
                archive.extractall(runtime)
    baseline_registry = dict(schema=1, runtime_release='b11718', runtime_dir=str(runtime),
                    model=dict(id='tencent/Hy-MT2-7B-GGUF:Q6_K', revision=REV, license='Apache-2.0',
                               path=str(ROOT/'models'/ASSETS[2]['name']), sha256=ASSETS[2]['sha256'],
                               role='experimental baseline; not quality promoted'), assets=ASSETS)
    # Reinstallation keeps the user's selected default and downloaded candidates.
    from papertranslator.storage import atomic_json,read_json
    existing=read_json(ROOT/'model_registry.json',{})
    registry={**baseline_registry,**existing,'runtime_dir':str(runtime),'runtime_release':'b11718'}
    atomic_json(ROOT/'model_registry.json',registry)
    for name in ['README.md', 'LICENSE.txt']:
        urllib.request.urlretrieve(f'https://huggingface.co/tencent/Hy-MT2-7B-GGUF/resolve/{REV}/{name}', ROOT/'models'/('Hy-MT2-'+name))
    print(json.dumps(dict(status='verified', root=str(ROOT))), flush=True)


if __name__ == '__main__':
    main()
