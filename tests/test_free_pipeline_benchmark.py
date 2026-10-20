import copy
import importlib
from pathlib import Path


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('benchmark_free_pipelines')


def registry():
    return dict(models={key: dict(id=key, adapter=adapter, sha256='synthetic',
                                 options=dict(temperature=.7, context=4096))
                        for key, adapter in [('qwen38', 'qwen'), ('hy7b', 'hy'),
                                             ('qwen30b', 'qwen'), ('translategemma27b', 'translategemma')]})


def test_production_sampling_without_mutating_registry(monkeypatch):
    m = module(monkeypatch); installed = registry(); original = copy.deepcopy(installed)
    for key in installed['models']:
        configured = m.configured(installed['models'][key], installed)
        assert configured['options']['temperature'] == (.7 if key == 'translategemma27b' else .2)
    assert m.configured(installed['models']['hy7b'], installed, auxiliary=True)['options']['temperature'] == .7
    editor = m.configured(installed['models']['qwen38'], installed, editor=True, workers=2)
    assert editor['options']['context'] == 8192 and editor['options']['temperature'] == .1
    assert editor['options']['parallel'] == 2
    assert installed == original


def test_all_cases_get_full_editor_and_shared_work_is_charged(monkeypatch, tmp_path):
    m = module(monkeypatch); installed = registry(); edited = []; calls = []
    def translation(backend, block, profile, adapter, folder):
        return {**block, 'target': '합성 번역', 'candidate': '합성 번역', 'status': 'translated', 'issues': []}
    def edit(backend, record, block, settings, stamp, folder, cancel):
        edited.append((folder.name, record['id'], bool(record.get('alternative'))))
        assert record['source'] == block['source']
        assert settings['editor_output'] == 'patch-v1'
        return {**record, 'refinement': dict(decision='keep')}
    def stage(folder, model, items, operation):
        calls.append(folder.name)
        results = [operation(object(), item, folder) for item in items]
        seconds = 3 if folder.name.startswith('edit-') else (2 if folder.name.startswith('auxiliary-') else 1)
        return results, dict(warm_seconds=seconds, startup_seconds=.5, failed=0, blocking=0, uncertain=0)
    monkeypatch.setattr(m, 'translate_record', translation)
    monkeypatch.setattr(m, 'edit_record', edit)
    monkeypatch.setattr(m, 'stage', stage)
    report = m.compare(installed, tmp_path, list(installed['models']), 3, 1)
    assert len(edited) == 12 and len({entry[1] for entry in edited}) == 3
    assert all(has_alt == (name != 'edit-hy7b') for name, _, has_alt in edited)
    assert calls.count('auxiliary-hy7b') == 1
    for pipeline in report['pipelines']:
        assert pipeline['warm_seconds'] == (4 if pipeline['primary'] == 'hy7b' else 6)
    assert report['automatic_promotion'] is False and report['human_approved'] is False
    assert (tmp_path / 'blind-comparison.json').is_file()


def test_new_candidate_registration_preserves_defaults_and_refreshes_settings(monkeypatch, tmp_path):
    module(monkeypatch)
    m = importlib.import_module('fetch_hy30')
    from papertranslator.storage import atomic_json, read_json
    monkeypatch.setattr(m, 'ROOT', tmp_path)
    initial = registry(); initial.update(default_key='qwen38', model=initial['models']['hy7b'])
    atomic_json(tmp_path / 'model_registry.json', initial)
    def fetch(asset):
        assert asset['sha256'] == 'bb44b11bb0f7cd3d1321645b41e911cd3de2e473731227fc6bf37aa18b543f88'
        assert asset['size'] == 18236702880 and m.REVISION in asset['url']
        current = read_json(tmp_path / 'model_registry.json')
        current['newer_setting'] = 'preserved'
        atomic_json(tmp_path / 'model_registry.json', current)
        return tmp_path / asset['name']
    monkeypatch.setattr(m, 'fetch', fetch)
    installed = m.install(); after = read_json(tmp_path / 'model_registry.json')
    assert after['default_key'] == 'qwen38' and after['newer_setting'] == 'preserved'
    assert after['model'] == initial['model']
    assert after['models']['hy30b'] == installed
    assert installed['adapter'] == 'hy' and installed['role'] == 'comparison candidate; not promoted'


def test_default_comparison_only_uses_installed_models_and_restores_home(monkeypatch, tmp_path):
    import os
    import sys
    from contextlib import nullcontext
    from papertranslator.storage import atomic_json
    m = module(monkeypatch); installed = registry()
    installed['models'].pop('qwen30b'); installed['models'].pop('translategemma27b')
    home = tmp_path / 'production'; home.mkdir()
    atomic_json(home / 'model_registry.json', installed)
    monkeypatch.setattr(m, 'root', lambda: home)
    monkeypatch.setattr(m, 'worker_lock', nullcontext)
    monkeypatch.setattr(m, 'gpu_idle', lambda: None)
    monkeypatch.setattr(m, '__file__', str(tmp_path / 'scripts/benchmark_free_pipelines.py'))
    monkeypatch.setattr(sys, 'argv', ['benchmark_free_pipelines.py'])
    monkeypatch.setenv('PAPERTRANSLATOR_HOME', str(home))
    compared = []
    def compare(registry, work, keys, limit, workers):
        compared.extend(keys)
        assert os.environ['PAPERTRANSLATOR_HOME'] != str(home)
    monkeypatch.setattr(m, 'compare', compare)
    m.main()
    assert compared == ['qwen38', 'hy7b']
    assert os.environ['PAPERTRANSLATOR_HOME'] == str(home)
