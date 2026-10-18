import importlib.util
from pathlib import Path
import pytest


def module(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parents[1]/'scripts'/(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def test_public_audit_rejects_generated_files_and_sensitive_values():
    audit=module('audit_public')
    secret='gh'+'p_'+'a'*36
    local='C'+':/Users/'+'example-person/Documents/private.txt'
    assert any(x[2]=='provider credential' for x in audit.findings('README.md',secret.encode()))
    assert any(x[2]=='personal machine path' for x in audit.findings('README.md',local.encode()))
    assert audit.findings('jobs/state.json',b'{}')
    assert audit.findings('src/papertranslator/secret.pdf',b'%PDF-1.7')
    assert audit.findings('src/papertranslator/model.onnx',b'synthetic weights')
    assert not audit.findings('README.md',b'No models or documents are bundled. SHA256 verifies downloads.')


def test_model_selection_keeps_job_snapshots_and_destination(tmp_path,monkeypatch):
    setup=module('select_model')
    from papertranslator.storage import atomic_json,read_json
    monkeypatch.setattr(setup,'root',lambda:tmp_path)
    baseline=dict(id='baseline',path='model.gguf',sha256='synthetic')
    atomic_json(tmp_path/'model_registry.json',dict(model=baseline,models={'qwen38':dict(id='large',options={'context':4096})}))
    atomic_json(tmp_path/'settings.json',dict(destination='selected-vault'))
    job=tmp_path/'jobs/snapshot.json';atomic_json(job,dict(model=baseline));before=job.read_bytes()
    setup.select('qwen38')
    assert read_json(tmp_path/'model_registry.json')['default_key']=='qwen38'
    assert read_json(tmp_path/'settings.json')==dict(destination='selected-vault',model_key='qwen38')
    assert job.read_bytes()==before
    setup.select('hy7b');assert read_json(tmp_path/'model_registry.json')['models']['hy7b']['adapter']=='hy'
    with pytest.raises(ValueError):setup.select('missing')
