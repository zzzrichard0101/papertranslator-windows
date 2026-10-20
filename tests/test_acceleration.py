import pytest
from papertranslator.acceleration import MTP_MODEL_SHA256,apply,arguments,capable
from papertranslator.models import options_for
from papertranslator.backend import LlamaBackend,Cancelled


def test_only_exact_model_runtime_pair_is_eligible_and_snapshots_are_independent():
    model=dict(sha256=MTP_MODEL_SHA256,adapter='qwen',options=dict(context=8192))
    registry=dict(runtime_release='b11718')
    assert capable(model,registry)
    assert not capable({**model,'sha256':'unknown'},registry)
    assert not capable(model,dict(runtime_release='unknown'))
    result=apply(model,registry,'mtp');assert 'speculation' not in model['options']
    assert arguments(options_for(result))==['--spec-type','draft-mtp','--spec-draft-n-max','3']
    assert arguments(options_for(model))==[] and 'speculation' not in options_for(model)
    assert apply(model,registry,'none')==model


@pytest.mark.parametrize('options',[dict(speculation='unsafe'),dict(spec_tokens=0),dict(spec_tokens=5),dict(spec_tokens=True)])
def test_no_unbounded_or_synthetic_acceptance_options(options):
    with pytest.raises(ValueError):options_for(dict(options=options))


def test_mtp_failure_then_parallel_failure_falls_back_without_reducing_context(monkeypatch,tmp_path):
    import papertranslator.backend as module
    monkeypatch.setattr(module,'read_json',lambda *_:dict(model={}))
    backend=LlamaBackend(tmp_path,model=dict(options=dict(context=8192,parallel=2,speculation='mtp')))
    tried=[]
    def start(progress):
        tried.append(dict(backend.options))
        if len(tried)<3:raise RuntimeError('synthetic allocation failure')
    monkeypatch.setattr(backend,'_start',start);backend.start()
    assert [(o['speculation'],o['parallel']) for o in tried]==[('mtp',2),('none',2),('none',1)]
    assert all(o['context']==8192 for o in tried)
    assert backend.log_name=='engine-no-mtp-sequential.log'


def test_cancel_does_not_disable_mtp_and_retry(monkeypatch,tmp_path):
    import papertranslator.backend as module
    monkeypatch.setattr(module,'read_json',lambda *_:dict(model={}))
    backend=LlamaBackend(tmp_path,model=dict(options=dict(speculation='mtp')));calls=[]
    def start(progress):calls.append(True);raise Cancelled()
    monkeypatch.setattr(backend,'_start',start)
    with pytest.raises(Cancelled):backend.start()
    assert len(calls)==1 and backend.options['speculation']=='mtp'


def test_gpu_probe_refuses_an_existing_engine_without_touching_it(monkeypatch):
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('benchmark_acceleration',Path(__file__).parents[1]/'scripts/benchmark_acceleration.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Process:info=dict(name='llama-server.exe')
    monkeypatch.setattr(module.psutil,'process_iter',lambda *_:[Process()])
    with pytest.raises(RuntimeError):module.gpu_idle()


def test_job_snapshots_only_opted_in_models_and_compact_policy(monkeypatch,tmp_path):
    import pymupdf as fitz
    from papertranslator.jobs import create_job
    from papertranslator.storage import atomic_json,read_json
    home=tmp_path/'home';home.mkdir();monkeypatch.setenv('PAPERTRANSLATOR_HOME',str(home))
    atomic_json(home/'model_registry.json',dict(runtime_release='b11718',default_key='hy7b',models={
        'hy7b':dict(id='synthetic-hy',adapter='hy',sha256='synthetic'),
        'qwen38':dict(id='synthetic-editor',adapter='qwen',sha256=MTP_MODEL_SHA256)}))
    source=tmp_path/'synthetic.pdf'
    with fitz.open() as doc:doc.new_page();doc.save(source)
    ordinary=create_job(source,tmp_path,quality_mode='ensemble')
    accelerated=create_job(source,tmp_path,quality_mode='ensemble',inference_acceleration='mtp',compact_editor=True)
    a=read_json(ordinary/'state.json');b=read_json(accelerated/'state.json')
    assert 'editor_output' not in a['quality_policy'] and 'speculation' not in a['quality_policy']['editor']['options']
    assert 'speculation' not in b['model']['options']  # Hy remains unchanged
    assert b['quality_policy']['editor']['options']['speculation']=='mtp'
    assert b['quality_policy']['editor_output']=='patch-v1'
