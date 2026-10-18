import json
from pathlib import Path
import httpx
import pytest

from papertranslator.models import catalog,options_for,resolve_model
from papertranslator.quality import make_prompt


def test_model_selection_is_a_deep_snapshot():
    registry=dict(default_key='large',models={'small':dict(id='s',options={'context':4096}), 'large':dict(id='l',options={'context':8192})})
    chosen=resolve_model(registry=registry);chosen['options']['context']=16384
    assert registry['models']['large']['options']['context']==8192
    assert resolve_model('small',registry)['id']=='s'
    assert catalog(dict(model=dict(id='legacy')))['hy7b']['adapter']=='hy'


def test_generation_options_are_separate_per_model():
    a=options_for(dict(options={'context':4096,'temperature':0,'top_k':1}))
    b=options_for({})
    assert a['temperature']==0 and b['temperature']==.7 and a['context']==4096 and b['context']==8192
    with pytest.raises(ValueError): options_for(dict(options={'parallel':4}))


def test_specialist_prompt_does_not_translate_generic_instructions():
    text='The occupational therapist used oil pastel.'
    prompt=make_prompt(text,{},'art_therapy','translategemma')
    assert prompt.endswith('\n\n\n'+text)
    assert 'Reference the following translations' not in prompt and 'PX identifier' not in prompt
    assert 'Reference the following translations' in make_prompt(text,{},'general','qwen')
    assert options_for(dict(adapter='translategemma'))['chat_template']=='model_metadata'


def test_glossary_is_relevant_bounded_and_context_sensitive():
    from papertranslator.quality import relevant_terms
    assert not relevant_terms('A braid is an algebraic object in this paper.')
    assert ('braid','땋은 머리') in relevant_terms('A gray braid rested on her shoulder.')
    assert not relevant_terms('Charcoal fuel was burned in a furnace.')
    assert ('charcoal','목탄') in relevant_terms('The art workshop offered charcoal and paint.')
    assert relevant_terms('The government agency funded the study.')==[]
    selected=relevant_terms('Use postcard-size paper.')
    assert selected==[('postcard-size','엽서 크기의')]


def test_streaming_uses_selected_sampling_and_rejects_truncation(monkeypatch,tmp_path):
    import papertranslator.backend as module
    monkeypatch.setattr(module,'read_json',lambda *args:dict(model=dict(id='legacy')))
    model=dict(id='qwen',adapter='qwen',options=dict(context=4096,temperature=0,top_p=.8,repeat_penalty=1))
    backend=module.LlamaBackend(tmp_path,model=model)
    seen=[]
    def handler(request):
        body=json.loads(request.content)
        if request.url.path=='/tokenize':return httpx.Response(200,json={'tokens':[1,2,3]})
        seen.append(body)
        return httpx.Response(200,text='data: '+json.dumps({'choices':[{'delta':{'content':'번역'},'finish_reason':'length'}]})+'\n\ndata: [DONE]\n\n')
    backend.client=httpx.Client(base_url='http://127.0.0.1:1',transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError,match='출력 미완료'):backend.translate('test',max_tokens=500)
    assert seen[0]['temperature']==0 and seen[0]['top_p']==.8 and seen[0]['repeat_penalty']==1
    backend.close()
