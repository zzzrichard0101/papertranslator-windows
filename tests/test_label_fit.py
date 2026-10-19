import json
from papertranslator.label_fit import run,fits


def test_label_fit_retries_invalid_evidence_and_keeps_the_raw_candidates(tmp_path):
    r=dict(id='label',kind='diagram_label',source='Clinical practice',target='임상에서 수행되는 실천 활동을 의미한다.',status='translated',
           source_fragments=[dict(bbox=[10,10,30,70],rotation=90)])
    calls=[]
    class Backend:
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def start(self,*args):pass
        def translate(self,*args,**kwargs):
            calls.append(1)
            return dict(text=json.dumps(dict(decision='revise',source_evidence='invented' if len(calls)==1 else 'Clinical',
                         draft_evidence='임상',explanation_ko='상자 안에 들어가는 간결한 문구.',revision='임상 실천'),ensure_ascii=False))
    assert not fits(r)
    run({'label':r},dict(id='synthetic'),tmp_path,Backend,lambda:None,lambda _:None,lambda:False)
    assert calls==[1,1] and fits(r) and r['target']=='임상 실천'
    assert r['layout_compaction']['accepted'] and not r['layout_compaction']['human_approved']
    assert len(list((tmp_path/'responses').glob('label-fit-*.json')))==2
