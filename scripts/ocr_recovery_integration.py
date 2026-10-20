"""Induced omission -> actual CPU crop OCR -> worker/export/replay.

The full-page omission is intentional. This tests a recovery path, not an OCR
accuracy rate on arbitrary scans or a translation quality score.
"""
import io
import json
import os
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

from papertranslator import jobs
from papertranslator.document import analyze
from papertranslator.ocr import LocalOCR
from papertranslator.storage import atomic_json, read_json, sha256


def main():
    project=Path(__file__).resolve().parents[1]
    work=project/'work/ocr-recovery-integration';work.mkdir(parents=True,exist_ok=True)
    home=work/'home';home.mkdir(exist_ok=True);os.environ['PAPERTRANSLATOR_HOME']=str(home)
    atomic_json(home/'model_registry.json',dict(runtime_release='synthetic',model=dict(
        id='synthetic',adapter='qwen',path='unused',sha256='synthetic')))
    image=Image.new('RGB',(1200,1600),'white');draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('arial.ttf',32)
    draw.text((90,220),'Continuing professional development (CPD).',font=font,fill='black')
    draw.text((90,430),'Missing resources and study failures.',font=font,fill='black')
    draw.text((90,650),'CPD requirements.',font=font,fill='black')
    stream=io.BytesIO();image.save(stream,format='PNG');source=work/'Synthetic omitted scan.pdf'
    with fitz.open() as doc:
        page=doc.new_page(width=600,height=800);page.insert_image(page.rect,stream=stream.getvalue());doc.save(source)
    engine=LocalOCR();calls=[]
    def induced(pixels):
        rows=engine(pixels);calls.append(pixels.shape)
        # Simulate one known full-page extractor omission. Actual crops use the
        # unmodified OCR engine and the unchanged original raster pixels.
        if pixels.shape[0]>800:rows=[r for r in rows if 'resources' not in r[1]]
        return rows
    old=analyze(source,[0],ocr_mode='force',ocr_engine=induced)
    assert 'Missing resources' not in ' '.join(b['source'] for b in old['blocks'])
    real_analyze=jobs.analyze
    jobs.analyze=lambda *args,**kwargs:real_analyze(*args,**kwargs,ocr_engine=induced)
    model_calls=[]
    class Backend:
        options=dict(context=4096)
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def start(self,*args):pass
        def translate(self,prompt,**kwargs):
            model_calls.append(prompt);text=prompt.split('\n\n')[-1]
            if 'professional development' in text:target='지속적 전문성 개발(CPD).'
            elif 'resources' in text:target='부족한 자원과 연구 실패.'
            elif text=='CPD requirements.':
                assert 'CPD translates to 지속적 전문성 개발(CPD)' in prompt
                target='지속적 전문성 개발(CPD) 요구사항.'
            else:raise AssertionError(text)
            return dict(text=target,seconds=.01)
    jobs.LlamaBackend=Backend
    folder=jobs.create_job(source,project/'test_vault',title='누락 영역 실제 OCR 합성 검사',ocr_mode='force')
    jobs.run_job(folder);state=read_json(folder/'state.json')
    assert state['status']=='completed_with_warnings',state.get('error')
    output=Path(state['output']);quality=read_json(output/'Quality.json')
    assert quality['failed_paragraphs']==0 and quality['layout']['validation']['status']=='passed'
    assert quality['ocr_recovery']['added_lines']>=1
    assert any('Missing resources' in r['source'] and '부족한 자원' in r['target'] for r in quality['records'])
    assert (output/'Terminology.json').exists() and sha256(source)==sha256(output/'Original.pdf')
    with fitz.open(source) as original,fitz.open(output/'Bilingual.pdf') as dual:
        assert original[0].get_pixmap().samples==dual[0].get_pixmap(clip=original[0].rect).samples
    before=len(model_calls);ocr_before=len(calls)
    jobs.run_job(folder)
    assert len(model_calls)==before and len(calls)==ocr_before
    assert read_json(folder/'state.json')['analysis_cache_hit']
    # A pre-upgrade job must retain the old policy and fingerprint on resume.
    jobs.analyze=real_analyze;native=work/'Synthetic legacy study.pdf'
    with fitz.open() as doc:
        page=doc.new_page();page.insert_text((50,150),'Continuing professional development (CPD).');doc.save(native)
    legacy=jobs.create_job(native,project/'test_vault',ocr_mode='off')
    saved=read_json(legacy/'state.json');saved.pop('processing_revision');atomic_json(legacy/'state.json',saved)
    jobs.run_job(legacy);first=read_json(legacy/'state.json');before=len(model_calls)
    assert first['status']=='completed_with_warnings',first.get('error')
    assert 'recovery' not in read_json(legacy/'ir.json')['ocr']
    assert not (legacy/'Terminology.json').exists()
    jobs.run_job(legacy)
    assert len(model_calls)==before and read_json(legacy/'state.json')['fingerprint']==first['fingerprint']
    result=dict(status='passed',fixture='intentional full-page omission',crop_ocr='actual local CPU engine',
        translation='deterministic test double',recovered_lines=quality['ocr_recovery']['added_lines'],
        source_pixels_equal=True,layout='passed',replay_without_ocr_or_model_calls=True,
        legacy_policy_preserved=True,human_approved=False)
    atomic_json(work/'result.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
