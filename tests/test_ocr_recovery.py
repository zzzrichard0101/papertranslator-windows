"""Synthetic omissions and geometry: recovery may add evidence, never replace it."""
import io

import numpy as np
import pymupdf as fitz
import pytest
from PIL import Image, ImageDraw, ImageFont

from papertranslator.backend import Cancelled
from papertranslator.document import analyze
from papertranslator.ocr import LocalOCR, signature
from papertranslator.ocr_recovery import box, join_fragments, missing_regions, recover


def fixture():
    image=Image.new('RGB',(1200,1000),'white');draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('arial.ttf',32)
    draw.text((70,100),'The known sentence remains unchanged.',font=font,fill='black')
    draw.text((70,250),'Missing resources and study failures.',font=font,fill='black')
    return image,np.array(image)[:,:,::-1].copy()


def row(rect,text,confidence=.95,**meta):
    x0,y0,x1,y1=rect
    return [[[x0,y0],[x1,y0],[x1,y1],[x0,y1]],text,confidence,meta]


def test_only_uncovered_ink_retried_with_exact_global_coordinates():
    image,pixels=fixture();known=row((65,100,800,138),'Known source.')
    regions=missing_regions(pixels,[known]);assert regions
    calls=[]
    def recognize(crop):
        calls.append(crop.shape)
        return [row((28,28,124,48),'Missing resources',words=[dict(text='Missing',bbox=[28,28,94,48])])]
    recovered,info=recover(pixels,[known],recognize)
    assert recovered[0] is known and known[1]=='Known source.'
    assert info['attempted_regions']==len(calls)<=6 and info['added_lines']>0
    added=next(r for r in recovered if r is not known)
    assert box(added).y0>200
    assert added[3]['words'][0]['bbox'][1]==box(added).y0
    assert added[3]['recovery'] and (box(known)&box(added)).get_area()==0
    assert sum(h*w for h,w,_ in calls)<=pixels.shape[0]*pixels.shape[1]*.25


def test_real_local_ocr_recovers_intentionally_missing_line():
    _,pixels=fixture();engine=LocalOCR()
    known=row((65,100,800,138),'The known sentence remains unchanged.')
    recovered,info=recover(pixels,[known],engine)
    text=' '.join(r[1] for r in recovered)
    assert info['added_lines']>0 and 'Missing resources' in text and 'study failures' in text
    assert text.count('known sentence')==1


def test_blank_rules_and_solid_shapes_are_not_retry_candidates():
    pixels=np.full((1000,1200,3),255,dtype=np.uint8)
    pixels[100:105,20:1100]=0;pixels[300:700,300:800]=0
    assert missing_regions(pixels,[])==[]
    assert recover(pixels,[],lambda _:pytest.fail('No OCR call expected'))[1]['attempted_regions']==0


def test_recovery_respects_cancel_and_rejects_low_score_out_of_crop():
    _,pixels=fixture()
    with pytest.raises(Cancelled):recover(pixels,[],lambda _:[],lambda:True)
    recovered,_=recover(pixels,[],lambda crop:[row((-1,0,40,20),'Out of bounds'),row((1,1,40,20),'Bad reading',.2)])
    assert recovered==[]


def blocks(*items):
    return [dict(type=0,bbox=rect,lines=[dict(bbox=rect,spans=[dict(text=text,bbox=rect,size=10,font='OCR',flags=0,extraction='ocr')],**meta)]) for rect,text,meta in items]


def test_split_baseline_restored_without_merging_columns_labels_or_native_code():
    source=blocks(([20,80,150,92],'The use',{}),([155,80,167,92],'of',{}),([172,80,260,92],'art therapy.',{}))
    restored=join_fragments(source)
    assert len(restored)==1
    assert ''.join(s['text'] for s in restored[0]['lines'][0]['spans'])=='The use of art therapy.'
    assert restored[0]['lines'][0]['reconstructed_fragments']==3
    assert source[0]['lines'][0]['spans'][0]['text']=='The use'
    assert len(join_fragments(blocks(([20,80,100,92],'First column',{}),([300,80,400,92],'Second column',{}))))==2
    assert len(join_fragments(blocks(([20,80,100,92],'First label',{'diagram_label':True}),([104,80,200,92],'Second label',{'diagram_label':True}))))==2
    native=blocks(([20,80,100,92],'def run():',{}))[0]
    native['lines'][0]['spans'][0]['extraction']='native'
    native['lines'].append(dict(bbox=[20,94,150,106],spans=[dict(text='return 42',extraction='native')]))
    assert join_fragments([native])==[native]
    captioned=blocks(([20,40,160,52],'Table 4.1 Strategies',{}),
                     ([20,80,150,92],'Treatment intensity',{}),([155,80,300,92],'Clinical outcome',{}))
    assert join_fragments(captioned)==captioned


def test_new_analysis_policy_is_opt_in_and_retains_raw_spans(tmp_path):
    path=tmp_path/'scan.pdf'
    with fitz.open() as doc:doc.new_page(width=600,height=800);doc.save(path)
    def recognize(pixels):
        scale=pixels.shape[1]/600
        return [row([v*scale for v in rect],text) for rect,text in [([30,120,130,132],'The use'),([135,120,147,132],'of'),([152,120,240,132],'art therapy.')]]
    old=analyze(path,[0],ocr_mode='force',ocr_engine=recognize)
    new=analyze(path,[0],ocr_mode='force',ocr_engine=recognize,recovery=True)
    assert any(b['preserve'] and b['source']=='of' for b in old['blocks'])
    assert len(new['blocks'])==1 and new['blocks'][0]['source']=='The use of art therapy.'
    assert len(new['blocks'][0]['style_runs'])==5
    assert new['pages'][0]['ocr']['reconstructed_fragments']==2
    assert signature('auto')!=signature('auto',recovery=True)
