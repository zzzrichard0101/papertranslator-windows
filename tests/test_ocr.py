"""Synthetic raster-only fixtures; never use private user documents."""
import io

import pymupdf as fitz
import pytest
from PIL import Image, ImageDraw, ImageFont

from papertranslator.backend import Cancelled
from papertranslator.document import analyze
from papertranslator.ocr import LocalOCR, MAX_PIXELS, MAX_SIDE, page_blocks, signature


def image_bytes(text='The study included 24 participants.'):
    image=Image.new('RGB',(1500,260),'white')
    font=ImageFont.truetype('arial.ttf',42)
    ImageDraw.Draw(image).text((35,75),text,font=font,fill='black')
    stream=io.BytesIO();image.save(stream,format='PNG');return stream.getvalue()


def scanned_page(doc, rotation=0):
    page=doc.new_page(width=600,height=800)
    page.insert_image(fitz.Rect(40,100,560,190),stream=image_bytes())
    page.set_rotation(rotation)
    return page


@pytest.fixture(scope='module')
def engine():
    return LocalOCR()


def test_real_cpu_ocr_scanned_and_mixed_pages(tmp_path,engine):
    path=tmp_path/'synthetic.pdf'
    with fitz.open() as doc:
        scanned_page(doc)
        page=scanned_page(doc)
        page.insert_text((40,60),'Native text remains available.',fontsize=12)
        doc.save(path)
    ir=analyze(path,[0,1],ocr_engine=engine)
    text=' '.join(b['source'] for b in ir['blocks'])
    assert text.count('24 participants')==2
    assert text.count('Native text remains available.')==1
    assert all(p['ocr']['lines']>=1 for p in ir['pages'])
    assert any(b['extraction'] in {'ocr','mixed'} for b in ir['blocks'])
    assert all(b['ocr_confidence_min']>.75 for b in ir['blocks'] if b['ocr_confidence_min'] is not None)


def test_modes_dedup_confidence_and_bounded_render():
    calls=[]
    with fitz.open() as doc:
        page=doc.new_page(width=600,height=800)
        page.insert_text((40,60),'Native words are preserved.',fontsize=12)
        def fake(pixels):
            calls.append(pixels.shape)
            scale=pixels.shape[1]/600
            def row(box,text,score):
                x0,y0,x1,y1=[v*scale for v in box]
                return [[[x0,y0],[x1,y0],[x1,y1],[x0,y1]],text,score]
            return [row((40,47,190,64),'Native words are preserved.',.98),
                    row((40,150,260,170),'Uncertain image text.',.4)]
        blocks,info,_=page_blocks(page,'off',fake)
        assert not calls and not info['attempted']
        # Native-only pages do not need OCR in automatic mode.
        page_blocks(page,'auto',fake);assert not calls
        page.insert_image(fitz.Rect(40,130,560,220),stream=image_bytes())
        blocks,info,warnings=page_blocks(page,'auto',fake)
        assert info['duplicate_lines']==1 and info['lines']==1
        assert info['low_confidence_lines']==1 and warnings
        assert sum('Native words' in s['text'] for b in blocks for l in b['lines'] for s in l['spans'])==1
        blocks,info,_=page_blocks(page,'force',fake)
        assert info['lines']==2 and info['duplicate_lines']==0
        assert all(s['extraction']=='ocr' for b in blocks for l in b['lines'] for s in l['spans'])
        page=doc.new_page(width=14400,height=14400)
        page_blocks(page,'force',lambda pixels: calls.append(pixels.shape) or [])
        h,w,_=calls[-1];assert h*w<=MAX_PIXELS and max(h,w)<=MAX_SIDE


def test_blank_cancel_and_invalid_mode(engine):
    with fitz.open() as doc:
        page=doc.new_page()
        blocks,info,warnings=page_blocks(page,'auto',engine)
        assert not blocks and info['attempted'] and warnings
        with pytest.raises(Cancelled):page_blocks(page,cancel=lambda:True)
    with pytest.raises(ValueError):signature('unknown')
    assert signature('off')!=signature('auto')


def test_cancel_after_recognition_and_partial_native_overlap():
    cancelled=[]
    with fitz.open() as doc:
        page=doc.new_page(width=600,height=800)
        page.insert_text((40,60),'Native',fontsize=12)
        def recognize(pixels):
            h,w,_=pixels.shape;scale=w/600
            return [[[[40*scale,47*scale],[480*scale,47*scale],[480*scale,64*scale],[40*scale,64*scale]],
                     'Native with additional image content.',.9]]
        blocks,info,_=page_blocks(page,'auto',recognize)
        assert info['lines']==1 and info['duplicate_lines']==0
        assert any('additional image content' in s['text'] for b in blocks for l in b['lines'] for s in l['spans'])
        def interrupted(pixels):
            cancelled.append(True);return []
        with pytest.raises(Cancelled):page_blocks(page,'force',interrupted,lambda:bool(cancelled))


def test_low_confidence_provenance_in_ir(tmp_path):
    path=tmp_path/'confidence.pdf'
    with fitz.open() as doc:
        doc.new_page();doc.save(path)
    def fake(pixels):
        return [[[[100,100],[500,100],[500,150],[100,150]],'Uncertain scanned sentence.',.3]]
    ir=analyze(path,[0],ocr_engine=fake)
    block=ir['blocks'][0]
    assert block['extraction']=='ocr' and block['ocr_confidence_min']==.3
    assert block['source_fragments'][0]['ocr_confidence_min']==.3
    assert ir['pages'][0]['ocr']['low_confidence_lines']==1
    assert any('낮은 신뢰도' in w for w in ir['pages'][0]['warnings'])


def test_rotation_and_crop_geometry():
    with fitz.open() as doc:
        page=scanned_page(doc,90)
        page.set_cropbox(fitz.Rect(20,20,580,780))
        def fake(pixels):
            h,w,_=pixels.shape
            return [[[[w*.1,h*.2],[w*.7,h*.2],[w*.7,h*.3],[w*.1,h*.3]],'Rotation test',.9]]
        blocks,_,_=page_blocks(page,'force',fake)
        rect=fitz.Rect(blocks[0]['bbox'])
        assert rect.x0==pytest.approx(page.rect.width*.1,abs=1)
        assert rect.y0==pytest.approx(page.rect.height*.2,abs=1)
        assert page.rect.contains(rect)


def test_ocr_cross_page_continuation(tmp_path):
    path=tmp_path/'boundary.pdf'
    with fitz.open() as doc:
        doc.new_page(width=600,height=800);doc.new_page(width=600,height=800);doc.save(path)
    calls=[]
    def fake(pixels):
        scale=pixels.shape[1]/600;first=not calls;calls.append(True)
        x0,y0,x1,y1=[v*scale for v in ((40,710,550,730) if first else (40,60,550,80))]
        return [[[[x0,y0],[x1,y0],[x1,y1],[x0,y1]],'The result continues with' if first else 'the next page of the study.',.91]]
    ir=analyze(path,[0,1],ocr_mode='force',ocr_engine=fake)
    assert len(ir['blocks'])==1
    assert ir['blocks'][0]['source']=='The result continues with the next page of the study.'
    assert ir['blocks'][0]['source_page_indices']==[0,1]


def test_real_rotated_scan(engine):
    with fitz.open() as doc:
        page=scanned_page(doc,90)
        blocks,info,_=page_blocks(page,'auto',engine)
        text=' '.join(s['text'] for b in blocks for l in b['lines'] for s in l['spans'])
        assert '24' in text and 'participants' in text
        assert info['lines']>0
