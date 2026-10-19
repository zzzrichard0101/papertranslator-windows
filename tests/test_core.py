import json
from pathlib import Path
import subprocess
import sys

import pymupdf as fitz
import pytest

from papertranslator.document import analyze, export, segments
from papertranslator.quality import check, protect, restore
from papertranslator.storage import atomic_json, fingerprint, pages, read_json, safe_title, sha256


def test_sparse_physical_page_selection():
    assert pages(' 1-2, 51,61- ',61)==[0,1,50,60]
    for bad in ['0','62','9-2','1,,3','-3','abc','1-100000000']:
        with pytest.raises(ValueError): pages(bad,61)


def test_source_segments_do_not_drop_words():
    original=' '.join(['A decimal 1.25 is not a list. '*90,'Next sentence.'])
    assert ' '.join(segments(original))==' '.join(original.split())
    assert all(len(s)<=1800 for s in segments(original))


def test_protected_delimiters_not_invented_or_lost():
    s='Lee et al., 2020 used `x < y` in O(n log n). See https://example.org/test [1, 3].'
    protected,mapping=protect(s)
    assert restore(protected,mapping)==s
    for token in mapping:
        with pytest.raises(ValueError): restore(protected.replace(token,''),mapping)
    with pytest.raises(ValueError): restore(protected+' {v1}',mapping)


def test_known_critical_mistranslations_are_flagged():
    assert any(x['code']=='p_vs_alpha' for x in check('p = 0.032','유의수준 p = 0.032'))
    assert any(x['code']=='added_random' for x in check('Participants were assigned.','참가자를 무작위로 배정하였다.'))
    assert any(x['code']=='numbers' for x in check('n = 24, -1.25','n = 25, 1.25'))
    assert any(x['code']=='kinship' for x in check('Her two aunts attended.','삼촌 두 명이 참석했다.'))
    assert not check('The value was 1.25 (n = 24).','값은 1.25였다(n = 24).')


def test_written_quantities_do_not_hide_original_decimal_or_sign_loss():
    assert not check('For the next five months.','다음 5개월 동안.')
    assert not check('On October 30, the value was -1.25.','10월 30일 값은 -1.25였다.')
    assert any(x['code']=='numbers' for x in check('One year later, the value was -1.25.','1년 후 값은 1.25였다.'))
    assert any(x['code']=='numbers' for x in check('Twenty years later.','30년 후.'))


def test_atomic_state_fingerprint_and_windows_titles(tmp_path):
    target=tmp_path/'한글 작업.json'; atomic_json(target,{'쪽':[50,60]}); assert read_json(target)=={'쪽':[50,60]}
    assert not list(tmp_path.glob('*.tmp'))
    assert safe_title('CON')=='_CON'; assert safe_title('a<>:/b. ')=='a____b'
    assert fingerprint({'model':'a','source':'x'})!=fingerprint({'model':'b','source':'x'})


def test_atomic_state_retries_windows_sharing_violation(tmp_path,monkeypatch):
    import papertranslator.storage as storage
    original=storage.os.replace; attempts=[]
    def locked(a,b):
        attempts.append(1)
        if len(attempts)<3: raise PermissionError('simulated Windows reader lock')
        return original(a,b)
    monkeypatch.setattr(storage.os,'replace',locked)
    atomic_json(tmp_path/'state.json',{'committed':True})
    assert len(attempts)==3 and read_json(tmp_path/'state.json')['committed']


def test_code_preserves_newlines_and_indent(tmp_path):
    source=tmp_path/'code.pdf'
    with fitz.open() as doc:
        p=doc.new_page(); p.insert_text((50,80),'def update(counter):\n    return counter + 1',fontname='cour')
        doc.save(source)
    block=analyze(source,[0])['blocks'][0]
    assert block['preserve'] and '\n    return' in block['source']


def test_ir_and_export_preserve_selected_originals(tmp_path):
    source=tmp_path/'source.pdf'
    with fitz.open() as doc:
        for index in range(61):
            p=doc.new_page(); p.insert_text((50,80),f'Physical page {index+1}. The value was 1.25 (n = 24).')
        doc.save(source)
    original_hash=sha256(source)
    ir=analyze(source,pages('51,61',61))
    assert [p['source_page_index'] for p in ir['pages']]==[50,60]
    records={b['id']:{**b,'target':'측정값은 1.25였다(n = 24).','status':'translated','issues':[]} for b in ir['blocks']}
    job=dict(source_sha256=original_hash,source_name=source.name,created_at='2026-10-18',total_pages=61,model={'id':'synthetic'},profile='general')
    quality=export(ir,records,source,tmp_path/'export','합성 시험',job)
    assert sha256(source)==sha256(tmp_path/'export'/'Original.pdf')==original_hash
    assert quality['semantic_review_coverage']==0 and quality['needs_review']
    assert quality['selected_source_pages']==[51,61]
    with fitz.open(tmp_path/'export'/'Korean.pdf') as doc:
        assert '측정값' in doc[0].get_text()
        assert len(doc)==2
        assert doc[0].get_fonts()
    from pypdf import PdfReader
    assert len(PdfReader(tmp_path/'export'/'Bilingual.pdf').pages)==2


@pytest.mark.parametrize('rotation',[0,90,180,270])
@pytest.mark.parametrize('cropped',[False,True])
def test_bilingual_original_face_keeps_rotation_and_crop(tmp_path,rotation,cropped):
    source=tmp_path/'rotated.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=400)
        p.draw_rect(fitz.Rect(50,45,250,100),fill=(1,0,0),color=None)
        p.draw_rect(fitz.Rect(430,250,540,330),fill=(0,0,1),color=None)
        p.insert_text((70,150),'Asymmetric original face.',fontsize=18)
        if cropped:p.set_cropbox(fitz.Rect(25,20,570,370))
        p.set_rotation(rotation);doc.save(source)
    before=sha256(source);ir=analyze(source,[0],ocr_mode='off')
    records={b['id']:{**b,'target':'원본 표시 검사. '+'가나다라 '*1500,'status':'translated','issues':[]} for b in ir['blocks']}
    job=dict(source_sha256=before,source_name=source.name,created_at='synthetic',total_pages=1,model={'id':'synthetic'},profile='general')
    out=tmp_path/'output';export(ir,records,source,out,'회전 표시 검사',job)
    with fitz.open(source) as original,fitz.open(out/'Bilingual.pdf') as dual:
        rect=original[0].rect
        a=original[0].get_pixmap();assert len(dual)>1
        for page in dual:
            b=page.get_pixmap(clip=rect)
            assert (a.width,a.height)==(b.width,b.height)
            assert a.samples==b.samples
    assert sha256(source)==sha256(out/'Original.pdf')==before


def test_pdf_wrap_does_not_split_decimals_or_code(tmp_path):
    from papertranslator.document import flow_pdf
    text=('측정값은 1.25이고 p = 0.032이며 `pending_count`를 유지한다. '*40)
    ir=dict(pages=[dict(source_page_index=0,warnings=[],block_ids=['p1'])])
    records={'p1':dict(status='translated',kind='paragraph',target=text)}
    path=tmp_path/'wrap.pdf';flow_pdf(ir,records,path,'숫자 줄바꿈 시험')
    with fitz.open(path) as doc:extracted=''.join(p.get_text() for p in doc)
    assert extracted.count('1.25')==40 and extracted.count('0.032')==40 and extracted.count('pending_count')==40


def test_network_guard_rejects_external_without_connecting():
    code='from papertranslator.offline import install_loopback_guard; import socket; install_loopback_guard(); socket.getaddrinfo("example.org", 443)'
    p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert p.returncode!=0 and 'External DNS is disabled' in p.stderr
