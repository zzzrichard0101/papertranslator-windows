"""Conservative text-PDF IR and legible reflow export. Original is never edited."""
import html
import json
from pathlib import Path
import re
import shutil

import pymupdf as fitz

from .storage import atomic_json, safe_title, sha256

PARSER = 'pymupdf-text-ir-1'


def segments(text, limit=1800):
    """Whitespace-only normalization; split at sentence/space, never discard text."""
    text = re.sub(r'\s+', ' ', text).strip()
    result = []
    while len(text) > limit:
        candidates = [m.end() for m in re.finditer(r'[.!?]\s+', text[:limit])]
        end = candidates[-1] if candidates and candidates[-1] > limit//3 else text.rfind(' ', 0, limit)
        if end < 1:
            end = limit
        result.append(text[:end].strip()); text = text[end:].strip()
    if text:
        result.append(text)
    return result


def analyze(source, selected, progress=lambda *_: None):
    doc = fitz.open(source); pages = []; blocks = []
    try:
        if doc.needs_pass:
            raise ValueError('암호화된 PDF입니다. 암호를 해제한 사본을 선택하세요.')
        for index in selected:
            page = doc[index]; progress(index+1)
            page_info = dict(source_page_index=index, mediabox=list(page.mediabox), cropbox=list(page.cropbox),
                             rotation=page.rotation, width=page.rect.width, height=page.rect.height,
                             warnings=[], block_ids=[])
            raw = page.get_text('dict', sort=True)
            visible = [b for b in raw['blocks'] if b['type'] == 0]
            if sum(len(s['text']) for b in visible for ln in b['lines'] for s in ln['spans']) < 20:
                page_info['warnings'].append('텍스트가 적거나 스캔된 쪽: OCR 미구현, 원문 페이지를 보존합니다.')
            for block in visible:
                lines = [''.join(s['text'] for s in line['spans']) for line in block['lines']]
                text = '\n'.join(lines).strip()
                if not text:
                    continue
                spans = [s for ln in block['lines'] for s in ln['spans']]
                preserve = bool(re.search(r'\b(def |class |import |return |for\s*\(|while\s*\(|SELECT |function\s)', text))
                preserve |= sum(c.isalpha() for c in text) < 4
                preserve |= bool(re.search(r'[∑∫√]|[μσ]\s*[=±]', text))
                kind = 'protected' if preserve else ('heading' if max(s['size'] for s in spans) >= 16 else 'paragraph')
                # Code/formula text retains line breaks and indentation verbatim.
                for part, segment in enumerate([text] if preserve else segments(text)):
                    bid = f'p{index+1}-b{block["number"]}-s{part}'
                    item = dict(id=bid, source_page_index=index, bbox=list(block['bbox']), kind=kind,
                                source=segment, style_runs=[dict(text=s['text'], font=s['font'], size=s['size'], flags=s['flags']) for s in spans],
                                reading_order=len(blocks), split_part=part, preserve=preserve)
                    blocks.append(item); page_info['block_ids'].append(bid)
            pages.append(page_info)
        return dict(parser=PARSER, pages=pages, blocks=blocks,
                    limitations=['읽기 순서는 PyMuPDF 기본 정렬입니다. 복잡한 2단·표·쪽을 넘는 문장은 검토가 필요합니다.',
                                 '이 버전은 한국어 가독성을 위한 재배치 출력입니다. 원문과 같은 조판은 아직 구현되지 않았습니다.',
                                 '스캔 OCR과 표 구조 인식은 아직 구현되지 않았습니다. 기계 의미 검수는 작업 옵션과 coverage를 확인하세요.'])
    finally:
        doc.close()


def flow_pdf(ir, records, path, title):
    """Embedded built-in Korean font; append pages instead of shrinking/clipping."""
    doc = fitz.open(); mapping = []
    font = fitz.Font('korea'); fontbuf = font.buffer
    page = None; y = 0

    def new_page(source_index, continued=False):
        nonlocal page, y
        page = doc.new_page(width=595.28, height=841.89)
        page.insert_font(fontname='KO', fontbuffer=fontbuf)
        page.insert_text((42, 35), f'원본 {source_index+1}쪽 / 한국어 재배치' + (' · 계속' if continued else ''),
                         fontname='KO', fontsize=9, color=(.25,.32,.40))
        page.draw_line((42, 46), (553,46), color=(.75,.8,.85))
        page.insert_text((42, 816), '초기 개발판 · 기계 번역 · 원문과 함께 확인하세요',
                         fontname='KO', fontsize=8, color=(.5,.35,.14))
        mapping.append(source_index); y = 67

    def lines_for(text, size):
        # Keep numbers, identifiers, URLs and short inline code on one line.
        # CJK prose can wrap between characters; long tokens still wrap safely.
        result=[]; line=''; width=0
        tokens=re.findall(r'https?://[^\s]+|`[^`]+`|O\([^\n)]*\)|[-−+]?\d+(?:[.,]\d+)*(?:%|[eE][-+]?\d+)?|[A-Za-z_][A-Za-z_0-9]*|\n|.',text)
        for token in tokens:
            w=font.text_length(token,fontsize=size)
            if token=='\n':
                result.append(line); line=''; width=0
                continue
            if width+w>505 and line:
                result.append(line);line='';width=0
            if w>505:
                for ch in token:
                    ch_width=font.text_length(ch,fontsize=size)
                    if width+ch_width>505:
                        result.append(line);line='';width=0
                    line+=ch;width+=ch_width
            else:
                line+=token;width+=w
        if line:
            result.append(line)
        return result

    for p in ir['pages']:
        index = p['source_page_index']; new_page(index)
        for warning in p['warnings']:
            for line in lines_for(warning, 10):
                page.insert_text((42,y), line, fontname='KO', fontsize=10, color=(.7,.34,.1)); y+=16
        if p['warnings']:
            y += 10
        for bid in p['block_ids']:
            record = records[bid]; failed=record['status']=='failed'
            text = record['target']
            label = '[번역 실패 · 원문 보존] ' if failed else ('[원문 보존] ' if record['status']=='preserved' else '')
            text = label + text
            size=14 if record['kind']=='heading' else 11
            for line in lines_for(text, size):
                if y+size > 789:
                    new_page(index, True)
                page.insert_text((42,y), line, fontname='KO', fontsize=size,
                                 color=(.65,.27,.08) if failed else (.08,.13,.20))
                y += size*1.55
            y += 10
    if not len(doc):
        raise RuntimeError('출력할 쪽이 없습니다.')
    doc.subset_fonts(); doc.save(path, garbage=4, deflate=True)
    doc.close()
    return mapping


def export(ir, records, original, folder, title, job, cancel=lambda: False):
    folder = Path(folder); folder.mkdir(parents=True)
    try:
        shutil.copyfile(original, folder/'Original.pdf')
        if sha256(folder/'Original.pdf') != job['source_sha256']:
            raise RuntimeError('원본 복사본 검증 실패')
        mapping = flow_pdf(ir, records, folder/'Korean.pdf', title)
        with fitz.open(original) as src, fitz.open(folder/'Korean.pdf') as ko, fitz.open() as dual:
            for n, source_index in enumerate(mapping):
                if cancel():
                    from .backend import Cancelled
                    raise Cancelled()
                a=src[source_index]; b=ko[n]
                height=max(a.rect.height,b.rect.height)
                out=dual.new_page(width=a.rect.width+b.rect.width,height=height)
                # Preserve original vectors and images on the left, with no edits.
                if a.get_contents():
                    out.show_pdf_page(fitz.Rect(0,0,a.rect.width,a.rect.height), src, source_index)
                out.show_pdf_page(fitz.Rect(a.rect.width,0,a.rect.width+b.rect.width,b.rect.height),ko,n)
            dual.save(folder/'Bilingual.pdf', garbage=4, deflate=True)
        failed=[r for r in records.values() if r['status']=='failed']
        warnings=[r for r in records.values() if r.get('issues') or r.get('review',{}).get('verdict') in {'review','fail','uncertain'}]
        translated=[r for r in records.values() if r['status']=='translated']
        reviewed=[r for r in translated if r.get('review',{}).get('evidence_validated')]
        coverage=len(reviewed)/len(translated) if translated else 0
        review_status='not_performed' if not reviewed else ('all_machine_reviewed' if len(reviewed)==len(translated) else 'partially_machine_reviewed')
        quality=dict(schema=2, needs_review=True, file_generation='completed', semantic_review=review_status,
                     semantic_review_coverage=coverage, reviewed_paragraphs=len(reviewed),human_approved=False,
                     reviewer=job.get('reviewer'),
                     model=job['model'], source_sha256=job['source_sha256'],
                     selected_source_pages=[p['source_page_index']+1 for p in ir['pages']],
                     korean_output_source_indices=mapping, output_layout='reflow',
                     failed_paragraphs=len(failed), warnings_paragraphs=len(warnings),
                     translated_paragraphs=sum(r['status']=='translated' for r in records.values()),
                     preserved_paragraphs=sum(r['status']=='preserved' for r in records.values()),
                     limitations=ir['limitations'], pages=ir['pages'], records=list(records.values()))
        atomic_json(folder/'Quality.json', quality)
        sections=[]
        for r in records.values():
            issues=' / '.join(i['message'] for i in r.get('issues', []))
            if r.get('review'):
                issues+=' / 검수 '+r['review'].get('verdict','uncertain')+': '+r['review'].get('explanation_ko','')
            sections.append(f'<article><h2>원본 {r["source_page_index"]+1}쪽 · {html.escape(r["id"])}</h2>'
                            f'<p class="warning">{html.escape(issues)}</p><div class="pair"><pre>{html.escape(r["source"])}</pre>'
                            f'<pre lang="ko">{html.escape(r["target"])}</pre></div></article>')
        review='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"><title>번역 검토</title><style>body{font:16px/1.7 'Malgun Gothic',sans-serif;margin:40px;background:#f4f6f8;color:#152233}header,article{max-width:1280px;margin:20px auto;padding:24px;background:white;border-radius:12px}h2{font-size:16px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:30px}pre{white-space:pre-wrap;font:inherit;overflow-wrap:anywhere}.warning{color:#9a4a10}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style><header><h1>영한 문단 검토</h1><p>초기 개발판 · 자동 검사는 의미 정확도를 보증하지 않습니다. 검수 범위와 판정은 Quality.json에 기록됩니다.</p></header>'''+''.join(sections)+'</html>'
        (folder/'Review.html').write_text(review,encoding='utf-8')
        fields=dict(제목=title, 원본파일=job['source_name'], 번역일=job['created_at'], 원본SHA256=job['source_sha256'],
                    전체쪽=job['total_pages'], 선택쪽=quality['selected_source_pages'], 모델=job['model'],
                    분야=job['profile'], 조판='한국어 재배치', 의미검수=review_status, 검수coverage=coverage,
                    사용자승인=False, 번역실패문단수=len(failed), tags=['논문','번역'])
        name=safe_title(title)
        note='---\n'+'\n'.join(k+': '+json.dumps(v,ensure_ascii=False) for k,v in fields.items())+'\n---\n\n# '+name+'\n\n'
        note+=f'> 초기 개발판: 기계 의미 검수 {len(reviewed)}/{len(translated)}문단 · 인간 승인 없음. 스캔 OCR 및 원문 조판 재현은 미지원입니다.\n\n'
        note+='\n'.join(f'- [[{f}|{label}]]' for f,label in [('Original.pdf','영어 원문'),('Korean.pdf','한국어'),('Bilingual.pdf','영한 대조'),('Review.html','문단 검토')])
        note+='\n\n![[Korean.pdf]]\n\n## 메모\n\n'
        (folder/(name+'.md')).write_text(note,encoding='utf-8')
        with fitz.open(folder/'Korean.pdf') as out:
            if len(out)!=len(mapping) or any(not p.get_text().strip() for p in out):
                raise RuntimeError('한국어 PDF 재열기/텍스트 검증 실패')
        with fitz.open(folder/'Bilingual.pdf') as out:
            if len(out)!=len(mapping):
                raise RuntimeError('대조 PDF 쪽수 검증 실패')
        return quality
    except Exception:
        # Caller owns staging folder; never touch committed user output.
        raise
