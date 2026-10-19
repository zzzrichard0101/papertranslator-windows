"""Conservative text-PDF IR and legible reflow export. Original is never edited."""
import html
import json
from pathlib import Path
import re
import shutil
from collections import Counter

import pymupdf as fitz

from .storage import atomic_json, safe_title, sha256, read_json
import hashlib
from .ocr import LocalOCR, page_blocks, signature

PARSER = 'pymupdf-logical-paragraph-ir-9-header-edge-fragments'
JOIN_WORDS=set('economic psychological paradigm practitioner practitioners particular individual individuals '
              'support supporting delivered delivery theories theory research professional professionals '
              'clinical objective subjective experience experiences observation observations systematic '
              'practice knowledge acquisition implementation effectiveness efficacy reliability validity '
              'intervention interventions condition conditions information interpretation development '
              'demonstrate demonstration representative representation organisation organization '
              'education educational population populations experimental experiment participation '
              'significant significance qualitative quantitative literature method methodology '
              'psychotherapy discipline disciplines construction constructivism epistemology '
              'contribution treatment treatments continuing inevitable technological technology '
              'university different difference differences provision authority contradiction '
              'contradictions situate situated depression uncertainty'.split())


def join_lines(left,right):
    if left.endswith('-') and right[:1].islower():
        a=re.search(r'([A-Za-z]+)-$',left);b=re.match(r'([a-z]+)',right)
        remove=bool(a and b and (a[1]+b[1]).lower() in JOIN_WORDS)
        return (left[:-1] if remove else left)+right,remove
    return left+(' ' if left else '')+right,False


def protected_kind(text, spans):
    """Protect actual code/formula regions, never ordinary uses of 'return'."""
    mono=sum(len(s['text'].strip()) for s in spans if s.get('flags',0)&8 or
             re.search(r'courier|consolas|mono',s.get('font',''),re.I))
    length=sum(len(s['text'].strip()) for s in spans)
    syntax=re.search(r'^\s*(?:def\s+\w+\s*\(|class\s+\w+[(:]|from\s+[\w.]+\s+import\s|'
                     r'import\s+[\w.]+\s*$|function\s+\w+\s*\(|SELECT\s+.+\s+FROM\s)',text,re.M)
    if syntax or (mono>=max(4,length*.7) and re.search(r'[=():{};]|^\s*return\b',text,re.M)):
        return 'code'
    if sum(c.isalpha() for c in text)<4:return 'symbol'
    if len(text)<220 and re.search(r'[∑∫√]|[μσ]\s*[=±]',text) and len(re.findall(r'\b[A-Za-z]{3,}\b',text))<4:
        return 'formula'
    return None


def paragraph_units(page, text_blocks=None):
    """Use blank lines, typography and bullets, including across PDF text blocks."""
    units=[];current=[];previous=None
    def flush():
        nonlocal current
        if not current:return
        text=''
        raw=' '.join(line['text'].strip() for line in current)
        for line in current:
            part=line['text'].strip()
            text,_=join_lines(text,part)
        bbox=[min(l['bbox'][0] for l in current),min(l['bbox'][1] for l in current),
              max(l['bbox'][2] for l in current),max(l['bbox'][3] for l in current)]
        spans=[s for l in current for s in l['spans']]
        units.append(dict(source=text,raw_source=raw,bbox=bbox,kind=current[0]['kind'],preserve=False,style_runs=spans))
        current=[]
    for block in (page.get_text('dict',sort=True)['blocks'] if text_blocks is None else text_blocks):
        if block['type']!=0:continue
        spans=[s for line in block['lines'] for s in line['spans']]
        text='\n'.join(''.join(s['text'] for s in line['spans']) for line in block['lines']).strip()
        if not text:continue
        region=protected_kind(text,spans)
        marker=bool(re.fullmatch(r'\d{1,4}',text)) and (block['bbox'][1]>page.rect.height*.9 or block['bbox'][3]<page.rect.height*.1)
        if region or marker:
            flush();previous=None
            units.append(dict(source=text,bbox=list(block['bbox']),kind='page_marker' if marker else 'protected',
                              preserve=True,preserve_reason='page_number' if marker else region,style_runs=spans))
            continue
        for line in block['lines']:
            spans=line['spans'];text=''.join(s['text'] for s in spans)
            if not text.strip():flush();previous=None;continue
            visible=[s for s in spans if s['text'].strip()]
            count=sum(len(s['text']) for s in visible)
            bold=sum(len(s['text']) for s in visible if s.get('flags',0)&16)
            heading=len(text)<180 and (bold>count*.75 or max(s['size'] for s in visible)>=16)
            kind='heading' if heading else 'paragraph'
            gap=previous and line['bbox'][1]-previous['bbox'][3]>max(s['size'] for s in visible)*.65
            bullet=bool(re.match(r'^\s*[•●▪◦]',text))
            numbered=bool(re.match(r'^\s*\d+[.)]\s',text))
            if numbered and current and not gap:
                # A wrapped age/value such as "age / 31. She ..." is not a list.
                numbered=bool(re.search(r'[.!?][\"\x27’”]*\s*$',current[-1]['text'])) or bool(re.match(r'^\s*\d+[.)]\s',current[0]['text']))
            bullet|=numbered
            if current and (kind!=current[0]['kind'] or heading or bullet or gap):flush()
            current.append(dict(text=text,bbox=line['bbox'],spans=spans,kind=kind));previous=line
    flush()
    return units


def continuation(left,right,left_page,right_page):
    if left['preserve'] or right['preserve'] or left['kind']!='paragraph' or right['kind']!='paragraph':return False
    if right_page['source_page_index']!=left_page['source_page_index']+1:return False
    a=left['source'].rstrip();b=right['source'].lstrip()
    if not a or not b or re.search(r'[.!?。！？…][\"\x27’”\])]*$',a):return False
    if re.match(r'[•●▪◦]|\d+[.)]\s',b):return False
    # An unpunctuated stage note after a finished sentence is not a continuation.
    prefix=re.sub(r'\s*[([][^()\[\]]{0,180}[)\]]\s*$','',a)
    if prefix!=a and re.search(r'[.!?][\"\x27’”\])]*$',prefix):return False
    connector=re.search(r'\b(?:the|a|an|of|to|for|and|or|with|in|by|from|as|through|not|couldn.t|must|can)\s*$',a,re.I)
    if not (b[:1].islower() or b[:1].isdigit() or connector or a.endswith(('-',',',':'))):return False
    last=left['fragments'][-1]
    if last['source_page_index']!=left_page['source_page_index']:return False
    # Require a real physical boundary, not a short caption ending mid-page.
    if last['bbox'][3]<left_page['height']*.68 or right['bbox'][1]>right_page['height']*.3:return False
    return True


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


def analyze(source, selected, progress=lambda *_: None, ocr_mode='auto', cancel=lambda: False, ocr_engine=None,cache_dir=None):
    doc = fitz.open(source); pages = []; blocks = []; extracted=[]
    engine=ocr_engine
    def recognize(pixels):
        nonlocal engine
        cached=None
        if cache_dir:
            cache=Path(cache_dir);cache.mkdir(parents=True,exist_ok=True)
            key=hashlib.sha256(json.dumps(signature(ocr_mode),sort_keys=True).encode()+pixels.tobytes()).hexdigest()
            cached=cache/(key+'.json');saved=read_json(cached,{})
            if saved.get('key')==key and isinstance(saved.get('rows'),list):return saved['rows']
        if engine is None:engine=LocalOCR()
        rows=engine(pixels)
        if cached:atomic_json(cached,dict(key=key,rows=rows))
        return rows
    try:
        if doc.needs_pass:
            raise ValueError('암호화된 PDF입니다. 암호를 해제한 사본을 선택하세요.')
        for index in selected:
            page = doc[index]; progress(index+1)
            page_info = dict(source_page_index=index, mediabox=list(page.mediabox), cropbox=list(page.cropbox),
                             rotation=page.rotation, width=page.rect.width, height=page.rect.height,
                             warnings=[], block_ids=[], continuations=[])
            text_blocks, extraction, warnings=page_blocks(page,ocr_mode,recognize,cancel)
            page_info['ocr']=extraction;page_info['warnings'].extend(warnings)
            if extraction['lines']:
                # An image can sit between lines in one native PDF text block.
                # Interleave body lines geometrically while retaining code blocks.
                ordered=[]
                for block in text_blocks:
                    spans=[s for line in block['lines'] for s in line['spans']]
                    text='\n'.join(''.join(s['text'] for s in line['spans']) for line in block['lines'])
                    if protected_kind(text,spans):ordered.append(block)
                    else:ordered.extend(dict(type=0,bbox=line['bbox'],lines=[line],
                        reading_top=line.get('reading_top',line['bbox'][1])) for line in block['lines'])
                text_blocks=sorted(ordered,key=lambda b:(b.get('reading_top',b['bbox'][1]),b['bbox'][0]))
            units=paragraph_units(page,text_blocks)
            for unit in units:
                unit['source_page_index']=index
                if unit['bbox'][1]>page.rect.height*.94 and len(unit['source'])<=20:
                    unit.update(kind='page_marker',preserve=True,preserve_reason='page_margin')
                # Keep OCR specks in the IR, but not between the last/first body
                # lines. Only very short non-word fragments at the outer margin.
                box=unit['bbox']
                outer=(box[3]<page.rect.height*.07 or box[1]>page.rect.height*.91 or
                       box[2]<page.rect.width*.04 or box[0]>page.rect.width*.96)
                if (outer and unit.get('preserve_reason')=='symbol' and len(unit['source'])<=20 and
                    all(s.get('extraction')=='ocr' for s in unit['style_runs'])):
                    unit.update(kind='page_marker',preserve=True,preserve_reason='ocr_margin_fragment')
                # A horizontal scan rule can be read as a few Latin letters.
                # Its box is far wider than those letters at the measured height.
                if (outer and unit['kind']!='page_marker' and len(unit['source'])<40 and
                    (box[2]-box[0])/max(1,box[3]-box[1])>max(6,len(unit['source'].replace(' ',''))*1.2) and
                    all(s.get('extraction')=='ocr' for s in unit['style_runs'])):
                    unit.update(kind='page_marker',preserve=True,preserve_reason='stretched_ocr_margin')
                scores=[s['ocr_confidence'] for s in unit['style_runs'] if isinstance(s.get('ocr_confidence'),(int,float))]
                methods=set(s.get('extraction','native') for s in unit['style_runs'])
                method='mixed' if len(methods)>1 else next(iter(methods))
                unit['fragments']=[dict(source_page_index=index,bbox=unit['bbox'],source=unit['source'],
                                        raw_source=unit.get('raw_source',unit['source']),extraction=method,ocr_confidence_min=min(scores) if scores else None)]
            extracted.append(units)
            pages.append(page_info)
        # Repeated running titles are metadata; do not let them sever sentences
        # across a physical page boundary. Chapter titles below the margin remain.
        def header_key(unit,page):
            box=unit['bbox'];text=unit['source']
            if box[1]<page['height']*.075 and box[3]<page['height']*.12 and box[3]-box[1]<=20 and 4<len(text)<160:
                return re.sub(r'\W+','',text).casefold()
            return None
        keys=Counter(key for units,page in zip(extracted,pages) for key in
                     set(filter(None,(header_key(u,page) for u in units))))
        metadata={'page_marker','running_header'}
        for page_number,(units,page_info) in enumerate(zip(extracted,pages)):
            page_text=' '.join(unit['source'] for unit in units)
            labels=[unit['source'].strip().casefold() for unit in units if unit['bbox'][1]<page_info['height']*.3]
            reference_section=None
            if any(re.fullmatch(r'(?:references|bibliography)(?:\s+\d{1,4})?',label) for label in labels) and len(re.findall(r'\([12]\d{3}[a-z]?\)',page_text))>=3:
                reference_section='references'
            elif any(re.match(r'^index\b',label) and len(label)<45 for label in labels) and len(re.findall(r'\b\d{1,3}\b',page_text))>=10:
                reference_section='index'
            elif page_number+1<len(pages) and pages[page_number+1]['source_page_index']==page_info['source_page_index']+1:
                # A scan can miss the large title on the first index page. Require
                # both strong index syntax and an immediately following title.
                next_labels=[u['source'].strip().casefold() for u in extracted[page_number+1]
                             if u['bbox'][1]<pages[page_number+1]['height']*.3]
                if (any(re.match(r'^index\b',label) and len(label)<45 for label in next_labels) and
                    len(re.findall(r'\b\d{1,3}\b',page_text))>=40 and
                    len(re.findall(r'\bsee also\b',page_text,re.I))>=2):
                    reference_section='index'
                    page_info['warnings'].append('색인 형식과 다음 쪽 제목을 확인해 항목을 원문으로 보존했습니다. 복잡한 열은 원본을 보세요.')
            for unit in units:
                key=header_key(unit,page_info)
                if key and keys[key]>=3:
                    unit.update(kind='running_header',preserve=True,preserve_reason='repeated_running_title')
                label=unit['source'].strip().casefold()
                if reference_section and label in {'references','bibliography','index'} and unit['kind']!='running_header':
                    unit.update(kind='heading',preserve=False)
                elif reference_section and unit['kind'] not in metadata:
                    unit.update(kind='reference',preserve=True,preserve_reason=reference_section)
            header_tops=[u['bbox'][1] for u in units if u['kind']=='running_header']
            if header_tops:
                header_top=min(header_tops)
                for unit in units:
                    if (unit['bbox'][3]<header_top and len(unit['source'])<40 and
                        unit['bbox'][3]-unit['bbox'][1]<20 and
                        all(s.get('extraction')=='ocr' for s in unit['style_runs'])):
                        unit.update(kind='page_marker',preserve=True,preserve_reason='ocr_above_running_header')
            body=[u for u in units if u['kind'] not in metadata]
            prior=next((u for u in reversed(blocks) if u['kind'] not in metadata),None)
            if page_number and prior and body and continuation(prior,body[0],pages[page_number-1],page_info):
                first=body[0]
                combined,removed=join_lines(prior['source'],first['source'])
                if removed:prior['fragments'][-1]['source']=prior['fragments'][-1]['source'][:-1]
                prior['source']=combined;prior['fragments']+=first['fragments']
                prior['style_runs']+=first['style_runs'];units.remove(first)
            blocks.extend(units)
        logical=[];by_page={p['source_page_index']:p for p in pages}
        for number,block in enumerate(blocks):
            source=block['source'];cursor=0;offset=0;spans=[]
            for fragment in block['fragments']:
                start=source.find(fragment['source'],offset)
                if start<0:raise RuntimeError('연결 문단의 원문 위치를 확인하지 못했습니다.')
                end=start+len(fragment['source']);spans.append((start,end,fragment));offset=end
            for part,segment in enumerate([source] if block['preserve'] else segments(source)):
                start=source.find(segment,cursor);end=start+len(segment);cursor=end
                fragments=[{**f,'source':source[max(a,start):min(b,end)]} for a,b,f in spans if a<end and b>start]
                indices=list(dict.fromkeys(f['source_page_index'] for f in fragments))
                bid=f'p{indices[0]+1}-u{number}-s{part}'
                item={k:v for k,v in block.items() if k not in {'fragments','source','source_page_index'}}
                item.update(id=bid,source=segment,source_page_index=indices[0],source_page_indices=indices,
                            source_fragments=fragments,reading_order=len(logical),split_part=part)
                methods=set(f.get('extraction','native') for f in fragments)
                scores=[f['ocr_confidence_min'] for f in fragments if f.get('ocr_confidence_min') is not None]
                item.update(extraction='mixed' if len(methods)>1 else next(iter(methods)),
                            ocr_confidence_min=min(scores) if scores else None)
                logical.append(item);by_page[indices[0]]['block_ids'].append(bid)
                for index in indices[1:]:by_page[index]['continuations'].append(dict(id=bid,source_page_index=indices[0],source_page_indices=indices))
        return dict(parser=PARSER, ocr=signature(ocr_mode), pages=pages, blocks=logical,
                    limitations=['빈 줄·제목·글머리표로 문단을 나누고 연속 선택한 쪽의 미완성 본문을 연결합니다. 복잡한 2단·표는 검토가 필요합니다.',
                                 '이 버전은 한국어 가독성을 위한 재배치 출력입니다. 원문과 같은 조판은 아직 구현되지 않았습니다.',
                                 'OCR은 영어 인쇄체 중심입니다. 손글씨·수식·흐린 스캔·복잡한 표의 인식 및 읽기 순서는 보장하지 않습니다. 기계 의미 검수는 coverage를 확인하세요.'])
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
        for continuation_info in p.get('continuations',[]):
            first=continuation_info['source_page_index']+1
            notice=f'[연결 안내] 이 쪽의 이어지는 본문은 원본 {first}쪽 번역면의 연결 문단에 함께 번역했습니다.'
            for line in lines_for(notice,9):
                if y+14>789:new_page(index,True)
                page.insert_text((42,y),line,fontname='KO',fontsize=9,color=(.15,.4,.42));y+=14
            y+=8
        for bid in p['block_ids']:
            record = records[bid]; failed=record['status']=='failed'
            if record['kind'] in {'page_marker','running_header'}:continue
            indices=record.get('source_page_indices',[index])
            connection_label=f'[연결 문단: 원본 {indices[0]+1}-{indices[-1]+1}쪽]' if len(indices)>1 else ''
            text = record.get('candidate') if failed and record.get('candidate') else record['target']
            label = ('[자동 검사 미통과 · 검토용 번역] ' if record.get('candidate') else '[번역 실패 · 원문 보존] ') if failed else ('[원문 보존] ' if record['status']=='preserved' else '')
            text = label + text
            size=13 if record['kind']=='heading' else 11
            lines=lines_for(text,size)
            # Keep a short paragraph together rather than stranding its last line.
            height=len(lines)*size*1.55
            if record['kind']!='heading' and height<650 and y+height+(17 if connection_label else 0)>789 and y>100:
                new_page(index,True)
            if connection_label:
                if y+17>789:new_page(index,True)
                page.insert_text((42,y),connection_label,fontname='KO',fontsize=9,color=(.15,.4,.42));y+=17
            for line in lines:
                if y+size > 789:
                    new_page(index, True)
                page.insert_text((42,y), line, fontname='KO', fontsize=size,
                                 color=(.65,.27,.08) if failed else (.08,.13,.20))
                y += size*1.55
            y += 8
    if not len(doc):
        raise RuntimeError('출력할 쪽이 없습니다.')
    doc.subset_fonts(); doc.save(path, garbage=4, deflate=True)
    doc.close()
    return mapping


def export(ir, records, original, folder, title, job, cancel=lambda: False):
    folder = Path(folder); folder.mkdir(parents=True)
    # Retried/cached records may have been inserted later than their neighbours.
    # Reports and quality records follow the IR, just like the PDF reading order.
    records={block['id']:records[block['id']] for block in ir['blocks']}
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
        quality=dict(schema=5, ocr=ir.get('ocr'), ocr_pages=sum(p.get('ocr',{}).get('lines',0)>0 for p in ir['pages']),
                     ocr_unscored_lines=sum(p.get('ocr',{}).get('unscored_lines',0) for p in ir['pages']),
                     ocr_low_confidence_lines=sum(p.get('ocr',{}).get('low_confidence_lines',0) for p in ir['pages']),
                     needs_review=True, file_generation='completed', semantic_review=review_status,
                     semantic_review_coverage=coverage, reviewed_paragraphs=len(reviewed),human_approved=False,
                     reviewer=job.get('reviewer'),
                     model=job['model'], source_sha256=job['source_sha256'],
                     selected_source_pages=[p['source_page_index']+1 for p in ir['pages']],
                     korean_output_source_indices=mapping, output_layout='reflow',
                     failed_paragraphs=len(failed), warnings_paragraphs=len(warnings),
                     translated_paragraphs=sum(r['status']=='translated' for r in records.values()),
                     preserved_paragraphs=sum(r['status']=='preserved' for r in records.values()),
                     page_markers=sum(r['kind']=='page_marker' for r in records.values()),
                     running_headers=sum(r['kind']=='running_header' for r in records.values()),
                     reference_units=sum(r['kind']=='reference' for r in records.values()),
                     quality_policy=job.get('quality_policy',{'mode':'legacy'}),
                     refinement_processed=sum(bool(r.get('refinement')) for r in records.values()),
                     refinement_accepted=sum(r.get('refinement',{}).get('accepted',False) for r in records.values()),
                     cloud_review_completed=sum(r.get('cloud_review',{}).get('status')=='completed' for r in records.values()),
                     cloud_review_attempted=sum(bool(r.get('cloud_review',{}).get('attempted')) for r in records.values()),
                     connected_paragraphs=sum(len(r.get('source_page_indices',[]))>1 for r in records.values()),
                     failed_candidate_paragraphs=sum(r['status']=='failed' and bool(r.get('candidate')) for r in records.values()),
                     assistant_edited_paragraphs=sum(bool(r.get('assistant_post_edit')) for r in records.values()),
                     limitations=ir['limitations'], pages=ir['pages'], records=list(records.values()))
        atomic_json(folder/'Quality.json', quality)
        sections=[]
        for r in records.values():
            issues=' / '.join(i['message'] for i in r.get('issues', []))
            if r.get('extraction') in {'ocr','mixed'}:
                score=r.get('ocr_confidence_min')
                issues+=' / OCR 인식 원문 · 이미지와 대조 필요'+(f' · 최소 신뢰도 {score:.0%}' if score is not None else '')
            if r.get('review'):
                issues+=' / 검수 '+r['review'].get('verdict','uncertain')+': '+r['review'].get('explanation_ko','')
            refinement=r.get('refinement',{})
            if refinement:
                issues+=' / 다중 모델 교정: '+('후보 반영 · 인간 승인 없음' if refinement.get('accepted') else '기존 번역 유지')
                issues+=' / '+refinement.get('explanation_ko','')
            if r.get('cloud_review'):
                issues+=' / 무료 API: '+r['cloud_review'].get('status','not_performed')+' · 인간 승인 없음'
            target=r['target']
            if r['status']=='failed' and r.get('candidate'):
                issues='자동 검사 미통과: 아래는 승인되지 않은 검토용 번역 후보입니다. / '+issues
                target=r['candidate']
            page_label=', '.join(str(i+1) for i in r.get('source_page_indices',[r['source_page_index']]))
            history=(f'<details><summary>교정 전 번역과 원문 인용 근거</summary><pre>{html.escape(refinement.get("draft",""))}</pre>'
                     f'<p>원문 근거: {html.escape(refinement.get("source_evidence",""))}</p>'
                     f'<p>번역 근거: {html.escape(refinement.get("draft_evidence",""))}</p></details>') if refinement else ''
            sections.append(f'<article id="{html.escape(r["id"],quote=True)}"><h2>원본 {page_label}쪽 · {html.escape(r["id"])}</h2>'
                            f'<p class="warning">{html.escape(issues)}</p><div class="pair"><pre>{html.escape(r["source"])}</pre>'
                            f'<pre lang="ko">{html.escape(target)}</pre></div>{history}</article>')
        review='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'"><title>번역 검토</title><style>body{font:16px/1.7 'Malgun Gothic',sans-serif;margin:40px;background:#f4f6f8;color:#152233}header,article{max-width:1280px;margin:20px auto;padding:24px;background:white;border-radius:12px}h2{font-size:16px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:30px}pre{white-space:pre-wrap;font:inherit;overflow-wrap:anywhere}.warning{color:#9a4a10}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style><header><h1>영한 문단 검토</h1><p>초기 개발판 · 자동 검사는 의미 정확도를 보증하지 않습니다. 검수 범위와 판정은 Quality.json에 기록됩니다.</p></header>'''+''.join(sections)+'</html>'
        (folder/'Review.html').write_text(review,encoding='utf-8')
        fields=dict(제목=title, 원본파일=job['source_name'], 번역일=job['created_at'], 원본SHA256=job['source_sha256'],
                    전체쪽=job['total_pages'], 선택쪽=quality['selected_source_pages'], 모델=job['model'],
                    분야=job['profile'], 조판='한국어 재배치', 의미검수=review_status, 검수coverage=coverage,
                    사용자승인=False, 번역실패문단수=len(failed), tags=['논문','번역'])
        name=safe_title(title)
        note='---\n'+'\n'.join(k+': '+json.dumps(v,ensure_ascii=False) for k,v in fields.items())+'\n---\n\n# '+name+'\n\n'
        note+=f'> 초기 개발판: 기계 의미 검수 {len(reviewed)}/{len(translated)}문단 · 인간 승인 없음. OCR {quality["ocr_pages"]}쪽 · 인식 오류 대조 필요. 원문 조판 재현은 미지원입니다.\n\n'
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
