"""Source-position bilingual rendering, with explicit readable overflow pages."""
import math,re,base64,statistics
from collections import defaultdict
import pymupdf as fitz
from .alignment import page_sources

VERSION='source-position-layout-3-validated-flow'
TOKEN=re.compile(r'https?://[^\s]+|`[^`]+`|[-−+]?\d+(?:[.,]\d+)*(?:%|[eE][-+]?\d+)?|[A-Za-z_][A-Za-z_0-9]*|\n|.')


def wrap(text,font,size,width):
    lines=[];line='';used=0
    for match in TOKEN.finditer(text):
        token=match[0]
        if token=='\n':lines.append(line);line='';used=0;continue
        pieces=[token] if font.text_length(token,fontsize=size)<=width else list(token)
        for piece in pieces:
            amount=font.text_length(piece,fontsize=size)
            if line and used+amount>width:lines.append(line);line='';used=0
            line+=piece;used+=amount
    if line:lines.append(line)
    return lines


def record_entries(ir,records):
    entries=defaultdict(list);uncertain=[]
    page_info={p['source_page_index']:p for p in ir['pages']}
    header_keys={re.sub(r'\W+','',r['source']).casefold() for r in records.values() if r['kind']=='running_header' and len(r['source'])>12}
    def metadata(fragment):
        if fragment.get('verified_margin_noise'):return True
        info=page_info.get(fragment['source_page_index'],{});height=info.get('height',842);box=fragment['bbox'];text=fragment['source'].strip()
        if box[3]<height*.12 and box[1]<height*.075:
            key=re.sub(r'\W+','',text).casefold()
            if len(text)<40 and re.search(r'[—_=\-]{3,}',text):return True
            return key in {'foreword','introduction','contents','references','index'} or any(key.startswith(k) and len(k)>=len(key)*.55 for k in header_keys)
        return box[1]>height*.91 and len(text)<40 and (sum(c.isalpha() for c in text)<len(text)*.5 or
            (bool(re.search(r'[—_=\-]{2,}',text)) and sum(c.isalpha() for c in text)<=10))

    translated_cells={(f['source_page_index'],tuple(f['cell_bbox'])) for r in records.values() if r['status']!='preserved'
                      for f in r.get('source_fragments',[]) if f.get('cell_bbox')}
    for block in ir['blocks']:
        r=records[block['id']]
        shared_cell=(r['kind']=='table_cell' and any((f['source_page_index'],tuple(f['cell_bbox'])) in translated_cells
                     for f in r.get('source_fragments',[]) if f.get('cell_bbox')))
        if (r['status']=='preserved' and not shared_cell) or r['kind'] in {'page_marker','running_header'}:continue
        target=(r.get('candidate') if r['status']=='failed' else r['target']) or r['target']
        sources=page_sources(r)
        if not sources:sources=[dict(page=r['source_page_index'],source=r['source'])]
        alignment=r.get('page_alignment',{})
        parts=alignment.get('parts',[])
        spans_valid=(bool(parts) and all(isinstance(p.get('start'),int) and isinstance(p.get('end'),int) and
                     p['target']==target[p['start']:p['end']] for p in parts) and parts[0]['start']==0 and
                     parts[-1]['end']==len(target) and all(a['start']<=b['start'] and a['end']<=b['end'] and
                     not target[a['end']:b['start']].strip() for a,b in zip(parts,parts[1:])))
        valid=(alignment.get('status') in {'aligned','proposed'} and (spans_valid or ''.join(p['target'] for p in parts)==target) and
               [p['source_page_index'] for p in parts]==[p['page'] for p in sources])
        for source in sources:
            index=source['page'];fragments=[f for f in r.get('source_fragments',[]) if f['source_page_index']==index]
            fragments=[f for f in fragments if not metadata(f)]
            if r.get('source_fragments') and not fragments:continue
            if not fragments:fragments=[dict(bbox=r['bbox'])]
            boxes=[fitz.Rect(f.get('cell_bbox') or f.get('region_bbox') or f['bbox']) for f in fragments]
            box=fitz.Rect(min(b.x0 for b in boxes),min(b.y0 for b in boxes),max(b.x1 for b in boxes),max(b.y1 for b in boxes))
            text=next(p['target'] for p in parts if p['source_page_index']==index) if valid else target
            uncertain_part=len(sources)>1 and not valid
            if uncertain_part:text='[쪽 대응 미확정 · 연결 문단 전체]\n'+text
            shared=valid and any(p.get('shared_boundary') for p in parts if p['source_page_index']==index)
            entries[index].append(dict(ids=[r['id']],bbox=box,erase_patches=[f['erase_patch'] for f in fragments if f.get('erase_patch')],erase_boxes=[fitz.Rect(l['bbox']) for f in fragments for l in (f.get('layout_lines') or [f])],text=text,kind=r['kind'],failed=r['status']=='failed',uncertain=uncertain_part,shared_boundary=shared,
                                      estimated=alignment.get('status')=='proposed',rotation=fragments[0].get('rotation',0),diagram=any(i.get('code')=='diagram_layout' for i in r.get('issues',[]))))
        if len(sources)>1 and not valid:uncertain.append(r['id'])
    return entries,uncertain


def group_overlaps(entries):
    groups=[]
    for entry in entries:
        overlaps=[g for g in groups if g['kind']!='diagram_label' and entry['kind']!='diagram_label' and (g['bbox']&entry['bbox']).height>2 and
                  (g['bbox']&entry['bbox']).get_area()>min(g['bbox'].get_area(),entry['bbox'].get_area())*.03]
        if not overlaps and entry['kind']=='paragraph':
            # A complete boundary sentence can be longer than its one source
            # line. Share the continuous body column below it before spilling.
            overlaps=[g for g in groups if g['kind']=='paragraph' and (g['shared_boundary'] or entry['shared_boundary']) and
                      -2.5<=entry['bbox'].y0-g['bbox'].y1<4 and
                      min(g['bbox'].x1,entry['bbox'].x1)-max(g['bbox'].x0,entry['bbox'].x0)>min(g['bbox'].width,entry['bbox'].width)*.7]
        if not overlaps:groups.append({**entry});continue
        first=overlaps[0]
        # Split model requests on the same printed line share a region. Keep
        # their original reading order instead of painting over one another.
        for other in [entry,*overlaps[1:]]:
            first['text']+=(' ' if first['kind'] in {'table_cell','list_label'} or other['kind']=='list_label' else '\n')+other['text'];first['ids']+=other['ids'];first['bbox']|=other['bbox']
            first['erase_boxes']+=other['erase_boxes'];first['erase_patches']+=other['erase_patches']
            first['failed']|=other['failed'];first['uncertain']|=other['uncertain']
            first['estimated']|=other['estimated'];first['shared_boundary']|=other['shared_boundary'];first['diagram']|=other['diagram']
        for other in overlaps[1:]:groups.remove(other)
    return groups


def bilingual_pdf(ir,records,original,path,cancel=lambda:False,min_font=9.):
    entries,uncertain=record_entries(ir,records);font=fitz.Font('korea');fontbuf=font.buffer
    placed=[];mapping=[];overflow_pages=0;validation_issues=[];drawn=defaultdict(list)
    with fitz.open(original) as src,fitz.open(original) as right_src,fitz.open() as out:
        faces={p['source_page_index']:(fitz.Rect(src[p['source_page_index']].rect),src[p['source_page_index']].rotation) for p in ir['pages']}
        def face(page,index,x):
            rect,rotation=faces[index];document=right_src if x else src
            source=document[index];source.set_rotation(0)
            if source.get_contents():page.show_pdf_page(fitz.Rect(x,0,x+rect.width,rect.height),document,index,rotate=-rotation)
        def new_page(index,extra=False):
            nonlocal overflow_pages
            rect,_=faces[index];page=out.new_page(width=rect.width*2,height=rect.height+25)
            page.insert_font(fontname='KO',fontbuffer=fontbuf);face(page,index,0)
            if not extra:face(page,index,rect.width)
            if extra:overflow_pages+=1
            page.draw_line((rect.width+.5,0),(rect.width+.5,rect.height),color=(.8,.83,.86),width=.4)
            label=f'원본 {index+1}쪽 · '+('번역 보충면 · 같은 원문과 대조' if extra else '원문 위치 대응 · 기계 번역 · 원문 대조 필요')
            if not extra and any(e.get('shared_boundary') for e in entries[index]):label+=' · 연결 문장 양쪽 표시'
            if not extra and any(e.get('estimated') for e in entries[index]):label+=' · 대응 추정'
            page.insert_text((rect.width+12,rect.height+15),label,fontname='KO',fontsize=7.5,color=(.3,.38,.43))
            mapping.append(index);return page
        for info in ir['pages']:
            if cancel():
                from .backend import Cancelled
                raise Cancelled()
            index=info['source_page_index'];original_rect,_=faces[index];w=original_rect.width;h=original_rect.height
            preview=src[index].get_pixmap(matrix=fitz.Matrix(.12,.12),colorspace=fitz.csRGB,alpha=False)
            background=tuple(statistics.median(preview.samples[c::3])/255 for c in range(3))
            dark=sum(background)<.9
            if not dark:background=(1,1,1)
            page=new_page(index);groups=group_overlaps(entries[index]);pending=[];plans=[]
            gutters=[fitz.Rect(f['bbox']) for r in records.values() if r['status']=='preserved' and r['kind']=='protected' and
                     len(r['source'])<12 and re.fullmatch(r'(?:[A-ZIVX]+[().a-z]*|[•*»]+)',r['source'].strip())
                     for f in r.get('source_fragments',[]) if f['source_page_index']==index]

            for group in groups:
                box=group['bbox']&original_rect
                if box.is_empty:box=fitz.Rect(16,40,w-16,60)
                allocated=fitz.Rect(box)
                if group['kind'] not in {'table_cell','diagram_label','form_question'}:
                    blockers=[g['bbox'] for g in groups if g is not group]
                    # A short English label is not a Korean column width. Use
                    # available column/row space, stopping before adjacent text.
                    if box.width<w*.35 or len(group['text'])<100:
                        columns=[g['bbox'] for g in groups if g['bbox'].width>w*.48 and g['bbox'].x0<=box.x0+20 and g['bbox'].x1>box.x1]
                        right=min((b.x1 for b in columns),default=w-18)
                        beside=[b.x0-5 for b in blockers if b.x0>=box.x1-1 and (b&box).height>2]
                        allocated.x1=max(box.x1,min([right,*beside]))
                        allocated.y0=max(0,box.y0-1.2)
                    below=[b.y0 for b in blockers if b.y0>box.y0+2 and min(b.x1,allocated.x1)-max(b.x0,allocated.x0)>min(b.width,allocated.width)*.3]
                    allocated.y1=min(min(below,default=h-24)-1,h-24)
                    allocated.y1=max(allocated.y0,allocated.y1)
                    if len(group['text'])<40 and box.height<12:
                        line_height=(font.ascender-font.descender)*min_font
                        allocated.y0=max(0,(box.y0+box.y1-line_height)/2)
                        allocated.y1=max(allocated.y1,allocated.y0+line_height+.1)
                elif group['kind'] in {'table_cell','form_question'}:
                    padding=.2 if group['kind']=='table_cell' and len(group['text'].strip())<=3 else 1.2
                    allocated=allocated+fitz.Rect(padding,.05,-padding,-.05)
                    if group['kind']=='table_cell':
                        line_height=(font.ascender-font.descender)*min_font
                        if allocated.height<line_height+.1:
                            center=(allocated.y0+allocated.y1)/2
                            allocated.y0=center-(line_height+.1)/2;allocated.y1=center+(line_height+.1)/2
                if group['kind'] not in {'table_cell','diagram_label'}:
                    left_gutters=[b.x1+5 for b in gutters if abs(b.x0-box.x0)<8 and b.y0<box.y0+20 and b.y1>box.y0]
                    if left_gutters:allocated.x0=max([allocated.x0,*left_gutters])
                # Clear source glyph regions, not the surrounding table borders
                # or the whitespace allocated for a longer Korean translation.
                masks=[(b+fitz.Rect(-.3,-.05,.3,.05))&original_rect for b in group['erase_boxes']]
                if group['kind'] not in {'table_cell','diagram_label'}:
                    for mask in masks:
                        for gutter in gutters:
                            if abs(gutter.x0-mask.x0)<8 and (gutter&mask).height>1:mask.x0=max(mask.x0,gutter.x1+2)
                if not group['diagram'] and not group['erase_patches']:
                    for mask in masks:
                        if not mask.is_empty:page.add_redact_annot(mask+fitz.Rect(w,0,w,0),fill=background,cross_out=False)
                rotated=group.get('rotation',0)
                width=allocated.height if rotated else allocated.width
                height=allocated.width if rotated else allocated.height
                sizes=[9.,8.5,8.] if group['kind']=='diagram_label' else [s for s in [13 if group['kind']=='heading' else 10.5,10.,9.5,min_font] if s>=min_font]
                size=min_font;lines=[];capacity=0
                for candidate in dict.fromkeys(sizes):
                    inset=.2 if group['kind']=='table_cell' and len(group['text'].strip())<=3 else 1.5
                    candidate_lines=wrap(group['text'],font,candidate,max(candidate,width-inset*2))
                    tolerance=.15
                    candidate_capacity=max(0,math.floor((height+tolerance-(font.ascender-font.descender)*candidate)/(candidate*1.35))+1)
                    if width<candidate+inset*2:candidate_capacity=0
                    size=candidate;lines=candidate_lines;capacity=candidate_capacity
                    if len(lines)<=capacity:break
                # A continuation receives the canonical whole text, never the
                # line breaks made for a narrow source box or a two-letter tail.
                spills=len(lines)>capacity or (not rotated and allocated.width<min(60,w*.12) and len(lines)>3) or group['diagram']
                if re.sub(r'\s+','',''.join(lines))!=re.sub(r'\s+','',group['text']):
                    validation_issues.append(dict(code='planned_text_mismatch',source_page_index=index,ids=group['ids']))
                visible=[] if spills else lines;remaining=group['text'] if spills else ''
                if group['diagram']:
                    validation_issues.append(dict(code='diagram_not_positioned',source_page_index=index,ids=group['ids']))

                color=((1,.73,.38) if dark else (.65,.28,.08)) if group['failed'] or group['uncertain'] or group['estimated'] else ((.94,.94,.9) if dark else (.08,.13,.2))
                plans.append((group,box,allocated,size,visible,remaining,color))
            if plans:page.apply_redactions(images=0,graphics=0,text=0)
            for group,box,allocated,size,visible,remaining,color in plans:
                for patch in group['erase_patches']:
                    page.insert_image(fitz.Rect(patch['bbox'])+fitz.Rect(w,0,w,0),stream=base64.b64decode(patch['png']))
                baseline=allocated.y0+font.ascender*size
                if group.get('rotation') and visible:
                    with fitz.open() as label_doc:
                        label=label_doc.new_page(width=allocated.height,height=allocated.width)
                        label.insert_font(fontname='KO',fontbuffer=fontbuf)
                        text_height=(font.ascender-font.descender)*size+(len(visible)-1)*size*1.35
                        local_y=(label.rect.height-text_height)/2+font.ascender*size
                        for line in visible:
                            label.insert_text((max(1.5,(label.rect.width-font.text_length(line,fontsize=size))/2),local_y),line,fontname='KO',fontsize=size,color=color);local_y+=size*1.35
                        page.show_pdf_page(allocated+fitz.Rect(w,0,w,0),label_doc,0,rotate=group['rotation'])
                    drawn[page.number].extend(visible)
                for line in ([] if group.get('rotation') else visible):
                    inset=.2 if group['kind']=='table_cell' and len(group['text'].strip())<=3 else 1.5
                    page.insert_text((w+allocated.x0+inset,baseline),line,fontname='KO',fontsize=size,color=color)
                    drawn[page.number].append(line);baseline+=size*1.35
                placed.append(dict(ids=group['ids'],source_page_index=index,output_page=page.number,
                                   source_bbox=list(box),target_bbox=list(allocated),font_size=size,overflow=bool(remaining),uncertain=group['uncertain'],estimated=group['estimated'],shared_boundary=group['shared_boundary']))
                if remaining:
                    # Never silently shrink below the chosen minimum or drop text.
                    pending.append(dict(ids=group['ids'],text=remaining,color=(.08,.13,.2) if dark else color))
                    if not group['diagram'] and allocated.width>80 and allocated.height>11:
                        page.insert_text((w+allocated.x0+1.5,allocated.y0+9),f'[번역 보충 {len(pending)}]',fontname='KO',fontsize=7.5,color=(.3,.4,.44))
                page.insert_link({'kind':fitz.LINK_GOTO,'from':box,'page':page.number,
                                  'to':fitz.Point(w+allocated.x0,allocated.y0)})
            extra=None;y=0
            if pending:
                page.insert_text((w+12,h+23),'일부 번역은 다음 보충면에 이어집니다.',fontname='KO',fontsize=6,color=(.4,.28,.1))
            for continuation_number,item in enumerate(pending,1):
                lines=wrap(item['text'],font,10.5,w-40)
                if extra is None or y>h-60:extra=new_page(index,True);y=40
                label=f'[번역 이어짐 · 원문 영역 {continuation_number}]'
                extra.insert_text((w+20,y),label,fontname='KO',fontsize=7.5,color=(.2,.4,.44));y+=20
                for line in lines:
                    if y+14>h-20:extra=new_page(index,True);y=40
                    extra.insert_text((w+20,y),line,fontname='KO',fontsize=10.5,color=item['color']);drawn[extra.number].append(line);y+=15
                y+=15
        # Extract every rendered body span, not a sample of pages. Missing or
        # clipped glyphs are a failed layout check, independent of model quality.
        for n,index in enumerate(mapping):
            width=faces[index][0].width;actual=[];boxes=[]
            for block in out[n].get_text('dict')['blocks']:
                if block['type']!=0:continue
                for row in block['lines']:
                    for span in row['spans']:
                        if span['font']==font.name and span['size']>=8-.01 and span['bbox'][0]>=width:
                            actual.append(span['text']);boxes.append(fitz.Rect(span['bbox']))
            normalize=lambda text:re.sub(r'\s+','',text)
            expected=normalize(''.join(drawn[n]));observed=normalize(''.join(actual))
            if expected!=observed:validation_issues.append(dict(code='rendered_text_mismatch',output_page=n,source_page_index=index))
            for i,box in enumerate(boxes):
                if any((box&other).height>min(box.height,other.height)*.3 and (box&other).get_area()>min(box.get_area(),other.get_area())*.15 for other in boxes[:i]):
                    validation_issues.append(dict(code='body_text_overlap',output_page=n,source_page_index=index));break
            if ir['pages'][next(i for i,p in enumerate(ir['pages']) if p['source_page_index']==index)].get('ocr',{}).get('fragmented_text'):
                validation_issues.append(dict(code='fragmented_ocr_requires_review',output_page=n,source_page_index=index))
            boundary=fitz.Rect(width,0,width*2,faces[index][0].height)
            if any(not boundary.contains(b) for b in boxes):validation_issues.append(dict(code='body_outside_page',output_page=n,source_page_index=index))
        out.subset_fonts();out.save(path,garbage=4,deflate=True)

    return dict(version=VERSION,mode='source_position',source_indices=mapping,regions=placed,overflow_pages=overflow_pages,
                unresolved_page_alignments=uncertain,minimum_font=min_font,
                proposed_page_alignments=[r['id'] for r in records.values() if r.get('page_alignment',{}).get('status')=='proposed'],
                validation=dict(status='review_required' if validation_issues or uncertain else 'passed',issues=validation_issues,all_output_pages_checked=len(mapping),semantic_accuracy='not_validated'),
                diagram_minimum_font=8.,table_regions=sum(len(p.get('table_regions',[])) for p in ir['pages']))
