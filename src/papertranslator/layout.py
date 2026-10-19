"""Source-position bilingual rendering, with explicit readable overflow pages."""
import math,re
from collections import defaultdict
import pymupdf as fitz
from .alignment import page_sources

VERSION='source-position-layout-1'
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
            if not fragments:fragments=[dict(bbox=r['bbox'])]
            boxes=[fitz.Rect(f.get('cell_bbox') or f['bbox']) for f in fragments]
            box=fitz.Rect(min(b.x0 for b in boxes),min(b.y0 for b in boxes),max(b.x1 for b in boxes),max(b.y1 for b in boxes))
            text=next(p['target'] for p in parts if p['source_page_index']==index) if valid else target
            uncertain_part=len(sources)>1 and not valid
            if uncertain_part:text='[쪽 대응 미확정 · 연결 문단 전체]\n'+text
            shared=valid and any(p.get('shared_boundary') for p in parts if p['source_page_index']==index)
            entries[index].append(dict(ids=[r['id']],bbox=box,erase_boxes=[fitz.Rect(f['bbox']) for f in fragments],text=text,kind=r['kind'],failed=r['status']=='failed',uncertain=uncertain_part,shared_boundary=shared,
                                      estimated=alignment.get('status')=='proposed'))
        if len(sources)>1 and not valid:uncertain.append(r['id'])
    return entries,uncertain


def group_overlaps(entries):
    groups=[]
    for entry in entries:
        overlaps=[g for g in groups if (g['bbox']&entry['bbox']).height>2 and
                  (g['bbox']&entry['bbox']).get_area()>min(g['bbox'].get_area(),entry['bbox'].get_area())*.03]
        if not overlaps:groups.append({**entry});continue
        first=overlaps[0]
        # Split model requests on the same printed line share a region. Keep
        # their original reading order instead of painting over one another.
        for other in [entry,*overlaps[1:]]:
            first['text']+=(' ' if first['kind']=='table_cell' else '\n')+other['text'];first['ids']+=other['ids'];first['bbox']|=other['bbox']
            first['erase_boxes']+=other['erase_boxes']
            first['failed']|=other['failed'];first['uncertain']|=other['uncertain']
            first['estimated']|=other['estimated'];first['shared_boundary']|=other['shared_boundary']
        for other in overlaps[1:]:groups.remove(other)
    return groups


def bilingual_pdf(ir,records,original,path,cancel=lambda:False,min_font=9.):
    entries,uncertain=record_entries(ir,records);font=fitz.Font('korea');fontbuf=font.buffer
    placed=[];mapping=[];overflow_pages=0
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
            page.insert_text((rect.width+12,rect.height+15),label,fontname='KO',fontsize=8,color=(.3,.38,.43))
            mapping.append(index);return page
        for info in ir['pages']:
            if cancel():
                from .backend import Cancelled
                raise Cancelled()
            index=info['source_page_index'];original_rect,_=faces[index];w=original_rect.width;h=original_rect.height
            page=new_page(index);groups=group_overlaps(entries[index]);pending=[];plans=[]
            for group in groups:
                box=group['bbox']&original_rect
                if box.is_empty:box=fitz.Rect(16,40,w-16,60)
                allocated=fitz.Rect(box)
                # Body can use whitespace below itself, up to the next source
                # region in the same column. Table cells keep their exact grid.
                if group['kind']!='table_cell':
                    below=[g['bbox'].y0 for g in groups if g is not group and g['bbox'].y0>=box.y1-1 and
                           min(g['bbox'].x1,box.x1)-max(g['bbox'].x0,box.x0)>min(g['bbox'].width,box.width)*.3]
                    allocated.y1=max(box.y1,min(min(below,default=h-15)-3,box.y1+40,h-12))
                if group['kind']=='table_cell':allocated=allocated+fitz.Rect(1.2,.05,-1.2,-.05)
                # Clear only this source region; figures and table rules remain.
                masks=[(b+fitz.Rect(-.4,-.1,.4,.1))&box for b in group['erase_boxes']] if group['kind']=='table_cell' else [allocated]
                for mask in masks:
                    if not mask.is_empty:page.add_redact_annot(mask+fitz.Rect(w,0,w,0),fill=(1,1,1),cross_out=False)
                sizes=[s for s in [13 if group['kind']=='heading' else 10.5,10.,9.5,min_font] if s>=min_font]
                size=min_font;lines=[];capacity=0
                for candidate in dict.fromkeys(sizes):
                    candidate_lines=wrap(group['text'],font,candidate,max(candidate,allocated.width-3))
                    tolerance=.15 if group['kind']=='table_cell' else 0
                    candidate_capacity=max(0,math.floor((allocated.height+tolerance-(font.ascender-font.descender)*candidate)/(candidate*1.35))+1)
                    if allocated.width<candidate+3:candidate_capacity=0
                    size=candidate;lines=candidate_lines;capacity=candidate_capacity
                    if len(lines)<=capacity:break
                visible=lines[:capacity];remaining=lines[capacity:]
                color=(.65,.28,.08) if group['failed'] or group['uncertain'] or group['estimated'] else (.08,.13,.2)
                plans.append((group,box,allocated,size,visible,remaining,color))
            if plans:page.apply_redactions(images=0,graphics=0,text=0)
            for group,box,allocated,size,visible,remaining,color in plans:
                baseline=allocated.y0+font.ascender*size
                for line in visible:
                    page.insert_text((w+allocated.x0+1.5,baseline),line,fontname='KO',fontsize=size,color=color)
                    baseline+=size*1.35
                placed.append(dict(ids=group['ids'],source_page_index=index,output_page=page.number,
                                   source_bbox=list(box),target_bbox=list(allocated),font_size=size,overflow=bool(remaining),uncertain=group['uncertain'],estimated=group['estimated'],shared_boundary=group['shared_boundary']))
                if remaining:
                    # Never silently shrink below the chosen minimum or drop text.
                    pending.append(dict(ids=group['ids'],text='\n'.join(remaining),color=color))
                page.insert_link({'kind':fitz.LINK_GOTO,'from':box,'page':page.number,
                                  'to':fitz.Point(w+allocated.x0,allocated.y0)})
            extra=None;y=0
            if pending:
                page.insert_text((w+12,h+23),'일부 번역은 다음 보충면에 이어집니다.',fontname='KO',fontsize=6,color=(.4,.28,.1))
            for continuation_number,item in enumerate(pending,1):
                lines=wrap(item['text'],font,10.5,w-40)
                if extra is None or y>h-60:extra=new_page(index,True);y=40
                label=f'[번역 이어짐 · 원문 영역 {continuation_number}]'
                extra.insert_text((w+20,y),label,fontname='KO',fontsize=8,color=(.2,.4,.44));y+=20
                for line in lines:
                    if y+14>h-20:extra=new_page(index,True);y=40
                    extra.insert_text((w+20,y),line,fontname='KO',fontsize=10.5,color=item['color']);y+=15
                y+=15
        out.subset_fonts();out.save(path,garbage=4,deflate=True)
    return dict(version=VERSION,mode='source_position',source_indices=mapping,regions=placed,overflow_pages=overflow_pages,
                unresolved_page_alignments=uncertain,minimum_font=min_font,
                proposed_page_alignments=[r['id'] for r in records.values() if r.get('page_alignment',{}).get('status')=='proposed'],
                table_regions=sum(len(p.get('table_regions',[])) for p in ir['pages']))
