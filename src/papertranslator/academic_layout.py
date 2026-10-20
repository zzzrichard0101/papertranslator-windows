"""Automatic academic columns, with independent full-width float bands."""
from collections import Counter
import copy
import re
import statistics

import pymupdf as fitz

from .structure import bands, text, words_for

VERSION='academic-column-bands-1'


def _side(line,x,width):
    box=line['bbox']
    if box[2]<=x:return 0
    if box[0]>=x:return 1
    words=words_for(line)
    if not words or any(w['bbox'][0]<x<w['bbox'][2] for w in words):return 2
    left=[w for w in words if w['bbox'][2]<=x]
    right=[w for w in words if w['bbox'][0]>=x]
    if (left and right and min(w['bbox'][0] for w in right)-max(w['bbox'][2] for w in left)>=max(10,width*.018)
        and min(sum(len(w['text']) for w in items) for items in (left,right))>=25):return 3
    return 2


def _prose(line,page):
    box=fitz.Rect(line['bbox']);value=text(line)
    return (page.rect.height*.08<box.y0<page.rect.height*.92 and box.height<30 and
        box.width>page.rect.width*.22 and len(re.findall(r'[A-Za-z]{2,}',value))>=4 and
        not line.get('rotation') and not line.get('diagram_label') and not line.get('layout_kind') and
        abs(line.get('dir',(1,0))[1])<.1 and
        not re.match(r'^\s*(?:Table|Figure|Fig\.|Box)\s*\d',value,re.I) and
        not any(s.get('flags',0)&8 or re.search(r'courier|consolas|mono',s.get('font',''),re.I) for s in line['spans']))


def _merge_intervals(intervals):
    result=[]
    for a,b in sorted(intervals):
        if result and a<=result[-1][1]+2:result[-1][1]=max(result[-1][1],b)
        else:result.append([a,b])
    return result


def _key(box,info):
    x=info['gutter'];y=(box[1]+box[3])/2
    for i,(a,b) in enumerate(info['spanning_bands']):
        if y<a:return (i*2,0 if box[2]<=x else 1,box[1],box[0])
        if y<=b:return (i*2+1,0,box[1],box[0])
    return (len(info['spanning_bands'])*2,0 if box[2]<=x else 1,box[1],box[0])


def order_units(units,info):
    """Include separated table cells in the same geometric reading plan."""
    units.sort(key=lambda u:_key(u['bbox'],info))
    for index,unit in enumerate(units):unit['flow_order']=index


def _algorithms(page,blocks):
    lines=[l for b in blocks for l in b.get('lines',[])]
    regions=[]
    for first in lines:
        if not re.match(r'^\s*Algorithm\s*\d',text(first),re.I):continue
        side=0 if first['bbox'][0]<page.rect.width/2 else 1
        left=0 if side==0 else page.rect.width/2;right=page.rect.width/2 if side==0 else page.rect.width
        endings=[l for l in lines if first['bbox'][1]+30<l['bbox'][1]<first['bbox'][1]+page.rect.height*.6 and
                 left<=l['bbox'][0]<right and re.fullmatch(r'\s*end\s*(?:function|procedure|algorithm)\s*',text(l),re.I)]
        if not endings:continue
        end=min(endings,key=lambda l:l['bbox'][1])['bbox'][3]+1
        members=[l for l in lines if left<=(l['bbox'][0]+l['bbox'][2])/2<right and first['bbox'][1]-1<=l['bbox'][1]<end]
        value=' '.join(text(l) for l in members)
        if sum(bool(re.search(r'\b'+word+r'\b',value,re.I)) for word in ('function','for','return','while','procedure'))<2:continue
        regions.append((left,first['bbox'][1]-1,right,end,members))
    if not regions:return blocks
    covered={id(l) for *_,members in regions for l in members};result=[]
    for block in blocks:
        kept=[l for l in block.get('lines',[]) if id(l) not in covered]
        if kept:result.append({**block,'lines':kept})
    for left,top,right,end,members in regions:
        result.append(dict(type=0,bbox=[left,top,right,end],lines=members,academic_preserve='algorithm'))
    return result


def join_column_continuations(units,height):
    """Join only an unfinished body paragraph at a verified column transition."""
    from .document import join_lines
    body=[u for u in units if u['kind'] not in {'page_marker','running_header','caption'}]
    for left,right in zip(body,body[1:]):
        a=left['source'].rstrip();b=right['source'].lstrip()
        if (left.get('column_index')!=0 or right.get('column_index')!=1 or
            left.get('flow_group',[None])[0]!=right.get('flow_group',[None])[0] or
            left['kind']!='paragraph' or right['kind']!='paragraph' or left['preserve'] or right['preserve'] or
            left['bbox'][3]<height*.62 or right['bbox'][1]>height*.45 or
            not a or not b or re.search(r'[.!?。！？…][\"\x27’”\])]*$',a) or
            not (b[:1].islower() or b[:1].isdigit() or a.endswith(('-',',')))):continue
        combined,removed=join_lines(a,b)
        if removed:left['fragments'][-1]['source']=left['fragments'][-1]['source'][:-1]
        left['source']=combined;left['fragments']+=right['fragments'];left['style_runs']+=right['style_runs']
        left['column_continuation']=True;units.remove(right)


def columns(page,blocks,regions=(),hint=None):
    blocks=_algorithms(page,blocks)
    lines=[l for b in blocks if b.get('type')==0 for l in b.get('lines',[])]
    prose=[l for l in lines if _prose(l,page)]
    if len(prose)<12:return blocks,None
    width=page.rect.width;candidates=[]
    for x in range(int(width*.36),int(width*.64),max(1,int(width/400))):
        sides=[[],[]];crossed=0
        for line in prose:
            side=_side(line,x,width)
            if side==2:crossed+=1;continue
            for index in ([0,1] if side==3 else [side]):
                items=[w for w in words_for(line) if (w['bbox'][0]+w['bbox'][2])/2<x] if index==0 else [w for w in words_for(line) if (w['bbox'][0]+w['bbox'][2])/2>=x]
                if items:sides[index].extend(items)
        rows=[bands(items) for items in sides]
        if min(map(len,rows))<6:continue
        counts=[sum(len(w['text']) for w in items) for items in sides]
        if min(counts)<max(counts)*.30:continue
        breadth=lambda rr:statistics.median(max(w['bbox'][2] for w in r)-min(w['bbox'][0] for w in r) for r in rr)
        if min(map(breadth,rows))<width*.22:continue
        gap=min(w['bbox'][0] for w in sides[1])-max(w['bbox'][2] for w in sides[0])
        if gap<max(10,width*.018):continue
        centers=[[statistics.median((w['bbox'][1]+w['bbox'][3])/2 for w in r) for r in rr] for rr in rows]
        support=sum(any(abs(a-b)<12 for b in centers[1]) for a in centers[0])
        if support<5 or min(max(c)-min(c) for c in centers)<page.rect.height*.10:continue
        candidates.append((-min(map(len,rows)),crossed,-gap,abs(x-width/2),x))
    inherited=False
    if not candidates and hint is not None:
        x=hint*width
        evidence=[l for l in lines if page.rect.height*.08<l['bbox'][1]<page.rect.height*.92 and
                  l['bbox'][2]-l['bbox'][0]>width*.22 and len(re.findall(r'[A-Za-z]{2,}',text(l)))>=5]
        sides=[[l for l in evidence if _side(l,x,width)==side] for side in (0,1)]
        # A nearby verified gutter can support unequal columns, but never a
        # full-width page or an empty column. Both sides need broad text lines.
        if min(map(len,sides))<1 or sum(map(len,sides))<8:return blocks,None
        if any(min(l['bbox'][0] for l in side)<(0 if i==0 else x)-3 for i,side in enumerate(sides)):return blocks,None
        middle=x;inherited=True
    elif not candidates:return blocks,None
    else:middle=min(candidates)[-1]
    # Rotated side stamps and page margins cannot cut both body columns.
    intervals=[(l['bbox'][1],l['bbox'][3]) for l in lines if _side(l,middle,width)==2 and
        abs(l.get('dir',(1,0))[1])<.1 and not l.get('rotation')]
    intervals.extend((r['bbox'][1],r['bbox'][3]) for r in regions if r['bbox'][0]<middle<r['bbox'][2])
    # A wide numeric table is one float, even when it has too many columns for
    # the conservative cell extractor. Do not read half of it as body prose.
    from .structure import horizontal_rules
    wide_tables=[]
    captions=[l for l in lines if re.match(r'^\s*Table\s+\d',text(l),re.I)]
    if captions:
        rules=horizontal_rules(page)
        for caption in captions:
            top=caption['bbox'][1]
            body_start=min((l['bbox'][1] for l in prose if l['bbox'][1]>caption['bbox'][3]+15 and
                _side(l,middle,width) in (0,1) and max(s.get('size',0) for s in l['spans'])>=8.5),default=top+page.rect.height*.35)
            table_rules=[r[1] for r in rules if top<r[1]<body_start and r[0]<middle<r[2] and r[2]-r[0]>width*.65]
            if not table_rules:continue
            numbers=[w for l in lines for w in words_for(l) if caption['bbox'][3]<w['bbox'][1]<body_start and re.fullmatch(r'[-+]?\d+(?:\.\d+)?%?',w['text'])]
            if len(numbers)<12 or len(bands(numbers))<3:continue
            if max(w['bbox'][2] for w in numbers)-min(w['bbox'][0] for w in numbers)<width*.55:continue
            bottom=max(w['bbox'][3] for w in numbers)+2
            if bottom>body_start-4:continue
            intervals.append((top,bottom));wide_tables.append([top,bottom])
    # A floating full-width image may be tiled into many adjacent image objects.
    images=[fitz.Rect(i['bbox'])*page.rotation_matrix for i in page.get_image_info()]
    image_groups=[]
    for box in sorted(images,key=lambda b:b.y0):
        if box.width<20 or box.height<20 or box.get_area()>page.rect.get_area()*.65:continue
        matches=[g for g in image_groups if min(g.y1,box.y1)-max(g.y0,box.y0)>min(g.height,box.height)*.5 and
                 max(g.x0,box.x0)-min(g.x1,box.x1)<width*.06]
        if matches:matches[0]|=box
        else:image_groups.append(fitz.Rect(box))
    intervals.extend((b.y0,b.y1) for b in image_groups if b.width>width*.65 and b.x0<middle<b.x1)
    info=dict(version=VERSION,columns=2,gutter=middle,spanning_bands=_merge_intervals(intervals),
              full_width_numeric_tables=len(wide_tables),
              evidence=dict(prose_lines=len(prose),bilateral_support=-min(candidates)[0] if candidates else 0,
                            document_supported=inherited))
    ordered=[]
    from .document import protected_kind
    for block_index,block in enumerate(blocks):
        if block.get('type')!=0:continue
        spans=[s for l in block['lines'] for s in l['spans']]
        # Keep a code/formula block intact, including its font/style runs.
        if block.get('academic_preserve') or protected_kind('\n'.join(text(l) for l in block['lines']),spans):
            item=copy.deepcopy(block);item['_atomic']=True;ordered.append(item);continue
        for line in block['lines']:
            side=_side(line,middle,width)
            if side==3 and abs(line.get('dir',(1,0))[1])<.1 and not line.get('rotation'):
                from .regions import combine_words
                pieces=[combine_words(line,[w for w in words_for(line) if ((w['bbox'][0]+w['bbox'][2])/2>=middle)==bool(i)]) for i in (0,1)]
            else:pieces=[copy.deepcopy(line)]
            for piece in pieces:
                box=piece['bbox'];value=text(piece)
                if abs(piece.get('dir',(1,0))[1])>.8 and (box[2]<width*.08 or box[0]>width*.92):
                    piece['layout_kind']='page_marker'
                elif (len(value)<140 and value.isupper() and len(re.findall('[A-Z]',value))>=5 and _prose(piece,page)):
                    piece['layout_kind']='heading'
                elif any(re.fullmatch(r'\d+(?:\.\d+)*',text(other)) and abs(other['bbox'][1]-box[1])<2 and
                         0<box[0]-other['bbox'][2]<25 for other in lines):
                    piece['layout_kind']='heading'
                ordered.append(dict(type=0,bbox=piece['bbox'],lines=[piece]))
    # Exact recognized-word counts, including formula/code and float labels.
    tokens=lambda bs:Counter(w['text'] for b in bs for l in b['lines'] for w in words_for(l))
    if tokens(blocks)!=tokens(ordered):return blocks,None
    ordered.sort(key=lambda b:_key(b['bbox'],info))
    for i,block in enumerate(ordered):
        key=_key(block['bbox'],info);side=0 if block['bbox'][2]<=middle else (1 if block['bbox'][0]>=middle else 2)
        block['flow_order']=i
        for line in block['lines']:
            line['flow_order']=i;line['flow_group']=(key[0],key[1])
            if key[0]%2==0 and side<2:
                line['column_index']=side
                line['column_bbox']=[0 if side==0 else middle,0,middle if side==0 else width,page.rect.height]
    return ordered,info
