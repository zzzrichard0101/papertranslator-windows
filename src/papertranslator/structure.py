"""Evidence-bounded prose tables and two-column reading order."""
import copy
import re
import statistics

import pymupdf as fitz

VERSION = 'captioned-prose-and-two-columns-1'


def text(line):
    return ''.join(s['text'] for s in line['spans']).strip()


def words_for(line):
    return line.get('layout_words') or []


def bands(words):
    """Group words by baseline centres, not overlapping invisible-font tops."""
    result=[]
    height=statistics.median(w['bbox'][3]-w['bbox'][1] for w in words) if words else 10
    for word in sorted(words,key=lambda w:((w['bbox'][1]+w['bbox'][3])/2,w['bbox'][0])):
        y=(word['bbox'][1]+word['bbox'][3])/2
        if result and abs(y-statistics.median((w['bbox'][1]+w['bbox'][3])/2 for w in result[-1]))<height*.4:
            result[-1].append(word)
        else:result.append([word])
    return result


def horizontal_rules(page):
    """Find long native or slightly skewed scan rules in display coordinates."""
    found=[]
    matrix=page.rotation_matrix
    for drawing in page.get_drawings():
        for item in drawing['items']:
            if item[0]!='l':continue
            a,b=item[1]*matrix,item[2]*matrix
            if abs(a.x-b.x)>page.rect.width*.45 and abs(a.y-b.y)<abs(a.x-b.x)*.03:
                found.append((min(a.x,b.x),(a.y+b.y)/2,max(a.x,b.x),(b.y-a.y)/(b.x-a.x)))
    if found:return found
    import cv2
    import numpy as np
    scale=min(1.5,1800/max(page.rect.width,page.rect.height))
    pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),colorspace=fitz.csGRAY,alpha=False)
    gray=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width)
    ink=cv2.threshold(gray,190,255,cv2.THRESH_BINARY_INV)[1]
    lines=cv2.HoughLinesP(ink,1,np.pi/1800,threshold=max(40,int(pix.width*.15)),
                         minLineLength=int(pix.width*.45),maxLineGap=22)
    for x0,y0,x1,y1 in ([] if lines is None else lines.reshape(-1,4)):
        if abs(y1-y0)<abs(x1-x0)*.03 and x1!=x0:
            found.append((min(x0,x1)/scale,(y0+y1)/2/scale,max(x0,x1)/scale,(y1-y0)/(x1-x0)))
    return found


def prose_tables(page,blocks):
    """Require caption, distinct headers, persistent gutter and a bottom rule.

    Only two-column prose tables with repeated aligned row labels qualify.
    Unknown row structure is left intact instead of guessing cell membership.
    """
    lines=[l for b in blocks for l in b.get('lines',[])]
    captions=[l for l in lines if re.match(r'^\s*Table\s+\d',text(l),re.I)]
    if not captions:return []
    rules=horizontal_rules(page);regions=[]
    for caption in captions:
        below=[l for l in lines if caption['bbox'][3]<=l['bbox'][1]<caption['bbox'][3]+32]
        candidates=[w for l in below for w in words_for(l)]
        header_bands=bands(candidates)
        if not header_bands:continue
        header=sorted(header_bands[0],key=lambda w:w['bbox'][0])
        gaps=[(b['bbox'][0]-a['bbox'][2],a['bbox'][2],b['bbox'][0]) for a,b in zip(header,header[1:])]
        if not gaps:continue
        width,g0,g1=max(gaps)
        letters=''.join(w['text'] for w in header)
        header_lines=[l for l in below if abs(l['bbox'][1]-min(w['bbox'][1] for w in header))<6]
        bold=any(s.get('flags',0)&16 for l in header_lines for s in l['spans'])
        if width<page.rect.width*.06 or not (letters.isupper() or bold):continue
        top=min(w['bbox'][1] for w in header)
        endings=[r for r in rules if r[1]>top+60 and r[0]<=header[0]['bbox'][0]+page.rect.width*.07 and r[2]>=header[-1]['bbox'][2]-10
                 and not any(w['bbox'][1]-2<r[1]<w['bbox'][3]+2 for l in lines for w in words_for(l))]
        if not endings:continue
        bottom_rule=min(endings,key=lambda r:r[1]);bottom=bottom_rule[1]-1
        words=[w for l in lines for w in words_for(l) if top-1<=(w['bbox'][1]+w['bbox'][3])/2<bottom]
        if len(words)<25:continue
        # The header gap locates a candidate region. All body word boxes then
        # determine the actual gutter, so long labels stay in their left cell.
        scores=[]
        for middle in range(int(g0)+3,int(g1)-2):
            crossing=sum(w['bbox'][0]<middle<w['bbox'][2] for w in words)
            clearance=min(min(abs(middle-w['bbox'][0]),abs(middle-w['bbox'][2])) for w in words)
            scores.append((crossing,-clearance,middle))
        if not scores:continue
        crossing,_,middle=min(scores)
        if crossing:continue
        left=[w for w in words if (w['bbox'][0]+w['bbox'][2])/2<middle]
        right=[w for w in words if (w['bbox'][0]+w['bbox'][2])/2>=middle]
        lb=bands(left);rb=bands(right)
        if len(lb)<4 or len(rb)<5:continue
        slope=bottom_rule[3]
        centres=lambda band:statistics.median((w['bbox'][1]+w['bbox'][3])/2 for w in band)
        body=[b for b in lb if centres(b)>max(w['bbox'][3] for w in header)+1]
        if len(body)<3:continue
        starts=[min(w['bbox'][0] for w in b)-slope*centres(b) for b in body]
        margin=min(starts)
        row_bands=[b for b,x in zip(body,starts) if x<=margin+3]
        row_starts=[centres(b) for b in row_bands]
        if len(row_starts)<3:continue
        # Repeated label indentation is essential: do not turn every physical
        # line of a wrapped left-cell paragraph into a new logical table row.
        if len(body)>len(row_starts) and max(starts)-margin<4:continue
        all_centres=sorted(centres(b) for b in bands(words))
        boundaries=[top-1]
        for y,band in zip(row_starts,row_bands):
            before=max((v for v in all_centres if v<y-3),default=top)
            # Anchor text near its first printed row, rather than at the centre
            # of a large blank gap above it. Word centres still determine cell
            # membership when invisible OCR font boxes overlap neighbouring rows.
            boundaries.append(min(y-1,max(before+.5,min(w['bbox'][1] for w in band)-1)))
        boundaries.append(bottom)
        if any(b-a<5 for a,b in zip(boundaries,boundaries[1:])):continue
        left_edge=min(w['bbox'][0] for w in words)-2
        right_edge=max(w['bbox'][2] for w in words)+2
        if middle-left_edge<page.rect.width*.15 or right_edge-middle<page.rect.width*.2:continue
        cells=[[a,y,b,end] for y,end in zip(boundaries,boundaries[1:])
               for a,b in [(left_edge,middle),(middle,right_edge)]]
        regions.append(dict(bbox=[left_edge,top-1,right_edge,bottom],cells=cells,
            method='captioned_prose_columns',evidence=dict(caption=text(caption),gutter=middle,
                rows=len(boundaries)-1,bottom_rule=list(bottom_rule),crossing_word_boxes=crossing)))
    return regions


def two_columns(page,blocks):
    """Order verified column bands; retain full-width headings between bands."""
    lines=[l for b in blocks for l in b.get('lines',[])]
    if any(l.get('rotation') or l.get('diagram_label') or l.get('layout_kind') or
           re.match(r'^\s*(?:Table|Figure|Box)\s+\d',text(l),re.I) or
           any(s.get('flags',0)&8 or re.search(r'courier|consolas|mono',s.get('font',''),re.I) for s in l['spans']) for l in lines):
        return blocks,None
    # Native PDF lines and OCR lines have exact word coordinates. A row may
    # contain both columns; never split a word at an inferred gutter.
    words=[w for l in lines for w in words_for(l) if page.rect.height*.1< w['bbox'][1]<page.rect.height*.9]
    if len(words)<50:return blocks,None
    def spans_gutter(line,x):
        words=words_for(line)
        if not words:return True
        if any(w['bbox'][0]-3<x<w['bbox'][2]+3 for w in words):return True
        left=[w['bbox'][2] for w in words if w['bbox'][2]<x]
        right=[w['bbox'][0] for w in words if w['bbox'][0]>x]
        return bool(left and right and min(right)-max(left)<page.rect.width*.035)
    step=max(1,int(page.rect.width/300));candidates=[]
    for x in range(int(page.rect.width*.35),int(page.rect.width*.65),step):
        crossed=[l for l in lines if spans_gutter(l,x)]
        if len(crossed)>max(3,len(lines)*.18):continue
        eligible=[w for l in lines if l not in crossed for w in words_for(l)]
        left=[w for w in eligible if w['bbox'][2]<x];right=[w for w in eligible if w['bbox'][0]>x]
        if not left or not right:continue
        # A short label column beside long explanations is not ordinary
        # newspaper-style prose. Without table evidence, retain its row order.
        if min(len(left),len(right))<max(len(left),len(right))*.55:continue
        gap=min(w['bbox'][0] for w in right)-max(w['bbox'][2] for w in left)
        if gap<page.rect.width*.035:continue
        lb=bands(left);rb=bands(right)
        if min(len(lb),len(rb))<6:continue
        breadth=lambda rows:statistics.median(max(w['bbox'][2] for w in row)-min(w['bbox'][0] for w in row) for row in rows)
        if min(breadth(lb),breadth(rb))<page.rect.width*.2:continue
        support=sum(any(abs(statistics.median(w['bbox'][1] for w in a)-statistics.median(w['bbox'][1] for w in b))<12 for b in rb) for a in lb)
        if support<5:continue
        candidates.append((len(crossed),-gap,abs(x-page.rect.width/2),x))
    if not candidates:return blocks,None
    middle=min(candidates)[-1]
    spanning=[l for l in lines if spans_gutter(l,middle)]
    # Overlapping spanning material is ambiguous (e.g. a figure within columns).
    ordinary=[l for l in lines if l not in spanning]
    if any(any(l['bbox'][1]<s['bbox'][3]-2 and l['bbox'][3]>s['bbox'][1]+2 for l in ordinary) for s in spanning):
        return blocks,None
    separators=sorted(spanning,key=lambda l:l['bbox'][1]);ordered=[];cursor=-1;group=0
    for separator in [*separators,None]:
        end=separator['bbox'][1] if separator else page.rect.height+1
        band=[l for l in ordinary if cursor<=l['bbox'][1]<end]
        for side in [0,1]:
            pieces=[]
            for line in band:
                items=[w for w in words_for(line) if ((w['bbox'][0]+w['bbox'][2])/2>=middle)==bool(side)]
                if items:
                    from .regions import combine_words
                    pieces.append(combine_words(line,items))
            for line in sorted(pieces,key=lambda l:(l.get('reading_top',l['bbox'][1]),l['bbox'][0])):
                line['flow_group']=group;ordered.append(line)
            group+=1
        if separator:
            item=copy.deepcopy(separator);item['flow_group']=group;ordered.append(item);group+=1
            cursor=separator['bbox'][3]-2
    # A strict word multiset check prevents source loss/duplication on odd boxes.
    from collections import Counter
    expected=Counter(w['text'] for l in lines for w in words_for(l))
    actual=Counter(w['text'] for l in ordered for w in words_for(l))
    if expected!=actual:return blocks,None
    output=[]
    for index,line in enumerate(ordered):
        line['flow_order']=index
        output.append(dict(type=0,bbox=line['bbox'],lines=[line],flow_order=index))
    return output,dict(version=VERSION,columns=2,gutter=middle,spanning_lines=len(spanning))
