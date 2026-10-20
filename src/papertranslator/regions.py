"""Conservative ruled-table regions and traceable source line geometry."""
import copy
import re,statistics
import pymupdf as fitz


def table_regions(page,scanned=False):
    regions=[]
    if not scanned:
        try:
            for table in page.find_tables(strategy='lines_strict').tables:
                cells=[list(c) for c in table.cells if c]
                if table.row_count>=2 and table.col_count>=2 and len(cells)>=4:
                    regions.append(dict(bbox=list(table.bbox),cells=cells,method='native_ruled'))
        except (ValueError,RuntimeError):pass
    if not regions and scanned:
        import cv2
        import numpy as np
        # Rendering is bounded independently of OCR, and uses display coordinates.
        scale=min(1.5,1800/max(page.rect.width,page.rect.height))
        pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),colorspace=fitz.csGRAY,alpha=False)
        gray=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width)
        polarity=cv2.THRESH_BINARY if np.mean(gray<90)>.55 else cv2.THRESH_BINARY_INV
        bw=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,polarity,21,12)
        horizontal=cv2.morphologyEx(bw,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(max(25,pix.width//25),1)))
        vertical=cv2.morphologyEx(bw,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(1,max(20,pix.height//35))))
        mask=cv2.dilate(horizontal|vertical,np.ones((3,3),np.uint8))
        contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is not None:
            for index,contour in enumerate(contours):
                x,y,width,height=cv2.boundingRect(contour)
                if width<pix.width*.15 or height<30:continue
                children=[i for i,h in enumerate(hierarchy[0]) if h[3]==index]
                cells=[]
                for child in children:
                    cx,cy,cw,ch=cv2.boundingRect(contours[child])
                    if cw>=20 and ch>=10 and cv2.contourArea(contours[child])>=cw*ch*.75:
                        cells.append([cx/scale,cy/scale,(cx+cw)/scale,(cy+ch)/scale])
                columns={round((c[0]+c[2])/2/12) for c in cells}
                rows={round((c[1]+c[3])/2/12) for c in cells}
                coverage=sum((c[2]-c[0])*(c[3]-c[1])*scale*scale for c in cells)/max(1,width*height)
                if len(cells)>=4 and len(columns)>=2 and len(rows)>=2 and coverage>=.45:
                    regions.append(dict(bbox=[x/scale,y/scale,(x+width)/scale,(y+height)/scale],
                                        cells=sorted(cells,key=lambda c:(c[1],c[0])),method='scanned_ruled'))
    return regions


def combine_words(line,items):
    items=sorted(items,key=lambda w:w['bbox'][0]);boxes=[w['bbox'] for w in items]
    box=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
    return {**line,'bbox':box,'layout_words':copy.deepcopy(items),
            'spans':[{**line['spans'][0],'text':' '.join(w['text'] for w in items),'bbox':box}]}


def separate_tables(page,blocks,scanned=False,structured=False):
    if any(l.get('rotation') or l.get('diagram_label') or l.get('layout_kind') for b in blocks for l in b.get('lines',[])):return blocks,[],[]
    regions=table_regions(page,scanned)
    if not regions:regions=borderless_numeric_table(page,blocks)
    if not regions:regions=borderless_comparison_table(page,blocks)
    if not regions and structured:
        from .structure import prose_tables
        regions=prose_tables(page,blocks)
    cells=[]
    for table_index,table in enumerate(regions):
        for cell_index,box in enumerate(table['cells']):
            cells.append(dict(bbox=box,lines=[],table=table_index,cell=cell_index))
    remaining=[]
    for block in blocks:
        if block.get('type')!=0:
            remaining.append(block);continue
        kept=[]
        for line in block['lines']:
            words=line.get('layout_words',[])
            if words and any((fitz.Rect(t['bbox'])&fitz.Rect(line['bbox'])).get_area()>0 for t in regions):
                buckets={};outside=[]
                for word in words:
                    rect=fitz.Rect(word['bbox']);center=(rect.tl+rect.br)/2
                    matches=[i for i,c in enumerate(cells) if fitz.Rect(c['bbox']).contains(center)]
                    if len(matches)==1:buckets.setdefault(matches[0],[]).append(word)
                    else:outside.append(word)
                for i,items in buckets.items():cells[i]['lines'].append(combine_words(line,items))
                if outside:kept.append(combine_words(line,outside))
                continue
            rect=fitz.Rect(line['bbox'])
            matches=[c for c in cells if fitz.Rect(c['bbox']).contains((rect.tl+rect.br)/2) and
                     (fitz.Rect(c['bbox'])&rect).get_area()>=rect.get_area()*.65]
            if len(matches)==1:matches[0]['lines'].append(copy.deepcopy(line))
            else:kept.append(line)
        if kept:remaining.append({**block,'lines':kept})
    return remaining,[c for c in cells if c['lines']],regions


def borderless_comparison_table(page,blocks):
    """Two-column qualitative/quantitative comparisons with a caption and source."""
    lines=[l for b in blocks if b.get('type')==0 for l in b['lines']]
    text=lambda l:' '.join(s['text'] for s in l['spans']).strip()
    regions=[]
    for left_header in [l for l in lines if text(l).casefold()=='qualitative']:
        right=next((l for l in lines if text(l).casefold()=='quantitative' and abs(l['bbox'][1]-left_header['bbox'][1])<4),None)
        if not right:continue
        y=left_header['bbox'][1]
        captions=[l for l in lines if re.match(r'^Table\s+\d',text(l),re.I) and 0<y-l['bbox'][1]<35]
        endings=[l for l in lines if re.match(r'^Source\s*:',text(l),re.I) and y<l['bbox'][1]<y+page.rect.height*.65]
        if not captions or not endings:continue
        end=min(l['bbox'][1] for l in endings);left=left_header['bbox'][0]-2;middle=right['bbox'][0]-8
        words=[w for l in lines if y<=l['bbox'][1]<end for w in l.get('layout_words',[])]
        if not words:continue
        left_words=sorted([w for w in words if w['bbox'][0]<middle],key=lambda w:(w['bbox'][1],w['bbox'][0]))
        bands=[]
        for word in left_words:
            if bands and abs(word['bbox'][1]-bands[-1][0]['bbox'][1])<3:bands[-1].append(word)
            else:bands.append([word])
        starts=[];multiline=False
        for band in bands:
            value=' '.join(w['text'] for w in sorted(band,key=lambda w:w['bbox'][0]));x=min(w['bbox'][0] for w in band)
            if len(value.strip())<3:continue
            new=not starts or value.startswith(('Methods:','Research genres:')) or (not multiline and x<left+8)
            if new:starts.append(band[0]['bbox'][1]-1)
            if value.startswith(('Methods:','Research genres:')):multiline=True
        if len(starts)<5:continue
        # A candidate must contain several substantive rows in both columns.
        if sum(w['bbox'][0]>=middle and len(w['text'])>=3 for w in words)<5:continue
        right_edge=min(page.rect.width-12,max(w['bbox'][2] for w in words)+3)
        cells=[[a,top,b,bottom] for top,bottom in zip(starts,[*starts[1:],end-2]) for a,b in [(left,middle),(middle,right_edge)]]
        regions.append(dict(bbox=[left,starts[0],right_edge,end-2],cells=cells,method='captioned_comparison_columns'))
    return regions


def borderless_numeric_table(page,blocks):
    """Require a Table caption, a clear end boundary and repeated score columns."""
    lines=[l for b in blocks if b.get('type')==0 for l in b['lines']]
    text=lambda l:' '.join(s['text'] for s in l['spans']).strip()
    captions=[l for l in lines if re.match(r'^Table\s+\d',text(l),re.I)]
    regions=[]
    for caption in captions:
        top=caption['bbox'][3]
        ends=[l['bbox'][1] for l in lines if re.match(r'^Source\s*:',text(l),re.I) and top<l['bbox'][1]<top+page.rect.height*.6]
        if not ends:
            ends=[l['bbox'][1] for l in lines if top+20<l['bbox'][1]<top+page.rect.height*.6 and
                  len(text(l).split())>=8 and l['bbox'][2]-l['bbox'][0]>=page.rect.width*.7]
        if not ends:continue
        bottom=min(ends);body=[l for l in lines if l['bbox'][1]>=top and l['bbox'][3]<=bottom+1]
        # Word boxes retain the wide gaps that a flattened OCR line loses.
        items=[w for l in body for w in (l.get('layout_words') or [dict(text=text(l),bbox=l['bbox'])])]
        numbers=[w for w in items if re.fullmatch(r'[-+]?\d+(?:\.\d+)?%?',w['text'])]
        clusters=[]
        for word in sorted(numbers,key=lambda w:(w['bbox'][0]+w['bbox'][2])/2):
            center=(word['bbox'][0]+word['bbox'][2])/2
            if clusters and abs(center-statistics.mean(clusters[-1]))<page.rect.width*.025:clusters[-1].append(center)
            else:clusters.append([center])
        columns=[statistics.mean(c) for c in clusters if len(c)>=3]
        if not 2<=len(columns)<=10:continue
        left=min(w['bbox'][0] for w in items);right=min(page.rect.width-8,columns[-1]+(columns[-1]-columns[-2])/2)
        edges=[left,columns[0]-(columns[1]-columns[0])/2]+[(a+b)/2 for a,b in zip(columns,columns[1:])]+[right]
        if edges[1]-left<page.rect.width*.12:continue
        bands=[]
        for word in sorted(items,key=lambda w:(w['bbox'][1]+w['bbox'][3])/2):
            center=(word['bbox'][1]+word['bbox'][3])/2
            if bands and abs(center-statistics.mean((v['bbox'][1]+v['bbox'][3])/2 for v in bands[-1]))<4:bands[-1].append(word)
            else:bands.append([word])
        if sum(sum(re.fullmatch(r'[-+]?\d+(?:\.\d+)?%?',w['text']) is not None for w in row)>=2 for row in bands)<3:continue
        centers=[statistics.mean((w['bbox'][1]+w['bbox'][3])/2 for w in row) for row in bands]
        rows=[min(w['bbox'][1] for w in bands[0])-1]+[(a+b)/2 for a,b in zip(centers,centers[1:])]+[max(w['bbox'][3] for w in bands[-1])+1]
        cells=[]
        for index,row in enumerate(bands):
            across=any((w['bbox'][0]+w['bbox'][2])/2>edges[1] for w in row)
            boundaries=edges if across else [left,right]
            cells.extend([[a,rows[index],b,rows[index+1]] for a,b in zip(boundaries,boundaries[1:])])
        regions.append(dict(bbox=[left,rows[0],right,rows[-1]],cells=cells,method='captioned_numeric_columns'))
    return regions


def slice_fragment(fragment,start,end):
    """Clip a logical segment to just the original lines it covers."""
    result={**fragment,'source':fragment['source'][start:end]}
    lines=[line for line in fragment.get('layout_lines',[]) if line['end']>start and line['start']<end]
    if lines:
        result['bbox']=[min(l['bbox'][0] for l in lines),min(l['bbox'][1] for l in lines),
                        max(l['bbox'][2] for l in lines),max(l['bbox'][3] for l in lines)]
        result['layout_lines']=[{**l,'start':max(0,l['start']-start),'end':min(end-start,l['end']-start)} for l in lines]
    return result
