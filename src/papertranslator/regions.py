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
        bw=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,21,12)
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
                if len(cells)>=4 and len(columns)>=2 and len(rows)>=2:
                    regions.append(dict(bbox=[x/scale,y/scale,(x+width)/scale,(y+height)/scale],
                                        cells=sorted(cells,key=lambda c:(c[1],c[0])),method='scanned_ruled'))
    return regions


def separate_tables(page,blocks,scanned=False):
    regions=table_regions(page,scanned)
    if not regions:regions=borderless_numeric_table(page,blocks)
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
                def combine(items):
                    items=sorted(items,key=lambda w:w['bbox'][0]);boxes=[w['bbox'] for w in items]
                    box=[min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)]
                    return {**line,'bbox':box,'layout_words':items,'spans':[{**line['spans'][0],'text':' '.join(w['text'] for w in items),'bbox':box}]}
                for i,items in buckets.items():cells[i]['lines'].append(combine(items))
                if outside:kept.append(combine(outside))
                continue
            rect=fitz.Rect(line['bbox'])
            matches=[c for c in cells if fitz.Rect(c['bbox']).contains((rect.tl+rect.br)/2) and
                     (fitz.Rect(c['bbox'])&rect).get_area()>=rect.get_area()*.65]
            if len(matches)==1:matches[0]['lines'].append(copy.deepcopy(line))
            else:kept.append(line)
        if kept:remaining.append({**block,'lines':kept})
    return remaining,[c for c in cells if c['lines']],regions


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
