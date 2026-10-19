"""Isolate numbered questionnaire text from adjacent Yes/No response grids."""
import re


def recover(pixels,initial,recognize):
    import cv2
    import numpy as np
    text=' '.join(r[1] for r in initial)
    if (len(re.findall(r'\b\d+\.\d+\b',text))<2 and text.count('?')<5) or not re.search(r'\bYes\b',text) or not re.search(r'\bKnow\b',text):return initial
    gray=cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY)
    binary=cv2.threshold(gray,180,255,cv2.THRESH_BINARY_INV)[1]
    vertical=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((100,1),np.uint8))
    count=np.sum(vertical>0,axis=0);xs=np.flatnonzero(count>pixels.shape[0]*.12)
    clusters=[]
    for x in xs:
        if clusters and x-clusters[-1][-1]<8:clusters[-1].append(int(x))
        else:clusters.append([int(x)])
    columns=[int(np.median(c)) for c in clusters if np.median(c)>pixels.shape[1]*.55]
    if len(columns)<4:return initial
    columns=columns[-4:];grid_left=columns[0]
    if columns[-1]-grid_left>pixels.shape[1]*.35:return initial
    left_rows=recognize(pixels[:,:max(1,grid_left-6)].copy())
    if len(re.findall(r'\b\d+\.\d+\b',' '.join(r[1] for r in left_rows)))<2:return initial
    def box(row):return [min(p[0] for p in row[0]),min(p[1] for p in row[0]),max(p[0] for p in row[0]),max(p[1] for p in row[0])]
    same_lines=[]
    for row in sorted(left_rows,key=lambda r:(box(r)[1],box(r)[0])):
        if same_lines and abs(box(row)[1]-box(same_lines[-1][-1])[1])<3:same_lines[-1].append(row)
        else:same_lines.append([row])
    left_rows=[]
    for line in same_lines:
        line.sort(key=lambda r:box(r)[0]);a=min(box(r)[0] for r in line);b=min(box(r)[1] for r in line)
        c=max(box(r)[2] for r in line);d=max(box(r)[3] for r in line)
        left_rows.append([[(a,b),(c,b),(c,d),(a,d)],' '.join(r[1] for r in line),None,dict(words=[])])
    groups=[]
    for row in left_rows:
        value=row[1].strip().strip('| ').strip();b=box(row)
        numbered=bool(re.match(r'^\d+(?:\.\d+)+(?:\s|(?=[A-Za-z])|$)|^\d{1,2}\s+(?:Is|Does|What|Are|Was|Were|Can|Do|If)\b',value))
        continuation=(groups and groups[-1]['question'] and not numbered and b[1]-groups[-1]['bbox'][3]<30 and
                      b[0]>groups[-1]['bbox'][0]+10 and len(value)>10 and not value.startswith(('Please','Appendix')))
        if continuation:
            g=groups[-1];g['text']+=' '+value;g['bbox']=[min(g['bbox'][0],b[0]),g['bbox'][1],max(g['bbox'][2],b[2]),max(g['bbox'][3],b[3])]
        else:groups.append(dict(text=value,bbox=b,question=numbered,row=row))
    result=[]
    for g in groups:
        a,b,c,d=g['bbox'];meta=dict(g['row'][3]) if len(g['row'])>3 else {}
        if g['question']:meta.update(layout_kind='form_question',region_bbox=[a,b,grid_left-5,d+3],words=[])
        result.append([[(a,b),(c,b),(c,d),(a,d)],g['text'],g['row'][2],meta])
    original_header_words=[w for row in initial if len(row)>3 for w in row[3].get('words',[])]
    # Reread each response column separately so a question can never absorb
    # a grid heading; preserve the original empty response cells and strokes.
    for col,(a,c) in enumerate(zip(columns,columns[1:])):
        crop=pixels[:,a+3:c-3].copy()
        rows=recognize(crop)
        # Page-segmentation can reject a column containing only repeated
        # two-letter "No" labels. Original word boxes remain valid evidence.
        for word in original_header_words:
            if not re.fullmatch(['Yes','No',r"Don['’]?t"][col],word['text'],re.I):continue
            x0,y0,x1,y1=word['bbox']
            if not a<(x0+x1)/2<c:continue
            if any(abs(box(row)[1]-y0)<10 for row in rows):continue
            rows.append([[(x0-a-3,y0),(x1-a-3,y0),(x1-a-3,y1),(x0-a-3,y1)],word['text'],None,{}])
        for row in rows:
            value=row[1].strip()
            if not re.search(r'Yes|No|Don|Know',value,re.I):continue
            x0,y0,x1,y1=box(row);label=['Yes','No',"Don't Know"][col]
            # The two-line Don't Know label is emitted only once per header.
            if col==2 and 'Know' in value and not re.search(r'Don',value,re.I):continue
            bottom=y1+(y1-y0)*1.4 if col==2 and not re.search(r'Know',value,re.I) else y1
            result.append([[(a+3,y0),(c-3,y0),(c-3,bottom),(a+3,bottom)],label,None,
                           dict(layout_kind='table_cell',region_bbox=[a+3,y0,c-3,bottom+2],words=[])])
    return sorted(result,key=lambda r:(min(p[1] for p in r[0]),min(p[0] for p in r[0])))
