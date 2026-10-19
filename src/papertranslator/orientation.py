"""Recover sideways diagram labels without replacing the graph with prose."""
import re,base64


def quality(rows):
    text=' '.join(r[1] for r in rows);letters=sum(c.isalpha() for c in text)
    words=re.findall(r'\b[A-Za-z]{3,}\b',text)
    return sum(map(len,words))/max(1,letters),sum(map(len,words))


def recover_upright(pixels,initial,recognize):
    """Split separated closed diagram nodes; a contiguous table is ineligible."""
    import cv2
    import numpy as np
    text=' '.join(r[1] for r in initial)
    if re.search(r'\bYes\b',text) and re.search(r'\bKnow\b',text):return initial
    gray=cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY)
    binary=cv2.threshold(gray,180,255,cv2.THRESH_BINARY_INV)[1]
    contours,_=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    shapes=[]
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        if min(w,h)>=30 and 1800<w*h<binary.size*.04 and cv2.contourArea(contour)>w*h*.88:
            box=(x,y,x+w,y+h)
            if not any(abs(x-b[0])+abs(y-b[1])+abs(x+w-b[2])+abs(y+h-b[3])<25 for b in shapes):shapes.append(box)
    def separated(a,b):
        overlap_x=min(a[2],b[2])-max(a[0],b[0]);overlap_y=min(a[3],b[3])-max(a[1],b[1])
        return (overlap_x>20 and max(a[1],b[1])-min(a[3],b[3])>pixels.shape[0]*.015 or
                overlap_y>20 and max(a[0],b[0])-min(a[2],b[2])>pixels.shape[1]*.015)
    if len(shapes)<4 or sum(any(separated(a,b) for b in shapes if a!=b) for a in shapes)<4:return initial
    def adjacent(a,b):
        overlap_x=min(a[2],b[2])-max(a[0],b[0]);overlap_y=min(a[3],b[3])-max(a[1],b[1])
        return (overlap_x>20 and 0<=max(a[1],b[1])-min(a[3],b[3])<14 or
                overlap_y>20 and 0<=max(a[0],b[0])-min(a[2],b[2])<14)
    if sum(adjacent(a,b) for i,a in enumerate(shapes) for b in shapes[:i])>=2:return initial
    guard=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((1,60),np.uint8))|cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((60,1),np.uint8))
    guard=cv2.dilate(guard,np.ones((3,3),np.uint8))
    result=[];accepted=[]
    for x0,y0,x1,y1 in shapes:
        crop=pixels[y0+5:y1-5,x0+5:x1-5]
        rows=recognize(cv2.copyMakeBorder(crop,16,16,16,16,cv2.BORDER_CONSTANT,value=(255,255,255)))
        text=' '.join(r[1] for r in rows).strip()
        if not re.search(r'[A-Za-z]{3,}',text) or quality(rows)[0]<.75:continue
        erase=np.zeros(binary.shape,np.uint8);erase[y0+5:y1-5,x0+5:x1-5]=255;erase[guard>0]=0
        rgba=np.full((y1-y0,x1-x0,4),255,np.uint8);rgba[:,:,3]=erase[y0:y1,x0:x1]
        patch=dict(bbox=[x0,y0,x1,y1],png=base64.b64encode(cv2.imencode('.png',rgba)[1]).decode('ascii'))
        result.append([[(x0+5,y0+5),(x1-5,y0+5),(x1-5,y1-5),(x0+5,y1-5)],text,None,
                       dict(diagram_label=True,region_bbox=[x0+5,y0+5,x1-5,y1-5],erase_patch=patch,words=[])])
        accepted.append((x0,y0,x1,y1))
    if len(accepted)<4:return initial
    for row in initial:
        words=row[3].get('words',[]) if len(row)>3 else []
        if not words:
            # Never silently discard an OCR row with no recoverable word boxes.
            result.append(row);continue
        outside=[w for w in words if not any(b[0]<(w['bbox'][0]+w['bbox'][2])/2<b[2] and b[1]<(w['bbox'][1]+w['bbox'][3])/2<b[3] for b in accepted)]
        if not outside:continue
        a=min(w['bbox'][0] for w in outside);b=min(w['bbox'][1] for w in outside)
        c=max(w['bbox'][2] for w in outside);d=max(w['bbox'][3] for w in outside)
        result.append([[(a,b),(c,b),(c,d),(a,d)],' '.join(w['text'] for w in outside),row[2],{**row[3],'words':outside}])
    return sorted(result,key=lambda r:(min(p[1] for p in r[0]),min(p[0] for p in r[0])))


def recover(pixels,initial,recognize):
    import cv2
    import numpy as np
    initial_score,initial_letters=quality(initial)
    if len(initial)<15 or initial_score>=.55:return initial
    candidates=[]
    for turn in [1,3]:
        upright=np.rot90(pixels,turn).copy();gray=cv2.cvtColor(upright,cv2.COLOR_BGR2GRAY)
        binary=cv2.threshold(gray,180,255,cv2.THRESH_BINARY_INV)[1]
        count,labels,stats,_=cv2.connectedComponentsWithStats(binary)
        cleaned=np.full_like(upright,255)
        for component in range(1,count):
            x,y,w,h,area=stats[component]
            if 3<=h<=55 and w<=300 and area<2500 and w/max(1,h)<20:
                crop=cleaned[y:y+h,x:x+w];original=upright[y:y+h,x:x+w]
                crop[labels[y:y+h,x:x+w]==component]=original[labels[y:y+h,x:x+w]==component]
        rows=recognize(cleaned);score,letters=quality(rows)
        if score>.7 and letters>max(80,initial_letters*1.6):candidates.append((score*letters,turn,rows,binary,upright))
    if not candidates:return initial
    _,turn,rows,binary,upright=max(candidates,key=lambda v:v[0])
    count,labels,stats,_=cv2.connectedComponentsWithStats(binary);letter_mask=np.zeros(binary.shape,np.uint8)
    for component in range(1,count):
        x,y,w,h,area=stats[component]
        if 3<=h<=55 and w<=300 and area<2500 and w/max(1,h)<20:
            region=letter_mask[y:y+h,x:x+w];region[labels[y:y+h,x:x+w]==component]=255
    contours,_=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    shapes=[];outlines={}
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour);area=w*h
        ratio=cv2.contourArea(contour)/max(1,area)
        shaped=ratio>.88 or (.65<ratio<.85 and .75<w/h<1.33) or (.4<ratio<.6 and .75<w/h<1.33)
        if min(w,h)>=40 and 1800<area<binary.size*.025 and shaped:
            box=(x,y,x+w,y+h);shapes.append(box);outlines[box]=contour
    groups={};height,width=pixels.shape[:2]
    for row in rows:
        words=row[3].get('words',[]) if len(row)>3 else []
        if not words:continue
        for word in words:
            if not (re.search(r'[A-Za-z]{3,}',word['text']) or word['text'].casefold() in {'in','of','on','to','by','as','at','or','an','is'} or re.search(r'\d',word['text'])):continue
            a,b,c,d=word['bbox'];cx=(a+c)/2;cy=(b+d)/2
            matches=[s for s in shapes if s[0]+2<cx<s[2]-2 and s[1]+2<cy<s[3]-2]
            shape=min(matches,key=lambda s:(s[2]-s[0])*(s[3]-s[1])) if matches else None
            if not shape and any(s[0]-12<cx<s[2]+12 and s[1]-12<cy<s[3]+12 for s in shapes):continue
            # Tesseract block IDs keep two lines of one outside-arrow label
            # together, while cells are grouped by the smallest enclosing shape.
            key=('shape',shape) if shape else ('line',id(row))
            groups.setdefault(key,[]).append(word)
    # Join only spatially adjacent outside labels, never a Tesseract block that
    # spans several diagram nodes and arrows.
    outside=[(k,v) for k,v in groups.items() if k[0]=='line']
    for key,words in outside:
        if key not in groups:continue
        box=[min(w['bbox'][0] for w in words),min(w['bbox'][1] for w in words),max(w['bbox'][2] for w in words),max(w['bbox'][3] for w in words)]
        for other,more in outside:
            if other==key or other not in groups:continue
            b=[min(w['bbox'][0] for w in more),min(w['bbox'][1] for w in more),max(w['bbox'][2] for w in more),max(w['bbox'][3] for w in more)]
            aligned=abs(box[0]-b[0])<20 or abs((box[0]+box[2]-b[0]-b[2])/2)<40
            similar=max(box[2]-box[0],b[2]-b[0])<min(box[2]-box[0],b[2]-b[0])*2.5
            stacked=-30<=b[1]-box[3]<25 and b[1]>box[1]+4 and aligned and similar and b[3]-box[1]<100
            same_line=min(box[3],b[3])-max(box[1],b[1])>min(box[3]-box[1],b[3]-b[1])*.5 and 0<=b[0]-box[2]<40
            if stacked or same_line:
                words.extend(more);box=[min(box[0],b[0]),min(box[1],b[1]),max(box[2],b[2]),max(box[3],b[3])];groups.pop(other)
    result=[]
    def map_point(x,y):return (width-y,x) if turn==1 else (y,height-x)
    for key,words in groups.items():
        text=' '.join(w['text'] for w in words)
        if key[0]=='shape':
            x0,y0,x1,y1=key[1];mask=np.zeros(binary.shape,np.uint8)
            cv2.drawContours(mask,[outlines[key[1]]],-1,255,cv2.FILLED)
            mask=cv2.erode(mask,np.ones((5,5),np.uint8));isolated=upright.copy();isolated[mask==0]=255
            crop=isolated[y0:y1,x0:x1]
            if crop.size:
                padded=cv2.copyMakeBorder(crop,16,16,16,16,cv2.BORDER_CONSTANT,value=(255,255,255))
                reread=recognize(padded)
                if quality(reread)[0]>.8:text=' '.join(r[1] for r in reread)
        text=' '.join(w for w in text.split() if len(re.sub(r'\W','',w))>=3 or w.casefold() in {'in','of','on','to','by','as','at','or','an','is'} or re.fullmatch(r'\d[\d.]*',w))
        if not re.search(r'\b[A-Za-z]{3,}\b',text):continue
        a=min(w['bbox'][0] for w in words);b=min(w['bbox'][1] for w in words)
        c=max(w['bbox'][2] for w in words);d=max(w['bbox'][3] for w in words)
        points=[map_point(x,y) for x,y in [(a,b),(c,b),(c,d),(a,d)]]
        # The empty interior of the source shape provides the translation box.
        region=key[1] if key[0]=='shape' else (a-5,b-6,c+5,d+6)
        if key[0]=='shape':
            x0,y0,x1,y1=region;ratio=cv2.contourArea(outlines[key[1]])/((x1-x0)*(y1-y0))
            if ratio<.88:
                inset=.22 if ratio>.65 else .24
                dx=(x1-x0)*inset;dy=(y1-y0)*inset;region=(x0+dx,y0+dy,x1-dx,y1-dy)
        rp=[map_point(x,y) for x,y in [(region[0]+3,region[1]+3),(region[2]-3,region[3]-3)]]
        region_box=[min(p[0] for p in rp),min(p[1] for p in rp),max(p[0] for p in rp),max(p[1] for p in rp)]
        # Transparent white pixels erase glyph components only. Rectangle masks
        # would cut the circle, diamond, box borders and adjacent arrow shafts.
        erase=np.zeros(binary.shape,np.uint8)
        if key[0]=='shape':
            shape_mask=np.zeros(binary.shape,np.uint8);cv2.drawContours(shape_mask,[outlines[key[1]]],-1,255,cv2.FILLED)
            erase=cv2.erode(shape_mask,np.ones((9,9),np.uint8))
        else:erase[max(0,int(b)-6):int(d)+7,max(0,int(a)-6):int(c)+7]=255
        horizontal=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((1,60),np.uint8))
        vertical=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((60,1),np.uint8))
        graph_guard=cv2.dilate(horizontal|vertical,np.ones((3,3),np.uint8))
        cv2.drawContours(graph_guard,list(outlines.values()),-1,255,5)
        erase[graph_guard>0]=0
        erase=np.rot90(erase,-turn).copy()
        yy,xx=np.where(erase>0)
        patch=None
        if len(xx):
            x0=int(xx.min());y0=int(yy.min());x1=int(xx.max())+1;y1=int(yy.max())+1
            rgba=np.full((y1-y0,x1-x0,4),255,np.uint8);rgba[:,:,3]=erase[y0:y1,x0:x1]
            patch=dict(bbox=[x0,y0,x1,y1],png=base64.b64encode(cv2.imencode('.png',rgba)[1]).decode('ascii'))
        result.append([points,text,None,dict(rotation=90 if turn==3 else 270,region_bbox=region_box,erase_patch=patch,words=[])])
    return result or initial
