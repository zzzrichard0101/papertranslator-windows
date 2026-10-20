"""Readable source-column flow, shared by Korean and bilingual PDFs.

Translations stay in their original source column. Figures, equations and
full-width material reserve space; excess text continues in the same column
on a matching continuation page. No translation is rewritten to make it fit.
"""
from collections import Counter, defaultdict
from pathlib import Path
import os
import re

import pymupdf as fitz

VERSION = 'source-column-flow-1'
IGNORABLE = re.compile('[\ufe00-\ufe0f\U000e0100-\U000e01ef]')


def normalized(text):
    return re.sub(r'\s+', '', IGNORABLE.sub('', text))


class Fonts:
    """Use system symbol glyphs and explicit UTF-16 ToUnicode maps for SMP math.

    MuPDF's generated five-digit destinations for mathematical alphanumerics
    are not valid UTF-16 CMap destinations. Map only our newly embedded fonts,
    never original PDF fonts. Font files are not redistributed with the app.
    """
    def __init__(self):
        self.fonts = [fitz.Font('korea')]
        directory = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
        for name in ['seguisym.ttf', 'cambria.ttc']:
            if (directory / name).is_file():
                self.fonts.append(fitz.Font(fontfile=str(directory / name)))
        self.used = {}; self.missing = set(); self.cache = {}

    def choice(self, char):
        if IGNORABLE.fullmatch(char): return None
        if char not in self.cache:
            self.cache[char] = next((i for i, font in enumerate(self.fonts)
                                     if font.has_glyph(ord(char))), 0)
            if not self.fonts[self.cache[char]].has_glyph(ord(char)):
                self.missing.add(ord(char))
        return self.cache[char]

    def width(self, text, size):
        return sum(self.fonts[i].text_length(c, fontsize=size) for c in text
                   if (i := self.choice(c)) is not None)

    def wrap(self, text, size, width):
        from .layout import TOKEN
        lines = []; line = ''; used = 0.
        for token in TOKEN.findall(IGNORABLE.sub('', text)):
            if token == '\n': lines.append(line); line = ''; used = 0.; continue
            pieces = [token] if self.width(token, size) <= width else list(token)
            for piece in pieces:
                amount = self.width(piece, size)
                if line and used + amount > width:
                    lines.append(line); line = ''; used = 0.
                line += piece; used += amount
        if line: lines.append(line)
        return lines

    def draw(self, page, x, top, text, size, color, expected):
        runs = []
        for c in text:
            i = self.choice(c)
            if i is None: continue
            if runs and runs[-1][0] == i: runs[-1][1] += c
            else: runs.append([i, c])
        baseline = top + self.fonts[0].ascender * size
        for i, value in runs:
            font = self.fonts[i]; name = 'PTColumn' + str(i)
            xref = page.insert_font(fontname=name, fontbuffer=font.buffer)
            entry = self.used.setdefault(xref, (font, set()))
            entry[1].update(value)
            page.insert_text((x, baseline), value, fontname=name, fontsize=size, color=color)
            expected[page.number].append(value)
            x += self.width(value, size)

    def finish(self, doc):
        for xref, (font, chars) in self.used.items():
            pairs = {}
            for char in sorted(chars):
                glyph = font.has_glyph(ord(char))
                if glyph: pairs.setdefault(glyph, char)
            if not pairs: continue
            body = ['/CIDInit /ProcSet findresource begin', '12 dict begin', 'begincmap',
                    '/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def',
                    '/CMapName /PTColumnUnicode def', '/CMapType 2 def',
                    '1 begincodespacerange', '<0000> <FFFF>', 'endcodespacerange']
            rows = [f'<{glyph:04X}> <{char.encode("utf-16-be").hex().upper()}>'
                    for glyph, char in sorted(pairs.items())]
            for start in range(0, len(rows), 100):
                chunk = rows[start:start+100]
                body += [f'{len(chunk)} beginbfchar', *chunk, 'endbfchar']
            body += ['endcmap', 'CMapName currentdict /CMap defineresource pop', 'end', 'end']
            target = doc.get_new_xref(); doc.update_object(target, '<<>>')
            doc.update_stream(target, '\n'.join(body).encode('ascii'))
            doc.xref_set_key(xref, 'ToUnicode', f'{target} 0 R')


def union(boxes):
    boxes = list(boxes)
    return fitz.Rect(min(b.x0 for b in boxes), min(b.y0 for b in boxes),
                     max(b.x1 for b in boxes), max(b.y1 for b in boxes))


def captions(groups):
    """Adjacent caption lines form one paragraph, rather than tiny line boxes."""
    result = []
    for group in groups:
        prior = result[-1] if result else None
        if (prior and prior['kind'] == group['kind'] == 'caption' and
            -1 <= group['bbox'].y0-prior['bbox'].y1 <= 5 and
            abs(group['bbox'].x0-prior['bbox'].x0) < 35 and
            bool(prior.get('column_bbox')) == bool(group.get('column_bbox'))):
            prior['text'] += ' ' + group['text']; prior['bbox'] |= group['bbox']
            prior['ids'] += group['ids']; prior['erase_boxes'] += group['erase_boxes']
            prior['failed'] |= group['failed']; prior['uncertain'] |= group['uncertain']
        else: result.append({**group})
    return result


def intervals(top, bottom, obstacles, left, right):
    occupied = sorted((max(top, b.y0-3), min(bottom, b.y1+3)) for b in obstacles
                      if b.y1 > top and b.y0 < bottom and min(b.x1, right)-max(b.x0, left) > 2)
    free = []; y = top
    for a, b in occupied:
        if a > y: free.append([y, a])
        y = max(y, b)
    if y < bottom: free.append([y, bottom])
    return free


def scientific_regions(page):
    """Native scientific objects are indivisible, including stacked math glyphs.

    Keep their original vectors. Their translated labels/prose are linked on
    the continuation face rather than painting across numbers or fractions.
    This is deliberately independent of a particular paper or page number.
    """
    blocks = [b for b in page.get_text('dict')['blocks'] if b['type'] == 0]
    result = []
    for block in blocks:
        spans = [s for row in block['lines'] for s in row['spans']]
        value = ' '.join(s['text'] for s in spans)
        box = fitz.Rect(block['bbox'])
        if not page.rect.height*.09 < box.y0 < page.rect.height*.9: continue
        math_count = sum(len(s['text']) for s in spans
                         if re.search(r'math|txsy|txmi|txex', s['font'], re.I))
        digits = sum(c.isdigit() for c in value)
        if digits >= 25 and digits/max(1,len(value.replace(' ',''))) > .26:
            result.append((box, '숫자 표'))
        elif math_count >= 7 and (math_count/max(1,len(value)) > .23 or
                                 len(re.findall(r'[A-Za-z]{3,}',value)) < 3):
            result.append((box, '수식 포함 영역'))
        if re.match(r'^Algorithm\s+\d+\b',value,re.I):
            # Ruled algorithms may begin with `while`, without a function/end
            # pair. Detect their enclosing horizontal rules geometrically.
            rules = [fitz.Rect(d['rect']) for d in page.get_drawings()
                     if d['rect'].width >= box.width*.8 and d['rect'].height < 1 and
                     abs(d['rect'].x0-box.x0) < 5]
            end = next((b.y0 for b in sorted(rules,key=lambda b:b.y0)
                        if b.y0 > box.y1+20), None)
            if end is not None:
                result.append((fitz.Rect(box.x0,box.y0,max(b.x1 for b in rules),end+2),'알고리즘'))
    return result


def label_color(source, box):
    """Photograph labels need contrast; embedded white panels need dark text."""
    pix=source.get_pixmap(clip=box & source.rect,matrix=fitz.Matrix(.5,.5),colorspace=fitz.csGRAY,alpha=False)
    light=sum(pix.samples)/max(1,len(pix.samples))
    return (1,1,1) if light<115 else (.08,.13,.2)


def compose(ir, records, original, bilingual_path, korean_path, cancel=lambda: False, min_font=9.):
    from .layout import record_entries, group_overlaps
    entries, uncertain = record_entries(ir, records)
    planned={index:Counter(normalized(''.join(g['text'] for g in groups))) for index,groups in entries.items()}
    fonts = Fonts(); expected = defaultdict(list); regions = []; mapping = []; extra_count = 0
    issues = []; bounds = {}; symbol_moves = []; preserved_objects=[]
    with fitz.open(original) as src, fitz.open() as ko:
        def new_page(index, extra=False):
            nonlocal extra_count
            rect = src[index].rect
            page = ko.new_page(width=rect.width, height=rect.height+22)
            if not extra and src[index].get_contents():
                page.show_pdf_page(fitz.Rect(0, 0, rect.width, rect.height), src, index)
                kind, group = src.xref_get_key(src[index].xref, 'Group')
                value = src.xref_object(int(group.split()[0])) if kind == 'xref' else group
                color = re.search(r'/CS\s*/(DeviceCMYK|DeviceRGB|DeviceGray)\b', value)
                if color and '/Transparency' in value:
                    ko.xref_set_key(page.xref, 'Group', f'<< /S /Transparency /CS /{color[1]} >>')
            notice_xref=page.insert_font(fontname='PTNotice', fontbuffer=fonts.fonts[0].buffer)
            label = f'원본 {index+1}쪽 · ' + ('같은 열 번역 계속' if extra else '원문 열 유지 · 기계 번역')
            fonts.used.setdefault(notice_xref,(fonts.fonts[0],set()))[1].update(label)
            page.insert_text((18, rect.height+13), label, fontname='PTNotice', fontsize=7,
                             color=(.3, .38, .43))
            mapping.append(index); bounds[page.number] = fitz.Rect(0, 0, rect.width, rect.height)
            if extra: extra_count += 1
            return page

        for info in ir['pages']:
            if cancel():
                from .backend import Cancelled
                raise Cancelled()
            index = info['source_page_index']; source = src[index]; w = source.rect.width; h = source.rect.height
            two = bool(info.get('reading_columns')); gutter = info.get('reading_columns',{}).get('gutter',w/2)
            groups = captions(group_overlaps(entries[index])); page = new_page(index); main = page.number
            # Images retain their original position. A full-page scan is a
            # background, not an obstacle covering the whole translated page.
            images = [fitz.Rect(i['bbox'])*source.rotation_matrix for i in source.get_image_info()]
            images = [b for b in images if b.get_area() < source.rect.get_area()*.65]
            protected = [fitz.Rect(f['bbox']) for r in records.values() if r['status']=='preserved'
                         or r['kind'] in {'page_marker','running_header'}
                         for f in r.get('source_fragments',[]) if f['source_page_index']==index]
            objects = scientific_regions(source)
            deferred = defaultdict(list)
            # Expand to complete paragraph rectangles so a scientific object
            # never keeps half of an English paragraph under Korean flow.
            changed = True
            while changed:
                changed = False
                for group in list(groups):
                    hits=[(b,label) for b,label in objects if (b&group['bbox']).get_area()>1]
                    if not hits: continue
                    groups.remove(group); changed=True
                    side=0 if not two or group['bbox'].x0 < gutter else 1
                    label=hits[0][1]
                    objects.append((group['bbox'],label))
                    deferred[side].append({**group,'object_label':label})
                    preserved_objects.append(dict(source_page_index=index,ids=group['ids'],kind=label,bbox=list(group['bbox'])))
            protected += [b for b,_ in objects]
            for side,items in deferred.items():
                merged=[]
                for group in sorted(items,key=lambda g:(g['bbox'].y0,g['bbox'].x0)):
                    if merged and merged[-1]['object_label']==group['object_label']:
                        prior=merged[-1]
                        prior['text']+='\n'+group['text'];prior['ids']+=group['ids']
                        prior['bbox']|=group['bbox']
                        prior['failed']|=group['failed'];prior['uncertain']|=group['uncertain']
                    else:merged.append({**group,'ids':list(group['ids'])})
                for group in merged:
                    group['text']=f'[{group["object_label"]} · 원문 {index+1}쪽]\n'+group['text']
                deferred[side]=merged
            # Keep preserved heading numerals with their translated headings.
            for group in groups:
                if group['kind'] != 'heading': continue
                nearby = [r for r in records.values() if r['status']=='preserved' and
                          re.fullmatch(r'(?:\d+(?:\.\d+)*|[A-Z])', r['source'].strip()) and
                          any(f['source_page_index']==index and abs(f['bbox'][1]-group['bbox'].y0)<4 and
                              0 <= group['bbox'].x0-f['bbox'][2] < 30 for f in r.get('source_fragments',[]))]
                if nearby:
                    group['text'] = ' '.join(r['source'].strip() for r in nearby)+'  '+group['text']
                    for r in nearby:
                        for f in r['source_fragments']:
                            if f['source_page_index']==index:
                                box=fitz.Rect(f['bbox']);group['erase_boxes'].append(box)
                                protected=[p for p in protected if p != box]
                                symbol_moves.append(r['id'])
            # Native caption prefixes are often separate preserved runs. Move
            # them with the translated caption instead of leaving `Fig. 4.`
            # underneath the first Korean line.
            for group in groups:
                if group['kind']!='caption':continue
                for r in records.values():
                    match=re.fullmatch(r'(Fig\.?|Figure|Table)\s*(\d+)\.?',r['source'].strip(),re.I)
                    if r['status']!='preserved' or not match:continue
                    for fragment in r.get('source_fragments',[]):
                        box=fitz.Rect(fragment['bbox'])
                        if (fragment['source_page_index']==index and abs(box.y0-group['bbox'].y0)<3
                            and (box&group['bbox']).height>2 and box.x1<group['bbox'].x1):
                            prefix=('표 ' if match[1].lower()=='table' else '그림 ')+match[2]+'. '
                            if not re.match(r'^(?:그림|표)\s*'+match[2]+r'\b',group['text']):group['text']=prefix+group['text']
                            group['erase_boxes'].append(box)
                            protected=[p for p in protected if p!=box]
                            symbol_moves.append(r['id'])
            flow = defaultdict(list); anchored = []
            for group in groups:
                box=group['bbox']; col=group.get('column_bbox')
                side = 0 if not two or (box.x0+box.x1)/2 < gutter else 1
                column_width=(gutter if side==0 else w-gutter) if two else w
                inside_image=any((box&b).get_area()>box.get_area()*.4 for b in images)
                small = box.width < column_width*.5 and box.height<35 and len(group['text'])<100
                if ((two and col is None) or group['kind'] in {'caption','table_cell','diagram_label','form_question'}
                    or group.get('rotation') or inside_image or (small and group['kind']!='heading')):
                    anchored.append((side,group))
                else: flow[side].append(group)
            obstacles=protected+images; pending=deferred; placed_boxes=[]; erase=[]; anchor_draw=[]
            native_words=[fitz.Rect(word[:4]) for word in source.get_text('words')]
            flow_boxes=[g['bbox'] for items in flow.values() for g in items]
            for side,group in anchored:
                box=group['bbox']; broad=box.width>w*.55
                left=box.x0; right=min(w-24 if broad or not two else (gutter-9 if side==0 else w-24),
                                      max(box.x1, left+60))
                if broad:right=w-24
                neighbours=[g['bbox'].x0-3 for _,g in anchored if g is not group and
                            g['bbox'].x0>=box.x1 and (g['bbox']&box).height>2]
                if neighbours:right=min(right,*neighbours)
                available=[b.y0-3 for b in obstacles if b.y0>=box.y1-1 and
                           min(b.x1,right)-max(b.x0,left)>5]
                next_anchor=[g['bbox'].y0-2 for _,g in anchored if g is not group and
                             g['bbox'].y0>=box.y1-1 and min(g['bbox'].x1,right)-max(g['bbox'].x0,left)>5]
                bottom=min([h-35,*available,*next_anchor])
                # Full-width captions may expand into the subsequently erased
                # body area. That space is then reserved before column flow.
                sizes=[12.,11.,10.,9.] if group['kind']=='heading' else [9.,8.5,8.]
                selected=None
                for size in sizes:
                    lines=fonts.wrap(group['text'],size,max(1,right-left))
                    height=len(lines)*size*1.34
                    cap=(bottom-box.y0 if group['kind']=='caption' or broad else max(box.height, size*1.34+1))
                    drawn_box=fitz.Rect(left,box.y0,right,box.y0+height)
                    peers=[b for b in native_words if not box.contains((b.tl+b.br)/2) and
                           not any(e.contains((b.tl+b.br)/2) for e in group['erase_boxes']+flow_boxes)]
                    if (height<=cap+.1 and not any((drawn_box&b).get_area()>2 for b in placed_boxes+protected)
                        and not any((drawn_box&b).get_area()>2 for b in peers)):
                        selected=(size,lines,drawn_box);break
                color=(label_color(source,box) if any((box&b).get_area()>box.get_area()*.4 for b in images)
                       else (.65,.28,.08) if group['failed'] or group['uncertain'] else (.08,.13,.2))
                if selected:
                    size,lines,box_out=selected
                    for n,line in enumerate(lines):
                        anchor_draw.append((left,box.y0+n*size*1.34,line,size,color))
                    obstacles.append(box_out);placed_boxes.append(box_out)
                    regions.append(dict(ids=group['ids'],source_page_index=index,output_page=main,
                        source_bbox=list(box),target_bbox=list(box_out),font_size=size,overflow=False,column=side,kind=group['kind']))
                    erase.extend(group['erase_boxes'])
                else:
                    source_label=' '.join(records[i]['source'] for i in group['ids'] if i in records)
                    pending[side].append({**group,'legend':True,'text':f'[{source_label[:65]}] '+group['text']})
                    obstacles.append(box)
            # Erase glyphs without painting over diagrams, rules or photos.
            # Native paragraph boxes include superscripts missed by line boxes.
            for items in flow.values():
                for group in items:
                    erase.append(group['bbox']);erase.extend(group['erase_boxes'])
            for box in erase:
                page.add_redact_annot((box+fitz.Rect(-.1,-.1,.1,.1)) & source.rect,
                    fill=(1,1,1) if info.get('ocr',{}).get('lines') else None,cross_out=False)
            if erase: page.apply_redactions(images=0,graphics=0,text=0)
            # Include surviving native text in preflight, not just new Korean
            # spans. Small copyright lines and preserved notation are obstacles.
            obstacles += [fitz.Rect(row['bbox']) for b in page.get_text('dict')['blocks'] if b['type']==0
                          for row in b['lines']]
            for x,y,line,size,color in anchor_draw:
                fonts.draw(page,x,y,line,size,color,expected)
            columns=[0,1] if two else [0]
            continuations=[]
            for side in columns:
                items=flow[side]; lo=0 if side==0 else gutter; hi=gutter if two and side==0 else w
                wide=[g['bbox'] for g in items if g['bbox'].width>(hi-lo)*.58]
                left=max(lo+15,min((b.x0 for b in wide),default=lo+35))
                right=min(hi-12,max((b.x1 for b in wide),default=hi-35))
                # Keep original margins, with a modest gap at the gutter.
                if right-left<80:left=lo+25;right=hi-25
                top=min((g['bbox'].y0 for g in items),default=65)
                footer=min((b.y0-5 for b in protected if b.y0>h*.88),default=h-35)
                free=intervals(top,min(footer,h-28),obstacles,left,right)
                queue=items; cursor=free[0][0] if free else h
                slot=0; spills=[]
                for group in queue:
                    size=9.5 if group['kind']=='heading' else min_font
                    text=group['text']; lines=fonts.wrap(text,size,right-left)
                    if normalized(''.join(lines))!=normalized(text):
                        issues.append(dict(code='planned_text_mismatch',source_page_index=index,ids=group['ids']))
                    color=(.65,.28,.08) if group['failed'] or group['uncertain'] else (.08,.13,.2)
                    drawn=[]
                    for line in lines:
                        while slot<len(free) and cursor+size*1.34>free[slot][1]:
                            slot+=1
                            if slot<len(free):cursor=free[slot][0]
                        if slot>=len(free):spills.append((group,line,size,color));continue
                        fonts.draw(page,left,cursor,line,size,color,expected)
                        drawn.append(fitz.Rect(left,cursor,left+fonts.width(line,size),cursor+size*1.31))
                        cursor+=size*1.34
                    if drawn:
                        regions.append(dict(ids=group['ids'],source_page_index=index,output_page=main,
                            source_bbox=list(group['bbox']),target_bbox=list(union(drawn)),font_size=size,
                            overflow=False,column=side,kind=group['kind']))
                    cursor+=4 if group['kind']=='heading' else 3
                for group in pending[side]:
                    color=(.65,.28,.08) if group['failed'] or group['uncertain'] else (.08,.13,.2)
                    for line in fonts.wrap(group['text'],min_font,right-left):
                        spills.append((group,line,min_font,color))
                    spills.append((group,'',min_font,color))
                # Preserve the source column on every continuation. Left text
                # cannot displace right-column content from the source page.
                continuation=0; y=55
                for group,line,size,color in spills:
                    if continuation>=len(continuations):continuations.append(new_page(index,True).number)
                    if y+size*1.34>h-35:
                        continuation+=1;y=55
                        if continuation>=len(continuations):continuations.append(new_page(index,True).number)
                    target=ko[continuations[continuation]]
                    fonts.draw(target,left,y,line,size,color,expected)
                    regions.append(dict(ids=group['ids'],source_page_index=index,output_page=target.number,
                        source_bbox=list(group['bbox']),target_bbox=[left,y,right,y+size*1.31],font_size=size,
                        overflow=True,column=side,kind=group['kind']))
                    y+=size*1.34
                # new_page invalidates Page handles; reload the main page.
                page=ko[main]
        for index,counts in planned.items():
            emitted=Counter(normalized(''.join(value for n,values in expected.items()
                                             if mapping[n]==index for value in values)))
            if counts-emitted:
                issues.append(dict(code='translation_text_not_placed',source_page_index=index))
        fonts.finish(ko)
        font_name=lambda value:re.sub(r'Regular$','',value.replace(' ','')).casefold()
        names={font_name(font.name) for font in fonts.fonts}
        for n in range(len(ko)):
            actual=[]; boxes=[]; native=[]
            for block in ko[n].get_text('dict')['blocks']:
                if block['type']!=0:continue
                for row in block['lines']:
                    for span in row['spans']:
                        if span['size']>=8-.01 and font_name(span['font']) in names:
                            actual.append(span['text']);boxes.append(fitz.Rect(span['bbox']))
                        elif font_name(span['font']) not in names:
                            native.append(fitz.Rect(span['bbox']))
            if Counter(normalized(''.join(actual)))!=Counter(normalized(''.join(expected[n]))):
                issues.append(dict(code='rendered_text_mismatch',output_page=n,source_page_index=mapping[n]))
            if any(not bounds[n].contains(b) for b in boxes):
                issues.append(dict(code='body_outside_page',output_page=n,source_page_index=mapping[n]))
            if any((a&b).height>min(a.height,b.height)*.35 and (a&b).get_area()>min(a.get_area(),b.get_area())*.18
                   for i,a in enumerate(boxes) for b in boxes[:i]):
                issues.append(dict(code='body_text_overlap',output_page=n,source_page_index=mapping[n]))
            if any((a&b).height>min(a.height,b.height)*.35 and
                   (a&b).get_area()>min(a.get_area(),b.get_area())*.18 for a in boxes for b in native):
                issues.append(dict(code='preserved_text_overlap',output_page=n,source_page_index=mapping[n]))
        if fonts.missing:issues.append(dict(code='unsupported_translation_glyphs',codepoints=sorted(fonts.missing)))
        ko.subset_fonts();ko.save(korean_path,garbage=4,deflate=True)
        with fitz.open() as dual:
            for n,index in enumerate(mapping):
                if cancel():
                    from .backend import Cancelled
                    raise Cancelled()
                rect=src[index].rect
                dual.insert_pdf(src,from_page=index,to_page=index,links=False,annots=False)
                page=dual[-1]
                # Copying the source page preserves its original transparency
                # group and CMYK rendering on the unchanged left face.
                page.set_mediabox(fitz.Rect(0,-22,rect.width*2,rect.height))
                page.show_pdf_page(fitz.Rect(rect.width,0,rect.width*2,rect.height+22),ko,n)
                kind,group=src.xref_get_key(src[index].xref,'Group')
                value=src.xref_object(int(group.split()[0])) if kind=='xref' else group
                color=re.search(r'/CS\s*/(DeviceCMYK|DeviceRGB|DeviceGray)\b',value)
                if color and '/Transparency' in value:
                    dual.xref_set_key(page.xref,'Group',f'<< /S /Transparency /CS /{color[1]} >>')
            dual.save(bilingual_path,garbage=4,deflate=True)
    return dict(version=VERSION,mode='source_position',source_indices=mapping,regions=regions,
        overflow_pages=extra_count,automatic_collision_reflows=0,minimum_font=min_font,diagram_minimum_font=8.,
        korean_layout='source_columns',column_pages=sum(bool(p.get('reading_columns')) for p in ir['pages']),
        unresolved_page_alignments=uncertain,proposed_page_alignments=[],moved_heading_labels=symbol_moves,
        preserved_scientific_objects=preserved_objects,
        table_regions=sum(len(p.get('table_regions',[])) for p in ir['pages']),
        validation=dict(status='review_required' if issues or uncertain else 'passed',issues=issues,
                        all_output_pages_checked=len(mapping),semantic_accuracy='not_validated'))
