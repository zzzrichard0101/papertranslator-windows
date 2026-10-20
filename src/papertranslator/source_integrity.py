"""Conservative source repairs before any model request; no inferred content."""
from collections import Counter
import copy
import re

VERSION = 'source-integrity-2'
SUPPORTED = {'source-integrity-1', VERSION}
CAPTION = re.compile(r'^\s*(?:Fig\.|Figure|Table|Box)\s*\d', re.I)
REFERENCE_TITLE = re.compile(r'^(?:references|bibliography)$', re.I)
# Only observed lexical forms: never remove arbitrary compound-word hyphens.
JOIN_WORDS = {'iterations', 'initialization', 'initialized', 'gaussians', 'rendering'}


def advanced(policy):
    return policy is True or policy == VERSION


def caption_start(value):
    """Distinguish a labelled title from prose such as 'Table 2 using ...'."""
    match=re.match(r'^\s*(?:Fig\.|Figure|Table|Box)\s*\d+(?:\.\d+)*(.*)$',value,re.I)
    if not match:return False
    tail=match[1].strip()
    if tail.startswith(('.',':')):return True
    if not tail:return True
    if re.match(r'^(?:using|shows?|presents?|provides?|illustrates?|lists?|compares?|'
                r'summarizes?|can|is|was|are|were|in|on|for|of|and|as|to|with|that|which|we|the)\b',tail,re.I):return False
    return tail[:1].isupper()


def annotate(blocks):
    """Keep standalone bibliography titles separate even without bold fonts."""
    for block in blocks:
        for line in block.get('lines', []):
            value = ''.join(s['text'] for s in line['spans']).strip()
            if REFERENCE_TITLE.fullmatch(value):
                line['layout_kind'] = 'heading'


def reference_units(units, previous=False):
    """Preserve bibliographic metadata, with an explicit heading and density.

    A following appendix closes the section, including on the same page. A
    page gap or absent bibliography evidence cannot inherit this state.
    """
    value = ' '.join(u['source'] for u in units)
    years = len(re.findall(r'\b(?:19|20)\d{2}[a-z]?(?:\.|\s*\))', value))
    title = any(REFERENCE_TITLE.fullmatch(u['source'].strip()) for u in units)
    active = previous and years >= 3
    for index,unit in enumerate(units):
        source = unit['source'].strip()
        if REFERENCE_TITLE.fullmatch(source):
            active = years >= 3
            continue
        if (active and re.fullmatch(r'[A-Z]',source) and index+1<len(units) and
            units[index+1]['kind']=='heading' and units[index+1]['source'].isupper()):
            active=False
        if (active and unit['kind'] == 'heading' and
            (re.match(r'^(?:appendix\b|supplement(?:ary)?\b|[A-Z](?:\s|[.])|\d+[. ]+)', source, re.I) or
             (source.isupper() and len(re.findall(r'[A-Z]{2,}',source))>=2))):
            active = False
        if active and unit['kind'] not in {'heading', 'caption', 'running_header', 'page_marker'}:
            unit.update(kind='reference', preserve=True, preserve_reason='references')
    return active if (title or previous) else False


def split_footnote_markers(blocks, height,classify=False):
    """Separate a native raised numeric prefix from its footnote body.

    Require a smaller, raised span at the lower page margin. Ordinary leading
    numbers, OCR guesses and inline superscripts are left alone.
    """
    result=[];active=None
    for block in blocks:
        lines=block.get('lines',[])
        first=lines[0] if lines else {}
        spans=first.get('spans',[])
        if active and spans:
            group,size,bottom,left=active
            if (first.get('flow_group')==group and 0<=first['bbox'][1]-bottom<=size*1.5 and
                abs(first['bbox'][0]-left)<12 and max(s['size'] for s in spans)<=size*1.15 and
                not block.get('academic_preserve') and
                all(l.get('layout_kind') in {None,'footnote'} and not l.get('rotation') for l in lines) and
                all(s.get('extraction','native')=='native' for l in lines for s in l['spans'])):
                block=copy.deepcopy(block);lines=block['lines'];first=lines[0];spans=first['spans']
                for line in lines:line['layout_kind']='footnote'
                active=(group,size,lines[-1]['bbox'][3],left)
            else:active=None
        if len(spans)<2 or block.get('academic_preserve'):
            result.append(block);continue
        marker,body=spans[:2]
        if (first['bbox'][1]<height*.7 or not re.fullmatch(r'\d{1,2}',marker['text']) or
            not re.match(r'[A-Za-z]',body['text'].lstrip()) or
            marker.get('extraction','native')!='native' or body.get('extraction','native')!='native' or
            marker['size']>body['size']*.85 or marker['bbox'][3]>=body['bbox'][3]-1 or
            abs(marker['bbox'][2]-body['bbox'][0])>3):
            result.append(block);continue
        item=copy.deepcopy(block);line=item['lines'][0]
        markline={**copy.deepcopy(line),'spans':[copy.deepcopy(marker)],'bbox':list(marker['bbox'])}
        result.append({**copy.deepcopy(block),'lines':[markline],'bbox':list(marker['bbox']),
                       'academic_preserve':'footnote_marker'})
        line['spans']=line['spans'][1:]
        line['bbox']=[min(s['bbox'][0] for s in line['spans']),min(s['bbox'][1] for s in line['spans']),
                      max(s['bbox'][2] for s in line['spans']),max(s['bbox'][3] for s in line['spans'])]
        item['bbox']=[min(l['bbox'][0] for l in item['lines']),min(l['bbox'][1] for l in item['lines']),
                      max(l['bbox'][2] for l in item['lines']),max(l['bbox'][3] for l in item['lines'])]
        if classify:
            for line in item['lines']:line['layout_kind']='footnote'
            active=(first.get('flow_group'),body['size'],item['lines'][-1]['bbox'][3],marker['bbox'][0])
        result.append(item)
    return result


def validate_segments(source, parts):
    """Segmentation may normalize whitespace, never omit/repeat any characters."""
    clean = lambda text: re.sub(r'\s+', '', text)
    if clean(source) != ''.join(clean(p) for p in parts):
        raise RuntimeError('원문 분할의 누락·중복을 발견해 번역을 중단했습니다.')


def validate_lines(blocks, units):
    """Paragraph construction retains every non-space source character.

    Only recorded discretionary-hyphen removals can change the count. This is
    extraction-to-input accounting, not an OCR or semantic accuracy score.
    """
    chars = lambda value: Counter(re.sub(r'\s+', '', value))
    before = chars(''.join(s['text'] for b in blocks for l in b.get('lines', []) for s in l['spans']))
    after = chars(''.join(u['source'] for u in units))
    removed = sum(u.get('removed_line_hyphens', 0) for u in units)
    after['-'] += removed
    if before != after:
        raise RuntimeError('문단 구성 중 원문 글자의 누락·중복을 발견했습니다.')
    return dict(status='passed', extracted_characters=before.total(),
                retained_characters=before.total()-removed, removed_line_hyphens=removed)
