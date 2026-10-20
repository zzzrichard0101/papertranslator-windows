"""Narrow scientific literals and exact, unit-aware compact count equivalence."""
from fractions import Fraction
import re

AUTHOR=r"[A-Z][A-Za-z’'-]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z’'-]+)?(?:\s+et al\.)?"
YEAR=r'(?:1[5-9]|20)\d{2}[a-f]?'
REFERENCE=re.compile(r'\['+AUTHOR+r',?\s+'+YEAR+r'(?:\s*;\s*'+AUTHOR+r',?\s+'+YEAR+r')*\]')
URL=re.compile(r'(?<![\w@])(?:https?://[^\s<>]+|(?:[A-Za-z0-9-]+\.)+(?:com|org|net|edu|io)/[^\s<>]+)')
IDENTIFIER=re.compile(r'\b(?:[A-Z][A-Za-z0-9]*-)+\d+[A-Za-z0-9]*\b')
EXTRA=re.compile('|'.join('(?:'+p.pattern+')' for p in [REFERENCE,URL,IDENTIFIER]))
DIMENSION=re.compile(r'(?<![\w.])\d+(?:\.\d+)?(?:\s*[×x]\s*\d+(?:\.\d+)?){1,2}(?![\w.])')
EXTRA_V2=re.compile('(?:'+EXTRA.pattern+')|(?:'+DIMENSION.pattern+')')


def validation_mode(value):
    return value if value in {'typed-v2','typed-v3'} else value=='typed-v1'


def compact_ranges(source,target):
    """Pair both endpoints and an explicit compatible unit, never bare digits.

    An unmatched source range remains an error even when a later scalar match
    could otherwise hide its upper endpoint. A trailing scale applies to both
    unscaled endpoints (6–10K / 20~50만), but explicit scales take precedence.
    """
    number=r'[-+−]?\d+(?:,\d{3})*(?:\.\d+)?'
    en=re.compile(r'(?<![\w.+−-])('+number+r')([KM]?)\s*[-–~]\s*('+number+r')([KM])\s+'
        r'(?:(?:uniformly|randomly|random|initialized|initial|meaningful|optimized|trainable)\s+){0,4}'
        r'(iterations?|steps?|points?|Gaussians?|participants?)\b',re.I)
    ko=re.compile(r'(?<![\w.+−-])('+number+r')\s*([천만억KM]?)\s*[-–~∼]\s*('+number+r')\s*([천만억KM]?)\s*'
        r'(회(?:의?\s*반복)?|번의?\s*반복|단계|개의?\s*'
        r'(?:(?:무작위로|균일하게|초기화된|무작위|유의미한|최적화된|학습된)\s+){0,4}(?:점|가우시안)|명)',re.I)
    scale={'':1,'K':1000,'M':1000000,'천':1000,'만':10000,'억':100000000}
    def values(match):
        def amount(value,unit):return Fraction(value.replace(',','').replace('−','-'))*scale[unit.upper()]
        def sign(value):return value[0].replace('−','-') if value.startswith(('+','-','−')) else ''
        unit=match[5]
        family=('steps' if re.match(r'iteration|step|회|번|단계',unit,re.I) else
                'points' if re.match(r'point',unit,re.I) or '점' in unit else
                'people' if re.match(r'participant',unit,re.I) or unit=='명' else 'gaussians')
        return (amount(match[1],match[2] or match[4]),amount(match[3],match[4]),family,sign(match[1]),sign(match[3]))
    targets=list(ko.finditer(target));used=set();a=[];b=[];unmatched=[]
    for match in en.finditer(source):
        expected=values(match)
        partner=next((i for i,m in enumerate(targets) if i not in used and values(m)==expected),None)
        if partner is None:unmatched.append(dict(lower=str(expected[0]),upper=str(expected[1]),unit=expected[2]))
        else:used.add(partner);a.append(match.span());b.append(targets[partner].span())
    def mask(text,boxes):
        chars=list(text)
        for start,end in boxes:chars[start:end]=' '*(end-start)
        return ''.join(chars)
    return mask(source,a),mask(target,b),unmatched


def preserved_kind(text):
    text=text.strip()
    if URL.fullmatch(text):return 'protected'
    if IDENTIFIER.fullmatch(text):return 'protected'
    tokens=re.findall(r'[A-Za-z]{2,}',text)
    if ('=' in text and len(text)<400 and '\n' not in text and
        not any(t.upper() in {'THE','RESULT','IS','AND','OR','NOT','FOR','WITH','WITHOUT','ONLY'} for t in tokens) and
        all(t.isupper() or t.lower() in {'sin','cos','tan','log','exp','min','max','sqrt','lim'} for t in tokens)):
        return 'formula'
    return None


def compact_counts(source,target,extended=False):
    # Only quantities with explicit compatible units are paired. Model labels
    # such as Ours-30K remain literal identifiers, not quantities.
    modifiers=r'(?:(?:uniformly|randomly|random|initialized|initial|meaningful|optimized|trainable)\s+){0,4}' if extended else ''
    en=re.compile(r'(?<![\w.+−-])([-+−]?\d+(?:\.\d+)?)([KM])\s+'+modifiers+r'(iterations?|steps?|points?|Gaussians?|participants?)\b',re.I)
    ko_modifiers=r'(?:(?:무작위로|균일하게|초기화된|무작위|유의미한|최적화된|학습된)\s+){0,4}' if extended else ''
    ko=re.compile(r'(?<![\w.−+-])([-+−]?\d+(?:,\d{3})*(?:\.\d+)?)\s*([천만억]?)\s*(회(?:의?\s*반복(?:\s*(?:연산|계산))?)?|번의?\s*반복|단계|개의?\s*'+ko_modifiers+r'(?:점|가우시안)|명)')
    def family(unit):
        if re.match(r'iteration|step',unit,re.I) or re.match(r'회|번|단계',unit):return 'steps'
        if re.match(r'participant',unit,re.I) or unit=='명':return 'people'
        if re.match(r'point',unit,re.I) or '점' in unit:return 'points'
        return 'gaussians'
    sign=lambda value:value[0].replace('−','-') if value.startswith(('+','-','−')) else ''
    originals=[(m.start(),m.end(),Fraction(m[1].replace('−','-'))*{'K':1000,'M':1000000}[m[2].upper()],family(m[3]),sign(m[1])) for m in en.finditer(source)]
    targets=[(m.start(),m.end(),Fraction(m[1].replace(',','').replace('−','-'))*{'':1,'천':1000,'만':10000,'억':100000000}[m[2]],family(m[3]),sign(m[1])) for m in ko.finditer(target)]
    a=[];b=[];used=set()
    for start,end,amount,unit,signed in originals:
        partner=next((i for i,t in enumerate(targets) if i not in used and t[2:]==(amount,unit,signed)),None)
        if partner is not None:used.add(partner);a.append((start,end));b.append(targets[partner][:2])
    def mask(text,boxes):
        chars=list(text)
        for start,end in boxes:chars[start:end]=' '*(end-start)
        return ''.join(chars)
    return mask(source,a),mask(target,b)
