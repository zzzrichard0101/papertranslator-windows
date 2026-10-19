"""Exact, narrow duration equivalence; never relax signs or unrelated numbers."""
from fractions import Fraction
import re

WORDS=dict(zero=0,one=1,two=2,three=3,four=4,five=5,six=6,seven=7,eight=8,nine=9,ten=10,
           eleven=11,twelve=12,thirteen=13,fourteen=14,fifteen=15,sixteen=16,seventeen=17,eighteen=18,nineteen=19,
           twenty=20,thirty=30,forty=40,fifty=50,sixty=60,seventy=70,eighty=80,ninety=90)
KOREAN=dict(한=1,두=2,세=3,네=4,다섯=5,여섯=6,일곱=7,여덟=8,아홉=9,열=10,스무=20)
WORD='(?:'+'|'.join(sorted(WORDS,key=len,reverse=True))+')'
NUMBER=r'[-+−]?(?:\d+[- ]\d+/\d+|\d+½|\d+/\d+|\d+(?:\.\d+)?)'
VALUE=r'(?:(?:'+NUMBER+'|'+WORD+r')\s+(?:thousand|million|billion)|'+NUMBER+'|'+WORD+r'\s+and\s+a\s+half|'+WORD+r'(?:[- ]'+WORD+r')?)'
EN=re.compile(r'(?<![\w])('+VALUE+r')[-\s]+(decades?|years?|months?|weeks?|days?|hours?|minutes?|seconds?)\b',re.I)
KO=re.compile(r'(?<![A-Za-z0-9_])((?:'+NUMBER+'|'+'|'.join(sorted(KOREAN,key=len,reverse=True))+r')(?:\s*[만억조])?)\s*(개월|시간|년|세(?!기)|해|달|주|일|분|초)(\s*반)?')
FACTORS={'decade':('months',120),'year':('months',12),'month':('months',1),
         'week':('days',7),'day':('days',1),'hour':('seconds',3600),'minute':('seconds',60),'second':('seconds',1),
         '년':('months',12),'해':('months',12),'세':('months',12),'개월':('months',1),'달':('months',1),
         '주':('days',7),'일':('days',1),'시간':('seconds',3600),'분':('seconds',60),'초':('seconds',1)}


def value(text):
    text=text.lower().replace('−','-')
    large=re.fullmatch(r'(.+?)\s*(thousand|million|billion|만|억|조)',text)
    if large:return value(large[1])*dict(thousand=1000,million=1000000,billion=1000000000,만=10000,억=100000000,조=1000000000000)[large[2]]
    if text in KOREAN:return Fraction(KOREAN[text])
    mixed=re.fullmatch(r'([-+]?\d+)[- ](\d+/\d+)',text)
    if mixed:return Fraction(mixed[1])+(-1 if mixed[1].startswith('-') else 1)*Fraction(mixed[2])
    if text.endswith('½'):return Fraction(text[:-1])+Fraction(1,2)
    if re.match(r'[-+]?\d',text):return Fraction(text)
    half=text.endswith(' and a half')
    if half:text=text[:-11]
    return Fraction(sum(WORDS[w] for w in re.split('[- ]',text)))+(Fraction(1,2) if half else 0)


def groups(text,pattern):
    found=[]
    for match in pattern.finditer(text):
        unit=match[2].lower().rstrip('s') if pattern is EN else match[2]
        family,factor=FACTORS[unit]
        try:amount=value(match[1])+(Fraction(1,2) if pattern is KO and match[3] else 0)
        except (ValueError,KeyError,ZeroDivisionError):continue
        item=dict(start=match.start(),end=match.end(),family=family,value=amount*factor)
        if found and found[-1]['family']==family and re.fullmatch(r'\s*(?:and\s*)?',text[found[-1]['end']:item['start']],re.I):
            found[-1]['end']=item['end'];found[-1]['value']+=item['value']
        else:found.append(item)
    return found


def equivalent_durations(source,target):
    """Mask only paired equal quantities, leaving all other numbers for strict checks."""
    # Narrative "One day, ..." is not a measured 24-hour duration. Permit the
    # exact Korean idiom only at a sentence boundary; keep "took one day" strict.
    if re.search(r'어느\s*날',target):
        source=re.sub(r'(^|[.!?]\s+)(one day)(?=,\s+(?:after|when|while|I|we|he|she|they|there)\b)',
                      lambda m:m[1]+' '*len(m[2]),source,flags=re.I)
    originals=groups(source,EN);targets=groups(target,KO);used=set();a=[];b=[];unmatched=[]
    for original in originals:
        match=next((i for i,t in enumerate(targets) if i not in used and t['family']==original['family'] and t['value']==original['value']),None)
        if match is None:unmatched.append(source[original['start']:original['end']]);continue
        used.add(match);a.append((original['start'],original['end']));b.append((targets[match]['start'],targets[match]['end']))
    def mask(text,spans):
        chars=list(text)
        for start,end in spans:chars[start:end]=' '*(end-start)
        return ''.join(chars)
    return mask(source,a),mask(target,b),unmatched


def equivalent_fractions(source,target):
    """Pair explicit word fractions with exact Korean/percent representations."""
    denominators={'half':2,'halves':2,'third':3,'thirds':3,'quarter':4,'quarters':4,
                  'fourth':4,'fourths':4,'fifth':5,'fifths':5,'sixth':6,'sixths':6,
                  'seventh':7,'sevenths':7,'eighth':8,'eighths':8,'ninth':9,'ninths':9,'tenth':10,'tenths':10}
    nums=r'(?:\d+(?:\.\d+)?|'+WORD+r'|an?|a)'
    pattern=re.compile(r'(?<!\w)(?P<sign>minus\s+|negative\s+|[+−-]\s*)?(?P<num>'+nums+r')(?P<sep>[-\s]+)(?P<den>'+
                       '|'.join(sorted(denominators,key=len,reverse=True))+r')\b',re.I)
    originals=[]
    for match in pattern.finditer(source):
        following=source[match.end():]
        if following.startswith('-') or re.match(r'\s*(?:of\s+(?:a|an|the)\s+)?(?:years?|days?|hours?|months?|weeks?|minutes?|seconds?|per\s+cent|percent)\b',following,re.I):continue
        if match['den'].lower().startswith('quarter') and '-' not in match['sep'] and not re.match(r'\s+of\b',following,re.I):continue
        if match['num'].lower() in {'a','an'} and re.search(r'\band\s+$',source[:match.start()],re.I):continue
        numerator=Fraction(1) if match['num'].lower() in {'a','an'} else value(match['num'])
        sign=-1 if match['sign'] and match['sign'].strip().lower() in {'minus','negative','-','−'} else 1
        originals.append((match.start(),match.end(),sign*numerator/denominators[match['den'].lower()]))
    sino=dict(일=1,이=2,삼=3,사=4,오=5,육=6,칠=7,팔=8,구=9,십=10)
    korean={**KOREAN,**sino}
    token=r'[+−-]?(?:\d+|'+'|'.join(sorted(korean,key=len,reverse=True))+r')'
    def number(text):
        sign=-1 if text.startswith(('-','−')) else 1;text=text.lstrip('+−-')
        return sign*(Fraction(korean[text]) if text in korean else Fraction(text))
    targets=[]
    fraction=re.compile(r'(?<!\d)(?P<sign>마이너스\s*|음의\s*)?(?P<den>'+token+r')\s*분의\s*(?P<num>'+token+r')')
    for match in fraction.finditer(target):
        if re.match(r'\s*%',target[match.end():]):continue
        try:amount=number(match['num'])/number(match['den'])*(-1 if match['sign'] else 1)
        except ZeroDivisionError:continue
        targets.append((match.start(),match.end(),amount))
    for match in re.finditer(r'(?<![\w/:.])([-+−]?\d+)\s*/\s*(\d+)(?![\w/]|\s*%)',target):
        if int(match[2]):targets.append((match.start(),match.end(),Fraction(match[1].replace('−','-'))/int(match[2])))
    for match in re.finditer(r'(?<![A-Za-z0-9_./])[-+−]?\d+(?:\.\d+)?\s*%',target):
        targets.append((match.start(),match.end(),Fraction(match[0][:-1].strip().replace('−','-'))/100))
    for match in re.finditer(r'절반|반수',target):targets.append((match.start(),match.end(),Fraction(1,2)))
    used=set();a=[];b=[];unmatched=[]
    for start,end,amount in originals:
        partner=next((i for i,t in enumerate(targets) if i not in used and t[2]==amount),None)
        if partner is None:
            if targets:unmatched.append(source[start:end])
            continue
        used.add(partner);a.append((start,end));b.append(targets[partner][:2])
    def mask(text,spans):
        chars=list(text)
        for start,end in spans:chars[start:end]=' '*(end-start)
        return ''.join(chars)
    return mask(source,a),mask(target,b),unmatched
