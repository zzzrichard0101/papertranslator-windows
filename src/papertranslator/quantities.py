"""Exact, narrow duration equivalence; never relax signs or unrelated numbers."""
from fractions import Fraction
import re

WORDS=dict(zero=0,one=1,two=2,three=3,four=4,five=5,six=6,seven=7,eight=8,nine=9,ten=10,
           eleven=11,twelve=12,thirteen=13,fourteen=14,fifteen=15,sixteen=16,seventeen=17,eighteen=18,nineteen=19,
           twenty=20,thirty=30,forty=40,fifty=50,sixty=60,seventy=70,eighty=80,ninety=90)
KOREAN=dict(한=1,두=2,세=3,네=4,다섯=5,여섯=6,일곱=7,여덟=8,아홉=9,열=10,스무=20)
WORD='(?:'+'|'.join(sorted(WORDS,key=len,reverse=True))+')'
NUMBER=r'[-+−]?(?:\d+[- ]\d+/\d+|\d+½|\d+/\d+|\d+(?:\.\d+)?)'
VALUE=r'(?:'+NUMBER+'|'+WORD+r'\s+and\s+a\s+half|'+WORD+r'(?:[- ]'+WORD+r')?)'
EN=re.compile(r'(?<![\w])('+VALUE+r')[-\s]+(decades?|years?|months?|weeks?|days?|hours?|minutes?|seconds?)\b',re.I)
KO=re.compile(r'(?<![A-Za-z0-9_])('+NUMBER+'|'+'|'.join(sorted(KOREAN,key=len,reverse=True))+r')\s*(개월|시간|년|세(?!기)|해|달|주|일|분|초)(\s*반)?')
FACTORS={'decade':('months',120),'year':('months',12),'month':('months',1),
         'week':('days',7),'day':('days',1),'hour':('seconds',3600),'minute':('seconds',60),'second':('seconds',1),
         '년':('months',12),'해':('months',12),'세':('months',12),'개월':('months',1),'달':('months',1),
         '주':('days',7),'일':('days',1),'시간':('seconds',3600),'분':('seconds',60),'초':('seconds',1)}


def value(text):
    text=text.lower().replace('−','-')
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
