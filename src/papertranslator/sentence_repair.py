"""Conservative repair windows anchored by unique, verbatim source citations."""
import re
from collections import Counter

from .alignment import sentence_spans
from .quality import PROTECTED, check

VERSION='unique-citation-sentence-repair-1'
LOCAL_CODES={'numbers','quantity','session_frequency','placeholder','adhd_term','embedded_han','foreign_script'}


def plan(source,target,issues):
    from .translation import blocking
    if not issues or any(i['code'] not in LOCAL_CODES for i in issues):return None
    sources=sentence_spans(source);targets=sentence_spans(target)
    if min(len(sources),len(targets))<3:return None
    proposals=[]
    for a,b in sources:
        piece=source[a:b].strip()
        if len(piece)>min(700,len(source)*.5):continue
        literals=set(PROTECTED.findall(piece))
        anchors=[s for s in literals if re.search(r'[A-Za-z]{3}',s) and source.count(s)==target.count(s)==1]
        if not anchors:continue
        matches=[(c,d) for c,d in targets if all(s in target[c:d] for s in anchors)]
        if len(matches)!=1:continue
        c,d=matches[0];candidate=target[c:d].strip()
        if Counter(PROTECTED.findall(piece))!=Counter(PROTECTED.findall(candidate)):continue
        if not blocking(check(piece,candidate)):continue
        # Everything outside the window must pass the existing safeguards.
        # This is a bounded machine alignment proposal, not semantic approval.
        if blocking(check(source[:a]+source[b:],target[:c]+target[d:])):continue
        proposals.append(dict(source_start=a,source_end=b,target_start=c,target_end=d,
                              source=piece,draft=candidate,anchors=sorted(anchors)))
    return proposals[0] if len(proposals)==1 else None


def splice(source,target,window,replacement):
    from .translation import blocking
    a,b=window['target_start'],window['target_end']
    old=target[a:b];tail=old[len(old.rstrip()):]
    result=target[:a]+replacement.strip()+tail+target[b:]
    if Counter(PROTECTED.findall(source))!=Counter(PROTECTED.findall(result)):
        raise ValueError('부분 교정 중 원문 인용·코드·링크의 누락 또는 중복')
    if blocking(check(source,result)):
        raise ValueError('부분 교정 후 전체 문단 검사 미통과')
    return result
