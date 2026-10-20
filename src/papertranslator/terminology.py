"""Document-scoped, curated terminology; never learn from unchecked targets."""
import re
from collections import Counter

VERSION = 'document-terminology-1'
LOCAL_ONLY = {'aunt','charcoal','braid','confounding','practice','evidence','client','practitioner'}
# Only terms with an explicit curated translation are shared across paragraphs.
# Unknown abbreviations and personal names are never expanded speculatively.
ACADEMIC = [
    ('continuing professional development', '지속적 전문성 개발'),
    ('systematic review', '체계적 문헌고찰'),
    ('meta-analysis', '메타분석'),
    ('confidence interval', '신뢰구간'),
    ('effect size', '효과 크기'),
    ('case-control study', '환자-대조군 연구'),
    ('cohort study', '코호트 연구'),
    ('informed consent', '충분한 설명에 근거한 동의'),
]


def occurrences(source, text):
    return list(re.finditer(r'\b'+re.escape(source)+r'(?:s|es)?\b', text, re.I))


def build(ir, profile):
    from .quality import relevant_terms
    academic=ACADEMIC+([('uniformly random','균일하게 무작위로')]
        if ir.get('source_integrity',{}).get('version')=='source-integrity-2' else [])
    counts = Counter(); pages = {}; targets = {}; definitions = {}
    for block in ir['blocks']:
        if block['preserve']:
            continue
        text = block['source']
        terms = relevant_terms(text, profile) + [(s,t) for s,t in academic if occurrences(s,text)]
        for source, target in terms:
            if source in LOCAL_ONLY:
                continue
            counts[source] += len(occurrences(source, text)); targets[source] = target
            pages.setdefault(source, set()).add(block['source_page_index'])
    # Inspect all literal initialisms, including unknown conflicting meanings.
    # An abbreviation with two source definitions must not inherit either one.
    canonical={re.sub(r'[^a-z]+',' ',s.lower()).strip():s for s in targets}
    for block in ir['blocks']:
        if block['preserve']:continue
        text=block['source']
        for match in re.finditer(r'\(([A-Z][A-Z0-9]{1,9})\)',text):
            abbreviation=match[1]
            prefix=re.split(r'[.;:!?\n()]',text[:match.start()])[-1]
            words=re.findall(r'[A-Za-z]+',prefix)[-10:]
            for length in range(2,len(words)+1):
                phrase=words[-length:]
                if ''.join(w[0] for w in phrase).upper()!=abbreviation:continue
                normalized=' '.join(phrase).lower();source=canonical.get(normalized,normalized)
                definitions.setdefault(abbreviation,set()).add((source,targets.get(source)))
    entries = [dict(source=s, target=targets[s], occurrences=counts[s], source_page_indices=sorted(pages[s]))
               for s in sorted(counts)]
    aliases = []
    for abbreviation, meanings in sorted(definitions.items()):
        if len(meanings) != 1:
            continue
        source, target = next(iter(meanings))
        if target is None:continue
        aliases.append(dict(source=abbreviation,target=target+'('+abbreviation+')',expansion=source))
    return dict(version=VERSION, origin='curated terms and literal source definitions',
                entries=entries, aliases=aliases, human_approved=False)


def for_source(text, glossary):
    terms = []; covered = []
    for item in sorted(glossary['entries']+glossary['aliases'],key=lambda i:len(i['source']),reverse=True):
        matches = [m for m in occurrences(item['source'],text)
                   if not any(m.start()<b and m.end()>a for a,b in covered)]
        if matches:
            terms.append((item['source'],item['target'])); covered.extend(m.span() for m in matches)
        if len(terms) == 10:
            break
    return terms


def terms_for(text, profile=None, context=None):
    from .quality import relevant_terms
    terms = relevant_terms(text, profile)
    supplied = (context or {}).get('terminology', [])
    # This field is generated locally, never from model responses or documents.
    extra = [(s,t) for s,t in supplied if occurrences(s,text)]
    return list(dict.fromkeys(extra+terms))[:16]
