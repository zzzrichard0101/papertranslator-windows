"""Independent Poppler renders, original-face pixel comparison and extract checks."""
import json
from pathlib import Path
import subprocess

import pymupdf as fitz
from PIL import Image, ImageChops

from papertranslator.storage import atomic_json, read_json, sha256

project=Path(__file__).resolve().parents[1]
latest=read_json(project/'benchmarks'/'runs'/'latest-job.json')
state=read_json(Path(latest['folder'])/'state.json'); folder=Path(state['output'])
out=project/'exports'/'validation'; out.mkdir(parents=True,exist_ok=True)
for name in ['Original','Korean','Bilingual']:
    subprocess.run(['pdftoppm','-r','72','-png',str(folder/(name+'.pdf')),str(out/name)],check=True,capture_output=True)
source=fitz.open(folder/'Original.pdf'); ko=fitz.open(folder/'Korean.pdf'); dual=fitz.open(folder/'Bilingual.pdf')
comparisons=[]
for index in range(len(source)):
    a=Image.open(out/f'Original-{index+1}.png').convert('RGB')
    b=Image.open(out/f'Bilingual-{index+1}.png').convert('RGB').crop((0,0,a.width,a.height))
    equal=ImageChops.difference(a,b).getbbox() is None
    comparisons.append(dict(source_page=index+1,left_face_pixel_equal=equal))
assert all(c['left_face_pixel_equal'] for c in comparisons)
assert sha256(folder/'Original.pdf')==sha256(latest['source'])
assert all('한국어 재배치' in page.get_text() for page in ko)
assert all(any('Droid' in f[3] or f[1] in {'ttf','otf','cid'} for f in p.get_fonts()) for p in ko)
result=dict(renderer='Poppler pdftoppm, 72 dpi RGB',source_faces=comparisons,
            source_hash_equal=True,korean_pages=len(ko),bilingual_pages=len(dual),
            korean_font_and_text='passed',visual_review='PNG files generated; agent must inspect every representative page')
atomic_json(out/'render-validation.json',result)
print(json.dumps(result,indent=2))
