"""Real offline OCR -> GPU translation -> export on synthetic scans only."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

from papertranslator.jobs import create_job
from papertranslator.storage import atomic_json, read_json, sha256


def raster(text):
    image=Image.new('RGB',(2100,180),'white')
    ImageDraw.Draw(image).text((20,35),text,font=ImageFont.truetype('arial.ttf',40),fill='black')
    stream=io.BytesIO();image.save(stream,format='PNG');return stream.getvalue()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--exe',type=Path);args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]
    work=project/'work/ocr';work.mkdir(parents=True,exist_ok=True)
    vault=project/'test_vault';vault.mkdir(exist_ok=True)
    source=work/'Synthetic OCR Study.pdf'
    with fitz.open() as doc:
        page=doc.new_page(width=600,height=800)
        page.insert_image(fitz.Rect(40,700,560,745),stream=raster('The study included 24 participants and continued with'))
        page=doc.new_page(width=600,height=800)
        page.insert_image(fitz.Rect(40,45,560,90),stream=raster('the same research question on the next page.'))
        page=doc.new_page(width=600,height=800)
        page.insert_text((40,60),'This native paragraph must appear only once.',fontsize=12)
        page.insert_image(fitz.Rect(40,150,560,195),stream=raster('The image reports two groups and 12 observations.'))
        doc.save(source)
    job=create_job(source,vault,title='합성 OCR 통합 시험',model_key='hy7b',ocr_mode='auto')
    command=[str(args.exe),'--worker',str(job)] if args.exe else [sys.executable,'-m','papertranslator','--worker',str(job)]
    began=time.monotonic()
    subprocess.run(command,check=True,timeout=240)
    state=read_json(job/'state.json');assert state['status']=='completed_with_warnings',state.get('error')
    output=Path(state['output']);ir=read_json(job/'ir.json');quality=read_json(output/'Quality.json')
    joined=[b for b in ir['blocks'] if len(b['source_page_indices'])>1]
    assert len(joined)==1 and '24 participants' in joined[0]['source']
    assert 'the same research question' in joined[0]['source']
    assert quality['ocr_pages']==3 and quality['failed_paragraphs']==0
    assert quality['connected_paragraphs']==1
    assert sum('This native paragraph' in r['source'] for r in quality['records'])==1
    assert sha256(source)==sha256(output/'Original.pdf')
    with fitz.open(output/'Korean.pdf') as doc:
        assert all('한국어 재배치' in p.get_text() for p in doc)
    result=dict(status='passed',frozen=bool(args.exe),seconds=round(time.monotonic()-began,2),
                ocr_pages=3,connected_paragraphs=1,failed_paragraphs=0,source_hash_equal=True,
                job=str(job),output=str(output))
    atomic_json(work/('frozen-result.json' if args.exe else 'source-result.json'),result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
