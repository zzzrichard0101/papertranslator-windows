import io
import pymupdf as fitz
import pytest
from PIL import Image,ImageDraw,ImageFont
from papertranslator.document import analyze,protected_kind
from papertranslator.ocr_assets import data_directory


def form_image():
    image=Image.new('RGB',(1800,2400),'white');draw=ImageDraw.Draw(image);font=ImageFont.truetype('arial.ttf',32)
    draw.text((100,100),'Synthetic appraisal checklist',font=font,fill='black')
    for x in [1250,1380,1530,1720]:draw.line((x,300,x,1700),fill='black',width=3)
    for y in [300,450,650,850,1050,1250,1450,1700]:draw.line((1250,y,1720,y),fill='black',width=3)
    for x,t in [(1270,'Yes'),(1400,'No'),(1550,"Don't")]:draw.text((x,350),t,font=font,fill='black')
    draw.text((1550,390),'Know',font=font,fill='black')
    for i in range(6):draw.text((100,490+i*200),f'1.{i+1} Is the study described clearly?',font=font,fill='black')
    return image


@pytest.mark.skipif(data_directory() is None,reason='Pinned English OCR data is required')
def test_questionnaire_keeps_six_questions_outside_the_response_grid(tmp_path):
    source=tmp_path/'form.pdf';stream=io.BytesIO();form_image().save(stream,format='PNG')
    with fitz.open() as doc:
        page=doc.new_page(width=600,height=800);page.insert_image(page.rect,stream=stream.getvalue());doc.save(source)
    ir=analyze(source,[0],ocr_mode='auto')
    questions=[b for b in ir['blocks'] if b['kind']=='form_question']
    assert len(questions)==6
    assert all(b['bbox'][2]<420 for b in questions)
    assert all('Yes' not in b['source'] and 'Know' not in b['source'] for b in questions)
    headers=[b for b in ir['blocks'] if b['kind']=='table_cell']
    assert {'Yes','No',"Don't Know"}<={b['source'] for b in headers}
    assert all(not b['preserve'] for b in headers)


def test_yes_and_no_are_translatable_words_not_symbols():
    assert protected_kind('Yes',[]) is None and protected_kind('No',[]) is None
    assert protected_kind('24',[])=='symbol'
