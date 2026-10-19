import numpy as np
import pytest
from PIL import Image,ImageDraw,ImageFont
from papertranslator.ocr import LocalOCR
from papertranslator.ocr_assets import data_directory


def diagram_image():
    image=Image.new('RGB',(1800,1200),'white');draw=ImageDraw.Draw(image);font=ImageFont.truetype('arial.ttf',32)
    labels=['Clinical\njudgement','Clinical\npractice','Patient\ncare',
            'Research\nevidence','Case series\nevaluation','Training\neducation']
    for i,label in enumerate(labels):
        x=[100,650,1200][i%3];y=150 if i<3 else 650
        draw.rectangle((x,y,x+300,y+130),outline='black',width=4);draw.multiline_text((x+20,y+20),label,font=font,fill='black')
    draw.line((400,215,650,215),fill='black',width=4);draw.polygon([(650,215),(632,205),(632,225)],fill='black')
    draw.line((950,215,1200,215),fill='black',width=4)
    return image.rotate(90,expand=True)


@pytest.mark.skipif(data_directory() is None,reason='Pinned English OCR data is required')
def test_sideways_diagram_recovers_complete_nodes_and_keeps_geometry():
    pixels=np.asarray(diagram_image())[:,:,::-1].copy();rows=LocalOCR()(pixels)
    texts={r[1] for r in rows}
    assert {'Clinical judgement','Clinical practice','Patient care','Research evidence','Case series evaluation','Training education'}<=texts
    for row in rows:
        assert row[3]['rotation'] in {90,270}
        box=row[3]['region_bbox'];assert 0<=box[0]<box[2]<=pixels.shape[1] and 0<=box[1]<box[3]<=pixels.shape[0]
        assert row[3]['erase_patch'] and row[3]['erase_patch']['png']


@pytest.mark.skipif(data_directory() is None,reason='Pinned English OCR data is required')
def test_upright_diagram_recovers_nodes_without_changing_direction():
    pixels=np.asarray(diagram_image().rotate(-90,expand=True))[:,:,::-1].copy()
    rows=LocalOCR()(pixels);labels=[r for r in rows if len(r)>3 and r[3].get('diagram_label')]
    assert len(labels)==6
    assert {'Clinical judgement','Clinical practice','Patient care','Research evidence','Case series evaluation','Training education'}=={r[1] for r in labels}
    assert all(not r[3].get('rotation') and r[3]['erase_patch'] for r in labels)
