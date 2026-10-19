import copy,io,json,re
from pathlib import Path
import pymupdf as fitz
import pytest
from PIL import Image,ImageDraw
from papertranslator.document import analyze
from papertranslator.layout import bilingual_pdf
from papertranslator.alignment import validate_boundaries,sentence_spans,run


def records_for(ir):
    return {b['id']:{**b,'status':'preserved' if b['preserve'] else 'translated',
            'target':b['source'] if b['preserve'] else '해당 영역의 한국어 번역입니다.','issues':[]} for b in ir['blocks']}


def simple(path):
    with fitz.open() as doc:
        page=doc.new_page(width=600,height=800)
        page.insert_text((50,90),'Alpha source paragraph.',fontsize=12)
        page.insert_text((50,410),'Beta source paragraph.',fontsize=12)
        page.draw_rect(fitz.Rect(80,600,180,680),fill=(1,0,0),color=None)
        doc.save(path)


def test_bilingual_paragraphs_follow_source_positions_and_left_pixels(tmp_path):
    source=tmp_path/'source.pdf';simple(source);ir=analyze(source,[0],ocr_mode='off');records=records_for(ir)
    for r in records.values():r['target']='위쪽 가 문단.' if 'Alpha' in r['source'] else '아래쪽 나 문단.'
    output=tmp_path/'aligned.pdf';layout=bilingual_pdf(ir,records,source,output)
    assert layout['overflow_pages']==0 and layout['source_indices']==[0]
    with fitz.open(source) as original,fitz.open(output) as dual:
        assert original[0].get_pixmap().samples==dual[0].get_pixmap(clip=original[0].rect).samples
        right=dual[0].get_text(clip=fitz.Rect(600,0,1200,800))
        assert 'Alpha' not in right and 'Beta' not in right
        upper=dual[0].search_for('위쪽')[0];lower=dual[0].search_for('아래쪽')[0]
        assert 600+45<=upper.x0<=600+60 and 70<=upper.y0<=100
        assert 390<=lower.y0<=420
        a=original[0].get_pixmap(clip=fitz.Rect(80,600,180,680))
        b=dual[0].get_pixmap(clip=fitz.Rect(680,600,780,680))
        assert a.samples==b.samples and len(dual[0].get_links())==2


def test_readable_overflow_keeps_every_character_and_same_source(tmp_path):
    source=tmp_path/'source.pdf';simple(source);ir=analyze(source,[0],ocr_mode='off');records=records_for(ir)
    for r in records.values():r['target']='누락검사 가나다. '*250
    output=tmp_path/'overflow.pdf';layout=bilingual_pdf(ir,records,source,output)
    assert layout['overflow_pages']>0 and set(layout['source_indices'])=={0}
    assert all(r['font_size']>=9 for r in layout['regions'])
    with fitz.open(output) as doc,fitz.open(source) as original:
        text=''.join(p.get_text(clip=fitz.Rect(600,0,1200,800)) for p in doc)
        assert ''.join(text.split()).count('누락검사')==500
        for page in doc:assert page.get_pixmap(clip=original[0].rect).samples==original[0].get_pixmap().samples


def table(path,scan=False):
    with fitz.open() as doc:
        page=doc.new_page(width=600,height=800)
        if scan:
            image=Image.new('RGB',(600,800),'white');draw=ImageDraw.Draw(image)
            for x in [50,250,450]:draw.line((x,150,x,300),fill='black',width=2)
            for y in [150,200,250,300]:draw.line((50,y,450,y),fill='black',width=2)
            stream=io.BytesIO();image.save(stream,format='PNG');page.insert_image(page.rect,stream=stream.getvalue())
        else:
            for x in [50,250,450]:page.draw_line((x,150),(x,300))
            for y in [150,200,250,300]:page.draw_line((50,y),(450,y))
            for x,y,text in [(70,175,'Group'),(270,175,'Count'),(70,225,'Treatment'),(270,225,'24'),(70,275,'Control'),(270,275,'24')]:page.insert_text((x,y),text)
        doc.save(path)


@pytest.mark.parametrize('scan',[False,True])
def test_ruled_tables_keep_cell_membership_and_translation_inside_cells(tmp_path,scan):
    source=tmp_path/'table.pdf';table(source,scan)
    def recognize(pixels):
        scale=pixels.shape[0]/800
        return [[[[x*scale,y*scale],[(x+70)*scale,y*scale],[(x+70)*scale,(y+12)*scale],[x*scale,(y+12)*scale]],text,None]
                for x,y,text in [(70,163,'Group'),(270,163,'Count'),(70,213,'Treatment'),(270,213,'24'),(70,263,'Control'),(270,263,'24')]]
    ir=analyze(source,[0],ocr_mode='force' if scan else 'off',ocr_engine=recognize if scan else None)
    assert len(ir['pages'][0]['table_regions'])==1
    cells=[b for b in ir['blocks'] if b['kind']=='table_cell']
    assert len(cells)==6 and all(len(b['source_page_indices'])==1 for b in cells)
    records=records_for(ir)
    for r in records.values():
        if not r['preserve']:r['target']={'Group':'집단','Count':'인원','Treatment':'치료군','Control':'대조군'}[r['source']]
    output=tmp_path/'table-aligned.pdf';layout=bilingual_pdf(ir,records,source,output)
    assert layout['table_regions']==1 and layout['overflow_pages']==0
    with fitz.open(output) as doc:
        group=doc[0].search_for('치료군')[0];control=doc[0].search_for('대조군')[0]
        assert 650<group.x0<850 and 200<group.y0<250
        assert 250<control.y0<300


def test_small_table_rows_keep_readable_labels_and_the_original_rules(tmp_path):
    source=tmp_path/'tight.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        for x in [50,250,450]:p.draw_line((x,150),(x,186))
        for y in [150,162,174,186]:p.draw_line((50,y),(450,y))
        for y,left,right in [(158,'Group','Count'),(170,'Treatment','24'),(182,'Control','24')]:
            p.insert_text((55,y),left,fontsize=7);p.insert_text((255,y),right,fontsize=7)
        doc.save(source)
    ir=analyze(source,[0],ocr_mode='off');records=records_for(ir)
    for r in records.values():
        if r['status']=='translated':r['target']='집단'
    output=tmp_path/'paired.pdf';layout=bilingual_pdf(ir,records,source,output)
    assert layout['overflow_pages']==0 and all(r['font_size']>=9 for r in layout['regions'])
    with fitz.open(source) as src,fitz.open(output) as doc:
        a=src[0].get_pixmap(clip=fitz.Rect(100,161.5,150,162.5))
        b=doc[0].get_pixmap(clip=fitz.Rect(700,161.5,750,162.5))
        assert a.samples==b.samples


def connected():
    return dict(id='connected',source='The study included 24 people. The treatment lasted two years.',
        target='연구에는 24명이 참여했다. 치료는 2년간 지속되었다.',status='translated',issues=[],
        source_page_index=0,source_page_indices=[0,1],source_fragments=[
            dict(source_page_index=0,source='The study included 24 people.',bbox=[50,700,550,730]),
            dict(source_page_index=1,source='The treatment lasted two years.',bbox=[50,70,550,95])])


def boundaries(shared=False,decision='matched'):
    return dict(boundaries=[dict(last_before=0,first_after=0 if shared else 1,
        before_quote='24 people',after_quote='two years',decision=decision)])


def test_page_partition_keeps_every_original_character():
    r=connected();parts,status=validate_boundaries(boundaries(),r)
    assert status=='aligned' and ''.join(p['target'] for p in parts)==r['target']
    assert [p['source_page_index'] for p in parts]==[0,1]


@pytest.mark.parametrize('change',['skip','negative','string_id','bool_id','source_quote','decision','extra','count'])
def test_page_partition_rejects_invalid_boundaries(change):
    value=boundaries();item=value['boundaries'][0]
    if change=='skip':item['first_after']=2
    if change=='negative':item['last_before']=-1
    if change=='string_id':item['last_before']='0'
    if change=='bool_id':item['last_before']=False
    if change=='source_quote':item['before_quote']='invented'
    if change=='decision':item['decision']='approved'
    if change=='extra':item['rewritten_text']='new translation'
    if change=='count':value['boundaries']=[]
    with pytest.raises(ValueError):validate_boundaries(value,connected())


def test_uncertain_model_proposal_remains_explicitly_unconfirmed():
    parts,status=validate_boundaries(boundaries(decision='uncertain'),connected())
    assert status=='proposed' and ''.join(p['target'] for p in parts)==connected()['target']


def test_multiple_page_boundaries_reject_backwards_sentence_ranges():
    r=connected();r['target']='첫 문장이다. 둘째 문장이다. 셋째 문장이다.'
    r['source_fragments'].append(dict(source_page_index=2,source='Next source.'))
    value=boundaries();value['boundaries'][0].update(last_before=1,first_after=2)
    value['boundaries'].append(dict(last_before=1,first_after=2,before_quote='two years',after_quote='Next',decision='matched'))
    with pytest.raises(ValueError):validate_boundaries(value,r)


def test_table_detection_leaves_plain_captioned_prose_as_body(tmp_path):
    source=tmp_path/'prose.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800);p.insert_text((50,80),'Table 1. A textual description')
        p.insert_text((50,140),'This description discusses research without repeated numeric columns.')
        p.insert_text((50,180),'Source: Synthetic text.');doc.save(source)
    ir=analyze(source,[0],ocr_mode='off')
    assert not ir['pages'][0]['table_regions']
    assert not any(b['kind']=='table_cell' for b in ir['blocks'])


def test_adjacent_columns_do_not_merge_into_one_paragraph(tmp_path):
    source=tmp_path/'columns.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        p.insert_text((50,100),'Left research paragraph continues')
        p.insert_text((330,100),'Right independent paragraph begins')
        doc.save(source)
    ir=analyze(source,[0],ocr_mode='off')
    assert len(ir['blocks'])==2


def test_indented_paragraph_after_short_finished_line_keeps_its_own_region(tmp_path):
    source=tmp_path/'indented.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        for x,y,text in [(50,100,'The first paragraph describes a long continuous research topic'),
                         (50,115,'and ends here.'),(65,130,'A new indented paragraph starts here.'),
                         (50,145,'It has another line about the research.')]:p.insert_text((x,y),text)
        doc.save(source)
    ir=analyze(source,[0],ocr_mode='off')
    assert len(ir['blocks'])==2
    assert ir['blocks'][0]['bbox'][1]<ir['blocks'][1]['bbox'][1]
    layout=bilingual_pdf(ir,records_for(ir),source,tmp_path/'paired.pdf')
    assert len(layout['regions'])==2


def test_split_requests_sharing_one_source_line_cannot_paint_over_each_other(tmp_path):
    source=tmp_path/'shared.pdf';simple(source)
    blocks=[];records={}
    for i,box in enumerate([[50,70,550,180],[50,174,550,210]]):
        r=dict(id=str(i),kind='paragraph',source='Synthetic source.',target='연결 번역 문장이다. '*12,
               source_page_index=0,source_page_indices=[0],bbox=box,status='translated',issues=[],
               source_fragments=[dict(source_page_index=0,source='Synthetic source.',bbox=box)])
        blocks.append(r);records[str(i)]=r
    layout=bilingual_pdf(dict(blocks=blocks,pages=[dict(source_page_index=0)]),records,source,tmp_path/'paired.pdf')
    assert len(layout['regions'])==1 and layout['regions'][0]['ids']==['0','1']


def test_alignment_replay_and_raw_response_retention(tmp_path):
    calls=[];record=connected()
    class Backend:
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def start(self,*args):pass
        def translate(self,*args,**kwargs):calls.append(1);return dict(text=json.dumps(boundaries(),ensure_ascii=False))
    for _ in range(2):run({'connected':record},dict(id='synthetic'),tmp_path,Backend,lambda:None,lambda _:None,lambda:False)
    assert calls==[1] and record['page_alignment']['status']=='aligned'
    assert (tmp_path/'responses/alignment-connected.json').exists()


def test_shared_boundary_sentence_is_repeated_unchanged_on_both_pages():
    r=connected();r['target']='연구에는 24명이 참여했고 치료는 2년간 지속되었다. 마지막 문장이다.'
    parts,status=validate_boundaries(boundaries(shared=True),r)
    assert status=='aligned' and all(p['shared_boundary'] for p in parts)
    assert parts[1]['target']==r['target'] and parts[0]['target']==r['target'][:parts[0]['end']]
    assert len(sentence_spans('Dr. Smith의 연구다. 다음 문장이다.'))==2


def test_verified_cross_page_parts_are_printed_only_on_their_source_pages(tmp_path):
    source=tmp_path/'two.pdf'
    with fitz.open() as doc:
        for text,y in [('First source',710),('Second source',85)]:
            p=doc.new_page(width=600,height=800);p.insert_text((50,y),text)
        doc.save(source)
    r=connected();r.update(kind='paragraph',bbox=[50,700,550,730],page_alignment=dict(status='aligned',parts=validate_boundaries(boundaries(),r)[0]))
    ir=dict(blocks=[dict(id=r['id'])],pages=[dict(source_page_index=0),dict(source_page_index=1)])
    output=tmp_path/'aligned.pdf';layout=bilingual_pdf(ir,{r['id']:r},source,output)
    assert not layout['unresolved_page_alignments']
    with fitz.open(output) as doc:
        first=doc[0].get_text(clip=fitz.Rect(600,0,1200,800));second=doc[1].get_text(clip=fitz.Rect(600,0,1200,800))
        assert '24명' in first and '2년' not in first
        assert '2년' in second and '24명' not in second


def test_split_logical_requests_keep_only_their_own_source_line_boxes(tmp_path):
    source=tmp_path/'long.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=1200)
        for index in range(42):p.insert_text((50,70+index*20),f'Line {index} describes the same continuous research topic and its context.')
        doc.save(source)
    ir=analyze(source,[0],ocr_mode='off');body=[b for b in ir['blocks'] if not b['preserve']]
    assert len(body)>1
    assert body[0]['source_fragments'][0]['bbox'][3]<body[-1]['source_fragments'][0]['bbox'][1]


@pytest.mark.parametrize('with_source',[False,True])
def test_captioned_borderless_table_uses_repeated_numeric_columns(tmp_path,with_source):
    source=tmp_path/'borderless.pdf'
    with fitz.open() as doc:
        page=doc.new_page(width=600,height=800);page.insert_text((50,80),'Table 1. Comparison of groups')
        for x,text in [(50,'Group'),(340,'Problems'),(440,'Risk')]:page.insert_text((x,130),text)
        for i,name in enumerate(['First treatment','Second treatment','Control treatment']):
            page.insert_text((50,160+i*25),f'{i+1}.');page.insert_text((75,160+i*25),name)
            page.insert_text((340,160+i*25),'3');page.insert_text((440,160+i*25),'2')
        if with_source:page.insert_text((50,260),'Source: Synthetic comparison.')
        page.insert_text((50,320),'This paragraph follows the table and describes the context of the study in detail.',fontsize=12)
        doc.save(source)
    ir=analyze(source,[0],ocr_mode='off')
    regions=ir['pages'][0]['table_regions'];assert len(regions)==1
    assert regions[0]['method']=='captioned_numeric_columns'
    assert any(b['kind']=='table_cell' and b['source']=='First treatment' for b in ir['blocks'])
    records=records_for(ir);output=tmp_path/'paired.pdf';bilingual_pdf(ir,records,source,output)
    with fitz.open(output) as doc:
        right=doc[0].get_text(clip=fitz.Rect(600,130,1200,240))
        assert '1.' in right and '2.' in right and '3.' in right


@pytest.mark.parametrize('rotation',[90,180,270])
def test_source_position_keeps_rotated_cropped_original_faces(tmp_path,rotation):
    source=tmp_path/'rotated.pdf'
    with fitz.open() as doc:
        page=doc.new_page(width=600,height=400);page.insert_text((60,110),'A source sentence is here.')
        page.set_cropbox(fitz.Rect(25,20,570,370));page.set_rotation(rotation);doc.save(source)
    ir=analyze(source,[0],ocr_mode='off');records=records_for(ir);output=tmp_path/'aligned.pdf'
    bilingual_pdf(ir,records,source,output)
    with fitz.open(source) as original,fitz.open(output) as doc:
        assert doc[0].get_pixmap(clip=original[0].rect).samples==original[0].get_pixmap().samples


def test_overflow_rewraps_the_whole_item_instead_of_reusing_narrow_line_breaks(tmp_path):
    source=tmp_path/'narrow.pdf';simple(source)
    r=dict(id='narrow',kind='table_cell',source='A label',target='보충면에서는 넓은 영역을 사용해 문장 전체를 읽을 수 있어야 한다. '*20,
           source_page_index=0,source_page_indices=[0],bbox=[50,100,82,125],status='translated',issues=[],
           source_fragments=[dict(source_page_index=0,source='A label',bbox=[50,100,82,125],cell_bbox=[50,100,82,125])])
    output=tmp_path/'paired.pdf';layout=bilingual_pdf(dict(blocks=[r],pages=[dict(source_page_index=0)]),{r['id']:r},source,output)
    assert layout['validation']['status']=='passed'
    with fitz.open(output) as doc:
        lines=[l for b in doc[1].get_text('dict')['blocks'] if b['type']==0 for l in b['lines']
               if l['bbox'][0]>600 and any(s['size']>=9 for s in l['spans'])]
        assert max(l['bbox'][2]-l['bbox'][0] for l in lines)>350
        text=''.join(s['text'] for l in lines for s in l['spans'])
        assert ''.join(text.split())==''.join(r['target'].split())


def test_short_list_items_stay_next_to_their_bullets(tmp_path):
    source=tmp_path/'list.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800)
        p.insert_text((75,300),'resources;',fontsize=8)
        p.insert_text((75,313),'failures and crises;',fontsize=8);doc.save(source)
    ir=analyze(source,[0],ocr_mode='off');records=records_for(ir)
    for r in records.values():r['target']='자원;' if r['source']=='resources;' else '실패와 위기;'
    output=tmp_path/'paired.pdf';layout=bilingual_pdf(ir,records,source,output)
    assert layout['overflow_pages']==0 and layout['validation']['status']=='passed'
    with fitz.open(output) as doc:assert '자원;' in doc[0].get_text() and '실패와 위기;' in doc[0].get_text()


def test_rotated_diagram_label_keeps_shape_and_uses_its_original_direction(tmp_path):
    source=tmp_path/'diagram.pdf'
    with fitz.open() as doc:
        p=doc.new_page(width=600,height=800);p.draw_rect(fitz.Rect(100,150,145,300));doc.save(source)
    r=dict(id='label',kind='diagram_label',source='Clinical judgement',target='임상적 판단',status='translated',issues=[],
           source_page_index=0,source_page_indices=[0],bbox=[110,165,130,280],
           source_fragments=[dict(source_page_index=0,source='Clinical judgement',bbox=[110,165,130,280],rotation=90,region_bbox=[102,152,143,298])])
    output=tmp_path/'paired.pdf';layout=bilingual_pdf(dict(blocks=[r],pages=[dict(source_page_index=0)]),{'label':r},source,output)
    assert layout['overflow_pages']==0 and layout['validation']['status']=='passed'
    with fitz.open(output) as doc:
        line=next(l for b in doc[0].get_text('dict')['blocks'] if b['type']==0 for l in b['lines'] if any('임상적' in s['text'] for s in l['spans']))
        assert line['dir']==(0.,-1.)


def test_preserved_row_marker_is_not_covered_by_the_korean_paragraph(tmp_path):
    source=tmp_path/'marker.pdf';simple(source)
    row=dict(id='row',kind='paragraph',source='Evidence from experts.',target='전문가의 의견에 따른 근거.',status='translated',issues=[],
             source_page_index=0,source_page_indices=[0],bbox=[70,200,500,225],source_fragments=[dict(source_page_index=0,source='Evidence from experts.',bbox=[70,200,500,225])])
    marker=dict(id='marker',kind='protected',source='IV.',target='IV.',status='preserved',issues=[],source_page_index=0,source_page_indices=[0],bbox=[70,200,85,215],
                source_fragments=[dict(source_page_index=0,source='IV.',bbox=[70,200,85,215])])
    layout=bilingual_pdf(dict(blocks=[row,marker],pages=[dict(source_page_index=0)]),{'row':row,'marker':marker},source,tmp_path/'paired.pdf')
    assert layout['regions'][0]['target_bbox'][0]>=90


def test_margin_rule_fragment_cannot_pull_a_connected_body_up_into_the_header(tmp_path):
    source=tmp_path/'header.pdf';simple(source)
    row=dict(id='row',kind='paragraph',source='disr---- A body paragraph.',target='본문 번역입니다.',status='translated',issues=[],
             source_page_index=0,source_page_indices=[0],bbox=[50,30,550,130],source_fragments=[
                 dict(source_page_index=0,source='disr----',bbox=[400,30,550,45]),
                 dict(source_page_index=0,source='A body paragraph.',bbox=[50,100,550,130])])
    layout=bilingual_pdf(dict(blocks=[row],pages=[dict(source_page_index=0)]),{'row':row},source,tmp_path/'paired.pdf')
    assert layout['regions'][0]['source_bbox'][1]==100


def test_footer_scan_rule_is_retained_as_metadata_and_never_translated(tmp_path):
    source=tmp_path/'footer.pdf'
    with fitz.open() as doc:
        doc.new_page(width=600,height=800);doc.save(source)
    def recognize(pixels):
        scale=pixels.shape[1]/600
        return [[[[x*scale,y*scale] for x,y in [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]],text,.9]
                for text,x0,y0,x1,y1 in [('A complete body sentence.',50,100,500,120),('—scan——',440,735,540,751)]]
    ir=analyze(source,[0],ocr_mode='force',ocr_engine=recognize)
    footer=next(b for b in ir['blocks'] if b['source']=='—scan——')
    assert footer['preserve'] and footer['kind']=='page_marker'
    records=records_for(ir)
    # An older checkpoint may still call the rule a translated paragraph.
    records[footer['id']].update(kind='paragraph',status='translated',target='—scan——')
    layout=bilingual_pdf(ir,records,source,tmp_path/'paired.pdf')
    assert all(footer['id'] not in region['ids'] for region in layout['regions'])


def test_evidence_levels_keep_their_individual_rows(tmp_path):
    source=tmp_path/'levels.pdf'
    with fitz.open() as doc:
        page=doc.new_page(width=600,height=800)
        for i in range(5):
            page.insert_text((60,200+i*50),f'Level {i+1}',fontsize=10)
            page.insert_text((130,200+i*50),f'Evidence from synthetic study number {i+1}.',fontsize=10)
        doc.save(source)
    ir=analyze(source,[0],ocr_mode='off')
    labels=[b for b in ir['blocks'] if b['kind']=='list_label']
    assert len(labels)==5 and {b['source'] for b in labels}=={f'Level {i+1}' for i in range(5)}
    bodies=[b for b in ir['blocks'] if b['kind']=='paragraph']
    assert len(bodies)==5
    records=records_for(ir)
    for r in records.values():
        r['target']=r['source'].replace('Level ','')+'단계' if r['kind']=='list_label' else '각 행의 근거 설명입니다.'
    output=tmp_path/'paired.pdf';layout=bilingual_pdf(ir,records,source,output)
    assert not layout['overflow_pages'] and layout['validation']['status']=='passed'
    with fitz.open(output) as doc:
        for i in range(5):
            right=doc[0].get_text(clip=fitz.Rect(600,180+i*50,1200,220+i*50))
            assert f'{i+1}단계' in right and '각 행의 근거 설명입니다.' in right


def test_dark_cover_text_is_not_a_ruled_table(tmp_path):
    from papertranslator.regions import table_regions
    source=tmp_path/'dark-cover.pdf'
    image=Image.new('RGB',(1000,1400),'#141414');draw=ImageDraw.Draw(image)
    from PIL import ImageFont
    font=ImageFont.truetype('arial.ttf',28)
    for i in range(18):draw.text((50,60+i*65),'Evidence and research in clinical practice.',font=font,fill='#e7c952')
    stream=io.BytesIO();image.save(stream,format='PNG')
    with fitz.open() as doc:
        page=doc.new_page(width=500,height=700);page.insert_image(page.rect,stream=stream.getvalue());doc.save(source)
    with fitz.open(source) as doc:assert table_regions(doc[0],scanned=True)==[]


def test_captioned_comparison_keeps_qualitative_and_quantitative_columns(tmp_path):
    source=tmp_path/'comparison.pdf'
    with fitz.open() as doc:
        page=doc.new_page(width=600,height=800)
        page.insert_text((60,100),'Table 1. Comparison of research approaches')
        for x,t in [(60,'Qualitative'),(340,'Quantitative')]:page.insert_text((x,125),t)
        for i,(left,right) in enumerate([('Flexible','Fixed'),('Subjective','Objective'),('Political','Value-free'),('Case study','Survey'),('Grounded','Abstract')]):
            page.insert_text((60,150+i*20),left);page.insert_text((340,150+i*20),right)
        page.insert_text((60,260),'Source: Synthetic comparison.');doc.save(source)
    ir=analyze(source,[0],ocr_mode='off')
    assert ir['pages'][0]['table_regions'][0]['method']=='captioned_comparison_columns'
    cells=[b for b in ir['blocks'] if b['kind']=='table_cell']
    assert len(cells)==12 and any(b['source']=='Flexible' for b in cells)
    assert not any('Flexible' in b['source'] and 'Fixed' in b['source'] for b in cells)


def test_dark_source_keeps_dark_translation_background_and_readable_text(tmp_path):
    source=tmp_path/'dark.pdf'
    with fitz.open() as doc:
        page=doc.new_page(width=600,height=800);page.draw_rect(page.rect,fill=(.05,.05,.05),color=None)
        page.insert_text((50,100),'A dark source paragraph.',color=(1,1,.8));doc.save(source)
    ir=analyze(source,[0],ocr_mode='off');output=tmp_path/'paired.pdf'
    layout=bilingual_pdf(ir,records_for(ir),source,output)
    assert layout['validation']['status']=='passed'
    with fitz.open(output) as doc:
        spans=[s for b in doc[0].get_text('dict')['blocks'] if b['type']==0 for l in b['lines'] for s in l['spans'] if '한국어' in s['text']]
        assert spans and spans[0]['color']>0xCCCCCC
        # Sample cleared whitespace beyond the last Korean glyph, avoiding
        # antialiased letter edges that legitimately are lighter than the fill.
        pix=doc[0].get_pixmap(clip=fitz.Rect(763,87,766,99));assert max(pix.samples)<80


def test_next_page_caption_does_not_complete_the_previous_body_sentence(tmp_path):
    source=tmp_path/'caption.pdf'
    with fitz.open() as doc:
        p=doc.new_page();p.insert_text((50,740),'Many approaches include methods such as')
        p=doc.new_page();p.insert_text((50,80),'Table 1. Features of research')
        p.insert_text((50,450),'observation and analysis of interviews.');doc.save(source)
    ir=analyze(source,[0,1],ocr_mode='off')
    assert all(len(b['source_page_indices'])==1 for b in ir['blocks'])
    assert next(b for b in ir['blocks'] if b['source'].startswith('Table 1.'))['kind']=='caption'


@pytest.mark.parametrize('gap',[-1,3])
def test_shared_boundary_uses_the_continuous_body_column_before_spilling(tmp_path,gap):
    source=tmp_path/'boundary-column.pdf'
    with fitz.open() as doc:
        for _ in range(2):doc.new_page(width=600,height=800)
        doc.save(source)
    text='경계 문장 전체의 의미를 유지하려면 다음 쪽에서도 완성된 문장을 읽을 수 있어야 한다. '*2
    r=dict(id='boundary',kind='paragraph',source='A sentence continues across pages',target=text,status='translated',issues=[],
           source_page_index=0,source_page_indices=[0,1],bbox=[50,700,550,750],source_fragments=[
               dict(source_page_index=0,source='A sentence continues',bbox=[50,700,550,750]),
               dict(source_page_index=1,source='across pages',bbox=[50,70,550,89])],page_alignment=dict(status='aligned',parts=[
                   dict(source_page_index=i,target=text,start=0,end=len(text),shared_boundary=True) for i in [0,1]]))
    after=dict(id='after',kind='paragraph',source='The next paragraph.',target='다음 본문도 빠짐없이 표시한다. '*10,status='translated',issues=[],
               source_page_index=1,source_page_indices=[1],bbox=[50,89+gap,550,240],source_fragments=[dict(source_page_index=1,source='The next paragraph.',bbox=[50,89+gap,550,240])])
    layout=bilingual_pdf(dict(blocks=[r,after],pages=[dict(source_page_index=i) for i in [0,1]]),{'boundary':r,'after':after},source,tmp_path/'paired.pdf')
    assert not layout['overflow_pages'] and layout['validation']['status']=='passed'
    assert any(set(region['ids'])=={'boundary','after'} for region in layout['regions'])
