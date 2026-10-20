"""Bounded, offline CPU OCR. Never download models or modify the source PDF."""
from pathlib import Path
import math
import re

import pymupdf as fitz

from .backend import Cancelled
from .storage import sha256
from .ocr_assets import data_directory, SHA256 as ENGLISH_SHA256

VERSION = 'english-lstm-best-rapid-fallback-6-rules-forms-diagrams'
MODES = {'auto', 'off', 'force'}
MODEL_HASHES = {
    'ch_PP-OCRv4_det_infer.onnx': 'd2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9',
    'ch_PP-OCRv4_rec_infer.onnx': '48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b',
    'ch_ppocr_mobile_v2.0_cls_infer.onnx': 'e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c',
}
LOW_CONFIDENCE = .75
MAX_PIXELS = 12_000_000
MAX_SIDE = 3500


def signature(mode, recovery=False):
    if mode not in MODES:
        raise ValueError('지원하지 않는 OCR 모드입니다.')
    result = dict(mode=mode, version=VERSION, models=MODEL_HASHES,
                english_lstm=ENGLISH_SHA256 if data_directory() else None,dpi=220,
                max_pixels=MAX_PIXELS, max_side=MAX_SIDE, low_confidence=LOW_CONFIDENCE)
    if recovery:
        from .ocr_recovery import VERSION as RECOVERY_VERSION
        result['recovery'] = RECOVERY_VERSION
    return result


class LocalOCR:
    def __init__(self):
        self.english=data_directory();self.engine=None
        if self.english:
            if sha256(self.english/'eng.traineddata')!=ENGLISH_SHA256:
                raise RuntimeError('영어 OCR 데이터 검증 실패: 다시 설치하세요.')

    def _rapid(self):
        if self.engine is not None:return self.engine
        import cv2
        import onnxruntime
        import rapidocr_onnxruntime
        from rapidocr_onnxruntime import RapidOCR
        onnxruntime.disable_telemetry_events()
        cv2.setNumThreads(4)
        folder = Path(rapidocr_onnxruntime.__file__).parent / 'models'
        for name, digest in MODEL_HASHES.items():
            path = folder / name
            if not path.is_file() or sha256(path) != digest:
                raise RuntimeError('OCR 모델 파일 검증 실패: 앱을 다시 설치하세요.')
        # Bundled models only: no downloader, CUDA or DirectML provider.
        self.engine = RapidOCR(intra_op_num_threads=4, inter_op_num_threads=1,
                               det_use_cuda=False, cls_use_cuda=False, rec_use_cuda=False,
                               det_use_dml=False, cls_use_dml=False, rec_use_dml=False,
                               max_side_len=MAX_SIDE, det_limit_side_len=1536,
                               det_limit_type='max', text_score=0.)
        return self.engine

    def _english_rows(self, pixels):
        if self.english:
            height,width=pixels.shape[:2]
            # Long scan/table rules confuse OCR into invented words. Remove
            # them only in the recognition copy, retaining source coordinates.
            if min(height,width)>500:
                import cv2
                import numpy as np
                gray=cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY)
                if np.mean(gray>180)>.65:
                    ink=cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV)[1]
                    rules=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(100,width//8)),np.uint8))
                    rules|=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((max(100,height//10),1),np.uint8))
                    pixels=pixels.copy();pixels[cv2.dilate(rules,np.ones((3,3),np.uint8))>0]=255
            pix=fitz.Pixmap(fitz.csRGB,width,height,pixels[:,:,::-1].copy().tobytes(),False)
            pix.set_dpi(220,220)
            data=pix.pdfocr_tobytes(language='eng',tessdata=str(self.english))
            rows=[]
            with fitz.open('pdf',data) as document:
                page=document[0];groups={}
                for word in page.get_text('words'):
                    groups.setdefault((word[5],word[6]),[]).append(word)
                for (block,line),words in groups.items():
                    words.sort(key=lambda w:w[7])
                    text=' '.join(w[4] for w in words)
                    x0=min(w[0] for w in words)*220/72;y0=min(w[1] for w in words)*220/72
                    x1=max(w[2] for w in words)*220/72;y1=max(w[3] for w in words)*220/72
                    rows.append([[[x0,y0],[x1,y0],[x1,y1],[x0,y1]],text,None,
                                 dict(block=block,words=[dict(text=w[4],bbox=[v*220/72 for v in w[:4]]) for w in words])])
            return rows
        return []

    def __call__(self,pixels):
        rows=self._english_rows(pixels)
        if rows:
            from .orientation import recover,recover_upright,quality
            import numpy as np
            if np.mean(np.mean(pixels,axis=2)<90)>.55:
                inverted=self._english_rows(255-pixels)
                if quality(inverted)[1]>quality(rows)[1]*1.08 and quality(inverted)[0]>.65:
                    rows=inverted
                    for row in rows:row[3]['inverted']=True
            rows=recover(pixels,rows,self._english_rows)
            if not any(len(row)>3 and row[3].get('rotation') for row in rows):
                from .forms import recover as recover_form
                rows=recover_form(pixels,rows,self._english_rows)
                if not any(len(row)>3 and row[3].get('layout_kind') for row in rows):
                    rows=recover_upright(pixels,rows,self._english_rows)
            if any(re.search(r'\b[A-Za-z]{3,}\b',r[1]) for r in rows):return rows
        result, _ = self._rapid()(pixels)
        return result or []


def overlap(a, b):
    a, b = fitz.Rect(a), fitz.Rect(b)
    area = min(a.get_area(), b.get_area())
    return (a & b).get_area() / area if area else 0.


def page_blocks(page, mode='auto', engine=None, cancel=lambda: False, recovery=False):
    """Return ordinary text blocks plus OCR lines in displayed page coordinates."""
    signature(mode)
    if cancel():
        raise Cancelled()
    native = page.get_text('dict', flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES, sort=True)['blocks']
    matrix = page.rotation_matrix
    native_words=[dict(text=w[4],bbox=list(fitz.Rect(w[:4])*matrix)) for w in page.get_text('words')]
    for block in native:
        block['bbox'] = list(fitz.Rect(block['bbox']) * matrix)
        for line in block.get('lines', []):
            line['bbox'] = list(fitz.Rect(line['bbox']) * matrix)
            line['layout_words']=[w for w in native_words if fitz.Rect(line['bbox']).contains((fitz.Rect(w['bbox']).tl+fitz.Rect(w['bbox']).br)/2)]
            for span in line['spans']:
                span['bbox'] = list(fitz.Rect(span['bbox']) * matrix)
                span['extraction'] = 'native'
    chars = sum(len(s['text'].strip()) for b in native for l in b.get('lines', []) for s in l['spans'])
    area = page.rect.get_area()
    images = [fitz.Rect(i['bbox']) * matrix for i in page.get_image_info()]
    has_image = any((r & page.rect).get_area() >= area * .005 and min(r.width, r.height) >= 20 for r in images)
    needed = mode == 'force' or (mode == 'auto' and (has_image or chars < 20))
    info = dict(mode=mode, attempted=needed, engine=VERSION if needed else None,
                lines=0, low_confidence_lines=0, unscored_lines=0,duplicate_lines=0)
    warnings = []
    if not needed:
        if chars < 20:
            warnings.append('추출 텍스트가 적습니다. OCR이 꺼져 있어 이미지 속 글자는 번역되지 않을 수 있습니다.')
        return native, info, warnings
    import numpy as np
    # Bound allocations before rendering even for unusually large PDF media boxes.
    scale = min(220/72, (MAX_SIDE-2)/max(page.rect.width, page.rect.height),
                (math.sqrt(MAX_PIXELS)-2)/math.sqrt(area))
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False, annots=False)
    pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)[:, :, ::-1].copy()
    if cancel():
        raise Cancelled()
    recognize = engine or LocalOCR()
    rows = recognize(pixels)
    if cancel():
        raise Cancelled()
    if recovery:
        from .ocr_recovery import recover
        rows, info['recovery'] = recover(pixels, rows, recognize, cancel)
    retained = [] if mode == 'force' else native
    native_spans = [s for b in native for l in b.get('lines', []) for s in l['spans'] if s['text'].strip()]
    previous_top=None;previous_height=0
    for points, text, confidence, *extra in rows:
        if not text.strip():
            continue
        bbox = [min(p[0] for p in points)/scale, min(p[1] for p in points)/scale,
                max(p[0] for p in points)/scale, max(p[1] for p in points)/scale]
        reading_top=bbox[1]
        # MuPDF's invisible OCR font can give a following line a slightly
        # earlier top (especially a few enlarged words). Tesseract's logical
        # order is retained for these overlapping rows; source boxes stay exact.
        if (confidence is None and previous_top is not None and reading_top<=previous_top and
            previous_top-reading_top<=max(previous_height,bbox[3]-bbox[1])*1.5):
            reading_top=previous_top+.01
            info['order_adjusted_lines']=info.get('order_adjusted_lines',0)+1
        previous_top=reading_top;previous_height=bbox[3]-bbox[1]
        # Prefer precise native text over a second OCR rendition of the same region.
        matched=[s for s in native_spans if overlap(bbox,s['bbox'])>=.65]
        normalize=lambda value: re.sub(r'\W+','',value).casefold()
        same_text=bool(matched) and bool(normalize(text)) and normalize(text)==normalize(' '.join(s['text'] for s in matched))
        coverage=sum((fitz.Rect(bbox)&fitz.Rect(s['bbox'])).get_area() for s in matched)/max(1,fitz.Rect(bbox).get_area())
        if mode != 'force' and (same_text or coverage>=.9):
            info['duplicate_lines'] += 1
            continue
        confidence = float(confidence) if confidence is not None else None
        meta=extra[0] if extra and isinstance(extra[0],dict) else {}
        rotation=meta.get('rotation',0)
        diagram_label=meta.get('diagram_label',False)
        layout_kind=meta.get('layout_kind')
        region_bbox=[v/scale for v in meta['region_bbox']] if meta.get('region_bbox') else None
        erase_patch={**meta['erase_patch'],'bbox':[v/scale for v in meta['erase_patch']['bbox']]} if meta.get('erase_patch') else None
        size = max(6., min(14., (bbox[3]-bbox[1])*.8))
        span = dict(text=text.strip(), bbox=bbox, size=size, flags=0, font='OCR',
                    extraction='ocr', ocr_confidence=confidence,rotation=rotation,diagram_label=diagram_label,region_bbox=region_bbox,erase_patch=erase_patch)
        if meta.get('recovery'):
            span['ocr_recovery']=dict(version=meta['recovery'],region=meta['recovered_region'])
        retained.append(dict(type=0, bbox=bbox, reading_top=reading_top,
                             lines=[dict(bbox=bbox, spans=[span],reading_top=reading_top,rotation=rotation,diagram_label=diagram_label,layout_kind=layout_kind,region_bbox=region_bbox,erase_patch=erase_patch,
                                         layout_words=[dict(text=w['text'],bbox=[v/scale for v in w['bbox']])
                                                       for w in (extra[0].get('words',[]) if extra and isinstance(extra[0],dict) else [])])]))
        info['lines'] += 1
        info['low_confidence_lines'] += confidence is not None and confidence < LOW_CONFIDENCE
        info['unscored_lines'] += confidence is None
    if info['lines']:
        warnings.append(f'로컬 OCR {info["lines"]}줄: 인식 오류가 번역에 반영될 수 있으므로 원문을 대조하세요.')
    if info['low_confidence_lines']:
        warnings.append(f'OCR 낮은 신뢰도 {info["low_confidence_lines"]}줄: 숫자·이름·수식을 특히 확인하세요.')
    if info['unscored_lines']:
        warnings.append('영어 LSTM OCR은 인식 확률을 제공하지 않습니다. 점수 없음은 정확하다는 뜻이 아닙니다.')
    if any(len(row)>3 and row[3].get('rotation') for row in rows):info['oriented_labels']=True
    if any(len(row)>3 and row[3].get('inverted') for row in rows):info['inverted_ocr']=True
    detected=' '.join(row[1] for row in rows)
    letters=sum(c.isalpha() for c in detected)
    word_letters=sum(len(word) for word in re.findall(r'\b[A-Za-z]{3,}\b',detected))
    if len(rows)>=15 and letters and word_letters/letters<.55:
        info['fragmented_text']=True
        warnings.append('OCR이 짧은 문자 조각으로 분절됐습니다. 회전 도식·표·잡음의 글자는 누락되거나 순서가 바뀔 수 있어 원본 확인이 필요합니다.')
    if not rows and (chars < 20 or mode == 'force'):
        warnings.append('OCR에서 글자를 찾지 못했습니다. 빈 쪽·도형·흐린 스캔인지 원문을 확인하세요.')
    info['raster_size'] = [pix.width, pix.height]
    return sorted(retained, key=lambda b: (b.get('reading_top',b['bbox'][1]), b['bbox'][0])), info, warnings
