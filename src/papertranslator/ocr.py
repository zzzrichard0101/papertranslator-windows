"""Bounded, offline CPU OCR. Never download models or modify the source PDF."""
from pathlib import Path
import math
import re

import pymupdf as fitz

from .backend import Cancelled
from .storage import sha256

VERSION = 'rapidocr-1.4.4-ppocr4-cpu-1'
MODES = {'auto', 'off', 'force'}
MODEL_HASHES = {
    'ch_PP-OCRv4_det_infer.onnx': 'd2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9',
    'ch_PP-OCRv4_rec_infer.onnx': '48fc40f24f6d2a207a2b1091d3437eb3cc3eb6b676dc3ef9c37384005483683b',
    'ch_ppocr_mobile_v2.0_cls_infer.onnx': 'e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c',
}
LOW_CONFIDENCE = .75
MAX_PIXELS = 12_000_000
MAX_SIDE = 3500


def signature(mode):
    if mode not in MODES:
        raise ValueError('지원하지 않는 OCR 모드입니다.')
    return dict(mode=mode, version=VERSION, models=MODEL_HASHES, dpi=220,
                max_pixels=MAX_PIXELS, max_side=MAX_SIDE, low_confidence=LOW_CONFIDENCE)


class LocalOCR:
    def __init__(self):
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

    def __call__(self, pixels):
        result, _ = self.engine(pixels)
        return result or []


def overlap(a, b):
    a, b = fitz.Rect(a), fitz.Rect(b)
    area = min(a.get_area(), b.get_area())
    return (a & b).get_area() / area if area else 0.


def page_blocks(page, mode='auto', engine=None, cancel=lambda: False):
    """Return ordinary text blocks plus OCR lines in displayed page coordinates."""
    signature(mode)
    if cancel():
        raise Cancelled()
    native = page.get_text('dict', flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES, sort=True)['blocks']
    matrix = page.rotation_matrix
    for block in native:
        block['bbox'] = list(fitz.Rect(block['bbox']) * matrix)
        for line in block.get('lines', []):
            line['bbox'] = list(fitz.Rect(line['bbox']) * matrix)
            for span in line['spans']:
                span['bbox'] = list(fitz.Rect(span['bbox']) * matrix)
                span['extraction'] = 'native'
    chars = sum(len(s['text'].strip()) for b in native for l in b.get('lines', []) for s in l['spans'])
    area = page.rect.get_area()
    images = [fitz.Rect(i['bbox']) * matrix for i in page.get_image_info()]
    has_image = any((r & page.rect).get_area() >= area * .005 and min(r.width, r.height) >= 20 for r in images)
    needed = mode == 'force' or (mode == 'auto' and (has_image or chars < 20))
    info = dict(mode=mode, attempted=needed, engine=VERSION if needed else None,
                lines=0, low_confidence_lines=0, duplicate_lines=0)
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
    retained = [] if mode == 'force' else native
    native_spans = [s for b in native for l in b.get('lines', []) for s in l['spans'] if s['text'].strip()]
    for points, text, confidence, *_ in rows:
        if not text.strip():
            continue
        bbox = [min(p[0] for p in points)/scale, min(p[1] for p in points)/scale,
                max(p[0] for p in points)/scale, max(p[1] for p in points)/scale]
        # Prefer precise native text over a second OCR rendition of the same region.
        matched=[s for s in native_spans if overlap(bbox,s['bbox'])>=.65]
        normalize=lambda value: re.sub(r'\W+','',value).casefold()
        same_text=bool(matched) and bool(normalize(text)) and normalize(text)==normalize(' '.join(s['text'] for s in matched))
        coverage=sum((fitz.Rect(bbox)&fitz.Rect(s['bbox'])).get_area() for s in matched)/max(1,fitz.Rect(bbox).get_area())
        if mode != 'force' and (same_text or coverage>=.9):
            info['duplicate_lines'] += 1
            continue
        confidence = float(confidence)
        size = max(6., min(14., (bbox[3]-bbox[1])*.8))
        span = dict(text=text.strip(), bbox=bbox, size=size, flags=0, font='OCR',
                    extraction='ocr', ocr_confidence=confidence)
        retained.append(dict(type=0, bbox=bbox, lines=[dict(bbox=bbox, spans=[span])]))
        info['lines'] += 1
        info['low_confidence_lines'] += confidence < LOW_CONFIDENCE
    if info['lines']:
        warnings.append(f'로컬 OCR {info["lines"]}줄: 인식 오류가 번역에 반영될 수 있으므로 원문을 대조하세요.')
    if info['low_confidence_lines']:
        warnings.append(f'OCR 낮은 신뢰도 {info["low_confidence_lines"]}줄: 숫자·이름·수식을 특히 확인하세요.')
    if not rows and (chars < 20 or mode == 'force'):
        warnings.append('OCR에서 글자를 찾지 못했습니다. 빈 쪽·도형·흐린 스캔인지 원문을 확인하세요.')
    info['raster_size'] = [pix.width, pix.height]
    return sorted(retained, key=lambda b: (b['bbox'][1], b['bbox'][0])), info, warnings
