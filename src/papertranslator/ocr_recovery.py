"""Bounded recovery of text-shaped, uncovered ink; never replace known text."""
import copy
import re

import pymupdf as fitz

from .backend import Cancelled

VERSION = 'uncovered-text-regions-1'
MAX_REGIONS = 6
MAX_AREA = .25
PADDING = 24


def box(row):
    return fitz.Rect(min(p[0] for p in row[0]), min(p[1] for p in row[0]),
                     max(p[0] for p in row[0]), max(p[1] for p in row[0]))


def missing_regions(pixels, rows):
    """Find horizontal clusters of glyph-sized components outside known rows.

    Photographs, solid shapes, scan rules and isolated marks are deliberately
    ineligible. This detects candidates, not proof of missing semantic content.
    """
    import cv2
    import numpy as np
    height, width = pixels.shape[:2]
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
    if np.mean(gray > 180) < .65:
        return []
    ink = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY_INV)[1]
    for row in rows:
        rect = box(row) + (-4, -4, 4, 4)
        x0, y0, x1, y1 = map(int, rect & fitz.Rect(0, 0, width, height))
        ink[y0:y1, x0:x1] = 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    glyphs = np.zeros_like(ink); heights=[]
    for x, y, w, h, area in stats[1:count]:
        if 5 <= h <= min(80, height * .045) and 1 <= w <= h * 2 and 5 <= area <= w*h*.88:
            glyphs[y:y+h, x:x+w] |= ink[y:y+h, x:x+w]
            heights.append(h)
    # Link nearby letters horizontally, leaving columns and vertically separated
    # rows independent. Reject clusters containing fewer than four glyphs.
    reach=max(19,min(65,int(np.median(heights)*1.4))) if heights else 19
    joined = cv2.dilate(glyphs, np.ones((3, reach), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    candidates = []
    for x, y, w, h, _ in stats[1:count]:
        if w < max(45, h * 2) or h > 100:
            continue
        region = glyphs[y:y+h, x:x+w]
        pieces = cv2.connectedComponentsWithStats(region, 8)[2][1:]
        if len(pieces) < 4 or not .025 <= np.count_nonzero(region)/(w*h) <= .6:
            continue
        rect = (fitz.Rect(x, y, x+w, y+h) + (-12, -12, 12, 12)) & fitz.Rect(0, 0, width, height)
        # Do not crop into a recognised row: a crop can otherwise read only the
        # unmasked tail and duplicate a known word with a shifted boundary.
        if any((rect & box(row)).get_area() > 0 for row in rows):
            continue
        candidates.append(rect)
    selected = []; area = 0
    for rect in sorted(candidates, key=lambda r: (r.y0, r.x0)):
        if any((rect & other).get_area() for other in selected):
            continue
        cost = (rect.width+2*PADDING)*(rect.height+2*PADDING)
        if area + cost > width * height * MAX_AREA:
            continue
        selected.append(rect); area += cost
        if len(selected) == MAX_REGIONS:
            break
    return selected


def recover(pixels, rows, recognize, cancel=lambda: False):
    """Retry up to six small crops once and add only nonoverlapping text rows."""
    import numpy as np
    result = list(rows); accepted = 0
    regions = missing_regions(pixels, rows)
    for index, rect in enumerate(regions):
        if cancel():
            raise Cancelled()
        x0, y0, x1, y1 = map(int, rect)
        crop = np.pad(pixels[y0:y1, x0:x1], ((PADDING,PADDING),(PADDING,PADDING),(0,0)), constant_values=255)
        candidates = recognize(crop)
        if cancel():
            raise Cancelled()
        for original in candidates:
            if not re.search(r'\b[A-Za-z]{2,}\b', original[1]):
                continue
            if original[2] is not None and original[2] < .75:
                continue
            row = copy.deepcopy(original)
            row[0] = [[x-PADDING, y-PADDING] for x,y in row[0]]
            local = box(row)
            if not fitz.Rect(0, 0, x1-x0, y1-y0).contains(local):
                continue
            row[0] = [[x+x0, y+y0] for x, y in row[0]]
            candidate = box(row)
            # Any substantial geometric collision is ambiguous. Keep existing
            # evidence instead of arbitrating between two OCR interpretations.
            if any((candidate & box(known)).get_area()/max(1, min(candidate.get_area(), box(known).get_area())) > .15 for known in result):
                continue
            meta = row[3] if len(row) > 3 and isinstance(row[3], dict) else {}
            if meta.get('rotation') or meta.get('layout_kind') or meta.get('diagram_label'):
                continue
            for word in meta.get('words', []):
                word['bbox'] = [v-PADDING + (x0 if i % 2 == 0 else y0) for i, v in enumerate(word['bbox'])]
            # Crop block numbers belong to a separate namespace.
            meta.update(recovered_region=index, recovery=VERSION)
            if len(row) > 3:
                row[3] = meta
            else:
                row.append(meta)
            result.append(row); accepted += 1
    # Recovery is only enabled by a snapshotted policy for new jobs. Restore
    # geometric order after appending crops, without moving structured labels.
    if accepted:
        additions = sorted(result[len(rows):], key=lambda row: (box(row).y0, box(row).x0))
        result = list(rows)
        for row in additions:
            index = next((i for i, known in enumerate(result) if box(known).y0 > box(row).y0), len(result))
            result.insert(index, row)
    return result, dict(version=VERSION, attempted_regions=len(regions), added_lines=accepted,
                        max_regions=MAX_REGIONS, max_area=MAX_AREA)


def join_fragments(blocks):
    """Reconstruct split OCR baselines after separating table cells/diagrams."""
    # An unrecognised borderless table can have word-sized column gaps.
    # Only reconstruct its individually identified cells, never the remaining
    # page body when it contains a Table caption. Abstain on unknown columns.
    if any(re.match(r'^\s*Table\s+\d',''.join(s['text'] for s in line['spans']),re.I)
           for block in blocks for line in block.get('lines', [])):
        return blocks
    output = []; previous = None
    for block in blocks:
        if block.get('type') != 0:
            output.append(block); previous = None; continue
        if any(s.get('extraction') != 'ocr' for line in block['lines'] for s in line['spans']):
            output.append(block); previous = None; continue
        for line in block['lines']:
            spans = line['spans']
            eligible = (spans and all(s.get('extraction') == 'ocr' for s in spans)
                        and not any(line.get(k) for k in ('rotation', 'layout_kind', 'diagram_label'))
                        and not re.fullmatch(r'\s*\d+[.,]?\s*',''.join(s['text'] for s in spans)))
            rect = fitz.Rect(line['bbox'])
            if eligible and previous is not None:
                prior = fitz.Rect(previous['bbox'])
                height = min(rect.height, prior.height)
                gap = rect.x0-prior.x1
                if (height > 0 and max(rect.height, prior.height) <= height * 1.6
                        and abs(rect.y1-prior.y1) <= height * .25
                        and 0 <= gap <= height * 1.6):
                    spacer = {**spans[0], 'text': ' ', 'bbox': [prior.x1, rect.y0, rect.x0, rect.y1]}
                    previous['spans'] += [spacer] + copy.deepcopy(spans)
                    previous['bbox'] = list(prior | rect)
                    previous.setdefault('layout_words', []).extend(copy.deepcopy(line.get('layout_words', [])))
                    output[-1]['bbox'] = previous['bbox']
                    previous['reconstructed_fragments'] = previous.get('reconstructed_fragments', 1)+1
                    continue
            new = copy.deepcopy(line)
            output.append(dict(type=0, bbox=new['bbox'], lines=[new]))
            previous = new if eligible else None
    return output
