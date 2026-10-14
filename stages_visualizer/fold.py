"""fold.pdf — where the fold falls inside each one-photo spread.

One page per one-photo spread, drawn at the size the album prints it, with
three things laid over the photo:

  * the **fold**, a solid red line down the centre of the spread;
  * each **face** found and each **head** estimated from a body with no face in
    it, as a box on the photo;
  * the **margin** the cost actually applies around each of those -- the face
    widened by `face_margin` of its own half-width, plus half the gutter.

The margin is the point of the drawing. A fold outside every face but inside
one of those margins is the case that reads as "too close" on paper and scores
as damage here, and the two are hard to reconcile without seeing them together.

Input is `files/stages_info/fold/fold.json`, written by
`src/stages_recorder/fold.py`. This module imports only `_shared`; the JSON
is the contract.
"""

from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from stages_visualizer._shared import find_image_for_photo, list_image_files

PAGE_SIZE = landscape(A4)

FOLD_COLOUR = (0.85, 0.1, 0.1)
FACE_COLOUR = (0.1, 0.5, 0.95)
BODY_COLOUR = (0.95, 0.6, 0.1)
MARGIN_DASH = (3, 2)


def _box_rect(box: Dict[str, Any], page_w: float, page_h: float) -> Tuple[float, float, float, float]:
    """The album's own mapping, so a page here is the page there."""
    return (box['x'] * page_w, box['y'] * page_h,
            box['width'] * page_w, box['height'] * page_h)


def _cropped_image(img_path: str, crop: Dict[str, Any], box_w: float, box_h: float) -> io.BytesIO:
    with Image.open(img_path) as img:
        width, height = img.size
        left, top = int(crop['x'] * width), int(crop['y'] * height)
        right = left + int(crop['width'] * width)
        bottom = top + int(crop['height'] * height)
        # Height doubled to match the album renderer's own resize.
        cropped = img.crop((left, top, right, bottom)).resize((max(1, int(box_w)),
                                                              max(1, int(box_h * 2))))
        buf = io.BytesIO()
        cropped.save(buf, format='PNG')
        buf.seek(0)
        return buf


def _image_x_to_page(u: float, page: Dict[str, Any], page_w: float) -> float:
    """An image column, through the crop and the box, onto the page."""
    crop, box = page['crop'], page['box']
    if crop['width'] <= 0:
        return box['x'] * page_w
    return (box['x'] + (u - crop['x']) / crop['width'] * box['width']) * page_w


def _draw_header(c: canvas.Canvas, page: Dict[str, Any], page_h: float) -> None:
    clearance = page.get('clearance')
    cost = page.get('cost') or 0.0
    bits = [f"page {page['page'] + 1}", f"design {page['design_id']}", f"photo {page['photo_id']}"]
    if page.get('crosses_fold'):
        bits.append(f"fold at {page['fold_u']:.3f} of the photo")
        if clearance is not None:
            bits.append(f"clearance {clearance:+.3f}")
        bits.append(f"cost {cost:.2f}")
    else:
        bits.append("box does not cross the fold")
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 9)
    c.drawString(12, page_h - 14, "   |   ".join(bits))


def _draw_spans(c: canvas.Canvas, page: Dict[str, Any], rect, page_w: float, page_h: float) -> None:
    _, box_y, _, box_h = rect
    top = page_h - box_y
    bottom = page_h - box_y - box_h

    for span in page.get('spans', ()):
        colour = FACE_COLOUR if span['kind'] == 'face' else BODY_COLOUR
        # the subject itself
        x1 = _image_x_to_page(span['x1'], page, page_w)
        x2 = _image_x_to_page(span['x2'], page, page_w)
        c.setStrokeColorRGB(*colour)
        c.setLineWidth(1.4)
        c.setDash()
        c.rect(x1, bottom, max(0.5, x2 - x1), box_h, stroke=1, fill=0)

        # the margin the cost applies around it
        m1 = _image_x_to_page(span['limit_x1'], page, page_w)
        m2 = _image_x_to_page(span['limit_x2'], page, page_w)
        c.setLineWidth(0.8)
        c.setDash(*MARGIN_DASH)
        c.line(m1, bottom, m1, top)
        c.line(m2, bottom, m2, top)
        c.setDash()

        c.setFillColorRGB(*colour)
        c.setFont("Helvetica", 7)
        c.drawString(x1 + 1, top - 9,
                     f"{span['kind']}{'' if span['kind'] == 'face' else ' (estimated)'}"
                     f" w={span['weight']:.2f}")


def _draw_fold(c: canvas.Canvas, page_w: float, page_h: float) -> None:
    """The fold is the middle of the spread, wherever the box happens to sit."""
    middle = page_w / 2
    c.setStrokeColorRGB(*FOLD_COLOUR)
    c.setLineWidth(1.6)
    c.setDash()
    c.line(middle, 0, middle, page_h)
    c.setFillColorRGB(*FOLD_COLOUR)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(middle + 3, 6, "fold")


def _draw_legend(c: canvas.Canvas, page_h: float) -> None:
    c.setFont("Helvetica", 7)
    entries = (("fold", FOLD_COLOUR), ("face", FACE_COLOUR),
               ("head estimated from a body", BODY_COLOUR),
               ("dashed = margin the cost applies", (0.4, 0.4, 0.4)))
    x = 12
    for label, colour in entries:
        c.setFillColorRGB(*colour)
        c.drawString(x, page_h - 26, label)
        x += c.stringWidth(label, "Helvetica", 7) + 14


def render(input_path: str, images_path: str, output_pdf_path: str) -> None:
    """Render fold.pdf, or nothing when the album has no one-photo spread."""
    with open(input_path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)

    pages: List[Dict[str, Any]] = data.get('pages') or []
    if not pages:
        print("[skip] no one-photo spreads -> fold.pdf not generated")
        return

    image_files = list_image_files(images_path)
    page_w, page_h = PAGE_SIZE
    c = canvas.Canvas(output_pdf_path, pagesize=PAGE_SIZE)

    for page in pages:
        rect = _box_rect(page['box'], page_w, page_h)
        box_x, box_y, box_w, box_h = rect

        name = find_image_for_photo(image_files, page['photo_id'])
        if name:
            try:
                buf = _cropped_image(os.path.join(images_path, name), page['crop'], box_w, box_h)
                c.drawImage(ImageReader(buf), box_x, page_h - box_y - box_h,
                            width=box_w, height=box_h)
            except Exception:  # noqa: BLE001 - a missing photo must not lose the page
                name = None
        if not name:
            c.setFillColorRGB(0.9, 0.9, 0.9)
            c.rect(box_x, page_h - box_y - box_h, box_w, box_h, stroke=0, fill=1)
            c.setFillColorRGB(0.4, 0.4, 0.4)
            c.setFont("Helvetica", 8)
            c.drawString(box_x + 4, page_h - box_y - 14, f"photo {page['photo_id']} not on disk")

        c.setStrokeColorRGB(0.3, 0.3, 0.3)
        c.setLineWidth(0.6)
        c.setDash()
        c.rect(box_x, page_h - box_y - box_h, box_w, box_h, stroke=1, fill=0)

        _draw_spans(c, page, rect, page_w, page_h)
        _draw_fold(c, page_w, page_h)
        _draw_header(c, page, page_h)
        _draw_legend(c, page_h)
        c.showPage()

    c.save()
