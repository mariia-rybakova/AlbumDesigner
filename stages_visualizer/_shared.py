"""Helpers shared across stage visualizers.

Keep this module free of pipeline imports — visualizers are decoupled from
the layouting/grouping code so the saved JSON is the only contract.
"""

from __future__ import annotations

import io
import os
from typing import Any, List, Optional

from PIL import Image
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


PAGE_SIZE = landscape(A4)


def list_image_files(images_path: str) -> List[str]:
    """Cache of filenames in `images_path`; empty list if the dir is missing."""
    if not os.path.isdir(images_path):
        return []
    return os.listdir(images_path)


def find_image_for_photo(image_files: List[str], photo_id: Any) -> Optional[str]:
    """First filename in `image_files` whose name starts with `photo_id`."""
    prefix = f"{photo_id}"
    for name in image_files:
        if name.startswith(prefix):
            return name
    return None


def format_general_time(t: Any) -> str:
    """Render `general_time` (assumed seconds-since-event-start) as HH:MM:SS.

    Falls back to the raw string if the value isn't numeric.
    """
    if t is None:
        return ""
    try:
        secs = float(t)
    except (TypeError, ValueError):
        return str(t)
    if secs < 0:
        return f"-{format_general_time(-secs)}"
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fit_image_to_box(img: Image.Image, target_w: float, target_h: float) -> io.BytesIO:
    """Center-crop `img` to the box aspect, resize to box pixel size, return PNG bytes."""
    target_w = max(1.0, float(target_w))
    target_h = max(1.0, float(target_h))
    box_ar = target_w / target_h
    iw, ih = img.size
    if ih == 0:
        ih = 1
    img_ar = iw / ih
    if img_ar > box_ar:
        new_w = int(round(ih * box_ar))
        off = max(0, (iw - new_w) // 2)
        img = img.crop((off, 0, off + new_w, ih))
    else:
        new_h = int(round(iw / box_ar)) if box_ar else ih
        off = max(0, (ih - new_h) // 2)
        img = img.crop((0, off, iw, off + new_h))
    img = img.resize((max(1, int(target_w)), max(1, int(target_h))))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


def draw_photo_thumb(c: canvas.Canvas,
                     rect: tuple,
                     photo_id: Any,
                     images_path: str,
                     image_files: List[str],
                     caption_lines: Optional[List[str]] = None) -> None:
    """Draw one thumbnail at `rect`=(x, y, w, h) (bottom-left origin).

    Border always shown. Image placed if found; otherwise a gray `id:<...>` stamp.
    `caption_lines` are drawn below the thumb in small gray text.
    """
    x, y, w, h = rect
    # caption area at the bottom
    cap_line_h = 8.0
    n_lines = len(caption_lines or [])
    cap_h = n_lines * cap_line_h
    img_h = max(1.0, h - cap_h - 2)
    img_y = y + cap_h + 2

    c.setStrokeColorRGB(0.2, 0.2, 0.2)
    c.setLineWidth(0.5)
    c.rect(x, img_y, w, img_h)

    if photo_id is not None:
        image_name = find_image_for_photo(image_files, photo_id)
        if image_name is not None:
            try:
                img_path = os.path.join(images_path, image_name)
                with Image.open(img_path) as img:
                    buf = fit_image_to_box(img, w, img_h)
                c.drawImage(ImageReader(buf), x, img_y, width=w, height=img_h)
            except Exception:
                c.setFont('Helvetica', 6)
                c.setFillColorRGB(0.8, 0, 0)
                c.drawString(x + 2, img_y + 2, f"err:{photo_id}")
                c.setFillColorRGB(0, 0, 0)
        else:
            c.setFont('Helvetica', 6)
            c.setFillColorRGB(0.5, 0.5, 0.5)
            c.drawString(x + 2, img_y + 2, f"id:{photo_id}")
            c.setFillColorRGB(0, 0, 0)

    if caption_lines:
        c.setFont('Helvetica', 6)
        c.setFillColorRGB(0.15, 0.15, 0.15)
        for i, line in enumerate(caption_lines):
            # lines drawn from bottom up
            ly = y + (n_lines - 1 - i) * cap_line_h + 1
            c.drawString(x, ly, line)
        c.setFillColorRGB(0, 0, 0)


def draw_photo_strip(c: canvas.Canvas,
                     rect: tuple,
                     photos: List[dict],
                     images_path: str,
                     image_files: List[str],
                     caption_fields: Optional[List[str]] = None,
                     label: Optional[str] = None) -> None:
    """Render a horizontal strip of photo thumbnails inside `rect`=(x, y, w, h).

    Each entry in `photos` should have `image_id` and any fields named in
    `caption_fields` (e.g. ['general_time', 'original_context']). The strip's
    thumb height is fixed by `rect[3]`; thumb width is rect.w / max(n, 1).
    A leading label (e.g. "merged") is drawn just above the strip if provided.
    """
    rx, ry, rw, rh = rect

    if label:
        c.setFont('Helvetica-Bold', 8)
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.drawString(rx, ry + rh + 2, label)
        c.setFillColorRGB(0, 0, 0)

    n = len(photos)
    if n == 0:
        c.setFont('Helvetica', 8)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.drawString(rx, ry + rh / 2, "(empty)")
        c.setFillColorRGB(0, 0, 0)
        return

    cell_w = rw / n
    pad = 2.0
    for i, p in enumerate(photos):
        x = rx + i * cell_w + pad / 2
        thumb_rect = (x, ry, cell_w - pad, rh)
        captions = []
        for field in (caption_fields or []):
            v = p.get(field)
            if field == 'general_time':
                captions.append(format_general_time(v))
            elif v is None:
                continue
            else:
                # Truncate long strings for readability.
                s = str(v)
                if len(s) > 18:
                    s = s[:15] + '...'
                captions.append(s)
        draw_photo_thumb(c, thumb_rect, p.get('image_id'),
                         images_path, image_files, caption_lines=captions)
