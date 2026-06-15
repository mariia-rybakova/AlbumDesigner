"""Helpers shared across stage visualizers.

Keep this module free of pipeline imports — visualizers are decoupled from
the layouting/grouping code so the saved JSON is the only contract.
"""

from __future__ import annotations

import io
import math
import os
from typing import Any, List, Optional, Sequence, Tuple

from PIL import Image
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


PAGE_SIZE = landscape(A4)

# Thumbnail cell size in PDF points. Fixed (never adapts to page content),
# so a 20-photo group always occupies twice the visible area of a 10-photo
# group — that's the whole reason for the wrapping-grid layout. Pages flow
# to fit their content rather than the cells shrinking to fit a page.
DEFAULT_CELL_SIZE = 84.0

CAP_LINE_H = 8.0  # height per caption line, in points


# ---------- file / lookup ----------

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


# ---------- time formatting ----------

def format_general_time(t: Any) -> str:
    """Render `general_time` (assumed seconds-since-event-start) as HH:MM:SS."""
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


def format_image_time_date(t: Any) -> str:
    """Render `image_time_date` (absolute wall-clock timestamp) as HH:MM:SS.

    Photos in a wedding album are virtually always all on the same date, so
    showing only the time-of-day keeps captions tight under thumbnails. The
    value is whatever was saved — typically an ISO string from pd.Timestamp.
    """
    if t is None or t == "":
        return ""
    s = str(t)
    for sep in ('T', ' '):
        if sep in s:
            time_part = s.split(sep, 1)[1]
            for cut in ('.', '+', '-', 'Z'):
                if cut in time_part:
                    time_part = time_part.split(cut, 1)[0]
            return time_part[:8]
    return s[:8]


def caption_fields_for(base_fields: Sequence[str], is_artificial_time: bool) -> Tuple[str, ...]:
    """Pick the per-photo caption fields for the gallery's time flavour.

    When `is_artificial_time`, the stored `image_time_date` is stale/identical
    across photos (the real timeline lives in the synthetic `general_time`), so
    swap that field for `general_time`. Otherwise the base fields are used as-is.
    Mirrors the source switch `process_gallery.py` makes for album1.pdf.
    """
    if not is_artificial_time:
        return tuple(base_fields)
    return tuple('general_time' if f == 'image_time_date' else f for f in base_fields)


def mean_time_label(mean_image_time_date: Any, mean_general_time: Any,
                    is_artificial_time: bool) -> str:
    """Header/meta mean-time label: wall-clock normally, elapsed when artificial."""
    if is_artificial_time:
        return format_general_time(mean_general_time)
    return format_image_time_date(mean_image_time_date)


# ---------- caption assembly ----------

def captions_for_photo(p: dict, caption_fields: Sequence[str]) -> List[str]:
    """Build the list of caption lines for one thumbnail, given a record dict."""
    out: List[str] = []
    for field in caption_fields:
        v = p.get(field)
        if field == 'image_time_date':
            out.append(format_image_time_date(v))
        elif field == 'general_time':
            out.append(format_general_time(v))
        elif v is None:
            continue
        else:
            s = str(v)
            if len(s) > 18:
                s = s[:15] + '...'
            out.append(s)
    return out


def caption_height(caption_fields: Optional[Sequence[str]]) -> float:
    """Vertical space (pt) needed for the captions below one thumbnail."""
    return CAP_LINE_H * len(caption_fields or ())


# ---------- image fit (no crop, no stretch) ----------

def fit_image_preserving_ar(img: Image.Image, max_w: float, max_h: float) -> Tuple[io.BytesIO, float, float]:
    """Resize `img` to fit within `(max_w, max_h)` preserving aspect ratio.

    Returns (png_bytes, final_w, final_h). Final dims may be smaller than the
    bounding box; caller centers the image in the available space.
    """
    max_w = max(1.0, float(max_w))
    max_h = max(1.0, float(max_h))
    iw, ih = img.size
    if iw <= 0 or ih <= 0:
        iw = ih = 1
    scale = min(max_w / iw, max_h / ih)
    final_w = max(1, int(round(iw * scale)))
    final_h = max(1, int(round(ih * scale)))
    img = img.resize((final_w, final_h))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf, float(final_w), float(final_h)


def fit_image_to_box(img: Image.Image, target_w: float, target_h: float) -> io.BytesIO:
    """Center-crop `img` to the box aspect, resize to box pixel size.

    Used by the spreads visualizer to render photos *inside* layout boxes —
    that's how the real album rendering places photos (the algorithm crops to
    box AR), so the analysis PDF matches.

    For the grouping visualizers (splits/merges/subgroups), prefer
    `fit_image_preserving_ar` so original orientations stay visible.
    """
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


# ---------- panel sizing ----------

def grid_cols_for_width(rect_w: float, cell_size: float = DEFAULT_CELL_SIZE) -> int:
    """How many cells fit across `rect_w` at `cell_size`."""
    return max(1, int(rect_w // cell_size))


def grid_height_for(n_photos: int, cols: int,
                    caption_fields: Optional[Sequence[str]] = None,
                    cell_size: float = DEFAULT_CELL_SIZE) -> float:
    """Total height (pt) a `draw_photo_grid` call will occupy.

    `n_photos` wraps to `cols` columns; vertical pitch per row is the cell
    size plus the caption strip beneath.
    """
    rows = max(1, math.ceil(max(0, n_photos) / cols)) if n_photos else 1
    return rows * (cell_size + caption_height(caption_fields))


# ---------- per-cell + grid drawing ----------

def _draw_cell(c: canvas.Canvas,
               cell_x: float, cell_y: float,
               cell_size: float, caption_h: float,
               photo_id: Any, caption_lines: List[str],
               images_path: str, image_files: List[str]) -> None:
    """Draw one cell: square image area on top, caption lines beneath.

    `(cell_x, cell_y)` is the bottom-left of the *cell*, which spans
    `cell_size × (cell_size + caption_h)` total height. Image is placed at
    its natural aspect ratio, centered within the image area. The image
    border is drawn at the actual image bounds (not the full square slot) so
    you can visually read the photo's orientation.
    """
    image_area_y = cell_y + caption_h  # bottom-left of the image area
    image_area_size = cell_size  # square

    if photo_id is None:
        # blank cell — just outline the slot
        c.setStrokeColorRGB(0.7, 0.7, 0.7)
        c.setLineWidth(0.3)
        c.rect(cell_x, image_area_y, image_area_size, image_area_size)
        return

    image_name = find_image_for_photo(image_files, photo_id)
    drew_image = False
    final_w = final_h = image_area_size  # used to position the border
    final_x = cell_x
    final_y = image_area_y

    if image_name is not None:
        try:
            img_path = os.path.join(images_path, image_name)
            with Image.open(img_path) as img:
                buf, final_w, final_h = fit_image_preserving_ar(img, image_area_size, image_area_size)
            final_x = cell_x + (image_area_size - final_w) / 2
            final_y = image_area_y + (image_area_size - final_h) / 2
            c.drawImage(ImageReader(buf), final_x, final_y, width=final_w, height=final_h)
            drew_image = True
        except Exception:
            drew_image = False

    # border around what was actually drawn (or the slot, if image is missing)
    if drew_image:
        c.setStrokeColorRGB(0.2, 0.2, 0.2)
        c.setLineWidth(0.4)
        c.rect(final_x, final_y, final_w, final_h)
    else:
        c.setStrokeColorRGB(0.6, 0.6, 0.6)
        c.setLineWidth(0.3)
        c.rect(cell_x, image_area_y, image_area_size, image_area_size)
        label = f"id:{photo_id}" if image_name is None else f"err:{photo_id}"
        color = (0.5, 0.5, 0.5) if image_name is None else (0.8, 0, 0)
        c.setFont('Helvetica', 6)
        c.setFillColorRGB(*color)
        c.drawString(cell_x + 2, image_area_y + 2, label[:max(3, int(cell_size / 4))])
        c.setFillColorRGB(0, 0, 0)

    if caption_lines:
        c.setFont('Helvetica', 6)
        c.setFillColorRGB(0.15, 0.15, 0.15)
        for i, line in enumerate(caption_lines):
            # lines stacked bottom-up inside caption strip
            ly = cell_y + (len(caption_lines) - 1 - i) * CAP_LINE_H + 1
            c.drawString(cell_x, ly, line)
        c.setFillColorRGB(0, 0, 0)


def draw_photo_grid(c: canvas.Canvas,
                    rect: Tuple[float, float, float, float],
                    photos: List[dict],
                    images_path: str,
                    image_files: List[str],
                    caption_fields: Optional[Sequence[str]] = None,
                    cell_size: float = DEFAULT_CELL_SIZE,
                    label: Optional[str] = None) -> None:
    """Render a wrapping grid of fixed-size thumbnails inside `rect`.

    Cells are uniform `cell_size`×`cell_size` (image area) plus a caption
    strip beneath each cell of `caption_height(caption_fields)`. Rows fill
    top-down, left-to-right. Cell size doesn't change with photo count, so
    a group with more photos visibly occupies more area on the page — which
    is the point of this layout.

    The caller is responsible for picking `cell_size` (typically via
    `compute_uniform_cell_size` across all panels on the page) so the scale
    is consistent across the comparison.
    """
    rx, ry, rw, rh = rect

    if label:
        c.setFont('Helvetica-Bold', 8)
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.drawString(rx, ry + rh + 2, label)
        c.setFillColorRGB(0, 0, 0)

    if not photos:
        c.setFont('Helvetica', 8)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.drawString(rx, ry + rh / 2, "(empty)")
        c.setFillColorRGB(0, 0, 0)
        return

    cap_h = caption_height(caption_fields)
    pitch_h = cell_size + cap_h
    cols = max(1, int(rw // cell_size))

    for i, p in enumerate(photos):
        col = i % cols
        row = i // cols
        cell_x = rx + col * cell_size
        cell_y = ry + rh - (row + 1) * pitch_h
        captions = captions_for_photo(p, caption_fields or [])
        _draw_cell(c, cell_x, cell_y, cell_size, cap_h,
                   p.get('image_id'), captions, images_path, image_files)
