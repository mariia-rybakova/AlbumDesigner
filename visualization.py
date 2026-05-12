"""
Render an analysis PDF from files/stages_info/spreads.

For each per-group JSON in `files/stages_info/spreads/`, draws one PDF page
showing every top_layouts candidate as a small illustration. Each illustration
renders the candidate's spread(s) with boxes positioned from `_layouts.json`'s
`boxes_info` and the available photos placed inside.

Read-only: no imports from the layout/grouping pipeline; never mutates state
outside the output PDF.

Box-to-photo pairing is sequential (box[i] gets photo[i] in saved index order).
This is not the same as `GroupSingleLayout.resolve_and_order` — that uses each
photo's rank, aspect-ratio, and orientation to match photos to boxes — but the
saved JSON only carries photo `id`, so a faithful re-run would need full Photo
records. For *analysis at a glance* (which layouts were considered, which
photos went into which side) the sequential pairing is sufficient.

Usage:
    python visualization.py <input_dir> <output_dir>
        [--request files/test_requests/request1.json]
        [--stages-dir files/stages_info/spreads]

Output: <output_dir>/<projectId>/album1_analysis.pdf
"""

from __future__ import annotations

import argparse
import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


DEFAULT_STAGES_DIR = os.path.join('files', 'stages_info', 'spreads')
LAYOUTS_FILE = '_layouts.json'

# Album-pages aspect ratio (spread width : page height). The pipeline uses
# album_ar=2 by default for "anyPage" (process_gallery / request_processing),
# i.e. a spread is twice as wide as it is tall. Mirror that here so the
# illustrations match the algorithm's geometry assumption.
SPREAD_AR = 2.0


# ---------- IO / lookups ----------

def _load_layouts(stages_dir: str) -> Tuple[Dict[int, dict], Dict[int, dict]]:
    """Load `_layouts.json` and build idx-keyed lookups.

    Returns:
        layouts_by_idx:        layout_idx (== layouts_df row index)
                               -> full layouts_df record (carries `boxes_info`)
        layout_id2data_by_idx: same key -> {'layout_id', 'left_box_ids',
                                            'right_box_ids', 'boxes_areas'}
    """
    path = os.path.join(stages_dir, LAYOUTS_FILE)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    layouts_by_idx = {r['index']: r for r in data['layouts_df']}
    layout_id2data_by_idx = {r['idx']: r for r in data['layout_id2data']}
    return layouts_by_idx, layout_id2data_by_idx


def _load_group_files(stages_dir: str) -> List[Tuple[str, dict]]:
    """List `(group_name, parsed_json)` for every per-group file.

    Skips `_layouts.json` and non-JSON files. The returned name is the bare
    stem (e.g. `0_bride_-1_0`).
    """
    out = []
    for fname in sorted(os.listdir(stages_dir)):
        if fname == LAYOUTS_FILE or not fname.endswith('.json'):
            continue
        with open(os.path.join(stages_dir, fname), 'r', encoding='utf-8') as f:
            out.append((fname[:-5], json.load(f)))
    return out


def _find_image_for_photo(image_files: List[str], photo_id: Any) -> Optional[str]:
    """First filename in `image_files` whose name starts with `photo_id`."""
    prefix = f"{photo_id}"
    for name in image_files:
        if name.startswith(prefix):
            return name
    return None


def _build_spread_geometry(layout_idx: int,
                           layouts_by_idx: Dict[int, dict],
                           layout_id2data_by_idx: Dict[int, dict]
                           ) -> Tuple[int, List[dict], List[dict]]:
    """Resolve a `layout_idx` to (`layout_id`, left_boxes, right_boxes).

    Each box is the raw entry from `boxes_info` (with relative `x`, `y`,
    `width`, `height`). Order matches `left_box_ids` / `right_box_ids`, which
    is also the order `resolve_and_order` uses when matching photos to slots.
    """
    layout_record = layouts_by_idx[layout_idx]
    layout_data = layout_id2data_by_idx[layout_idx]
    boxes_by_id = {b['id']: b for b in layout_record['boxes_info']}
    left_boxes = [boxes_by_id[bid] for bid in layout_data['left_box_ids']]
    right_boxes = [boxes_by_id[bid] for bid in layout_data['right_box_ids']]
    return layout_data['layout_id'], left_boxes, right_boxes


# ---------- drawing primitives ----------

def _fit_image_to_box(img: Image.Image, target_w: float, target_h: float) -> io.BytesIO:
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


def _draw_box(c: canvas.Canvas, box: dict, photo_id: Any,
              spread_rect: Tuple[float, float, float, float],
              images_path: str, image_files: List[str]) -> None:
    """Draw one box outline at its relative position; place its photo if available.

    `spread_rect` is (x, y, w, h) in canvas coords (bottom-left origin).
    `box`'s `x`/`y`/`width`/`height` are spread-relative with y running top-down.
    """
    sx, sy, sw, sh = spread_rect
    bw = box['width'] * sw
    bh = box['height'] * sh
    bx = sx + box['x'] * sw
    # boxes_info y is top-origin; reportlab is bottom-origin → flip.
    by = sy + sh - (box['y'] + box['height']) * sh

    c.setStrokeColorRGB(0.1, 0.1, 0.1)
    c.setLineWidth(0.7)
    c.rect(bx, by, bw, bh)

    if photo_id is None:
        return

    image_name = _find_image_for_photo(image_files, photo_id)
    if image_name is None:
        c.setFont('Helvetica', 5)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.drawString(bx + 1, by + 1, f"id:{photo_id}")
        c.setFillColorRGB(0, 0, 0)
        return

    try:
        img_path = os.path.join(images_path, image_name)
        with Image.open(img_path) as img:
            buf = _fit_image_to_box(img, bw, bh)
        c.drawImage(ImageReader(buf), bx, by, width=bw, height=bh)
    except Exception:
        c.setFont('Helvetica', 5)
        c.setFillColorRGB(0.8, 0, 0)
        c.drawString(bx + 1, by + 1, f"err:{photo_id}")
        c.setFillColorRGB(0, 0, 0)


def _draw_spread(c: canvas.Canvas, spread_rect: Tuple[float, float, float, float],
                 left_boxes: List[dict], right_boxes: List[dict],
                 left_photo_ids: List[Any], right_photo_ids: List[Any],
                 images_path: str, image_files: List[str]) -> None:
    """Render one spread inside `spread_rect`. Boxes drawn from `boxes_info`,
    photos paired sequentially with `left_box_ids` / `right_box_ids` order."""
    sx, sy, sw, sh = spread_rect
    # Outer spread frame + spine.
    c.setStrokeColorRGB(0.55, 0.55, 0.55)
    c.setLineWidth(0.4)
    c.rect(sx, sy, sw, sh)
    c.line(sx + sw / 2, sy, sx + sw / 2, sy + sh)

    for i, box in enumerate(left_boxes):
        pid = left_photo_ids[i] if i < len(left_photo_ids) else None
        _draw_box(c, box, pid, spread_rect, images_path, image_files)
    for i, box in enumerate(right_boxes):
        pid = right_photo_ids[i] if i < len(right_photo_ids) else None
        _draw_box(c, box, pid, spread_rect, images_path, image_files)


def _fit_spread_rect_in_cell(cell_x: float, cell_y: float,
                             cell_w: float, cell_h: float) -> Tuple[float, float, float, float]:
    """Return the largest SPREAD_AR-ratio rect that fits inside the cell, centered."""
    if cell_w / cell_h > SPREAD_AR:
        sh = cell_h
        sw = sh * SPREAD_AR
    else:
        sw = cell_w
        sh = sw / SPREAD_AR
    sx = cell_x + (cell_w - sw) / 2
    sy = cell_y + (cell_h - sh) / 2
    return sx, sy, sw, sh


# ---------- candidate / page layout ----------

def _format_metric(v: Any) -> str:
    if v is None:
        return 'None'
    try:
        return f"{float(v):.4g}"
    except (TypeError, ValueError):
        return str(v)


def _draw_candidate(c: canvas.Canvas,
                    cell_rect: Tuple[float, float, float, float],
                    candidate: dict, photos: List[dict],
                    layouts_by_idx: Dict[int, dict],
                    layout_id2data_by_idx: Dict[int, dict],
                    images_path: str, image_files: List[str],
                    candidate_idx: int) -> None:
    """Draw one top_layouts candidate inside `cell_rect`.

    Layout: a small title strip at the top, then the candidate's spreads
    stacked vertically beneath it. Single-spread candidates fill the whole
    cell; multi-spread candidates share the vertical space evenly so the
    user sees them as one unit (per the brief: "if more than one spread in
    group — draw them together").
    """
    cx, cy, cw, ch = cell_rect

    # Title strip.
    title_h = 12.0
    c.setFont('Helvetica-Bold', 8)
    c.setFillColorRGB(0, 0, 0)
    title = (f"#{candidate_idx}  "
             f"score={_format_metric(candidate.get('score'))}  "
             f"weight={_format_metric(candidate.get('weight'))}")
    c.drawString(cx, cy + ch - 10, title)

    spreads = candidate.get('spreads_layouts', [])
    if not spreads:
        return

    body_y = cy
    body_h = ch - title_h
    per_h = body_h / len(spreads)
    pad = 2.0

    for s_idx, spread in enumerate(spreads):
        sub_cell_y = body_y + (len(spreads) - 1 - s_idx) * per_h
        sub_cell = (cx + pad, sub_cell_y + pad,
                    cw - 2 * pad, per_h - 2 * pad)

        layout_idx = spread['layout_idx']
        layout_id, left_boxes, right_boxes = _build_spread_geometry(
            layout_idx, layouts_by_idx, layout_id2data_by_idx)

        # caption above this spread
        caption_h = 9.0
        c.setFont('Helvetica', 6)
        c.setFillColorRGB(0.25, 0.25, 0.25)
        caption = (f"spread {s_idx + 1}/{len(spreads)}  "
                   f"layout_idx={layout_idx} layout_id={layout_id}  "
                   f"score={_format_metric(spread.get('score'))} "
                   f"weight={_format_metric(spread.get('weight'))}")
        c.drawString(sub_cell[0], sub_cell[1] + sub_cell[3] - caption_h + 1, caption)
        c.setFillColorRGB(0, 0, 0)

        ill_cell = (sub_cell[0], sub_cell[1],
                    sub_cell[2], max(1.0, sub_cell[3] - caption_h))
        spread_rect = _fit_spread_rect_in_cell(*ill_cell)

        left_pids = [photos[i]['id'] for i in sorted(spread['left_page_photo_idxs'])
                     if 0 <= i < len(photos)]
        right_pids = [photos[i]['id'] for i in sorted(spread['right_page_photo_idxs'])
                      if 0 <= i < len(photos)]

        _draw_spread(c, spread_rect, left_boxes, right_boxes,
                     left_pids, right_pids, images_path, image_files)


def _pick_grid(n: int) -> Tuple[int, int]:
    """Choose (cols, rows) for `n` candidates. Up to 3 cols, adds rows as needed."""
    if n <= 1:
        return 1, 1
    if n == 2:
        return 2, 1
    cols = 3 if n >= 3 else n
    rows = (n + cols - 1) // cols
    return cols, rows


def _draw_group_page(c: canvas.Canvas, page_size: Tuple[float, float],
                     group_name: str, group_data: dict,
                     layouts_by_idx: Dict[int, dict],
                     layout_id2data_by_idx: Dict[int, dict],
                     images_path: str, image_files: List[str]) -> None:
    """One PDF page per group: header + grid of top_layouts candidates."""
    page_w, page_h = page_size
    margin = 18.0
    header_h = 36.0

    photos = group_data.get('photos', []) or []
    candidates = group_data.get('top_layouts', []) or []

    # Header.
    c.setFont('Helvetica-Bold', 13)
    c.drawString(margin, page_h - margin - 12, f"Group: {group_name}")
    c.setFont('Helvetica', 9)
    c.drawString(margin, page_h - margin - 26,
                 f"{len(photos)} photos  |  {len(candidates)} candidates")

    if not candidates:
        c.setFont('Helvetica', 10)
        c.drawString(margin, page_h - margin - header_h - 20, "(no candidates)")
        c.showPage()
        return

    # Candidate grid.
    cols, rows = _pick_grid(len(candidates))
    grid_x = margin
    grid_y = margin
    grid_w = page_w - 2 * margin
    grid_h = page_h - margin - header_h - margin
    cell_w = grid_w / cols
    cell_h = grid_h / rows

    for i, cand in enumerate(candidates):
        col = i % cols
        row = i // cols
        # Top-down placement (row 0 at top of grid).
        cell_x = grid_x + col * cell_w
        cell_y = grid_y + (rows - 1 - row) * cell_h
        inner = (cell_x + 3, cell_y + 3, cell_w - 6, cell_h - 6)
        _draw_candidate(c, inner, cand, photos,
                        layouts_by_idx, layout_id2data_by_idx,
                        images_path, image_files, i)

    c.showPage()


# ---------- entry point ----------

def visualize_stages_info_to_pdf(stages_dir: str, images_path: str,
                                  output_pdf_path: str) -> None:
    """Read every per-group file under `stages_dir`, render one page each."""
    layouts_by_idx, layout_id2data_by_idx = _load_layouts(stages_dir)
    group_files = _load_group_files(stages_dir)
    image_files = os.listdir(images_path) if os.path.isdir(images_path) else []
    if not image_files:
        print(f"[warn] no images found under {images_path}; boxes will show photo ids only")

    page_size = landscape(A4)
    c = canvas.Canvas(output_pdf_path, pagesize=page_size)
    for group_name, group_data in group_files:
        _draw_group_page(c, page_size, group_name, group_data,
                         layouts_by_idx, layout_id2data_by_idx,
                         images_path, image_files)
    c.save()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a per-group layout-analysis PDF from files/stages_info.")
    parser.add_argument("input_dir",
                        help="Directory holding per-project image subdirs (matches process_gallery.py).")
    parser.add_argument("output_dir",
                        help="Where the analysis PDF should be written.")
    parser.add_argument("--request", default=os.path.join('files', 'test_requests', 'request1.json'),
                        help="Request file used to look up projectId (default: request1.json).")
    parser.add_argument("--stages-dir", default=DEFAULT_STAGES_DIR,
                        help=f"Stages-info spreads dir (default: {DEFAULT_STAGES_DIR}).")
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    with open(args.request, 'r', encoding='utf-8') as f:
        request = json.load(f)
    project_id = str(request['projectId'])

    images_path = os.path.join(args.input_dir, project_id)
    output_dir = os.path.join(args.output_dir, project_id)
    os.makedirs(output_dir, exist_ok=True)
    output_pdf = os.path.join(output_dir, 'album1_analysis.pdf')

    visualize_stages_info_to_pdf(args.stages_dir, images_path, output_pdf)
    print(f"saved: {output_pdf}")