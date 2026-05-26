"""Render an analysis PDF from files/stages_info/spreads/.

For each per-group JSON, draws one PDF page showing every `top_layouts`
candidate as a small illustration. Each illustration renders the candidate's
spread(s) with boxes positioned from `_layouts.json`'s `boxes_info` and the
available photos placed inside.

Read-only: no imports from the layout/grouping pipeline.

Box-to-photo pairing is sequential (box[i] gets photo[i] in saved index order).
This is not the same as `GroupSingleLayout.resolve_and_order`, which uses each
photo's rank / aspect / orientation to match photos to boxes — but the saved
JSON only carries photo `id`, so a faithful re-run would need full Photo
records. For *analysis at a glance* (which layouts were considered, which
photos went into which side) the sequential pairing is sufficient.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from stages_visualizer._shared import (
    PAGE_SIZE,
    find_image_for_photo,
    fit_image_to_box,
    list_image_files,
)


LAYOUTS_FILE = '_layouts.json'

# Album-pages aspect ratio (spread width : page height). The pipeline uses
# album_ar=2 by default for "anyPage" (process_gallery / request_processing),
# i.e. a spread is twice as wide as it is tall.
SPREAD_AR = 2.0


# ---------- loading ----------

def _load_layouts(stages_dir: str) -> Tuple[Dict[int, dict], Dict[int, dict]]:
    """Load `_layouts.json` and build idx-keyed lookups."""
    path = os.path.join(stages_dir, LAYOUTS_FILE)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    layouts_by_idx = {r['index']: r for r in data['layouts_df']}
    layout_id2data_by_idx = {r['idx']: r for r in data['layout_id2data']}
    return layouts_by_idx, layout_id2data_by_idx


def _load_group_files(stages_dir: str) -> List[Tuple[str, dict]]:
    """List `(group_name, parsed_json)` for every per-group file."""
    out = []
    for fname in sorted(os.listdir(stages_dir)):
        if fname == LAYOUTS_FILE or not fname.endswith('.json'):
            continue
        with open(os.path.join(stages_dir, fname), 'r', encoding='utf-8') as f:
            out.append((fname[:-5], json.load(f)))
    return out


def _build_spread_geometry(layout_idx: int,
                           layouts_by_idx: Dict[int, dict],
                           layout_id2data_by_idx: Dict[int, dict]
                           ) -> Tuple[int, List[dict], List[dict]]:
    """Resolve `layout_idx` to (`layout_id`, left_boxes, right_boxes)."""
    layout_record = layouts_by_idx[layout_idx]
    layout_data = layout_id2data_by_idx[layout_idx]
    boxes_by_id = {b['id']: b for b in layout_record['boxes_info']}
    left_boxes = [boxes_by_id[bid] for bid in layout_data['left_box_ids']]
    right_boxes = [boxes_by_id[bid] for bid in layout_data['right_box_ids']]
    return layout_data['layout_id'], left_boxes, right_boxes


# ---------- drawing ----------

def _draw_box(c: canvas.Canvas, box: dict, photo_id: Any,
              spread_rect: Tuple[float, float, float, float],
              images_path: str, image_files: List[str]) -> None:
    """Draw one box outline at its relative position; place its photo if available."""
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

    image_name = find_image_for_photo(image_files, photo_id)
    if image_name is None:
        c.setFont('Helvetica', 5)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.drawString(bx + 1, by + 1, f"id:{photo_id}")
        c.setFillColorRGB(0, 0, 0)
        return

    try:
        img_path = os.path.join(images_path, image_name)
        with Image.open(img_path) as img:
            buf = fit_image_to_box(img, bw, bh)
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
    """Render one spread inside `spread_rect`."""
    sx, sy, sw, sh = spread_rect
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
    if cell_w / cell_h > SPREAD_AR:
        sh = cell_h
        sw = sh * SPREAD_AR
    else:
        sw = cell_w
        sh = sw / SPREAD_AR
    sx = cell_x + (cell_w - sw) / 2
    sy = cell_y + (cell_h - sh) / 2
    return sx, sy, sw, sh


def _format_metric(v: Any) -> str:
    if v is None:
        return 'None'
    try:
        return f"{float(v):.4g}"
    except (TypeError, ValueError):
        return str(v)


# Short codes for compact display, mapped to nicer one-line labels for the
# breakdown block. Keeping it terse so the cell can show every applied factor.
_PENALTY_LABELS = {
    'left_color_mix':       'L color_mix',
    'right_color_mix':      'R color_mix',
    'left_class_mix':       'L class_mix',
    'right_class_mix':      'R class_mix',
    'left_bride_groom_mix': 'L bride/groom mix',
    'right_bride_groom_mix':'R bride/groom mix',
    'left_context_mix':     'L context_mix',
    'right_context_mix':    'R context_mix',
    'left_orientation_mix': 'L orientation_mix',
    'right_orientation_mix':'R orientation_mix',
    'double_color_mix':     'both pages color mix',
    'crop':                 'crop (square slots)',
    'time_order':           'time order inversions',
}


def _draw_penalty_breakdown(c: canvas.Canvas, rect: Tuple[float, float, float, float],
                            breakdown: Optional[dict]) -> None:
    """Render the spread-score breakdown inside `rect` (text only, no images).

    `rect` is (x, y, w, h) with y as the bottom edge (reportlab convention).
    Layout:
      • Header line 1: "dominant: <name>×<factor>" — the harshest multiplier
        that hit, so the eye lands on what mattered most for this score.
      • Header line 2: page-state flags (color/class/contexts on each side).
      • Then one row per applied penalty, top-down, with factor (and exponent
        when it's a power-style penalty like crop or time_order).
    Rows beyond what fits in `rect` are silently dropped — the dominant header
    still tells the headline story even if the tail is clipped.
    """
    if not breakdown:
        return
    x, y, w, h = rect
    if h <= 0:
        return

    applied = breakdown.get('penalties_applied') or []
    flags = breakdown.get('spread_flags') or {}
    left = breakdown.get('left_page') or {}
    right = breakdown.get('right_page') or {}

    # Dominant = the entry with the smallest factor (most punishing).
    dominant = None
    for entry in applied:
        try:
            f = float(entry.get('factor', 1.0))
        except (TypeError, ValueError):
            continue
        if dominant is None or f < dominant[1]:
            dominant = (entry, f)

    line_h = 7.0
    cursor_y = y + h - 6  # top edge minus a tiny lead

    # ---- header line 1: dominant ----
    c.setFont('Helvetica-Bold', 6)
    c.setFillColorRGB(0.45, 0.0, 0.0)
    if dominant is None:
        c.drawString(x, cursor_y, "dominant: (none — score=1.0)")
    else:
        entry, factor = dominant
        label = _PENALTY_LABELS.get(entry.get('name'), entry.get('name', '?'))
        exp = entry.get('exponent')
        suffix = f"^{exp}" if exp else ""
        c.drawString(x, cursor_y, f"dom: {label}{suffix}  ×{factor:.3g}")
    c.setFillColorRGB(0, 0, 0)
    cursor_y -= line_h

    # ---- header line 2: page-state flags ----
    def _page_state_summary(p: dict) -> str:
        parts = []
        if not p.get('is_same_color', True):
            parts.append('!color')
        if not p.get('is_same_class', True):
            parts.append('!class')
        if p.get('is_bride_groom_mix'):
            parts.append('b/g')
        n_ctx = p.get('number_of_unique_contexts')
        if n_ctx and n_ctx > 1:
            parts.append(f'ctx={n_ctx}')
        return ','.join(parts) if parts else 'clean'

    c.setFont('Helvetica', 6)
    c.setFillColorRGB(0.2, 0.2, 0.2)
    n_squares = flags.get('number_of_squares', 0)
    n_inv = flags.get('time_inversions', 0)
    flag_bits = []
    if flags.get('layout_left_mixed_orientation'):
        flag_bits.append('L mix-orient')
    if flags.get('layout_right_mixed_orientation'):
        flag_bits.append('R mix-orient')
    if flags.get('double_color_mix'):
        flag_bits.append('both gray')
    state = (f"L[{_page_state_summary(left)}]  R[{_page_state_summary(right)}]"
             + (f"  sq={n_squares}" if n_squares else "")
             + (f"  inv={n_inv}" if n_inv else "")
             + ("  " + " ".join(flag_bits) if flag_bits else ""))
    c.drawString(x, cursor_y, state[:120])
    c.setFillColorRGB(0, 0, 0)
    cursor_y -= line_h

    # ---- per-penalty rows ----
    c.setFont('Helvetica', 6)
    c.setFillColorRGB(0.15, 0.15, 0.15)
    for entry in applied:
        if cursor_y < y + 2:
            # cell would clip the next line; stop early
            break
        name = entry.get('name', '?')
        label = _PENALTY_LABELS.get(name, name)
        try:
            f = float(entry.get('factor', 1.0))
            f_str = f"{f:.3g}"
        except (TypeError, ValueError):
            f_str = str(entry.get('factor'))
        exp = entry.get('exponent')
        suffix = f"^{exp}" if exp else ""
        c.drawString(x + 4, cursor_y, f"• {label}{suffix}  ×{f_str}")
        cursor_y -= line_h
    c.setFillColorRGB(0, 0, 0)


def _draw_candidate(c: canvas.Canvas,
                    cell_rect: Tuple[float, float, float, float],
                    candidate: dict, photos: List[dict],
                    layouts_by_idx: Dict[int, dict],
                    layout_id2data_by_idx: Dict[int, dict],
                    images_path: str, image_files: List[str],
                    candidate_idx: int) -> None:
    """Draw one top_layouts candidate inside `cell_rect`. Multi-spread = vertical stack."""
    cx, cy, cw, ch = cell_rect

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

        caption_h = 9.0
        c.setFont('Helvetica', 6)
        c.setFillColorRGB(0.25, 0.25, 0.25)
        caption = (f"spread {s_idx + 1}/{len(spreads)}  "
                   f"layout_idx={layout_idx} layout_id={layout_id}  "
                   f"score={_format_metric(spread.get('score'))} "
                   f"weight={_format_metric(spread.get('weight'))}")
        c.drawString(sub_cell[0], sub_cell[1] + sub_cell[3] - caption_h + 1, caption)
        c.setFillColorRGB(0, 0, 0)

        # Reserve ~40% of the sub-cell height for the breakdown block when one
        # is present. With per_h ≈ 180pt that's ~70pt = ~10 lines of text at
        # 7pt pitch — enough room to list every applied penalty. The spread
        # illustration takes the remaining space above.
        breakdown = spread.get('penalty_breakdown')
        breakdown_h = 0.0
        if breakdown:
            breakdown_h = min(max(0.35 * (sub_cell[3] - caption_h), 40.0),
                              sub_cell[3] - caption_h - 20.0)
            breakdown_h = max(0.0, breakdown_h)

        ill_h = max(1.0, sub_cell[3] - caption_h - breakdown_h)
        ill_cell = (sub_cell[0], sub_cell[1] + breakdown_h,
                    sub_cell[2], ill_h)
        spread_rect = _fit_spread_rect_in_cell(*ill_cell)

        left_pids = [photos[i]['id'] for i in sorted(spread['left_page_photo_idxs'])
                     if 0 <= i < len(photos)]
        right_pids = [photos[i]['id'] for i in sorted(spread['right_page_photo_idxs'])
                      if 0 <= i < len(photos)]

        _draw_spread(c, spread_rect, left_boxes, right_boxes,
                     left_pids, right_pids, images_path, image_files)

        if breakdown and breakdown_h > 0:
            breakdown_rect = (sub_cell[0], sub_cell[1],
                              sub_cell[2], breakdown_h)
            _draw_penalty_breakdown(c, breakdown_rect, breakdown)


def _pick_grid(n: int) -> Tuple[int, int]:
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
        cell_x = grid_x + col * cell_w
        cell_y = grid_y + (rows - 1 - row) * cell_h
        inner = (cell_x + 3, cell_y + 3, cell_w - 6, cell_h - 6)
        _draw_candidate(c, inner, cand, photos,
                        layouts_by_idx, layout_id2data_by_idx,
                        images_path, image_files, i)

    c.showPage()


# ---------- entry point ----------

def render(stages_dir: str, images_path: str, output_pdf_path: str) -> None:
    """Read every per-group file under `stages_dir`, render one page each."""
    layouts_by_idx, layout_id2data_by_idx = _load_layouts(stages_dir)
    group_files = _load_group_files(stages_dir)
    image_files = list_image_files(images_path)
    if not image_files:
        print(f"[warn] no images found under {images_path}; boxes will show photo ids only")

    c = canvas.Canvas(output_pdf_path, pagesize=PAGE_SIZE)
    for group_name, group_data in group_files:
        _draw_group_page(c, PAGE_SIZE, group_name, group_data,
                         layouts_by_idx, layout_id2data_by_idx,
                         images_path, image_files)
    c.save()
