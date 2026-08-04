"""Render a catalogue PDF of every layout in `layouts_df`.

Reads `files/stages_info/spreads/_layouts.json` and draws one small scheme per
row of `layouts_df` — all boxes empty (outlines only), no photos involved. This
is the "what could the algorithm have picked" companion to
`spreads_layouts.pdf`'s "what it actually considered".

Schemes are drawn at exactly the same size as the spread illustrations in
`spreads_layouts.pdf` (see `SPREAD_W`), and via the same `_draw_spread` helper,
so a scheme here is visually interchangeable with a candidate there.

Layouts keep their `layouts_df` order, so the `#idx` under each scheme is the
same `layout_idx` printed in `spreads_layouts.pdf` captions.

Read-only: no imports from the layout/grouping pipeline.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from reportlab.pdfgen import canvas

from stages_visualizer._shared import PAGE_SIZE
from stages_visualizer.spreads import SPREAD_AR, _draw_spread


MARGIN = 18.0
HEADER_H = 36.0

# Same spread width as one `spreads_layouts.pdf` candidate cell: that page uses
# a 3-column grid over `PAGE_W - 2*MARGIN`, minus the 3pt cell inset and the
# 2pt sub-cell pad on each side. Height follows from the album aspect ratio,
# which is what `_fit_spread_rect_in_cell` lands on there too (those cells are
# always width-limited, never height-limited).
GRID_COLS = 3
SPREAD_W = (PAGE_SIZE[0] - 2 * MARGIN) / GRID_COLS - 2 * 3.0 - 2 * 2.0
SPREAD_H = SPREAD_W / SPREAD_AR

CAPTION_H = 9.0
ROW_GAP = 6.0


# ---------- loading ----------

def _load_layouts_df(layouts_json_path: str) -> List[dict]:
    """Return `layouts_df` rows in saved order."""
    with open(layouts_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('layouts_df', []) or []


def _boxes_for(record: dict) -> Tuple[List[dict], List[dict]]:
    """Split a `layouts_df` row's `boxes_info` into (left_boxes, right_boxes)."""
    boxes_by_id: Dict[Any, dict] = {b['id']: b for b in record.get('boxes_info', [])}
    left = [boxes_by_id[bid] for bid in record.get('left_box_ids', []) if bid in boxes_by_id]
    right = [boxes_by_id[bid] for bid in record.get('right_box_ids', []) if bid in boxes_by_id]
    return left, right


# ---------- drawing ----------

def _draw_scheme(c: canvas.Canvas, x: float, y: float, record: dict) -> None:
    """Draw one empty-box scheme with its caption; `(x, y)` = bottom-left of cell."""
    left_boxes, right_boxes = _boxes_for(record)

    # No photo ids -> `_draw_spread` outlines every box and places nothing.
    _draw_spread(c, (x, y + CAPTION_H, SPREAD_W, SPREAD_H),
                 left_boxes, right_boxes, [], [], '', [])

    c.setFont('Helvetica', 6)
    c.setFillColorRGB(0.25, 0.25, 0.25)
    caption = (f"#{record.get('index')}  id={record.get('id')}  "
               f"boxes={record.get('number of boxes')} "
               f"({len(left_boxes)}+{len(right_boxes)})")
    if record.get('is_mirrored'):
        caption += "  mirrored"
    c.drawString(x, y + 1, caption)
    c.setFillColorRGB(0, 0, 0)


def _rows_per_page(page_h: float) -> int:
    usable = page_h - MARGIN - HEADER_H - MARGIN
    pitch = SPREAD_H + CAPTION_H + ROW_GAP
    return max(1, int(usable // pitch))


def _draw_page(c: canvas.Canvas, page_size: Tuple[float, float],
               records: List[dict], total: int, first_pos: int) -> None:
    """One page: header + grid of schemes for `records`."""
    page_w, page_h = page_size
    pitch = SPREAD_H + CAPTION_H + ROW_GAP
    top = page_h - MARGIN - HEADER_H

    c.setFont('Helvetica-Bold', 13)
    c.drawString(MARGIN, page_h - MARGIN - 12, "All available layouts")
    c.setFont('Helvetica', 9)
    c.drawString(MARGIN, page_h - MARGIN - 26,
                 f"layouts_df rows {first_pos + 1}-{first_pos + len(records)} of {total}  "
                 f"|  empty boxes, spread size as in spreads_layouts.pdf")

    for i, record in enumerate(records):
        col = i % GRID_COLS
        row = i // GRID_COLS
        x = MARGIN + col * (page_w - 2 * MARGIN) / GRID_COLS
        y = top - (row + 1) * pitch + ROW_GAP
        _draw_scheme(c, x, y, record)

    c.showPage()


# ---------- entry point ----------

def render(layouts_json_path: str, images_path: str, output_pdf_path: str) -> None:
    """Render every `layouts_df` row as an empty-box scheme. `images_path` unused."""
    records = _load_layouts_df(layouts_json_path)
    if not records:
        print(f"[warn] no layouts_df rows in {layouts_json_path}")

    per_page = _rows_per_page(PAGE_SIZE[1]) * GRID_COLS
    c = canvas.Canvas(output_pdf_path, pagesize=PAGE_SIZE)
    for start in range(0, max(len(records), 1), per_page):
        _draw_page(c, PAGE_SIZE, records[start:start + per_page], len(records), start)
    c.save()