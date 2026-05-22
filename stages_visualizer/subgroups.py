"""Render subgroups_*.pdf from files/stages_info/groups/subgroups_{0,1,2}.json.

Three snapshots of the grouping pipeline, in order:
    subgroups_0 -> initial groups (before split)
    subgroups_1 -> after handle_wedding_splitting
    subgroups_2 -> after all merges + singleton resolution + rebalance

Subgroups are listed in chronological order (sorted by `mean_general_time`
in the saved JSON). Per-subgroup we draw:
  - Header: group_key, photo count, mean wall-clock time and the relative
    `mean_general_time` (the value `merge.py` actually compares groups by).
  - A wrapping grid of photos at natural aspect ratio in **fixed-size cells**.

Cell size is the same on every page of every PDF, so relative photo counts
are visible as relative area. Pages flow dynamically: as many subgroups pack
onto a page as fit at their natural heights; the rest spill onto a new page.
"""

from __future__ import annotations

import json
import os
from typing import List

from reportlab.pdfgen import canvas

from stages_visualizer._shared import (
    DEFAULT_CELL_SIZE,
    PAGE_SIZE,
    draw_photo_grid,
    format_general_time,
    format_image_time_date,
    grid_cols_for_width,
    grid_height_for,
    list_image_files,
)


CAPTION_FIELDS = ('image_time_date', 'original_context')

HEADER_H = 16.0
PAD = 10.0
PAGE_MARGIN = 20.0


def _load_subgroups(json_path: str) -> List[dict]:
    if not os.path.isfile(json_path):
        return []
    with open(json_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return d.get('subgroups', []) or []


def _draw_header(c: canvas.Canvas, x: float, y_top: float, w: float,
                 sg: dict) -> None:
    """Bold group key on the left, right-aligned mean-time / count metadata."""
    gk = sg.get('group_key', [])
    n = sg.get('n_photos', 0)
    mean_date = format_image_time_date(sg.get('mean_image_time_date'))
    mean_general = format_general_time(sg.get('mean_general_time'))

    header_y = y_top - 11
    c.setFont('Helvetica-Bold', 10)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(x, header_y, f"{tuple(gk)}")

    c.setFont('Helvetica', 8)
    c.setFillColorRGB(0.3, 0.3, 0.3)
    meta = f"n={n}   mean t={mean_date}   (mean general_time={mean_general})"
    text_w = c.stringWidth(meta, 'Helvetica', 8)
    c.drawString(x + w - text_w, header_y, meta)
    c.setFillColorRGB(0, 0, 0)


def render(json_path: str, images_path: str, output_pdf_path: str) -> None:
    """Render the PDF for one subgroups_*.json file."""
    subgroups = _load_subgroups(json_path)
    image_files = list_image_files(images_path)
    if not image_files:
        print(f"[warn] no images found under {images_path}; cells will show photo ids only")

    page_w, page_h = PAGE_SIZE
    panel_w = page_w - 2 * PAGE_MARGIN
    cols = grid_cols_for_width(panel_w, DEFAULT_CELL_SIZE)
    top = page_h - PAGE_MARGIN
    bottom = PAGE_MARGIN

    c = canvas.Canvas(output_pdf_path, pagesize=PAGE_SIZE)

    if not subgroups:
        c.setFont('Helvetica', 12)
        c.drawString(36, page_h - 50,
                     f"(no subgroups recorded in {os.path.basename(json_path)})")
        c.showPage()
        c.save()
        return

    cursor_y = top
    page_started = False

    for sg in subgroups:
        photos = sg.get('photos') or []
        n = sg.get('n_photos') or len(photos)
        grid_h = grid_height_for(n, cols, CAPTION_FIELDS, DEFAULT_CELL_SIZE)
        panel_h = HEADER_H + grid_h

        # If this panel won't fit on the current page, start a new one. A
        # subgroup that's larger than a full page still gets placed at the
        # top (and overflows off the bottom) — that's acceptable here, the
        # alternative is shrinking cells which is exactly what we're avoiding.
        if page_started and cursor_y - panel_h < bottom:
            c.showPage()
            cursor_y = top
            page_started = False

        _draw_header(c, PAGE_MARGIN, cursor_y, panel_w, sg)
        grid_top = cursor_y - HEADER_H
        grid_bottom = grid_top - grid_h
        grid_rect = (PAGE_MARGIN, grid_bottom, panel_w, grid_h)
        draw_photo_grid(c, grid_rect, photos, images_path, image_files,
                        caption_fields=CAPTION_FIELDS,
                        cell_size=DEFAULT_CELL_SIZE, label=None)
        cursor_y = grid_bottom - PAD
        page_started = True

    if page_started:
        c.showPage()
    c.save()
