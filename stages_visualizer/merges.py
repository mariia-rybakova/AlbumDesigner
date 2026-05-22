"""Render merge.pdf from files/stages_info/groups/merge.json.

For each successful merge:
  1. Header (merge type and the two source keys) on a fresh page.
  2. A grid of the merged group's photos.
  3. If a reminder group exists, its grid below.

Cells are a fixed size globally so the visual area of a group is always
proportional to its photo count. Panels that overflow start a fresh page.
Photos are drawn at their natural aspect ratio — no cropping, no stretching.
Captions show `image_time_date` (matching album1.pdf) and `original_context`.

Only successful merges appear; failed/skipped attempts aren't saved.
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
    grid_cols_for_width,
    grid_height_for,
    list_image_files,
)


MERGE_FILE = 'merge.json'

CAPTION_FIELDS = ('image_time_date', 'original_context')

PAGE_MARGIN = 24.0
HEADER_H = 36.0
LABEL_H = 12.0
GRID_PAD = 8.0


def _load_merges(stages_dir: str) -> List[dict]:
    path = os.path.join(stages_dir, MERGE_FILE)
    if not os.path.isfile(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('merges', []) or []


def _draw_title(c: canvas.Canvas, x: float, y_top: float, merge: dict) -> None:
    merge_type = merge.get('merge_type', '?')
    src_key = merge.get('src_key', [])
    partner_key = merge.get('partner_key', [])
    n_merged = len(merge.get('merged_photos', []) or [])
    n_reminder = len(merge.get('reminder_photos', []) or [])
    c.setFont('Helvetica-Bold', 13)
    c.drawString(x, y_top - 12,
                 f"Merge ({merge_type}):  {tuple(src_key)}  +  {tuple(partner_key)}")
    c.setFont('Helvetica', 9)
    c.setFillColorRGB(0.25, 0.25, 0.25)
    c.drawString(x, y_top - 26,
                 f"{n_merged} photos merged"
                 + (f"  |  {n_reminder} in reminder" if n_reminder else ""))
    c.setFillColorRGB(0, 0, 0)


def _draw_merge(c: canvas.Canvas, merge: dict,
                images_path: str, image_files: List[str]) -> None:
    """Draw one merge entry across as many pages as it needs."""
    page_w, page_h = PAGE_SIZE
    panel_w = page_w - 2 * PAGE_MARGIN
    cols = grid_cols_for_width(panel_w, DEFAULT_CELL_SIZE)
    top = page_h - PAGE_MARGIN
    bottom = PAGE_MARGIN

    _draw_title(c, PAGE_MARGIN, top, merge)
    cursor_y = top - HEADER_H

    merged_photos = merge.get('merged_photos', []) or []
    reminder_photos = merge.get('reminder_photos', []) or []
    panels = [('Merged group', merged_photos)]
    if reminder_photos:
        panels.append(('Reminder (left over after balancing)', reminder_photos))

    for label_text, photos in panels:
        n = len(photos)
        grid_h = grid_height_for(n, cols, CAPTION_FIELDS, DEFAULT_CELL_SIZE)
        panel_h = LABEL_H + grid_h

        if cursor_y - panel_h < bottom:
            c.showPage()
            cursor_y = top

        c.setFont('Helvetica-Bold', 8)
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.drawString(PAGE_MARGIN, cursor_y - LABEL_H + 2,
                     f"{label_text}  (n={n})")
        c.setFillColorRGB(0, 0, 0)

        grid_top = cursor_y - LABEL_H
        grid_bottom = grid_top - grid_h
        grid_rect = (PAGE_MARGIN, grid_bottom, panel_w, grid_h)
        draw_photo_grid(c, grid_rect, photos, images_path, image_files,
                        caption_fields=CAPTION_FIELDS,
                        cell_size=DEFAULT_CELL_SIZE, label=None)
        cursor_y = grid_bottom - GRID_PAD


def render(stages_dir: str, images_path: str, output_pdf_path: str) -> None:
    merges = _load_merges(stages_dir)
    image_files = list_image_files(images_path)
    if not image_files:
        print(f"[warn] no images found under {images_path}; cells will show photo ids only")

    c = canvas.Canvas(output_pdf_path, pagesize=PAGE_SIZE)
    if not merges:
        c.setFont('Helvetica', 12)
        c.drawString(36, PAGE_SIZE[1] - 50, "(no merges recorded)")
        c.showPage()
    else:
        for m in merges:
            _draw_merge(c, m, images_path, image_files)
            c.showPage()
    c.save()
