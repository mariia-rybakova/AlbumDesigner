"""Render split.pdf from files/stages_info/groups/split.json.

For each split entry:
  1. Header (group key, split method, counts) on a fresh page.
  2. A grid of the original group's photos, in time order.
  3. One grid per resulting sub-group, stacked below.

Cells are a fixed size globally so the visual area of a group is always
proportional to its photo count. Panels that overflow start a fresh page.
Photos are drawn at their natural aspect ratio — no cropping, no stretching.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

from reportlab.pdfgen import canvas

from stages_visualizer._shared import (
    DEFAULT_CELL_SIZE,
    PAGE_SIZE,
    draw_photo_grid,
    grid_cols_for_width,
    grid_height_for,
    list_image_files,
)


SPLIT_FILE = 'split.json'

# Match album1.pdf's caption source so the same photo shows the same time.
CAPTION_FIELDS = ('image_time_date',)

PAGE_MARGIN = 24.0
HEADER_H = 36.0     # title strip at the top of each split entry
LABEL_H = 12.0      # bold label above each grid
GRID_PAD = 8.0      # gap between consecutive grids


def _load_splits(stages_dir: str) -> List[dict]:
    path = os.path.join(stages_dir, SPLIT_FILE)
    if not os.path.isfile(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('splits', []) or []


def _draw_title(c: canvas.Canvas, x: float, y_top: float, split: dict) -> None:
    group_key = split.get('group_key', [])
    method = split.get('split_method', '?')
    original_photos = split.get('original_photos', []) or []
    sub_groups = split.get('sub_groups', []) or []
    c.setFont('Helvetica-Bold', 13)
    c.drawString(x, y_top - 12,
                 f"Split: {tuple(group_key)}  |  method: {method}")
    c.setFont('Helvetica', 9)
    c.setFillColorRGB(0.25, 0.25, 0.25)
    c.drawString(x, y_top - 26,
                 f"{len(original_photos)} photos  ->  {len(sub_groups)} sub-groups  "
                 f"(spread_size={split.get('group_spread_size')}, "
                 f"number_of_spreads={split.get('number_of_spreads')})")
    c.setFillColorRGB(0, 0, 0)


def _draw_split(c: canvas.Canvas, split: dict,
                images_path: str, image_files: List[str]) -> None:
    """Draw one split entry across as many pages as it needs."""
    page_w, page_h = PAGE_SIZE
    panel_w = page_w - 2 * PAGE_MARGIN
    cols = grid_cols_for_width(panel_w, DEFAULT_CELL_SIZE)
    top = page_h - PAGE_MARGIN
    bottom = PAGE_MARGIN

    # Fresh page for this entry, header at top.
    _draw_title(c, PAGE_MARGIN, top, split)
    cursor_y = top - HEADER_H

    original_photos = split.get('original_photos', []) or []
    sub_groups = split.get('sub_groups', []) or []

    panels = [('Original group', original_photos)]
    for sg in sub_groups:
        panels.append((f"Sub-group {sg.get('sub_index')}", sg.get('photos') or []))

    if not sub_groups:
        # still draw the original at least
        pass

    for label_text, photos in panels:
        n = len(photos)
        grid_h = grid_height_for(n, cols, CAPTION_FIELDS, DEFAULT_CELL_SIZE)
        panel_h = LABEL_H + grid_h

        if cursor_y - panel_h < bottom:
            c.showPage()
            cursor_y = top  # continuation page, no header repeat

        # bold label
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

    if not sub_groups:
        c.setFont('Helvetica', 10)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.drawString(PAGE_MARGIN, cursor_y - 12, "(no sub-groups recorded)")
        c.setFillColorRGB(0, 0, 0)


def render(stages_dir: str, images_path: str, output_pdf_path: str) -> None:
    splits = _load_splits(stages_dir)
    image_files = list_image_files(images_path)
    if not image_files:
        print(f"[warn] no images found under {images_path}; cells will show photo ids only")

    c = canvas.Canvas(output_pdf_path, pagesize=PAGE_SIZE)
    if not splits:
        c.setFont('Helvetica', 12)
        c.drawString(36, PAGE_SIZE[1] - 50, "(no splits recorded)")
        c.showPage()
    else:
        for split in splits:
            _draw_split(c, split, images_path, image_files)
            c.showPage()
    c.save()
