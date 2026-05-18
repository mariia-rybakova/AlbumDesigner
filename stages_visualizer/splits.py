"""Render split.pdf from files/stages_info/groups/split.json.

For each split entry, draws one PDF page showing:
  1. A header strip with the group key and split method.
  2. A horizontal strip of the original group's photos in their saved
     (time-sorted) order, each captioned with its general_time.
  3. Below that, one horizontal strip per resulting sub-group (typically two:
     "left" and "right" after a binary split, but the renderer handles any N).

Read-only: only depends on the saved JSON and on-disk images.
"""

from __future__ import annotations

import json
import os
from typing import List, Tuple

from reportlab.pdfgen import canvas

from stages_visualizer._shared import (
    PAGE_SIZE,
    draw_photo_strip,
    list_image_files,
)


SPLIT_FILE = 'split.json'

CAPTION_FIELDS = ('general_time',)


def _load_splits(stages_dir: str) -> List[dict]:
    """Load split.json (`{'splits': [...]}`). Empty list if the file is missing."""
    path = os.path.join(stages_dir, SPLIT_FILE)
    if not os.path.isfile(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('splits', []) or []


def _draw_split_page(c: canvas.Canvas, page_size: Tuple[float, float],
                     split: dict, images_path: str, image_files: List[str]) -> None:
    page_w, page_h = page_size
    margin = 24.0

    # ---- header ----
    group_key = split.get('group_key', [])
    method = split.get('split_method', '?')
    n_photos = len(split.get('original_photos', []))
    sub_groups = split.get('sub_groups', []) or []
    n_subs = len(sub_groups)

    c.setFont('Helvetica-Bold', 13)
    c.drawString(margin, page_h - margin - 12,
                 f"Split: {tuple(group_key)}  |  method: {method}")
    c.setFont('Helvetica', 9)
    c.setFillColorRGB(0.25, 0.25, 0.25)
    c.drawString(margin, page_h - margin - 26,
                 f"{n_photos} photos  ->  {n_subs} sub-groups  "
                 f"(spread_size={split.get('group_spread_size')}, "
                 f"number_of_spreads={split.get('number_of_spreads')})")
    c.setFillColorRGB(0, 0, 0)

    # ---- strips ----
    # Top half: original. Bottom half: sub-groups side-by-side (or stacked).
    header_h = 36.0
    body_top = page_h - margin - header_h
    body_bottom = margin
    body_h = body_top - body_bottom

    # Strip caption labels eat ~12pt above each strip; account for that.
    label_h = 12.0
    # 40% of body for original, 60% for sub-groups.
    orig_band_h = body_h * 0.40
    subs_band_h = body_h - orig_band_h
    inner_pad = 8.0

    # Original strip.
    orig_strip_rect = (margin,
                       body_top - orig_band_h + inner_pad,
                       page_w - 2 * margin,
                       max(40.0, orig_band_h - label_h - 2 * inner_pad))
    draw_photo_strip(c, orig_strip_rect, split.get('original_photos', []) or [],
                     images_path, image_files,
                     caption_fields=list(CAPTION_FIELDS),
                     label="Original group (sorted by general_time)")

    # Sub-group strips: lay them side by side horizontally. Each sub-group gets
    # a column proportional to its photo count so visually-large sub-groups
    # don't squash visually-small ones (and vice versa).
    if not sub_groups:
        c.setFont('Helvetica', 10)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.drawString(margin, body_top - orig_band_h - 20, "(no sub-groups recorded)")
        c.setFillColorRGB(0, 0, 0)
        return

    subs_band_top = body_top - orig_band_h
    subs_strip_h = max(40.0, subs_band_h - label_h - 2 * inner_pad)
    subs_strip_y = subs_band_top - subs_band_h + inner_pad
    column_pad = 12.0
    total_w = page_w - 2 * margin - column_pad * (n_subs - 1)
    sizes = [max(1, len(sg.get('photos', []) or [])) for sg in sub_groups]
    total_size = sum(sizes) or 1
    x_cursor = margin
    for i, sg in enumerate(sub_groups):
        col_w = total_w * (sizes[i] / total_size)
        col_rect = (x_cursor, subs_strip_y, col_w, subs_strip_h)
        sub_idx = sg.get('sub_index', i)
        draw_photo_strip(c, col_rect, sg.get('photos', []) or [],
                         images_path, image_files,
                         caption_fields=list(CAPTION_FIELDS),
                         label=f"Sub-group {sub_idx}  (n={len(sg.get('photos') or [])})")
        x_cursor += col_w + column_pad


def render(stages_dir: str, images_path: str, output_pdf_path: str) -> None:
    splits = _load_splits(stages_dir)
    image_files = list_image_files(images_path)
    if not image_files:
        print(f"[warn] no images found under {images_path}; strips will show photo ids only")

    c = canvas.Canvas(output_pdf_path, pagesize=PAGE_SIZE)
    if not splits:
        c.setFont('Helvetica', 12)
        c.drawString(36, PAGE_SIZE[1] - 50, "(no splits recorded)")
        c.showPage()
    else:
        for split in splits:
            _draw_split_page(c, PAGE_SIZE, split, images_path, image_files)
            c.showPage()
    c.save()
