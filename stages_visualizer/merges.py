"""Render merge.pdf from files/stages_info/groups/merge.json.

For each successful merge, draws one PDF page showing:
  1. Header: merge type (`bridegroom` / `other` / `singleton`) and the two
     source group keys.
  2. The merged group as a single horizontal strip of photo thumbnails.
     Each thumb is captioned with `general_time` (formatted HH:MM:SS) and
     `original_context` (the photo's pre-merge cluster), per the brief.
  3. If the merge produced a reminder group (bridegroom case), it's shown
     beneath the merged strip.

Only records present in `merge.json` are rendered — failed/skipped merge
attempts aren't saved (and aren't shown). Read-only: only depends on the
saved JSON and on-disk images.
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


MERGE_FILE = 'merge.json'

# Per-thumb captions: time first, then the original cluster (per the brief).
CAPTION_FIELDS = ('general_time', 'original_context')


def _load_merges(stages_dir: str) -> List[dict]:
    """Load merge.json (`{'merges': [...]}`). Empty list if the file is missing."""
    path = os.path.join(stages_dir, MERGE_FILE)
    if not os.path.isfile(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('merges', []) or []


def _draw_merge_page(c: canvas.Canvas, page_size: Tuple[float, float],
                     merge: dict, images_path: str, image_files: List[str]) -> None:
    page_w, page_h = page_size
    margin = 24.0

    merge_type = merge.get('merge_type', '?')
    src_key = merge.get('src_key', [])
    partner_key = merge.get('partner_key', [])
    merged_photos = merge.get('merged_photos', []) or []
    reminder_photos = merge.get('reminder_photos', []) or []

    # ---- header ----
    c.setFont('Helvetica-Bold', 13)
    c.drawString(margin, page_h - margin - 12,
                 f"Merge ({merge_type}):  {tuple(src_key)}  +  {tuple(partner_key)}")
    c.setFont('Helvetica', 9)
    c.setFillColorRGB(0.25, 0.25, 0.25)
    c.drawString(margin, page_h - margin - 26,
                 f"{len(merged_photos)} photos merged"
                 + (f"  |  {len(reminder_photos)} in reminder" if reminder_photos else ""))
    c.setFillColorRGB(0, 0, 0)

    # ---- strips ----
    header_h = 36.0
    body_top = page_h - margin - header_h
    body_bottom = margin
    body_h = body_top - body_bottom
    label_h = 12.0
    inner_pad = 8.0

    if reminder_photos:
        # Two strips: merged (top, ~60%) + reminder (bottom, ~40%).
        merged_band_h = body_h * 0.60
        rem_band_h = body_h - merged_band_h
    else:
        merged_band_h = body_h
        rem_band_h = 0.0

    merged_rect = (margin,
                   body_top - merged_band_h + inner_pad,
                   page_w - 2 * margin,
                   max(40.0, merged_band_h - label_h - 2 * inner_pad))
    draw_photo_strip(c, merged_rect, merged_photos,
                     images_path, image_files,
                     caption_fields=list(CAPTION_FIELDS),
                     label="Merged group (sorted by general_time)")

    if reminder_photos:
        rem_rect = (margin,
                    body_bottom + inner_pad,
                    page_w - 2 * margin,
                    max(40.0, rem_band_h - label_h - 2 * inner_pad))
        draw_photo_strip(c, rem_rect, reminder_photos,
                         images_path, image_files,
                         caption_fields=list(CAPTION_FIELDS),
                         label="Reminder (left over after balancing)")


def render(stages_dir: str, images_path: str, output_pdf_path: str) -> None:
    merges = _load_merges(stages_dir)
    image_files = list_image_files(images_path)
    if not image_files:
        print(f"[warn] no images found under {images_path}; strips will show photo ids only")

    c = canvas.Canvas(output_pdf_path, pagesize=PAGE_SIZE)
    if not merges:
        c.setFont('Helvetica', 12)
        c.drawString(36, PAGE_SIZE[1] - 50, "(no merges recorded)")
        c.showPage()
    else:
        for m in merges:
            _draw_merge_page(c, PAGE_SIZE, m, images_path, image_files)
            c.showPage()
    c.save()
