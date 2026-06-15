"""Render merge.pdf from files/stages_info/groups/merge.json.

Renders the chronological merge event stream:

  search           - one src group's hunt for a partner; lists every candidate
                     evaluated by `merge_illegal_group_by_time` with raw and
                     adjusted time distances, plus the selected one (or None).
                     Text-only block, no photos.

  merge_skipped    - a previously-proposed merge that didn't go through. Reason
                     and (when applicable) the partner_key + time_diff. Single
                     text line.

  merge_succeeded  - actual merge that happened. Full photo grid for the merged
                     group plus a reminder grid if the merge produced one.

Events render top-down on each page; long blocks (search tables with many
candidates, big merged groups) flow onto fresh pages without shrinking the
fixed cell size or compressing the candidate table.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, List, Optional, Tuple

from reportlab.pdfgen import canvas

from stages_visualizer._shared import (
    DEFAULT_CELL_SIZE,
    PAGE_SIZE,
    caption_height,
    draw_photo_grid,
    caption_fields_for,
    format_general_time,
    mean_time_label,
    grid_cols_for_width,
    grid_height_for,
    list_image_files,
)


MERGE_FILE = 'merge.json'

CAPTION_FIELDS = ('image_time_date', 'original_context')

PAGE_MARGIN = 24.0
LABEL_H = 12.0
GRID_PAD = 8.0
LINE_H = 10.0           # text-row pitch for search/skipped blocks
EVENT_GAP = 14.0        # gap between two consecutive events
CAND_TABLE_PAD = 4.0


# ---------- load ----------

def _load_events(stages_dir: str) -> tuple:
    """Load `(events, is_artificial_time)`. Falls back to legacy `merges` format gracefully."""
    path = os.path.join(stages_dir, MERGE_FILE)
    if not os.path.isfile(path):
        return [], False
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    is_artificial_time = bool(data.get('is_artificial_time', False))
    events = data.get('events')
    if events is not None:
        return events, is_artificial_time
    # Legacy format: a list of successful merges only.
    return ([dict(e, type='merge_succeeded') for e in (data.get('merges') or [])],
            is_artificial_time)


# ---------- helpers ----------

def _fmt_time(t: Any) -> str:
    """Format a time-distance value (seconds) as HH:MM:SS or '-' if missing."""
    if t is None:
        return "-"
    try:
        return format_general_time(float(t))
    except (TypeError, ValueError):
        return str(t)


def _fmt_key(k: Any) -> str:
    if k is None:
        return "None"
    try:
        return str(tuple(k))
    except TypeError:
        return str(k)


# ---------- height computation (for page-flow) ----------

def _search_height(ev: dict) -> float:
    """Total vertical space a search event needs (header + candidate rows)."""
    candidates = ev.get('candidates') or []
    header_lines = 2  # title + src meta
    table_header_lines = 1
    n_rows = max(1, len(candidates))
    return (header_lines + table_header_lines + n_rows) * LINE_H + CAND_TABLE_PAD * 2


def _skipped_height(_ev: dict) -> float:
    return 2 * LINE_H


def _succeeded_height(ev: dict, cols: int, caption_fields: tuple) -> float:
    merged = ev.get('merged_photos') or []
    reminder = ev.get('reminder_photos') or []
    h = LABEL_H + grid_height_for(len(merged), cols, caption_fields, DEFAULT_CELL_SIZE)
    if reminder:
        h += GRID_PAD + LABEL_H + grid_height_for(len(reminder), cols, caption_fields, DEFAULT_CELL_SIZE)
    # title above
    return h + 28.0


def _event_height(ev: dict, cols: int, caption_fields: tuple) -> float:
    t = ev.get('type')
    if t == 'search':
        return _search_height(ev)
    if t == 'merge_skipped':
        return _skipped_height(ev)
    if t == 'merge_succeeded':
        return _succeeded_height(ev, cols, caption_fields)
    return LINE_H


# ---------- drawing per event ----------

def _draw_search(c: canvas.Canvas, x: float, y_top: float, w: float, ev: dict,
                 is_artificial_time: bool = False) -> float:
    """Draw a search event; return the y of the bottom edge below the block."""
    candidates = ev.get('candidates') or []
    selected_key = ev.get('selected_partner_key')

    # ---- title ----
    cursor = y_top
    c.setFont('Helvetica-Bold', 10)
    c.setFillColorRGB(0.05, 0.05, 0.3)
    c.drawString(x, cursor - 10,
                 f"SEARCH [{ev.get('merge_type','?')}]  src={_fmt_key(ev.get('src_key'))}  "
                 f"n={ev.get('src_n_photos','?')}")
    cursor -= LINE_H

    # subtitle: mean times
    c.setFont('Helvetica', 8)
    c.setFillColorRGB(0.3, 0.3, 0.3)
    c.drawString(x, cursor - 8,
                 f"src mean t={mean_time_label(ev.get('src_mean_image_time_date'), ev.get('src_mean_general_time'), is_artificial_time)}  "
                 f"(general_time={format_general_time(ev.get('src_mean_general_time'))})    "
                 f"selected={_fmt_key(selected_key)}  "
                 f"time_diff={_fmt_time(ev.get('selected_time_diff'))}")
    c.setFillColorRGB(0, 0, 0)
    cursor -= LINE_H

    # ---- candidates table ----
    cursor -= CAND_TABLE_PAD
    # column header
    c.setFont('Helvetica-Bold', 7)
    c.setFillColorRGB(0.1, 0.1, 0.1)
    header_y = cursor - 8
    col_x = {
        'mark': x,
        'partner': x + 14,
        'n': x + 195,
        'raw': x + 225,
        'adj': x + 295,
        'between': x + 365,
        'long': x + 410,
        'fits': x + 445,
    }
    c.drawString(col_x['mark'], header_y, '')
    c.drawString(col_x['partner'], header_y, 'partner_key')
    c.drawString(col_x['n'], header_y, 'n')
    c.drawString(col_x['raw'], header_y, 'raw Δt')
    c.drawString(col_x['adj'], header_y, 'adj Δt')
    c.drawString(col_x['between'], header_y, 'between')
    c.drawString(col_x['long'], header_y, 'long?')
    c.drawString(col_x['fits'], header_y, 'fits?')
    c.setFillColorRGB(0, 0, 0)
    cursor -= LINE_H

    if not candidates:
        c.setFont('Helvetica-Oblique', 8)
        c.setFillColorRGB(0.4, 0.4, 0.4)
        c.drawString(x + 14, cursor - 8, "(no candidates considered)")
        c.setFillColorRGB(0, 0, 0)
        return cursor - LINE_H - CAND_TABLE_PAD

    # rows
    c.setFont('Helvetica', 7)
    for cand in candidates:
        row_y = cursor - 8
        # mark winner with an arrow
        if cand.get('was_selected'):
            c.setFillColorRGB(0.05, 0.5, 0.05)
            c.drawString(col_x['mark'], row_y, '>')
            c.setFillColorRGB(0, 0, 0)
        c.drawString(col_x['partner'], row_y, _fmt_key(cand.get('partner_key')))
        c.drawString(col_x['n'], row_y, str(cand.get('n_photos', '?')))
        c.drawString(col_x['raw'], row_y, _fmt_time(cand.get('time_diff_raw')))
        c.drawString(col_x['adj'], row_y, _fmt_time(cand.get('time_diff_adjusted')))
        c.drawString(col_x['between'], row_y, str(cand.get('images_in_between', '?')))
        c.drawString(col_x['long'], row_y, 'yes' if cand.get('long_distance') else 'no')
        c.drawString(col_x['fits'], row_y, 'yes' if cand.get('within_size_limit', True) else 'NO')
        cursor -= LINE_H

    return cursor - CAND_TABLE_PAD


def _draw_skipped(c: canvas.Canvas, x: float, y_top: float, ev: dict) -> float:
    c.setFont('Helvetica-Bold', 9)
    c.setFillColorRGB(0.55, 0.2, 0.0)
    c.drawString(x, y_top - 10,
                 f"SKIPPED [{ev.get('merge_type','?')}]  src={_fmt_key(ev.get('src_key'))} "
                 f"+ partner={_fmt_key(ev.get('partner_key'))}")
    c.setFillColorRGB(0.3, 0.3, 0.3)
    c.setFont('Helvetica', 8)
    c.drawString(x + 14, y_top - 10 - LINE_H,
                 f"reason: {ev.get('reason','?')}    time_diff={_fmt_time(ev.get('time_diff'))}")
    c.setFillColorRGB(0, 0, 0)
    return y_top - 2 * LINE_H


def _draw_succeeded(c: canvas.Canvas, x: float, y_top: float, w: float,
                    ev: dict, cols: int,
                    images_path: str, image_files: List[str],
                    caption_fields: tuple) -> float:
    merged_photos = ev.get('merged_photos') or []
    reminder_photos = ev.get('reminder_photos') or []

    # title
    c.setFont('Helvetica-Bold', 11)
    c.setFillColorRGB(0.05, 0.35, 0.05)
    c.drawString(x, y_top - 11,
                 f"MERGED [{ev.get('merge_type','?')}]  "
                 f"{_fmt_key(ev.get('src_key'))}  +  {_fmt_key(ev.get('partner_key'))}")
    c.setFont('Helvetica', 8)
    c.setFillColorRGB(0.25, 0.25, 0.25)
    c.drawString(x, y_top - 22,
                 f"{len(merged_photos)} photos merged"
                 + (f"  |  {len(reminder_photos)} reminder" if reminder_photos else "")
                 + f"    time_diff={_fmt_time(ev.get('time_diff'))}")
    c.setFillColorRGB(0, 0, 0)

    cursor = y_top - 28.0
    panels = [('Merged group', merged_photos)]
    if reminder_photos:
        panels.append(('Reminder', reminder_photos))

    for label_text, photos in panels:
        n = len(photos)
        grid_h = grid_height_for(n, cols, caption_fields, DEFAULT_CELL_SIZE)
        # label
        c.setFont('Helvetica-Bold', 8)
        c.setFillColorRGB(0.1, 0.1, 0.1)
        c.drawString(x, cursor - LABEL_H + 2, f"{label_text}  (n={n})")
        c.setFillColorRGB(0, 0, 0)

        grid_top = cursor - LABEL_H
        grid_bottom = grid_top - grid_h
        grid_rect = (x, grid_bottom, w, grid_h)
        draw_photo_grid(c, grid_rect, photos, images_path, image_files,
                        caption_fields=caption_fields,
                        cell_size=DEFAULT_CELL_SIZE, label=None)
        cursor = grid_bottom - GRID_PAD

    return cursor


# ---------- main render ----------

def render(stages_dir: str, images_path: str, output_pdf_path: str) -> None:
    events, is_artificial_time = _load_events(stages_dir)
    caption_fields = caption_fields_for(CAPTION_FIELDS, is_artificial_time)
    image_files = list_image_files(images_path)
    if not image_files:
        print(f"[warn] no images found under {images_path}; cells will show photo ids only")

    page_w, page_h = PAGE_SIZE
    panel_w = page_w - 2 * PAGE_MARGIN
    cols = grid_cols_for_width(panel_w, DEFAULT_CELL_SIZE)
    top = page_h - PAGE_MARGIN
    bottom = PAGE_MARGIN

    c = canvas.Canvas(output_pdf_path, pagesize=PAGE_SIZE)

    if not events:
        c.setFont('Helvetica', 12)
        c.drawString(36, page_h - 50, "(no merge events recorded)")
        c.showPage()
        c.save()
        return

    cursor_y = top
    page_started = False

    for ev in events:
        needed = _event_height(ev, cols, caption_fields)
        if page_started and cursor_y - needed < bottom:
            c.showPage()
            cursor_y = top
            page_started = False

        t = ev.get('type')
        if t == 'search':
            cursor_y = _draw_search(c, PAGE_MARGIN, cursor_y, panel_w, ev, is_artificial_time)
        elif t == 'merge_skipped':
            cursor_y = _draw_skipped(c, PAGE_MARGIN, cursor_y, ev)
        elif t == 'merge_succeeded':
            cursor_y = _draw_succeeded(c, PAGE_MARGIN, cursor_y, panel_w, ev,
                                       cols, images_path, image_files, caption_fields)
        else:
            # Unknown event types render as a single annotation line.
            c.setFont('Helvetica-Oblique', 8)
            c.setFillColorRGB(0.5, 0.5, 0.5)
            c.drawString(PAGE_MARGIN, cursor_y - 10, f"(unknown event: {t})")
            c.setFillColorRGB(0, 0, 0)
            cursor_y -= LINE_H

        cursor_y -= EVENT_GAP
        page_started = True

    if page_started:
        c.showPage()
    c.save()
