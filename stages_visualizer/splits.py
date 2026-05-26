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
    format_general_time,
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

LOG_LINE_H = 9.0    # vertical pitch for the decision-log monospace block
LOG_FONT = 'Courier'
LOG_FONT_SIZE = 7.5


def _load_splits(stages_dir: str) -> List[dict]:
    path = os.path.join(stages_dir, SPLIT_FILE)
    if not os.path.isfile(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('splits', []) or []


def _decision_log_lines(split: dict) -> List[str]:
    """Build the monospaced decision-log lines, mirroring `get_split_points`'s prints.

    Returns the same trace the console output emits — `...getting split points`,
    per-interval start time + between-times + `point appended!`, ending with
    `Split points: [...]`. Times are formatted HH:MM:SS for readability.
    For non-time-based attempts (size-based, or rejected before evaluation),
    returns a single-line description of why no per-interval trace exists.
    """
    lines: List[str] = []
    notes = split.get('notes') or ''
    if notes:
        lines.append(f"notes: {notes}")

    method = split.get('split_method')
    log = split.get('split_points_log')

    if method == 'size_based':
        lines.append("(size-based path — no time-interval decision log)")
        return lines

    if not log:
        lines.append("(no decision log recorded)")
        return lines

    group_key = log.get('group_key')
    n = log.get('n_group_times')
    lines.append(f"...getting split points  (group_key={group_key!r}, n={n})")

    if n is not None and n < 2:
        lines.append("len(group_time_list) < 2")
        return lines

    if not log.get('group_key_matched'):
        allowed = ', '.join(log.get('allowed_keys', []) or [])
        lines.append(f"group key doesn't match  (allowed: {allowed})")
        return lines

    # Prefer wall-clock HH:MM:SS strings (matches image captions / album1.pdf);
    # fall back to the relative general_time format if the log was written
    # before the wall-clock fields were added.
    def _show(clock: Any, raw: Any) -> str:
        if clock:
            return str(clock)
        return format_general_time(raw)

    intervals = log.get('intervals') or []
    end_clock_last = ''
    end_raw_last = None
    for interval in intervals:
        start_str = _show(interval.get('start_clock'), interval.get('start'))
        end_clock_last = interval.get('end_clock', '')
        end_raw_last = interval.get('end')
        n_between = interval.get('count_between', 0)
        between_clocks = interval.get('between_clocks') or []
        between_raw = interval.get('between_times') or []
        between_str = ', '.join(
            _show(between_clocks[i] if i < len(between_clocks) else None, between_raw[i])
            for i in range(len(between_raw))
        )
        appended = interval.get('appended')
        suffix = "   ----> point appended!" if appended else ""
        lines.append(f"{start_str}")
        lines.append(f"  => {n_between} between: [{between_str}]{suffix}")

    if end_raw_last is not None or end_clock_last:
        lines.append(_show(end_clock_last, end_raw_last))

    sp = log.get('split_points')
    sp_clocks = log.get('split_points_clock') or []
    if sp:
        lines.append(f"...got {len(sp)} split points")
        sp_strs = [
            _show(sp_clocks[i] if i < len(sp_clocks) else None, sp[i])
            for i in range(len(sp))
        ]
        lines.append("Split points: [" + ', '.join(sp_strs) + "]")
    else:
        lines.append("...got 0 split points")
        lines.append("Split points: None")

    return lines


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


def _draw_decision_log(c: canvas.Canvas, x: float, y_top: float,
                       lines: List[str], page_top: float, page_bottom: float) -> float:
    """Render the monospaced decision log; page-break if it overflows.

    Returns the y-cursor just below the last drawn line.
    """
    if not lines:
        return y_top
    cursor = y_top
    for line in lines:
        if cursor - LOG_LINE_H < page_bottom:
            c.showPage()
            cursor = page_top
        c.setFont(LOG_FONT, LOG_FONT_SIZE)
        c.setFillColorRGB(0.15, 0.15, 0.15)
        # Truncate ultra-long single lines so they never run off-page.
        max_chars = 180
        text = line if len(line) <= max_chars else (line[:max_chars - 1] + '…')
        c.drawString(x, cursor - LOG_LINE_H + 1, text)
        c.setFillColorRGB(0, 0, 0)
        cursor -= LOG_LINE_H
    return cursor


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

    # Decision-log block, mirroring the console prints. Useful both for
    # successful splits ("appended at 02:11:00 because 3 photos were
    # between...") and for skipped ones ("group key doesn't match").
    log_lines = _decision_log_lines(split)
    if log_lines:
        cursor_y = _draw_decision_log(c, PAGE_MARGIN, cursor_y, log_lines, top, bottom)
        cursor_y -= 6.0  # gap before the photo grids

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
