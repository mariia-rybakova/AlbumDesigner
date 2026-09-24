"""Render combinations.pdf from files/stages_info/combinations/.

Explains, per sub-group, **why it was cut into these spreads with these
photos** — the decision stages 1-2 of the layout search make before any design
is chosen (`spreads_layouts.pdf` picks up from there).

Each sub-group gets:

  1. A decision page. Header states the outcome, then the *partition funnel*:
     every integer partition of n_photos that was scored, its Gaussian fit
     against the class's photos-per-spread parameter, and the stage that
     dropped it (layout-infeasible / length filter / early stop). This is the
     "why 2 spreads of 7 and not 3 of 5" table.
  2. Assignment pages. Per surviving partition, the top scoring photo-to-spread
     assignments, drawn as one photo strip per spread with the two divisors
     that produced the score (time spread, duplicate cluster labels).

The combination the album actually shipped is flagged `SELECTED`. It is often
*not* the stage-2 leader: stage 3 re-ranks candidates by layout fit, so a
lower-weight assignment can win. That gap is the point of putting the ranks
on the page.

Read-only: no imports from the layout/grouping pipeline.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from reportlab.pdfgen import canvas

from stages_visualizer._shared import (
    PAGE_SIZE,
    caption_fields_for,
    draw_photo_grid,
    grid_cols_for_width,
    grid_height_for,
    list_image_files,
)


# Photo captions inside the spread strips: when a photo landed here and which
# cluster label it carries — the two inputs to the combination score.
#
# `image_time_date` is the wall clock album1.pdf prints, so a photo can be found
# in both documents by the same label. `caption_fields_for` swaps it back to the
# relative `general_time` on an artificial-time gallery, where the stored
# absolute time is stale and album1.pdf shows elapsed time too.
CAPTION_FIELDS = ('image_time_date', 'cluster_label')

# Smaller than the grouping PDFs' DEFAULT_CELL_SIZE: an assignment page shows
# the same photos several times over (once per candidate), so the strips have
# to stay compact enough that a candidate fits on one page.
CELL_SIZE = 46.0

PAGE_MARGIN = 22.0
LINE_H = 10.0
ROW_H = 11.0
PAD = 6.0

# Volume caps. Everything recorded is in the JSON; the PDF shows the head of
# each list and says out loud how much it dropped.
MAX_PARTITION_ROWS = 26
MAX_PARTITIONS_WITH_COMBS = 6
MAX_COMBS_PER_PARTITION = 3

_STATUS_STYLE = {
    'kept':              ((0.0, 0.35, 0.0), 'kept — combinations built'),
    'length_filtered':   ((0.75, 0.45, 0.0), 'dropped — spread-count filter'),
    'layout_infeasible': ((0.65, 0.0, 0.0), 'dropped — no layout fits its portraits/landscapes'),
    'early_stopped':     ((0.6, 0.6, 0.6), 'never examined — weight below threshold'),
}


# ---------- loading ----------

def _load_records(stages_dir: str) -> List[Tuple[str, dict]]:
    """List `(filename_stem, parsed_json)` for every per-sub-group file."""
    out = []
    for fname in sorted(os.listdir(stages_dir)):
        if not fname.endswith('.json') or fname.startswith('_'):
            continue
        with open(os.path.join(stages_dir, fname), 'r', encoding='utf-8') as f:
            out.append((fname[:-5], json.load(f)))
    return out


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return '-'
    try:
        return f"{float(v):.{digits}g}"
    except (TypeError, ValueError):
        return str(v)


def _sizes_label(sizes: Sequence[int]) -> str:
    return '[' + ', '.join(str(int(s)) for s in sizes) + ']'


# ---------- page flow ----------

class _Flow:
    """Top-down cursor that starts a new page when the next block won't fit."""

    def __init__(self, c: canvas.Canvas):
        self.c = c
        page_w, page_h = PAGE_SIZE
        self.x = PAGE_MARGIN
        self.width = page_w - 2 * PAGE_MARGIN
        self.top = page_h - PAGE_MARGIN
        self.bottom = PAGE_MARGIN
        self.y = self.top

    def new_page(self) -> None:
        self.c.showPage()
        self.y = self.top

    def reserve(self, height: float) -> float:
        """Make room for `height`; returns the y of the block's top edge."""
        if self.y - height < self.bottom and self.y < self.top:
            self.new_page()
        top = self.y
        self.y -= height
        return top

    def text(self, line: str, font: str = 'Helvetica', size: float = 8.0,
             color: Tuple[float, float, float] = (0, 0, 0), indent: float = 0.0) -> None:
        top = self.reserve(LINE_H)
        self.c.setFont(font, size)
        self.c.setFillColorRGB(*color)
        self.c.drawString(self.x + indent, top - LINE_H + 2.5, line)
        self.c.setFillColorRGB(0, 0, 0)

    def gap(self, height: float = PAD) -> None:
        self.y -= height


# ---------- section 1: how many spreads ----------

def _draw_summary(flow: _Flow, group_name: str, record: dict) -> None:
    """Headline: what this sub-group is, and what the search decided."""
    partitions = record.get('partitions') or {}
    winner = record.get('winner') or {}
    spread_params = record.get('spread_params') or {}

    flow.text(f"Sub-group: {record.get('group_id', group_name)}", 'Helvetica-Bold', 13)
    flow.gap(2)

    n_photos = partitions.get('n_photos', record.get('n_photos', 0))
    flow.text(f"{n_photos} photos "
              f"({partitions.get('n_portraits', '?')} portrait / {partitions.get('n_landscapes', '?')} landscape)   "
              f"target photos-per-spread: mean={_fmt(spread_params.get('mean'), 3)}, "
              f"std={_fmt(spread_params.get('std'), 3)}",
              'Helvetica', 9, (0.25, 0.25, 0.25))

    if partitions.get('std_widened'):
        # evaluate_list widens std when every partition scores exactly 0, i.e.
        # the group is nowhere near its class's expected spread size.
        after = partitions.get('spread_params_after_evaluation') or []
        flow.text(f"note: every partition scored 0 against the target — std widened to "
                  f"{_fmt(after[1] if len(after) > 1 else None, 3)} and all partitions rescored",
                  'Helvetica-Oblique', 8, (0.75, 0.45, 0.0))

    if winner.get('spread_sizes'):
        flow.text(f"DECISION: {len(winner['spread_sizes'])} spread(s), sizes "
                  f"{_sizes_label(winner['spread_sizes'])}",
                  'Helvetica-Bold', 10, (0.0, 0.35, 0.0))
        rank = winner.get('combination_rank')
        rank_text = (f"stage-2 rank {rank + 1} within its partition" if rank is not None
                     else "not in the stage-2 top cut — stage 3 promoted it")
        flow.text(f"chosen assignment: {rank_text}.   "
                  f"final layout score={_fmt(winner.get('final_layout_score'))} "
                  f"weight={_fmt(winner.get('final_layout_weight'))}",
                  'Helvetica', 8, (0.25, 0.25, 0.25))
    else:
        flow.text("DECISION: (no winning layout recorded)", 'Helvetica-Bold', 10, (0.5, 0.5, 0.5))


def _draw_funnel(flow: _Flow, partitions: dict) -> None:
    """One line summarising how many candidates each filter removed."""
    flow.gap(4)
    flow.text("Stage 1 — how many spreads, and how big", 'Helvetica-Bold', 10)
    # Plain ASCII for the exponent: Helvetica's standard encoding in reportlab
    # has no superscript-2 glyph and would draw a placeholder box.
    flow.text("partition score = product over spreads of exp(-0.5 * ((size - mean) / std)^2)"
              "   ·   weight = score / best score",
              'Helvetica-Oblique', 8, (0.3, 0.3, 0.3))
    boxes = partitions.get('available_box_counts') or []
    flow.text(f"{partitions.get('n_integer_partitions', '?')} integer partitions of "
              f"{partitions.get('n_photos', '?')}"
              f"  →  {partitions.get('n_with_available_layout_sizes', '?')} using only available layout sizes"
              f"  →  {partitions.get('n_examined_by_layout_filter', '?')} examined (rest cut by early stop)"
              f"  →  {partitions.get('n_layout_feasible', '?')} layout-feasible"
              f"  →  {partitions.get('n_kept', '?')} kept",
              'Helvetica', 8, (0.2, 0.2, 0.2))
    if boxes:
        flow.text(f"available layout sizes (boxes per spread): {', '.join(str(b) for b in boxes)}",
                  'Helvetica', 7.5, (0.45, 0.45, 0.45))
    flow.text(f"early stop: after 2 feasible partitions, once weight < best/"
              f"{_fmt(partitions.get('weight_threshold_divisor'), 4)}",
              'Helvetica', 7.5, (0.45, 0.45, 0.45))


def _draw_partition_table(flow: _Flow, partitions: dict,
                          winner_sizes: Optional[List[int]]) -> None:
    """Weight-ordered table of every scored partition and its fate."""
    candidates = partitions.get('candidates') or []
    if not candidates:
        flow.text("(no partition candidates recorded)", 'Helvetica', 9, (0.5, 0.5, 0.5))
        return

    col_sizes = flow.x
    col_n = flow.x + 210
    col_score = flow.x + 250
    col_weight = flow.x + 310
    col_bar = flow.x + 370
    bar_w = 70.0
    col_status = flow.x + 455

    flow.gap(4)
    top = flow.reserve(ROW_H)
    flow.c.setFont('Helvetica-Bold', 8)
    flow.c.setFillColorRGB(0.1, 0.1, 0.1)
    base = top - ROW_H + 3
    flow.c.drawString(col_sizes, base, "spread sizes")
    flow.c.drawString(col_n, base, "n")
    flow.c.drawString(col_score, base, "score")
    flow.c.drawString(col_weight, base, "weight")
    flow.c.drawString(col_bar, base, "")
    flow.c.drawString(col_status, base, "outcome")
    flow.c.setFillColorRGB(0, 0, 0)

    winner_key = sorted(winner_sizes) if winner_sizes else None

    for cand in candidates[:MAX_PARTITION_ROWS]:
        status = cand.get('status', '?')
        color, status_text = _STATUS_STYLE.get(status, ((0.3, 0.3, 0.3), status))
        is_winner = winner_key is not None and sorted(cand.get('spread_sizes') or []) == winner_key

        top = flow.reserve(ROW_H)
        base = top - ROW_H + 3
        font = 'Helvetica-Bold' if is_winner else 'Helvetica'
        flow.c.setFont(font, 8)
        flow.c.setFillColorRGB(*color)
        prefix = '★ ' if is_winner else '   '
        flow.c.drawString(col_sizes, base, prefix + _sizes_label(cand.get('spread_sizes') or []))
        flow.c.drawString(col_n, base, str(cand.get('n_spreads', '')))
        flow.c.drawString(col_score, base, _fmt(cand.get('score')))
        flow.c.drawString(col_weight, base, _fmt(cand.get('weight')))
        flow.c.drawString(col_status, base,
                          status_text + ('   ← SELECTED' if is_winner else ''))

        weight = cand.get('weight')
        if weight is not None:
            try:
                frac = max(0.0, min(1.0, float(weight)))
            except (TypeError, ValueError):
                frac = 0.0
            flow.c.setStrokeColorRGB(0.75, 0.75, 0.75)
            flow.c.setLineWidth(0.3)
            flow.c.rect(col_bar, base - 1, bar_w, 5.5)
            if frac > 0:
                flow.c.setFillColorRGB(*color)
                flow.c.rect(col_bar, base - 1, bar_w * frac, 5.5, stroke=0, fill=1)
        flow.c.setFillColorRGB(0, 0, 0)

    if len(candidates) > MAX_PARTITION_ROWS:
        flow.text(f"... {len(candidates) - MAX_PARTITION_ROWS} further candidates omitted "
                  f"(all ranked below the rows above)",
                  'Helvetica-Oblique', 7.5, (0.5, 0.5, 0.5))


# ---------- section 2: which photos in which spread ----------

def _spread_note(breakdown_entry: Optional[dict]) -> str:
    """One-line explanation of a spread's contribution to the combination score."""
    if not breakdown_entry:
        return ''
    parts = [f"n={breakdown_entry.get('n_photos', '?')}"]
    time_div = breakdown_entry.get('time_divisor')
    parts.append(f"time std={_fmt(breakdown_entry.get('time_std_minutes'), 3)}min"
                 f" (span {_fmt(breakdown_entry.get('time_span_minutes'), 3)}min)"
                 f" ÷{_fmt(time_div, 3)}")
    if breakdown_entry.get('labels_scored'):
        parts.append(f"labels {breakdown_entry.get('n_unique_labels')}/{breakdown_entry.get('n_labels')} unique,"
                     f" {breakdown_entry.get('duplicate_labels')} duplicate ÷{_fmt(breakdown_entry.get('label_divisor'), 3)}")
    else:
        parts.append("labels: none set (not scored)")
    return '   ·   '.join(parts)


def _draw_combination(flow: _Flow, comb: dict, photos_by_idx: Dict[int, dict],
                      images_path: str, image_files: List[str],
                      cols: int, caption_fields: Tuple[str, ...]) -> None:
    """One candidate assignment: a header line plus a photo strip per spread."""
    spreads = comb.get('spreads') or []
    breakdown = (comb.get('score_breakdown') or {}).get('per_spread') or []
    paired = list(zip(spreads, list(breakdown) + [None] * max(0, len(spreads) - len(breakdown))))
    # Chronological display order — index order is time order within a sub-group.
    paired.sort(key=lambda pair: min(pair[0]) if pair[0] else 0)

    is_winner = bool(comb.get('is_winner'))
    rank = comb.get('rank')
    rank_label = f"#{rank + 1}" if rank is not None else "#-"
    header = (f"{rank_label}  combination score={_fmt(comb.get('score'))}   "
              f"weight={_fmt(comb.get('weight'))}  (= score × partition weight)")
    if comb.get('outside_top'):
        header += "   [below the recorded top cut — rescued as the selected one]"
    flow.gap(3)

    # Keep the header with at least its first strip, so a candidate never ends
    # up as an orphan line at the foot of a page.
    if paired:
        first_h = grid_height_for(len(paired[0][0]), cols, caption_fields, CELL_SIZE)
        if flow.y - (2 * LINE_H + first_h) < flow.bottom:
            flow.new_page()

    flow.text(header + ('        ★ SELECTED' if is_winner else ''),
              'Helvetica-Bold' if is_winner else 'Helvetica', 8.5,
              (0.0, 0.35, 0.0) if is_winner else (0.15, 0.15, 0.15))

    for s_idx, (spread_idxs, entry) in enumerate(paired):
        spread_photos = [photos_by_idx[i] for i in spread_idxs if i in photos_by_idx]
        grid_h = grid_height_for(len(spread_photos), cols, caption_fields, CELL_SIZE)

        # Keep the label and its strip on one page.
        if flow.y - (LINE_H + grid_h) < flow.bottom:
            flow.new_page()

        flow.text(f"spread {s_idx + 1}/{len(paired)}   {_spread_note(entry)}",
                  'Helvetica', 7.5, (0.3, 0.3, 0.3), indent=6)
        top = flow.reserve(grid_h)
        draw_photo_grid(flow.c, (flow.x + 6, top - grid_h, flow.width - 6, grid_h),
                        spread_photos, images_path, image_files,
                        caption_fields=caption_fields, cell_size=CELL_SIZE)


def _pick_combinations(partition: dict) -> List[dict]:
    """Head of the partition's candidates, with the selected one always present."""
    combs = partition.get('top_combinations') or []
    shown = list(combs[:MAX_COMBS_PER_PARTITION])
    for comb in combs:
        if comb.get('is_winner') and comb not in shown:
            shown.append(comb)
    return shown


def _draw_assignments(flow: _Flow, record: dict, photos_by_idx: Dict[int, dict],
                      images_path: str, image_files: List[str],
                      caption_fields: Tuple[str, ...]) -> None:
    partitions = record.get('combinations') or []
    cols = grid_cols_for_width(flow.width - 6, CELL_SIZE)

    flow.gap(6)
    flow.text("Stage 2 — which photos go into which spread", 'Helvetica-Bold', 10)
    flow.text("combination score = Π over spreads of 1/(time std in minutes) · 1/(1 + duplicate cluster labels)",
              'Helvetica-Oblique', 8, (0.3, 0.3, 0.3))
    flow.text("stage 3 then re-ranks every candidate by layout fit, so the top-weighted assignment "
              "here is not always the selected one",
              'Helvetica-Oblique', 7.5, (0.45, 0.45, 0.45))

    if not partitions:
        flow.text("(no combinations recorded — no partition survived stage 1)",
                  'Helvetica', 9, (0.5, 0.5, 0.5))
        return

    for partition in partitions[:MAX_PARTITIONS_WITH_COMBS]:
        flow.gap(5)
        flow.text(f"partition {_sizes_label(partition.get('spread_sizes') or [])}   "
                  f"partition weight={_fmt(partition.get('partition_weight'))}   ·   "
                  f"{partition.get('search')} search   ·   "
                  f"{partition.get('n_generated')} generated → {partition.get('n_sampled')} kept "
                  f"(budget {partition.get('max_combs')})",
                  'Helvetica-Bold', 9, (0.1, 0.1, 0.4))

        shown = _pick_combinations(partition)
        if not shown:
            flow.text("(no candidate assignments recorded)", 'Helvetica', 8, (0.5, 0.5, 0.5))
            continue
        for comb in shown:
            _draw_combination(flow, comb, photos_by_idx, images_path, image_files, cols,
                              caption_fields)

        n_recorded = len(partition.get('top_combinations') or [])
        if n_recorded > len(shown):
            flow.text(f"... {n_recorded - len(shown)} further recorded candidates not drawn",
                      'Helvetica-Oblique', 7.5, (0.5, 0.5, 0.5))

    if len(partitions) > MAX_PARTITIONS_WITH_COMBS:
        flow.gap(4)
        flow.text(f"... {len(partitions) - MAX_PARTITIONS_WITH_COMBS} further partitions had candidates too "
                  f"(all with lower partition weight)",
                  'Helvetica-Oblique', 8, (0.5, 0.5, 0.5))


# ---------- entry point ----------

def _draw_record(c: canvas.Canvas, group_name: str, record: dict,
                 images_path: str, image_files: List[str]) -> None:
    flow = _Flow(c)
    partitions = record.get('partitions') or {}
    winner = record.get('winner') or {}
    photos_by_idx = {p['idx']: p for p in (record.get('photos') or []) if 'idx' in p}

    _draw_summary(flow, group_name, record)
    _draw_funnel(flow, partitions)
    _draw_partition_table(flow, partitions, winner.get('spread_sizes'))
    _draw_assignments(flow, record, photos_by_idx, images_path, image_files,
                      caption_fields_for(CAPTION_FIELDS,
                                         bool(record.get('is_artificial_time', False))))
    c.showPage()


def render(stages_dir: str, images_path: str, output_pdf_path: str) -> None:
    """Read every per-sub-group file under `stages_dir`, render its section."""
    records = _load_records(stages_dir)
    image_files = list_image_files(images_path)
    if not image_files:
        print(f"[warn] no images found under {images_path}; cells will show photo ids only")

    c = canvas.Canvas(output_pdf_path, pagesize=PAGE_SIZE)
    if not records:
        c.setFont('Helvetica', 12)
        c.drawString(36, PAGE_SIZE[1] - 50, "(no sub-group combination records found)")
        c.showPage()
    else:
        for group_name, record in records:
            _draw_record(c, group_name, record, images_path, image_files)
    c.save()
