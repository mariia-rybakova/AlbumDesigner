"""Layout bypass for predefined-spreads mode.

Each predefined spread already fixes stages 1 (partitions) and 2 (combinations)
of the normal pipeline -- a ``Combination`` is exactly ``List[Set[int]]`` of
photo indices per spread. So we build a single-spread ``Combination`` per spread
and feed it straight into stage 3 (``get_group_single_layouts``), reusing the
layout sampling, penalty scoring and box assignment unchanged.

One spread == one single-spread ``Combination`` (never all spreads in one):
stage 3's ``process_group_lists`` takes a cartesian product across a
combination's spreads, so a single combination holding N predefined spreads
would explode. Per-spread is both correct (spreads are independent) and cheap.
"""
from __future__ import annotations

import ast
import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.core.models import (AlbumDesignResources, GroupProcessingResult,
                             Spread, SpreadSearchParams)
from src.core.photos import Photo, get_photos_from_df
from src.covers.pages import find_single_box_layout
from src.spreads_layout.combinations import Combination
from src.spreads_layout.group_layouts import get_group_single_layouts
from src.spreads_layout.main import select_best_layout_for_subgroup
from src.predefined.models import PredefinedLayoutInput, PredefinedSpread


def _printed_box_ratio(layout_row: pd.Series, album_ar: float) -> Optional[float]:
    """The width:height a single-box layout's box is printed at, or None.

    A box's own width and height are fractions of the spread, so the ratio the
    crop is fitted to is the box's scaled by the spread's -- the same arithmetic
    as ``request_processing.box_target_ar``.
    """
    boxes = layout_row['boxes_info']
    if isinstance(boxes, str):
        boxes = ast.literal_eval(boxes)
    if not boxes:
        return None
    box = boxes[0]
    if not box.get('height'):
        return None
    return box['width'] * album_ar / box['height']


def _single_box_fallback(photo: Photo, layouts_df: pd.DataFrame, album_ar: float) -> Optional[Spread]:
    """Place a lone photo in a one-box layout whatever that box's orientation.

    Stage 3 only ever puts a photo in a box of its own orientation, and a
    design's one-box layouts are its full-spread frames, which classify as
    landscape on a 2:1 album -- in `files/layouts.csv` all four of them carry
    ``max portraits == 0``. So a predefined spread holding a single portrait
    photo matched no layout at all and the spread was dropped: 4,613 of them in
    two days of production, every one but one a single photo. The normal
    pipeline never hits this because `find_spreads_layouts_for_group` retries
    the whole group with a dummy photo appended; this path calls stage 3 once.

    The cover path already crosses the same line -- `find_single_box_layout`
    ends at "any single-box layout" and lets the crop absorb the mismatch -- and
    a predefined spread names its photo explicitly, so a cropped frame is the
    album that was asked for and a missing spread is not.

    Of the candidates, takes the box printed closest to the photo's own ratio,
    so a portrait does not land in the full-spread frame across the fold while a
    narrower box sits unused.
    """
    orientation = 'landscape' if photo.ar >= 1 else 'portrait'
    candidates = find_single_box_layout(layouts_df, orientation)
    if not candidates:
        return None

    def mismatch(layout_idx: int) -> float:
        ratio = _printed_box_ratio(layouts_df.loc[layout_idx], album_ar)
        if not ratio or ratio <= 0 or photo.ar <= 0:
            return math.inf
        return abs(math.log(ratio / photo.ar))

    layout_idx = min(candidates, key=mismatch)
    # Pairing is positional over `left_box_ids + right_box_ids`
    # (`assembly_output`), so the photo belongs to the page holding the box.
    on_left = len(layouts_df.loc[layout_idx]['left_box_ids']) > 0
    return Spread(layout_id=layout_idx,
                  left_photos=[photo] if on_left else [],
                  right_photos=[] if on_left else [photo])


def _layout_one_predefined_spread(spread: PredefinedSpread, gallery_df: pd.DataFrame,
                                  resources: AlbumDesignResources, params: SpreadSearchParams,
                                  is_wedding: bool, logger) -> Optional[Spread]:
    """Choose the best layout + box assignment for one fixed spread.

    Reuses stage 3 verbatim: wraps the spread's photos as a single-spread
    ``Combination`` and runs the normal candidate search / scoring, then takes
    the top candidate.

    Returns None if the spread has no photos in the gallery or no layout fits.
    """
    rows = gallery_df[gallery_df["image_id"].isin(spread.photo_ids)]
    if rows.empty:
        logger.warning(f"Predefined spread has no matching photos in gallery: {spread.photo_ids}")
        return None

    photos = get_photos_from_df(rows, is_wedding)

    # Stages 1+2 are given: one combination, one spread, all photos in it.
    comb = Combination(spreads=[set(range(len(photos)))])
    comb.weight = 1.0

    layouts_df = resources.printlab_data.layouts_df
    layout_id2data = resources.printlab_data.layout_id2data
    box_id2data = resources.printlab_data.box_id2data

    candidates = get_group_single_layouts([comb], photos, layouts_df, params, layout_id2data)
    best = select_best_layout_for_subgroup(candidates, photos, layout_id2data, box_id2data)
    if best is None:
        if len(photos) == 1:
            fallback = _single_box_fallback(photos[0], layouts_df, resources.printlab_data.album_ar)
            if fallback is not None:
                logger.info(f"Predefined spread {spread.photo_ids}: no layout holds a single "
                            f"{'landscape' if photos[0].ar >= 1 else 'portrait'} photo; "
                            f"placed it in single-box layout {fallback.layout_id}")
                return fallback
        logger.warning(f"No layout found for predefined spread: {spread.photo_ids}")
        return None

    single = best.spreads_layouts[0]
    return Spread(layout_id=single.layout_idx,
                  left_photos=single.left_page_photos,
                  right_photos=single.right_page_photos)


def predefined_layout_processing(gallery_df: pd.DataFrame, designs_info: Dict[str, Any],
                                 predefined: PredefinedLayoutInput, params: SpreadSearchParams,
                                 is_wedding: bool, logger) -> Tuple[List[Dict[str, GroupProcessingResult]], pd.DataFrame]:
    """Lay out every predefined spread, bypassing selection / partitions / combinations.

    Returns ``(result_list, gallery_df)`` in the same shape ``album_processing``
    returns, so ``assembly_output`` consumes it unchanged. Each spread becomes
    its own single-spread ``GroupProcessingResult`` so the downstream
    time-reorder can order them; input order is otherwise preserved here.
    """
    resources = AlbumDesignResources.from_dict(designs_info)  # LookUpTable defaulted, unused

    result_list: List[Dict[str, GroupProcessingResult]] = []
    for idx, spread in enumerate(predefined.spreads):
        laid_out = _layout_one_predefined_spread(spread, gallery_df, resources, params, is_wedding, logger)
        if laid_out is None:
            continue
        group_id = f"predefined_{idx}"
        result_list.append({group_id: GroupProcessingResult(group_name=group_id,
                                                            spreads=[laid_out], score=1.0)})

    logger.info(f"Predefined layout: produced {len(result_list)}/{len(predefined.spreads)} spreads.")
    return result_list, gallery_df


def _build_cover(photo_ids: List[str], gallery_df: pd.DataFrame, layouts_df: pd.DataFrame,
                 position: str, logger) -> Optional[Dict[str, Any]]:
    """Build one cover entry (firstPage/lastPage) for ``first_last_pages_data_dict``.

    Keeps the design-driven cover-layout pick (``find_single_box_layout`` by
    orientation) but takes the photos from the contract instead of the selection
    heuristic. ``position`` is 'first' or 'last' -- it only names the dict keys
    that ``assembly_output`` expects.
    """
    rows = gallery_df[gallery_df["image_id"].isin(photo_ids)]
    if rows.empty:
        logger.warning(f"Predefined {position} cover has no matching photos: {photo_ids}")
        return None

    orientation = rows["image_orientation"].values[0]
    candidates = find_single_box_layout(layouts_df, orientation)
    if not candidates:
        logger.warning(f"No single-box {position}Page layout for orientation {orientation}")
        return None

    key = "first_images" if position == "first" else "last_images"
    return {"design_id": candidates[0], f"{key}_ids": photo_ids, f"{key}_df": rows}


def build_first_last_pages(predefined: PredefinedLayoutInput, gallery_df: pd.DataFrame,
                           message, logger) -> Dict[str, Any]:
    """Build ``first_last_pages_data_dict`` from the input's cover photos.

    Replaces ``generate_first_last_pages`` in this mode: no selection heuristic,
    and crucially no removal of photos from the gallery (body spreads are
    explicit, so a cover photo simply never appears in one). Covers are only
    built when the design has the corresponding slot (``pagesInfo``).
    """
    data: Dict[str, Any] = {}

    if message.pagesInfo.get("firstPage") and predefined.first_page_photo_ids:
        cover = _build_cover(predefined.first_page_photo_ids, gallery_df,
                             message.designsInfo["firstPage_layouts_df"], "first", logger)
        if cover is not None:
            data["firstPage"] = cover

    if message.pagesInfo.get("lastPage") and predefined.last_page_photo_ids:
        cover = _build_cover(predefined.last_page_photo_ids, gallery_df,
                             message.designsInfo["lastPage_layouts_df"], "last", logger)
        if cover is not None:
            data["lastPage"] = cover

    return data