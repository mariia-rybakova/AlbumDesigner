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

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.core.models import (AlbumDesignResources, GroupProcessingResult,
                             Spread, SpreadSearchParams)
from src.core.photos import get_photos_from_df
from src.core.key_pages import _find_single_box_layout
from src.spreads_layout.combinations import Combination
from src.spreads_layout.group_layouts import get_group_single_layouts
from src.spreads_layout.main import select_best_layout_for_subgroup
from src.predefined.models import PredefinedLayoutInput, PredefinedSpread


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

    Keeps the design-driven cover-layout pick (``_find_single_box_layout`` by
    orientation) but takes the photos from the contract instead of the selection
    heuristic. ``position`` is 'first' or 'last' -- it only names the dict keys
    that ``assembly_output`` expects.
    """
    rows = gallery_df[gallery_df["image_id"].isin(photo_ids)]
    if rows.empty:
        logger.warning(f"Predefined {position} cover has no matching photos: {photo_ids}")
        return None

    orientation = rows["image_orientation"].values[0]
    candidates = _find_single_box_layout(layouts_df, orientation)
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