"""Choosing each page's treatment once the layout is set.

A photo uploaded in colour and in black and white is one shot in two
treatments (`enrich.treatment_twins`). The layout scores such a photo as
colour-neutral -- it can be either -- so it never makes a page mixed. This is
where that promise is kept: once every spread is fixed, each page's twins are
set to the colour its other photos share, by swapping the placed version for
its twin where the two differ.

One colour per page, for all of its twins together:

* the colour of the page's photos that have no twin, when they agree -- or
  their majority when they do not, since a mixed page cannot be mended and
  should at least not get worse;
* the majority of the twins' own colours when every photo on the page is a
  twin;
* colour on a tie, which is how the photographer delivered the shot first.

The swap changes the photo id the reply places, nothing else. The twin is the
same shot at the same aspect ratio, so the crop computed for the placed version
frames it identically, and the placement keeps it.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from typing import Iterable, List, Optional, Tuple

from src.core.photos import Photo


def _target(page: List[Photo]) -> Optional[bool]:
    fixed = [p.color for p in page if p is not None and p.twin_id is None]
    pool = fixed or [p.color for p in page if p is not None]
    if not pool:
        return None
    counts = Counter(bool(c) for c in pool)
    return counts[True] >= counts[False]


def settle_page(page: List[Photo]) -> Tuple[List[Photo], int]:
    """The page with every twin in the page's colour, and how many were swapped."""
    if not any(p is not None and p.twin_id is not None for p in page):
        return page, 0
    target = _target(page)
    settled, swapped = [], 0
    for photo in page:
        if (photo is not None and photo.twin_id is not None and target is not None
                and bool(photo.color) != target and photo.twin_color is not None
                and bool(photo.twin_color) == target):
            photo = dataclasses.replace(photo, color=target, output_id=photo.twin_id)
            swapped += 1
        settled.append(photo)
    return settled, swapped


def apply_twin_colors(result_list: Iterable, logger=None) -> int:
    """Settle every page of a laid-out album. Returns the number of swaps."""
    swapped = 0
    for group_dict in result_list or ():
        for result in (group_dict or {}).values():
            for spread in getattr(result, 'spreads', None) or ():
                for side in ('left_photos', 'right_photos'):
                    page, n = settle_page(list(getattr(spread, side) or []))
                    if n:
                        setattr(spread, side, page)
                        swapped += n
    if logger and swapped:
        logger.info(f"Treatment twins: {swapped} photo(s) placed in their other treatment "
                    f"to match their page")
    return swapped
