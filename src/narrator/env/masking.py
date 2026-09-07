"""Pure action-masking. Hard "don't-mix" constraints and album-shape bounds are
enforced here by zeroing invalid actions, so the policy can never emit an illegal
album. A safety net guarantees at least one legal action (END) to avoid deadlock.
"""

from __future__ import annotations

import numpy as np

from ..config import EnvConfig
from .state import AlbumState, PHASE_ASSIGN, STATUS_CLOSED, STATUS_OPEN

# special action offsets within the (N+2)-wide action vector
CLOSE = 0   # index N + CLOSE
END = 1     # index N + END


def action_dim(n: int) -> int:
    return n + 2


def page_size_bounds(open_style: int, cfg: EnvConfig) -> tuple[int, int]:
    """(min, max) photos allowed on a page, by its locked style.

    style_class: 1 = formal (2-3), 0 = candid (4-5); an unset page (-1) uses the candid
    bounds by default. Clamped to the overall [min_photos_per_page, max_photos_per_page]
    so tighter configs stay self-consistent.
    """
    if open_style == 1:
        mn, mx = cfg.formal_page_min, cfg.formal_page_max
    else:
        mn, mx = cfg.candid_page_min, cfg.candid_page_max
    lo, hi = cfg.min_photos_per_page, cfg.max_photos_per_page
    mn = min(max(mn, lo), hi)
    mx = min(max(mx, lo), hi)
    if mn > mx:
        mn = mx
    return mn, mx


def compute_mask(state: AlbumState, cfg: EnvConfig, *, page_size_enforced: bool = True,
                 is_dup: np.ndarray | None = None) -> np.ndarray:
    n = state.pg.n
    mask = np.zeros(n + 2, dtype=bool)
    # Hero pages occupy two slots of the final album, so the body budget shrinks by two and
    # the finished album still lands inside [min_pages, max_pages].
    n_hero = 2 if state.hero_first >= 0 else 0

    if state.phase == PHASE_ASSIGN:
        open_page = state.open_page
        nopen = len(open_page)
        max_pages = max(cfg.max_pages - n_hero, 1)

        # ASSIGN candidates
        cand = state.remaining.copy()
        if nopen == 0:
            # empty open page -> a new page; block if that would exceed max_pages.
            # (its first photo locks the page's style, and thus its size bounds.)
            if state.n_closed >= max_pages:
                cand[:] = False
        else:
            _, pmax = (page_size_bounds(state.open_style, cfg) if page_size_enforced
                       else (cfg.min_photos_per_page, cfg.max_photos_per_page))
            if nopen >= pmax:               # page full for its style
                cand[:] = False
            else:                           # must match the page's B&W + style signature
                cand &= (state.pg.bw == state.open_bw)
                cand &= (state.pg.style_class == state.open_style)

        # Near-duplicate exclusion as a HARD constraint (cfg.reward.dedup_enforced): a photo is
        # unassignable while a near-duplicate of it is already in the album, anywhere -- the
        # `dedup` term is album-wide, so the mask must be too. This makes the term 1.0 by
        # construction rather than something the policy must trade against `visual`, which reads
        # the same cosine matrix and outweighs it 12:1 at a 1/6 global scale (see
        # RewardConfig.dedup_enforced). Hero photos count: they are STATUS_CLOSED from reset.
        if is_dup is not None:
            included = (state.status == STATUS_CLOSED) | (state.status == STATUS_OPEN)
            if included.any():
                cand &= ~is_dup[:, included].any(axis=1)
        mask[:n] = cand

        # CLOSE_PAGE: only once the open page meets its style's minimum (never undersized).
        pmin = ((page_size_bounds(state.open_style, cfg)[0] if page_size_enforced
                 else cfg.min_photos_per_page) if nopen > 0 else 0)
        can_close = (nopen >= pmin) and (nopen > 0) and (state.n_closed < max_pages)
        mask[n + CLOSE] = can_close

        # END_ALBUM: allowed once >= min_pages would be finalized (open page counts only
        # if it already meets its style minimum; otherwise finalize drops it).
        open_counts = 1 if (nopen > 0 and nopen >= pmin) else 0
        finalized = state.n_closed + open_counts
        mask[n + END] = finalized >= max(cfg.min_pages - n_hero, 1)

        # safety net: never deadlock (e.g. a page stuck below its min with no matches left).
        if not mask.any():
            mask[n + END] = True

    else:  # PHASE_ORDER: place an unplaced page
        n_final = len(state.final_pages)
        if n_final > 0:
            mask[:n_final] = ~state.placed
        if not mask.any():  # all placed (env should have ended); keep well-formed
            mask[n + END] = True

    return mask
