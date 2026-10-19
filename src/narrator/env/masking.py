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


N_SIG = 4   # page signatures: (bw, style_class) in {0,1}^2 -- a page holds one signature only


def signature(pg) -> np.ndarray:
    """Per-photo page signature 2*bw + style_class. Photos may share a page iff they share it."""
    return 2 * pg.bw.astype(np.int64) + pg.style_class.astype(np.int64)


def close_min(sig: int, cfg: EnvConfig, page_size_enforced: bool) -> int:
    """Fewest photos a page of signature ``sig`` needs before CLOSE_PAGE is legal."""
    if page_size_enforced:
        return page_size_bounds(sig % 2, cfg)[0]
    return cfg.min_photos_per_page


# Album Designer prunes up to `state.prune_cap` body photos that resemble a cover -- the most similar
# ones IN THE FINISHED BODY (cfg.reward.hero_risk_sim). Every page must keep this many photos after that
# pruning, so the album can never ship a 1-photo spread.
PRUNE_SAFE_MIN = 2


def prune_set(state: AlbumState) -> np.ndarray:
    """(N,) bool: the photos AD would prune if the album ended now.

    The ``prune_cap`` included photos with the highest ``risk_rank`` (similarity to the nearest cover,
    ties by photoId, exactly as AD's `drop_cover_duplicates` sorts). Heroes are never ranked.

    MONOTONE in the one way the mask relies on: adding a photo can only push it into the set and push
    the least similar member out, so a page that is not ASSIGNed to never gains a member. A page that
    was safe when it closed stays safe. Removing photos is what breaks this -- see ``page_min``.
    """
    out = np.zeros(state.pg.n, dtype=bool)
    if state.risk_rank is None or state.prune_cap <= 0:
        return out
    inc = ((state.status == STATUS_OPEN) | (state.status == STATUS_CLOSED)) & (state.risk_rank >= 0)
    idx = np.nonzero(inc)[0]
    if idx.size:
        out[idx[np.argsort(state.risk_rank[idx])[::-1][:state.prune_cap]]] = True
    return out


def _risk_ctx(state: AlbumState):
    """(prune set, its members on the open page, mask of remaining photos that would ENTER it).

    A remaining photo enters the set iff the set has room or it outranks the set's least similar
    member. That bar only rises as photos are added, so a photo that would not enter now never will.
    """
    pruned = prune_set(state)
    on_open = int(pruned[state.open_page].sum()) if state.open_page else 0
    rr = state.risk_rank
    if rr is None or state.prune_cap <= 0:
        return pruned, 0, np.zeros(state.pg.n, dtype=bool)
    enter = state.remaining & (rr >= 0)
    if pruned.sum() >= state.prune_cap:
        enter &= rr > rr[pruned].min()
    return pruned, on_open, enter


def page_min(state: AlbumState, base: int) -> int:
    """``base`` raised so the open page keeps PRUNE_SAFE_MIN photos after AD's cover pruning.

    The single rule CLOSE, END and ``finalize`` all apply, so none of them can disagree about whether
    the open page survives.
    """
    if state.risk_rank is None or not state.open_page:
        return base
    return max(base, PRUNE_SAFE_MIN + _risk_ctx(state)[1])


def reachable_pages(n_closed: int, open_size: int, open_sig: int, rem: np.ndarray,
                    mins: np.ndarray, max_pages: int, extra: int = 0,
                    extra_sigs: np.ndarray | None = None, open_need: int | None = None) -> int:
    """Most body pages the album can still end with, given remaining photos per signature.

    An unfinished open page must be completed from its own signature before any other photo is
    assignable, so if it cannot be completed nothing more is reachable. ``open_need`` is how many
    photos completing it takes (-1 = it cannot be completed; None = up to its signature's minimum).
    ``extra`` photos are then charged ADVERSARIALLY -- each taken where it costs a page -- for
    prune-set members that future pages may receive: each needs one photo more than its page would
    otherwise, and is charged to a signature that still has photos able to join the set
    (``extra_sigs``; None = any). Conservative by design: the floor this feeds must be reachable,
    not merely plausible. A page counts once it can close; capped at ``max_pages``.
    """
    rem = rem.copy()
    if open_size == 0:
        total = n_closed
    else:
        need = max(int(mins[open_sig]) - open_size, 0) if open_need is None else open_need
        if need < 0 or rem[open_sig] < need:
            return min(n_closed, max_pages)
        rem[open_sig] -= need
        total = n_closed + 1
    left = None if extra_sigs is None else np.asarray(extra_sigs).copy()
    for _ in range(extra):
        ok = rem > 0 if left is None else (rem > 0) & (left > 0)
        live = np.nonzero(ok)[0]
        if not live.size:
            break
        hit = [g for g in live if rem[g] % mins[g] == 0]
        g = hit[0] if hit else live[0]
        rem[g] -= 1
        if left is not None:
            left[g] -= 1
    total += int(np.floor_divide(rem, mins).sum())
    return min(total, max_pages)


def _sig_state(state: AlbumState, cfg: EnvConfig, page_size_enforced: bool):
    sig = signature(state.pg)
    rem = np.bincount(sig[state.remaining], minlength=N_SIG)
    mins = np.array([close_min(g, cfg, page_size_enforced) for g in range(N_SIG)], dtype=np.int64)
    open_sig = (2 * state.open_bw + state.open_style) if state.open_page else -1
    return sig, rem, mins, open_sig


def _assign_max(sig: int, cfg: EnvConfig, page_size_enforced: bool) -> int:
    return (page_size_bounds(sig % 2, cfg)[1] if page_size_enforced
            else cfg.max_photos_per_page)


def _completion_need(state: AlbumState, cfg: EnvConfig, page_size_enforced: bool, page: list[int],
                     page_sig: int, sig: np.ndarray) -> tuple[int, int]:
    """(photos it takes to make ``page`` closable with PRUNE_SAFE_MIN photos outside the prune set,
    prune-set members on it once it is), or (-1, 0) if no completion exists within its maximum size.

    Simulated, not counted: the page is filled with its signature's remaining photos LEAST similar
    to a cover first (non-risky ones before any risky one), and the prune set is recomputed after
    every addition, so a photo that would join the set raises the need instead of meeting it -- and
    one that would not, because the set is already full of more similar photos, counts as safe.
    Least-similar-first is the order least likely to feed the set. ``page`` may include a photo not
    yet assigned (the candidate being tested); it is treated as included.
    """
    rr, cap = state.risk_rank, state.prune_cap
    inc = ((state.status == STATUS_OPEN) | (state.status == STATUS_CLOSED)) & (rr >= 0)
    members = list(page)
    inc[members] = rr[members] >= 0
    pool = np.nonzero(state.remaining & (sig == page_sig))[0]
    pool = pool[~np.isin(pool, members)]
    pool = pool[np.argsort(rr[pool], kind="stable")]
    base = close_min(page_sig, cfg, page_size_enforced)
    pmax = _assign_max(page_sig, cfg, page_size_enforced)
    used = 0
    while True:
        idx = np.nonzero(inc)[0]
        top = idx[np.argsort(rr[idx])[::-1][:cap]] if cap > 0 else idx[:0]
        on = int(np.isin(members, top).sum())
        if len(members) >= max(base, PRUNE_SAFE_MIN + on):
            return used, on
        if len(members) >= pmax or used >= pool.size:
            return -1, 0
        x = int(pool[used])
        members.append(x)
        if rr[x] >= 0:
            inc[x] = True
        used += 1


def _enter_sigs(state: AlbumState, sig: np.ndarray) -> np.ndarray:
    """Remaining photos per signature that could still join the prune set."""
    _, _, enter = _risk_ctx(state)
    return np.bincount(sig[enter], minlength=N_SIG)


def _reach_now(state: AlbumState, cfg: EnvConfig, page_size_enforced: bool, max_pages: int) -> int:
    sig, rem, mins, open_sig = _sig_state(state, cfg, page_size_enforced)
    nopen = len(state.open_page)
    if state.risk_rank is None:
        return reachable_pages(state.n_closed, nopen, open_sig, rem, mins, max_pages)
    enter_sigs = _enter_sigs(state, sig)
    need, on = (_completion_need(state, cfg, page_size_enforced, state.open_page, open_sig, sig)
                if nopen else (0, 0))
    return reachable_pages(state.n_closed, nopen, open_sig, rem, mins, max_pages,
                           extra=_future_extra(state, on, enter_sigs),
                           extra_sigs=enter_sigs, open_need=need)


def _future_extra(state: AlbumState, on_open: int, enter_sigs: np.ndarray) -> int:
    """Extra photos to reserve for prune-set members on FUTURE pages: the whole cap, as long as that
    many photos could still join the set. NOT reduced by the members the open page pays for -- later
    entries can evict those onto future pages, and reserving less made CLOSE lower the reach
    (52714373: 10 -> 9 on closing a page holding three members). With the full cap, closing a page
    can only raise it. ``on_open`` is kept so every caller states what it knows."""
    del on_open
    return min(state.prune_cap, int(enter_sigs.sum()))


def initial_page_floor(state: AlbumState, cfg: EnvConfig, page_size_enforced: bool) -> int:
    """The body-page floor for this episode: min_pages (less the bookends), capped at what the
    gallery can supply from its assignable photos."""
    n_hero = 2 if state.hero_first >= 0 else 0
    max_pages = max(cfg.max_pages - n_hero, 1)
    return min(max(cfg.min_pages - n_hero, 1),
               _reach_now(state, cfg, page_size_enforced, max_pages))


def _risk_safe_assign(state: AlbumState, cfg: EnvConfig, page_size_enforced: bool,
                      cand: np.ndarray, max_pages: int) -> np.ndarray:
    """``cand`` minus every ASSIGN that breaks the floor or leaves the open page uncompletable.

    Candidates are tested by class -- signature, and whether the photo would join the prune set --
    so at most eight tests run per step, not one per photo. Each class is represented by its
    candidate LEAST similar to a cover: within a class that is the worst case, since a weaker
    entering member lowers the bar for the photos after it, and a non-entering one is the photo the
    completion would otherwise have used first. The floor test does nothing when the floor is
    already out of reach (it then only constrains END).
    """
    sig, rem, mins, open_sig = _sig_state(state, cfg, page_size_enforced)
    nopen = len(state.open_page)
    risk = state.risk_rank is not None
    check_floor = (state.page_floor >= 0
                   and _reach_now(state, cfg, page_size_enforced, max_pages) >= state.page_floor)
    if not risk and not check_floor:
        return cand
    if risk:
        _, _, enter = _risk_ctx(state)
        enter_sigs = np.bincount(sig[enter], minlength=N_SIG)
    else:
        enter = np.zeros(state.pg.n, dtype=bool)
        enter_sigs = np.zeros(N_SIG, dtype=np.int64)
    out = cand.copy()
    reach_of: dict[tuple[int, bool], int] = {}       # completable classes -> reach after
    for g in range(N_SIG):
        page_sig = g if nopen == 0 else open_sig
        after = rem.copy()
        after[g] -= 1
        for entering in (False, True):
            kind = (sig == g) & (enter if entering else ~enter) & cand
            if not kind.any():
                continue
            if risk:
                idx = np.nonzero(kind)[0]
                x = int(idx[np.argmin(state.risk_rank[idx])])
                need, on = _completion_need(state, cfg, page_size_enforced,
                                            state.open_page + [x], page_sig, sig)
                bad = need < 0
                enter_after = enter_sigs.copy()
                enter_after[g] -= int(entering)
                if not bad and check_floor:
                    reach = reachable_pages(state.n_closed, nopen + 1, page_sig, after, mins,
                                            max_pages, extra=_future_extra(state, on, enter_after),
                                            extra_sigs=enter_after, open_need=need)
                    reach_of[(g, entering)] = reach
                    bad = reach < state.page_floor
            else:
                reach = reachable_pages(state.n_closed, nopen + 1, page_sig, after, mins, max_pages)
                reach_of[(g, entering)] = reach
                bad = reach < state.page_floor
            if bad:
                out &= ~kind
    base = (page_size_bounds(state.open_style, cfg)[0] if page_size_enforced
            else cfg.min_photos_per_page)
    closable = nopen > 0 and nopen >= page_min(state, base) and state.n_closed < max_pages
    if check_floor and not out.any() and reach_of and not closable:
        # The lookahead is conservative, so it can judge every assign short of a floor it judged
        # reachable a moment ago. When CLOSE is legal that is the right move and nothing is stuck;
        # otherwise never let it deadlock the album: keep the completable classes that come closest
        # to the floor.
        best = max(reach_of.values())
        for (g, entering), reach in reach_of.items():
            if reach == best:
                out |= cand & (sig == g) & (enter if entering else ~enter)
    return out


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
        if (state.page_floor >= 0 or state.risk_rank is not None) and cand.any():
            cand = _risk_safe_assign(state, cfg, page_size_enforced, cand, max_pages)
        mask[:n] = cand

        # CLOSE_PAGE: only once the open page meets its style's minimum (never undersized).
        pmin = (page_min(state, page_size_bounds(state.open_style, cfg)[0] if page_size_enforced
                         else cfg.min_photos_per_page) if nopen > 0 else 0)
        can_close = (nopen >= pmin) and (nopen > 0) and (state.n_closed < max_pages)
        mask[n + CLOSE] = can_close

        # END_ALBUM: allowed once >= min_pages would be finalized (open page counts only
        # if it already meets its style minimum; otherwise finalize drops it).
        if state.page_floor >= 0:
            # Count the open page by the rule `finalize` applies (its style's minimum), not the
            # CLOSE rule, so END can never drop a page the floor was counting on. Once the floor is
            # out of reach (only possible with dedup_enforced), END waits for every page that is
            # still reachable instead.
            keep = page_min(state, page_size_bounds(state.open_style, cfg)[0])
            finalized = state.n_closed + (1 if nopen and nopen >= keep else 0)
            reach = _reach_now(state, cfg, page_size_enforced, max_pages)
            mask[n + END] = finalized >= min(state.page_floor, reach)
        else:
            open_counts = 1 if (nopen > 0 and nopen >= pmin) else 0
            finalized = state.n_closed + open_counts
            mask[n + END] = finalized >= max(cfg.min_pages - n_hero, 1)
        if state.risk_rank is not None and nopen and \
                nopen < page_min(state, page_size_bounds(state.open_style, cfg)[0]):
            # END would drop the open page, and REMOVING photos is the one thing that can pull a new
            # member into the prune set -- possibly on a closed 2-photo page. Finish the page first.
            mask[n + END] = False

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
