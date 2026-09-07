"""Album construction state and the observation emitted to the policy.

Action layout (fixed size ``A = N + 2``):
  * ``0 .. N-1``  -- phase A: ASSIGN photo i to the open page;
                     phase B: PLACE page i next in the narrative (i < n_pages).
  * ``N``          -- CLOSE_PAGE (phase A): finalize the open page, open a new one.
  * ``N + 1``      -- END_ALBUM (phase A): finalize and move to ordering (phase B).

Status codes per photo: 0 remaining, 1 in-open-page, 2 in-closed-page, 3 excluded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.schema import PackedGallery

PHASE_ASSIGN = 0
PHASE_ORDER = 1

STATUS_REMAINING = 0
STATUS_OPEN = 1
STATUS_CLOSED = 2
STATUS_EXCLUDED = 3


@dataclass
class Observation:
    """Compact, picklable snapshot the policy featurizes. One per timestep."""
    phase: int
    status: np.ndarray          # (N,) int8
    page_of: np.ndarray         # (N,) int16 (page index or -1)
    open_bw: int                # -1 if open page empty/unset
    open_style: int             # -1 if unset
    action_mask: np.ndarray     # (N+2,) bool
    progress: np.ndarray        # (4,) float32 [inc_frac, pages_frac, open_frac, phase]
    pages: list[list[int]]      # current pages (closed + open in phase A; ordered-so-far context in B)
    placed_mask: np.ndarray     # (n_pages,) bool: page already placed (phase B); empty in A
    # Album-level state for the ACTOR (cfg.model.album_state_features; None when disabled).
    # `progress` above reaches only the value head, so without these the actor is blind to
    # everything the global reward terms depend on -- see AlbumNarratorEnv._album_features.
    album: np.ndarray | None = None   # (6,)   float32, album-global
    cand: np.ndarray | None = None    # (N, 2) float32, per-candidate


@dataclass
class AlbumState:
    """Mutable album-construction state plus low-level, mask-free mutators."""
    pg: PackedGallery
    phase: int = PHASE_ASSIGN
    pages: list[list[int]] = field(default_factory=list)   # closed pages + open page (last)
    remaining: np.ndarray = None                            # (N,) bool
    status: np.ndarray = None                               # (N,) int8
    page_of: np.ndarray = None                              # (N,) int16
    open_bw: int = -1
    open_style: int = -1
    # phase B ordering
    # Forced single-photo opening/closing pages (cfg.reward.hero_pages). Held outside the
    # assignable pool and bookended onto every view of the album, so the policy only ever
    # composes the body. -1 = disabled.
    hero_first: int = -1
    hero_last: int = -1
    final_pages: list[list[int]] = field(default_factory=list)
    order: list[int] = field(default_factory=list)          # placed page indices, in order
    placed: np.ndarray = None                               # (n_final,) bool

    def __post_init__(self):
        n = self.pg.n
        if self.remaining is None:
            self.remaining = np.ones(n, dtype=bool)
        if self.status is None:
            self.status = np.full(n, STATUS_REMAINING, dtype=np.int8)
        if self.page_of is None:
            self.page_of = np.full(n, -1, dtype=np.int16)
        if not self.pages:
            self.pages = [[]]  # start with one empty open page

    # ----- convenience views -----
    @property
    def open_page(self) -> list[int]:
        return self.pages[-1]

    @property
    def n_closed(self) -> int:
        return len(self.pages) - 1

    @property
    def n_pages_total(self) -> int:
        """Pages that would exist if the album ended now (open counts if non-empty)."""
        return self.n_closed + (1 if self.open_page else 0)

    @property
    def n_included(self) -> int:
        return int((self.status == STATUS_OPEN).sum() + (self.status == STATUS_CLOSED).sum())

    # ----- mutators (no validity checks; masking guarantees legality) -----
    def assign(self, i: int) -> None:
        if not self.open_page:  # first photo locks the page signature
            self.open_bw = int(self.pg.bw[i])
            self.open_style = int(self.pg.style_class[i])
        self.open_page.append(i)
        self.remaining[i] = False
        self.status[i] = STATUS_OPEN
        self.page_of[i] = self.n_closed  # index of the (current) open page

    def close_page(self) -> None:
        for i in self.open_page:
            self.status[i] = STATUS_CLOSED
        self.pages.append([])  # new empty open page
        self.open_bw = -1
        self.open_style = -1

    def finalize(self, min_photos_per_page: int) -> None:
        """End phase A: keep the open page iff full enough; exclude leftovers."""
        if self.open_page and len(self.open_page) >= min_photos_per_page:
            for i in self.open_page:
                self.status[i] = STATUS_CLOSED
            closed = self.pages
        else:
            for i in self.open_page:
                self.status[i] = STATUS_EXCLUDED
                self.page_of[i] = -1
            closed = self.pages[:-1]
        self.final_pages = [p for p in closed if p]
        # every still-remaining photo is excluded
        self.status[self.remaining] = STATUS_EXCLUDED
        self.remaining[:] = False
        self.placed = np.zeros(len(self.final_pages), dtype=bool)
        self.phase = PHASE_ORDER

    def place(self, page_idx: int) -> None:
        self.order.append(page_idx)
        self.placed[page_idx] = True

    def ordered_pages(self) -> list[list[int]]:
        if self.phase == PHASE_ORDER and self.order:
            body = [self.final_pages[j] for j in self.order]
        else:
            # phase A view: closed + open (non-empty)
            body = [p for p in self.pages if p]
        if self.hero_first < 0:
            return body
        # Bookends are structural, not chosen: they frame whatever the policy has built.
        return [[self.hero_first]] + body + [[self.hero_last]]

    def order_done(self) -> bool:
        return self.phase == PHASE_ORDER and bool(self.placed is not None and self.placed.all())
