"""AlbumNarratorEnv: the sequential-construction MDP.

An episode = one gallery. The policy interleaves ASSIGN / CLOSE_PAGE / END_ALBUM to
select photos and group them into pages (phase A), then PLACEs pages into a narrative
order (phase B). Reward = potential-based shaping of the page-local terms each step
(dense credit) + a terminal reward of the global terms on the finished album.

This is a light ``reset``/``step`` interface (no gymnasium dependency).
"""

from __future__ import annotations

import numpy as np

from ..config import Config
from ..data.preprocess import build_cache
from ..data.schema import Gallery, pack_gallery
from ..reward.reward_fn import RewardEngine
from . import hero, masking
from .state import (
    AlbumState,
    Observation,
    PHASE_ASSIGN,
    PHASE_ORDER,
    STATUS_CLOSED,
    STATUS_EXCLUDED,
    STATUS_OPEN,
)


class AlbumNarratorEnv:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.env_cfg = cfg.env
        self.reward_cfg = cfg.reward
        self.state: AlbumState | None = None
        self.engine: RewardEngine | None = None
        self.n = 0
        self._steps = 0
        self._budget = 0

    # ------------------------------------------------------------------ #
    def reset(self, gallery: Gallery) -> Observation:
        pg = pack_gallery(gallery, self.env_cfg)
        cache = build_cache(pg, self.reward_cfg)
        self.pg = pg
        # Heroes are picked before the engine exists so the engine can be told which photos are
        # structural (it exempts their pages from the page-local terms under
        # cfg.hero_exempt_local). pick_hero_photos returns (-1, -1) when the gallery cannot
        # supply two distinct bookends, which disables the whole mechanism for that gallery.
        first, last = hero.pick_hero_photos(pg) if self.reward_cfg.hero_pages else (-1, -1)
        self.engine = RewardEngine(pg, cache, self.reward_cfg, hero_photos=(first, last))
        self.state = AlbumState(pg=pg)
        if first >= 0:
            # Reserve the opening/closing photos before the policy sees anything: withheld from
            # `remaining` so they can never be assigned to a body page, and marked included so
            # coverage/selection/dedup count them. state.ordered_pages() bookends their pages.
            self.state.hero_first, self.state.hero_last = first, last
            for i in (first, last):
                self.state.remaining[i] = False
                self.state.status[i] = STATUS_CLOSED
        self.n = pg.n
        self._steps = 0
        self._budget = int(self.env_cfg.step_budget_factor * self.n) + 10
        return self._obs()

    @property
    def action_dim(self) -> int:
        return self.n + 2

    # ------------------------------------------------------------------ #
    def _potential(self) -> float:
        return self.engine.potential(self.state.ordered_pages())

    def _album_features(self) -> tuple[np.ndarray, np.ndarray]:
        """Album-level state for the actor (cfg.model.album_state_features).

        Returns ``(album (6,), cand (N, 2))``:

        album -- what the actor needs to steer the whole-album terms:
          0 inc_frac        included / N                          (selection pressure)
          1 pages_vs_target n_pages_total / shape.target_pages    (`shape`)
          2 open_size_frac  |open page| / max_photos_per_page      (`pagesize`, and when to CLOSE)
          3 open_need       photos still needed before it may close, normalized
          4 open_over       photos beyond its style's max, normalized
          5 coverage_now    identities present in the album / total identities   (`coverage`)

        cand -- per-candidate marginal effects, the part a static per-photo prior cannot express:
          0 new_id_frac     fraction of this photo's identities NOT yet in the album (`coverage`;
                            a state-blind actor can only learn "prefer crowded photos", which
                            measured 0.754 against 0.980 for a marginal-gain picker)
          1 dup_included    1.0 if a near-duplicate of this photo is already included (`dedup`)

        Recomputed from ``status`` every step rather than tracked incrementally, so it cannot go
        stale when ``finalize`` retroactively excludes an undersized open page. Costs ~100k numpy
        ops per step against a ~35-step episode, i.e. nothing.
        """
        s, pg = self.state, self.pg
        n = pg.n
        inc = (s.status == STATUS_CLOSED) | (s.status == STATUS_OPEN)
        mh = pg.cast_multihot
        n_ids = max(mh.shape[1], 1)
        covered = mh[inc].any(0) if inc.any() and mh.shape[1] else np.zeros(n_ids, dtype=bool)

        nopen = len(s.open_page)
        pmin, pmax = masking.page_size_bounds(s.open_style, self.env_cfg)
        mx = max(self.env_cfg.max_photos_per_page, 1)
        album = np.array([
            s.n_included / max(n, 1),
            s.n_pages_total / max(pg.shape.target_pages, 1e-6),
            nopen / mx,
            (max(pmin - nopen, 0) / mx) if nopen else 0.0,
            max(nopen - pmax, 0) / mx,
            float(covered.sum()) / n_ids if mh.shape[1] else 0.0,
        ], dtype=np.float32)

        cand = np.zeros((n, 2), dtype=np.float32)
        if mh.shape[1]:
            per_photo = mh.sum(1)                              # identities in each photo
            new_ids = (mh & ~covered).sum(1)                    # of those, how many are new
            np.divide(new_ids, np.maximum(per_photo, 1), out=cand[:, 0], casting="unsafe")
        if inc.any():
            cand[:, 1] = self.engine.cache.is_dup[:, inc].any(1).astype(np.float32)
        return album, cand

    def _obs(self) -> Observation:
        s = self.state
        mask = masking.compute_mask(
            s, self.env_cfg,
            page_size_enforced=self.reward_cfg.page_size_enforced,
            is_dup=(self.engine.cache.is_dup if self.reward_cfg.dedup_enforced else None))
        inc_frac = s.n_included / max(self.n, 1)
        pages_frac = s.n_pages_total / max(self.env_cfg.max_pages, 1)
        open_frac = len(s.open_page) / max(self.env_cfg.max_photos_per_page, 1) \
            if s.phase == PHASE_ASSIGN else 1.0
        progress = np.array([inc_frac, pages_frac, open_frac, float(s.phase)], dtype=np.float32)
        if s.phase == PHASE_ORDER:
            pages = s.final_pages
            placed = s.placed.copy() if s.placed is not None else np.zeros(0, dtype=bool)
        else:
            pages = [list(p) for p in s.pages if p]
            placed = np.zeros(0, dtype=bool)
        album, cand = (self._album_features()
                       if self.cfg.model.album_state_features else (None, None))
        return Observation(
            phase=s.phase,
            status=s.status.copy(),
            page_of=s.page_of.copy(),
            open_bw=s.open_bw,
            open_style=s.open_style,
            action_mask=mask,
            progress=progress,
            pages=[list(p) for p in pages],
            placed_mask=placed,
            album=album,
            cand=cand,
        )

    # ------------------------------------------------------------------ #
    def step(self, action: int) -> tuple[Observation, float, bool, dict]:
        s = self.state
        n = self.n
        self._steps += 1
        p_prev = self._potential()
        info: dict = {}

        truncate = self._steps > self._budget

        if s.phase == PHASE_ASSIGN:
            if action < n:
                s.assign(int(action))
            elif action == n + masking.CLOSE:
                s.close_page()
            else:  # END_ALBUM
                self._finalize_and_maybe_finish()
            if truncate and s.phase == PHASE_ASSIGN:
                self._finalize_and_maybe_finish()
                info["truncated"] = True
        else:  # PHASE_ORDER
            if action < len(s.final_pages) and not s.placed[action]:
                s.place(int(action))
            else:
                # mask should prevent this; if forced, place first unplaced
                remaining = np.where(~s.placed)[0]
                if len(remaining):
                    s.place(int(remaining[0]))
            if truncate:
                for j in np.where(~s.placed)[0]:
                    s.place(int(j))
                info["truncated"] = True

        p_new = self._potential()
        reward = self.reward_cfg.gamma_shaping * p_new - p_prev

        done = s.order_done() or (s.phase == PHASE_ORDER and len(s.final_pages) == 0)
        if done:
            ordered = s.ordered_pages()
            reward += self.engine.terminal(ordered)
            info["breakdown"] = self.engine.breakdown(ordered)
            info["n_pages"] = len([p for p in ordered if p])
            info["n_included"] = s.n_included
            info["hard_violations"] = self.count_hard_violations(ordered)

        return self._obs(), float(reward), bool(done), info

    # ------------------------------------------------------------------ #
    def _page_median_time(self, page: list[int]) -> float:
        """Median ``time_norm`` over the page's dated photos; +inf when it has none.

        +inf rather than NaN so undated pages sort to the end deterministically instead of
        landing wherever the sort happens to leave them.
        """
        valid = [i for i in page if self.pg.time_valid[i]]
        return float(np.median(self.pg.time_norm[valid])) if valid else float("inf")

    def _finalize_and_maybe_finish(self) -> None:
        s = self.state
        # keep the open page only if it meets ITS style's minimum (else exclude it)
        page_min = masking.page_size_bounds(s.open_style, self.env_cfg)[0]
        s.finalize(page_min)
        if self.reward_cfg.order_by_time:
            # Narrative order is structural, not learned (cfg.order_by_time): place every page in
            # chronological order here, so phase B emits no decisions and the episode ends. Worth
            # +0.126 album_score for free -- see RewardConfig.order_by_time.
            for j in sorted(range(len(s.final_pages)),
                            key=lambda k: self._page_median_time(s.final_pages[k])):
                s.place(j)
        elif len(s.final_pages) <= 1:
            # trivial ordering: auto-place when 0 or 1 page so we never emit a
            # single-choice phase-B step.
            for j in range(len(s.final_pages)):
                s.place(j)

    # ------------------------------------------------------------------ #
    def count_hard_violations(self, ordered_pages: list[list[int]]) -> int:
        """Pages that mix candid+formal or B&W+color. Must be 0 (masking guarantees)."""
        pg = self.pg
        v = 0
        for page in ordered_pages:
            if len(page) < 2:
                continue
            if len(set(pg.bw[i] for i in page)) > 1:
                v += 1
            elif len(set(pg.style_class[i] for i in page)) > 1:
                v += 1
        return v
