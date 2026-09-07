"""RewardEngine: assembles per-term functions into the shaped step reward and the
terminal reward, with tunable weights and a per-term breakdown for logging.

Design (see plan):
  * page-local terms feed a potential Phi(pages) = scale * sum_pages sum_terms w*term.
    The env awards potential-based shaping F = gamma*Phi(s') - Phi(s) each step, which
    is policy-invariant (does not move the optimum) yet gives dense credit.
  * global terms are awarded once, on the finished ordered album.

Weights not present for a term default to 0. Terms are grouped by name in
``terms.PAGE_LOCAL_TERMS`` / ``terms.GLOBAL_TERMS``.
"""

from __future__ import annotations

from ..config import RewardConfig
from ..data.preprocess import GalleryCache
from ..data.schema import PackedGallery
from . import terms


class RewardEngine:
    def __init__(self, pg: PackedGallery, cache: GalleryCache, cfg: RewardConfig,
                 hero_photos: tuple[int, int] = (-1, -1)):
        self.pg = pg
        self.cache = cache
        self.cfg = cfg
        self.w = cfg.weights
        self.scale = cfg.reward_scale
        # Which photos the env reserved as structural bookends, so `_local_pages` can tell a
        # forced single-photo page from one the policy built. Empty unless cfg.hero_exempt_local.
        self._hero = (frozenset(i for i in hero_photos if i >= 0)
                      if cfg.hero_exempt_local else frozenset())
        # Global terms moved into Phi (cfg.shape_globals). `narrative` is handled by its own flag,
        # so it is excluded here to make double-listing harmless rather than double-counted.
        shaped = frozenset(getattr(cfg, "shape_globals", ()) or ()) - {"narrative"}
        unknown = shaped - set(terms.GLOBAL_TERMS)
        if unknown:
            raise ValueError(f"reward.shape_globals names non-global term(s) {sorted(unknown)}; "
                             f"valid: {sorted(terms.GLOBAL_TERMS)}")
        self._shaped = shaped

    # ------------------------------ potential ------------------------------ #

    def page_potential(self, idx: list[int]) -> float:
        local = terms.page_local(idx, self.pg, self.cache, self.cfg.visual_mode,
                                 self.cfg.pagesize_mode, self.cfg.cast_mode)
        return sum(self.w.get(k, 0.0) * v for k, v in local.items())

    def _local_pages(self, nonempty: list[list[int]]) -> list[list[int]]:
        """The pages the page-local terms are computed over.

        Drops the hero bookends when ``cfg.hero_exempt_local`` is set: they are single photos
        placed by construction, every page-local term is a pairwise mean and so scores them 0,
        and counting them both understates page coherence and creates a spurious page-count
        gradient (see RewardConfig.hero_exempt_local for the measured sizes). Identified by
        CONTENT rather than by position, so it holds wherever the page list came from.
        """
        if not self._hero:
            return nonempty
        return [p for p in nonempty if not (len(p) == 1 and p[0] in self._hero)]

    def potential(self, pages: list[list[int]]) -> float:
        """Phi(s): weighted page-local terms combined over all (open + closed) pages.

        ``cfg.page_local_reduce`` selects sum (historical) or mean. Summing multiplies every
        page-local term by the page count, which is why they came to ~94% of album_score while
        all six global terms shared ~6% (docs/reward-audit.md); the mean makes album_score
        length-independent and restores the nominal weight balance.

        Potential-based shaping stays policy-invariant either way -- Phi may be any function
        of state -- but the two choices define different notions of a good album.
        """
        nonempty = [p for p in pages if p]
        if not nonempty:
            return 0.0
        # page-local terms score only the pages the policy composed; narrative below is a
        # whole-album property and so still reads every page, bookends included.
        local_pages = self._local_pages(nonempty)
        if self.cfg.page_local_reduce == "photo":
            # Weight each page by the photos it commits, then normalise by the photo count: a mean
            # over PHOTOS rather than over pages. Removes the small-page arbitrage -- see
            # RewardConfig.page_local_reduce.
            n_ph = sum(len(p) for p in local_pages)
            total = (sum(len(p) * self.page_potential(p) for p in local_pages) / n_ph
                     if n_ph else 0.0)
        else:
            total = sum(self.page_potential(p) for p in local_pages)
            if self.cfg.page_local_reduce == "mean" and local_pages:
                total /= len(local_pages)
        if self.cfg.shape_narrative:
            # Not divided by page count: narrative is a whole-album property. In phase B
            # ``ordered_pages()`` returns only the pages placed so far, so this evaluates the
            # growing prefix and every PLACE earns dense credit. At termination the prefix is
            # the full order, so Phi carries exactly the value ``terminal()`` used to award.
            # narrative is a GLOBAL term even when shaped into Phi, so it follows global_scale
            total += (self.cfg.global_scale * self.w.get("narrative", 0.0)
                      * terms.narrative_order(nonempty, self.pg))
        if self._shaped:
            # Same treatment, generalized: evaluated on the album SO FAR, so every step earns
            # attributable credit instead of one lump at termination. Not divided by page count --
            # these are whole-album properties. At termination `nonempty` is the finished album, so
            # Phi carries exactly the value `terminal()` would have awarded, which is what keeps
            # album_score identical (test_shape_globals_preserves_album_score).
            g = self._global_terms(nonempty, only=self._shaped)
            total += self.cfg.global_scale * sum(self.w.get(k, 0.0) * v for k, v in g.items())
        return self.scale * total

    def shaping_step(self, prev_pages: list[list[int]], new_pages: list[list[int]]) -> float:
        """Potential-based shaping reward F = gamma*Phi(s') - Phi(s)."""
        return self.cfg.gamma_shaping * self.potential(new_pages) - self.potential(prev_pages)

    # ------------------------------ terminal ------------------------------- #

    def _global_terms(self, ordered_pages: list[list[int]],
                      only: frozenset[str] | None = None) -> dict[str, float]:
        """The six global terms. ``only`` restricts the computation -- Phi calls this every step,
        so it must not pay for `diversity`'s per-page CLIP means when it is not shaping them."""
        included = [i for p in ordered_pages for i in p]
        fns = {
            "coverage": lambda: terms.people_coverage(included, self.pg),
            "narrative": lambda: terms.narrative_order(ordered_pages, self.pg),
            "diversity": lambda: terms.page_diversity(ordered_pages, self.pg),
            "shape": lambda: terms.shape_preference(ordered_pages, self.pg.shape,
                                                    self.cfg.shape_use_avg_pp),
            "selection": lambda: terms.selection_quality(included, self.pg),
            "dedup": lambda: terms.dedup(included, self.cache, self.cfg.dedup_mode),
            "scene": lambda: terms.scene_coverage(included, self.pg),
            "photocov": lambda: terms.photo_coverage(included, self.cache),
        }
        return {k: f() for k, f in fns.items() if only is None or k in only}

    def terminal(self, ordered_pages: list[list[int]]) -> float:
        g = self._global_terms(ordered_pages)
        drop = set(self._shaped)
        if self.cfg.shape_narrative:
            drop.add("narrative")
        if drop:
            # Awarded through Phi instead; counting it here too would double its weight.
            g = {k: v for k, v in g.items() if k not in drop}
        return self.scale * self.cfg.global_scale * sum(self.w.get(k, 0.0) * v
                                                        for k, v in g.items())

    def album_score(self, ordered_pages: list[list[int]]) -> float:
        """Order-independent-of-shaping quality of a finished album (page-local +
        global, weighted). Used to compare policy vs baselines apples-to-apples."""
        return self.potential(ordered_pages) + self.terminal(ordered_pages)

    # ------------------------------ breakdown ------------------------------ #

    def breakdown(self, ordered_pages: list[list[int]]) -> dict[str, float]:
        """Full per-term report (unweighted term values) for logging/eval.

        Page-local terms are averaged over non-empty pages (excluding the hero bookends when
        ``cfg.hero_exempt_local``, so the figure is body-only); global terms as-is.
        Also reports the total shaped-if-terminal potential and terminal reward.
        """
        pages = self._local_pages([p for p in ordered_pages if p])
        out: dict[str, float] = {}
        if pages:
            for name in terms.PAGE_LOCAL_TERMS:
                vals = [terms.page_local(p, self.pg, self.cache, self.cfg.visual_mode,
                                         self.cfg.pagesize_mode, self.cfg.cast_mode)[name]
                        for p in pages]
                out[name] = float(sum(vals) / len(vals))
        else:
            for name in terms.PAGE_LOCAL_TERMS:
                out[name] = 0.0
        out.update(self._global_terms(ordered_pages))
        out["_potential"] = self.potential(ordered_pages)
        out["_terminal"] = self.terminal(ordered_pages)
        return out
