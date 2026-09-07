"""Baselines and the harness that runs any action-selector through the env.

All baselines act through the same env (so masking / hard constraints apply equally),
producing a finished album that is scored with ``RewardEngine.album_score`` for an
apples-to-apples comparison with the learned policy.

Selectors have signature ``select(env, obs, rng) -> int`` (action index).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from ..config import Config
from ..data.schema import Gallery
from ..env.album_env import AlbumNarratorEnv
from ..env.masking import CLOSE, END
from ..env.state import Observation, PHASE_ASSIGN, STATUS_OPEN

Selector = Callable[[AlbumNarratorEnv, Observation, np.random.Generator], int]


def run_episode(env: AlbumNarratorEnv, gallery: Gallery, select: Selector,
                rng: np.random.Generator) -> tuple[list[list[int]], dict]:
    obs = env.reset(gallery)
    done = False
    info: dict = {}
    steps = 0
    while not done and steps < 20 * env.n + 100:
        a = select(env, obs, rng)
        obs, r, done, info = env.step(a)
        steps += 1
    return env.state.ordered_pages(), info


# --------------------------------- selectors -------------------------------- #

def random_valid(env, obs, rng) -> int:
    valid = np.where(obs.action_mask)[0]
    return int(rng.choice(valid))


def greedy_coherence(env, obs, rng) -> int:
    """Grow the open page with the most visually-similar compatible photo; close at
    the target page size; order pages chronologically."""
    n = env.n
    cfg = env.env_cfg
    if obs.phase == PHASE_ASSIGN:
        open_idx = np.where(obs.status == STATUS_OPEN)[0]
        assign_valid = np.where(obs.action_mask[:n])[0]
        can_close = obs.action_mask[n + CLOSE]
        can_end = obs.action_mask[n + END]

        # close when the page is big enough
        if len(open_idx) >= cfg.target_photos_per_page and can_close:
            return n + CLOSE
        if len(assign_valid) > 0:
            if len(open_idx) == 0:
                # seed a page with the highest-selection compatible photo
                best = assign_valid[np.argmax(env.pg.selection[assign_valid])]
                return int(best)
            sim = env.engine.cache.sim
            scores = sim[assign_valid][:, open_idx].mean(1)
            return int(assign_valid[int(np.argmax(scores))])
        if can_close:
            return n + CLOSE
        return n + END if can_end else int(np.where(obs.action_mask)[0][0])
    else:
        # place unplaced page with the earliest median time
        pages = obs.pages
        placed = obs.placed_mask
        cand = np.where(~placed)[0]
        meds = []
        for j in cand:
            valid_t = [i for i in pages[j] if env.pg.time_valid[i]]
            meds.append(np.median(env.pg.time_norm[valid_t]) if valid_t else np.inf)
        return int(cand[int(np.argmin(meds))])


def cluster_heuristic(env, obs, rng) -> int:
    """Group by the precomputed content ``cluster_id``; order chronologically.

    Mirrors the 'use the existing signals' baseline: keep a page to one cluster.
    """
    n = env.n
    cfg = env.env_cfg
    if obs.phase == PHASE_ASSIGN:
        open_idx = np.where(obs.status == STATUS_OPEN)[0]
        assign_valid = np.where(obs.action_mask[:n])[0]
        can_close = obs.action_mask[n + CLOSE]
        can_end = obs.action_mask[n + END]
        if len(assign_valid) == 0:
            return n + CLOSE if can_close else (n + END if can_end else int(np.where(obs.action_mask)[0][0]))
        if len(open_idx) == 0:
            best = assign_valid[np.argmax(env.pg.selection[assign_valid])]
            return int(best)
        page_cluster = int(env.pg.cluster_id[open_idx[0]])
        same = [i for i in assign_valid if int(env.pg.cluster_id[i]) == page_cluster]
        if same and len(open_idx) < cfg.max_photos_per_page:
            # highest selection within the same cluster
            same = np.array(same)
            return int(same[int(np.argmax(env.pg.selection[same]))])
        if can_close:
            return n + CLOSE
        return n + END if can_end else int(np.where(obs.action_mask)[0][0])
    else:
        return greedy_coherence(env, obs, rng)


def policy_greedy(net) -> Selector:
    """Argmax selector for a trained policy (deterministic evaluation).

    Note: argmax decoding of this pointer policy is myopic — it tends to fire
    CLOSE/END too early, collapsing to a degenerate short album. Prefer
    ``policy_sample`` (mean over samples / best-of-N) for a faithful measure.
    """
    @torch.no_grad()
    def select(env, obs, rng) -> int:
        gt = _gt_cache.get(env)
        if gt is None or gt[0] is not env.pg:
            from ..models.policy import build_gallery_tensors
            g_tensors = build_gallery_tensors(env.pg, net.device)
            h, g = net.encode(g_tensors)
            _gt_cache[env] = (env.pg, h, g, g_tensors)
        _, h, g, g_t = _gt_cache[env]
        logits, _ = net.act(g_t, h, g, obs)
        return int(torch.argmax(logits).item())

    _gt_cache: dict = {}
    return select


def policy_sample(net, temperature: float = 1.0) -> Selector:
    """Stochastic selector: sample from the (masked) policy distribution.

    Sampling is drawn from ``rng`` (numpy) over the softmax of the masked logits, so
    reproducibility is tied to the passed generator. Masked actions have -inf logits
    and thus zero probability, so a sampled action is always valid.
    """
    @torch.no_grad()
    def select(env, obs, rng) -> int:
        gt = _gt_cache.get(env)
        if gt is None or gt[0] is not env.pg:
            from ..models.policy import build_gallery_tensors
            g_tensors = build_gallery_tensors(env.pg, net.device)
            h, g = net.encode(g_tensors)
            _gt_cache[env] = (env.pg, h, g, g_tensors)
        _, h, g, g_t = _gt_cache[env]
        logits, _ = net.act(g_t, h, g, obs)
        p = torch.softmax(logits / temperature, dim=-1).cpu().numpy().astype(np.float64)
        p = p / p.sum()
        return int(rng.choice(len(p), p=p))

    _gt_cache: dict = {}
    return select


def policy_sample_scores(env: AlbumNarratorEnv, gallery: Gallery, net, n_samples: int,
                         rng: np.random.Generator, temperature: float = 1.0
                         ) -> tuple[list[float], list[list[int]]]:
    """Run ``n_samples`` stochastic rollouts of the policy on one gallery.

    Returns (scores, best_ordered): the album_score of every sample and the page
    grouping of the highest-scoring one (for best-of-N reporting and structure eval).
    """
    select = policy_sample(net, temperature)
    scores: list[float] = []
    best_score, best_ordered = -np.inf, []
    for _ in range(n_samples):
        ordered, _ = run_episode(env, gallery, select, rng)
        s = env.engine.album_score(ordered)
        scores.append(s)
        if s > best_score:
            best_score, best_ordered = s, ordered
    return scores, best_ordered


def random_search(env: AlbumNarratorEnv, gallery: Gallery, n_samples: int,
                  rng: np.random.Generator) -> float:
    """Best album_score over n random-valid rollouts.

    A weak *reference*, NOT an upper bound: at this combinatorial scale best-of-N
    random barely dents the search space, so a trained policy is expected to exceed it.
    Swap in CEM / simulated annealing here for a genuine upper-bound oracle later.
    """
    best = -np.inf
    for _ in range(n_samples):
        ordered, _ = run_episode(env, gallery, random_valid, rng)
        best = max(best, env.engine.album_score(ordered))
    return float(best)
