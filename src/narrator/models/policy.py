"""Pointer-based actor-critic policy.

The gallery encoder runs once per gallery to produce static contextual embeddings
``h`` and summary ``g``. Per timestep, cheap pointer heads score the variable action
set:

  * phase A: ASSIGN photo i scored from [h_i(+status), q_open, g]; CLOSE / END scored
    from learned token queries against [q_open, g].
  * phase B: PLACE page j scored from [page_emb_j, placed_context, g].

Action masks (from the env) are applied as additive -inf before the categorical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..config import Config
from ..data.schema import PackedGallery
from ..env.state import Observation, PHASE_ASSIGN, PHASE_ORDER, STATUS_OPEN
from .set_encoder import GalleryEncoder

SCALAR_NAMES = (
    "time_norm", "bw", "candid", "indoor", "lighting", "bgcolor", "selection",
    "composition", "flatness", "aspect", "blob_diameter", "blob_cx", "blob_cy",
    "cast_count",
)
N_SCALAR = len(SCALAR_NAMES)

# Scalar features are all normalized quantities (mostly [0,1]; aspect is a ratio ~[0.5,2]).
# A value far outside this band means an un-normalized source field or a sentinel leaked
# through. Because the encoder's input stack ends in LayerNorm, one oversized column silently
# erases every other per-photo feature -- so this must be an error, not a warning.
# See docs/convergence-hypotheses.md (flatnessScore = 1e6).
SCALAR_ABS_MAX = 10.0


@dataclass
class GalleryTensors:
    clip: torch.Tensor
    face: torch.Tensor
    body: torch.Tensor
    scalar: torch.Tensor
    cluster_onehot: torch.Tensor   # (n, C) float32; all-zero row = unclustered
    n: int


def _check_scalar_range(scalar: np.ndarray, gallery_id: str) -> None:
    bad = ~np.isfinite(scalar) | (np.abs(scalar) > SCALAR_ABS_MAX)
    if not bad.any():
        return
    cols = np.unique(np.nonzero(bad)[1])
    detail = ", ".join(
        f"{SCALAR_NAMES[c]}[col {c}] range=[{np.nanmin(scalar[:, c]):.4g}, "
        f"{np.nanmax(scalar[:, c]):.4g}]" for c in cols)
    raise ValueError(
        f"gallery {gallery_id}: scalar feature(s) outside +/-{SCALAR_ABS_MAX} or non-finite: "
        f"{detail}. Un-normalized or sentinel source values destroy all per-photo signal "
        f"downstream of the encoder's LayerNorm -- normalize at ingest (see data/ingest.py)."
    )


def build_gallery_tensors(pg: PackedGallery, device: torch.device) -> GalleryTensors:
    cast_count = pg.cast_multihot.sum(1).astype(np.float32)
    cast_count = cast_count / max(cast_count.max(), 1.0)
    scalar = np.stack([
        pg.time_norm, pg.bw.astype(np.float32), pg.candid, pg.indoor, pg.lighting,
        pg.bgcolor, pg.selection, pg.composition, pg.flatness, pg.aspect,
        pg.blob_diameter, pg.blob_centroid[:, 0], pg.blob_centroid[:, 1], cast_count,
    ], axis=1).astype(np.float32)
    _check_scalar_range(scalar, pg.gallery_id)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    # One-hot content clusters. content_cluster.pb already groups by similarity AND time; the
    # probe in docs/representation-probe.md showed the encoder destroys that signal (1-NN
    # cluster precision 0.804 in raw CLIP -> 0.299 in h), so the head reads it directly.
    cl = np.asarray(pg.cluster_id)
    uniq = np.unique(cl[cl >= 0])
    onehot = np.zeros((pg.n, len(uniq)), dtype=np.float32)
    if len(uniq):
        pos = {int(c): j for j, c in enumerate(uniq)}
        for i, c in enumerate(cl):
            if c >= 0:
                onehot[i, pos[int(c)]] = 1.0
    return GalleryTensors(
        clip=t(pg.clip), face=t(pg.face_emb), body=t(pg.body_emb),
        scalar=t(scalar), cluster_onehot=t(onehot), n=pg.n,
    )


def _mlp(din: int, dh: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(din, dh), nn.GELU(), nn.Linear(dh, 1))


class ActorCritic(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        m, e = cfg.model, cfg.env
        d = m.d_model
        self.d = d
        self.encoder = GalleryEncoder(
            clip_dim=e.clip_dim, face_dim=e.face_dim, body_dim=e.body_dim,
            n_scalar=N_SCALAR, d_model=d, n_isab=m.n_isab, n_heads=m.n_heads,
            n_induced=m.n_induced,
            input_skip=bool(getattr(m, "input_skip", False)),
            input_block_norm=bool(getattr(m, "input_block_norm", False)),
        )
        self.status_emb = nn.Embedding(4, d)
        self.close_tok = nn.Parameter(torch.randn(d) * 0.02)
        self.end_tok = nn.Parameter(torch.randn(d) * 0.02)
        self.empty_open = nn.Parameter(torch.randn(d) * 0.02)
        self.empty_placed = nn.Parameter(torch.randn(d) * 0.02)
        # Two extra per-candidate scalars when enabled: cosine to the open page's mean RAW
        # CLIP, and the fraction of the open page sharing this candidate's content cluster.
        self.pairwise = bool(getattr(m, "pairwise_features", False))
        self.detach_critic = bool(getattr(m, "detach_critic", False))
        self.n_pair = 2 if self.pairwise else 0
        # Album-level state for the actor (docs/research-log.md 1f): 6 album-global scalars reach
        # every phase-A head, 2 per-candidate scalars reach the assign head. Without them the actor
        # is blind to page count, open-page size, covered identities, closed pages and included
        # near-duplicates -- i.e. to everything the global reward terms are functions of.
        self.album_state = bool(getattr(m, "album_state_features", False))
        self.n_alb_g = 6 if self.album_state else 0
        self.n_alb_c = 2 if self.album_state else 0
        self.assign_mlp = _mlp(3 * d + self.n_pair + self.n_alb_c + self.n_alb_g, m.head_hidden)
        self.special_mlp = _mlp(3 * d + self.n_alb_g, m.head_hidden)
        self.place_mlp = _mlp(3 * d, m.head_hidden)
        self.value_mlp = nn.Sequential(
            nn.Linear(d + 4, m.head_hidden), nn.GELU(), nn.Linear(m.head_hidden, 1)
        )

    @property
    def device(self) -> torch.device:
        return self.close_tok.device

    def encode(self, gt: GalleryTensors) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(gt.clip, gt.face, gt.body, gt.scalar)

    def _pairwise(self, gt: "GalleryTensors", open_mask: torch.Tensor) -> torch.Tensor:
        """(B, n, 2) per-candidate features read straight off the raw inputs.

        col 0: cosine between the candidate's CLIP vector and the open page's mean CLIP
        col 1: fraction of the open page sharing the candidate's content cluster

        Both are 0 when the page is empty. These bypass the encoder deliberately: the raw
        input carries cluster identity at 1-NN precision 0.804 while `h` carries 0.299.
        """
        cnt = open_mask.sum(-1, keepdim=True)                      # (B,1)
        has = (cnt > 0).to(gt.clip.dtype)
        denom = cnt.clamp_min(1.0)
        mean_clip = (open_mask @ gt.clip) / denom                  # (B, clip_dim)
        mean_clip = mean_clip / mean_clip.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cos_raw = (mean_clip @ gt.clip.T) * has                    # (B, n)
        if gt.cluster_onehot.shape[1]:
            cnt_cl = open_mask @ gt.cluster_onehot                 # (B, C)
            same = (cnt_cl @ gt.cluster_onehot.T) / denom * has    # (B, n)
        else:
            same = torch.zeros_like(cos_raw)
        return torch.stack([cos_raw, same], dim=-1)                # (B, n, 2)

    def act(self, gt: "GalleryTensors", h: torch.Tensor, g: torch.Tensor, obs: Observation
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (masked_logits (N+2,), value scalar) for one observation."""
        d = self.d
        n = h.shape[0]
        dev = h.device
        status = torch.from_numpy(np.ascontiguousarray(obs.status.astype(np.int64))).to(dev)
        hd = h + self.status_emb(status)
        progress = torch.from_numpy(np.ascontiguousarray(obs.progress)).to(dev)

        logits = torch.full((n + 2,), float("-inf"), device=dev)

        if obs.phase == PHASE_ASSIGN:
            open_idx = np.where(obs.status == 1)[0]
            if len(open_idx):
                q_open = hd[torch.from_numpy(open_idx).to(dev)].mean(0)
            else:
                q_open = self.empty_open
            gb = g.expand(n, d)
            qb = q_open.expand(n, d)
            parts = [hd, qb, gb]
            if self.pairwise:
                om = torch.zeros(1, n, device=dev, dtype=h.dtype)
                if len(open_idx):
                    om[0, torch.from_numpy(open_idx).to(dev)] = 1.0
                parts.append(self._pairwise(gt, om)[0])
            extra = []
            if self.album_state:
                alb = torch.from_numpy(np.ascontiguousarray(obs.album)).to(dev).to(h.dtype)
                cnd = torch.from_numpy(np.ascontiguousarray(obs.cand)).to(dev).to(h.dtype)
                parts += [cnd, alb.unsqueeze(0).expand(n, self.n_alb_g)]
                extra = [alb]
            assign_logits = self.assign_mlp(torch.cat(parts, dim=-1)).squeeze(-1)
            close_logit = self.special_mlp(torch.cat([self.close_tok, q_open, g] + extra))
            end_logit = self.special_mlp(torch.cat([self.end_tok, q_open, g] + extra))
            logits[:n] = assign_logits
            logits[n] = close_logit.squeeze(-1)
            logits[n + 1] = end_logit.squeeze(-1)
        else:  # phase B: order pages
            pages = obs.pages
            n_final = len(pages)
            if n_final:
                page_emb = torch.stack([
                    hd[torch.from_numpy(np.asarray(p)).to(dev)].mean(0) for p in pages
                ])  # (n_final, d)
                placed = obs.placed_mask
                if placed.any():
                    placed_ctx = page_emb[torch.from_numpy(np.where(placed)[0]).to(dev)].mean(0)
                else:
                    placed_ctx = self.empty_placed
                ctxb = placed_ctx.expand(n_final, d)
                gb = g.expand(n_final, d)
                place_logits = self.place_mlp(torch.cat([page_emb, ctxb, gb], dim=-1)).squeeze(-1)
                logits[:n_final] = place_logits

        mask = torch.from_numpy(np.ascontiguousarray(obs.action_mask)).to(dev)
        logits = torch.where(mask, logits, torch.full_like(logits, float("-inf")))

        gv = g.detach() if self.detach_critic else g
        value = self.value_mlp(torch.cat([gv, progress])).squeeze(-1)
        return logits, value

    def evaluate_actions(self, gt: GalleryTensors, obs_list: list[Observation],
                         actions: list[int], return_h: bool = False
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched re-evaluation of an episode's taken actions.

        Numerically equivalent to calling :meth:`act` per step and stacking the results,
        but computes all ``T`` timesteps in a handful of batched forward passes instead of
        a Python loop. The gallery is encoded once (gradients flow through the encoder).

        Returns ``(logps (T,), values (T,), entropies (T,))``.
        """
        d = self.d
        h, g = self.encode(gt)                       # h: (n, d), g: (d,)
        n = h.shape[0]
        dev = h.device
        T = len(obs_list)

        status_all = torch.from_numpy(
            np.stack([o.status for o in obs_list]).astype(np.int64)).to(dev)   # (T, n)
        progress_all = torch.from_numpy(
            np.ascontiguousarray(np.stack([o.progress for o in obs_list]))).to(dev)  # (T, 4)
        mask_all = torch.from_numpy(
            np.stack([o.action_mask for o in obs_list])).to(dev)              # (T, n+2) bool
        actions_t = torch.tensor(actions, dtype=torch.long, device=dev)       # (T,)
        phases = np.array([o.phase for o in obs_list])

        # Value head is phase-independent (only g + progress).
        # Detaching stops the critic's gradient (measured ~99.7% of the encoder's total) from
        # shaping the shared trunk, which the probe showed degrades cluster identity below what
        # a random-init encoder preserves.
        gv = g.detach() if self.detach_critic else g
        values = self.value_mlp(torch.cat([gv.expand(T, d), progress_all], dim=-1)).squeeze(-1)

        logits_all = torch.full((T, n + 2), float("-inf"), device=dev)
        hd_all = h.unsqueeze(0) + self.status_emb(status_all)                  # (T, n, d)

        a_idx = np.where(phases == PHASE_ASSIGN)[0]
        if len(a_idx):
            ai = torch.as_tensor(a_idx, dtype=torch.long, device=dev)
            A = len(a_idx)
            hd_a = hd_all[ai]                                                  # (A, n, d)
            open_mask = (status_all[ai] == STATUS_OPEN).unsqueeze(-1).to(hd_a.dtype)  # (A, n, 1)
            cnt = open_mask.sum(1)                                             # (A, 1)
            has_open = (cnt > 0).to(hd_a.dtype)                                # (A, 1)
            q_open = has_open * ((hd_a * open_mask).sum(1) / cnt.clamp_min(1.0)) \
                + (1.0 - has_open) * self.empty_open                          # (A, d)
            parts = [hd_a, q_open.unsqueeze(1).expand(A, n, d), g.expand(A, n, d)]
            if self.pairwise:
                parts.append(self._pairwise(gt, open_mask.squeeze(-1)))
            extra = []
            if self.album_state:
                alb_a = torch.from_numpy(np.stack([obs_list[j].album for j in a_idx])
                                         ).to(dev).to(hd_a.dtype)              # (A, 6)
                cnd_a = torch.from_numpy(np.stack([obs_list[j].cand for j in a_idx])
                                         ).to(dev).to(hd_a.dtype)              # (A, n, 2)
                parts += [cnd_a, alb_a.unsqueeze(1).expand(A, n, self.n_alb_g)]
                extra = [alb_a]
            logits_all[ai, :n] = self.assign_mlp(torch.cat(parts, dim=-1)).squeeze(-1)  # (A, n)
            close_in = torch.cat([self.close_tok.expand(A, d), q_open, g.expand(A, d)] + extra,
                                 dim=-1)
            end_in = torch.cat([self.end_tok.expand(A, d), q_open, g.expand(A, d)] + extra, dim=-1)
            logits_all[ai, n] = self.special_mlp(close_in).squeeze(-1)        # (A,)
            logits_all[ai, n + 1] = self.special_mlp(end_in).squeeze(-1)      # (A,)

        o_idx = np.where(phases == PHASE_ORDER)[0]
        if len(o_idx):
            oi = torch.as_tensor(o_idx, dtype=torch.long, device=dev)
            O = len(o_idx)
            pages = obs_list[int(o_idx[0])].pages     # constant across phase B (final_pages)
            n_final = len(pages)
            hd_b = hd_all[int(o_idx[0])]              # (n, d); status constant in phase B
            page_emb = torch.stack([
                hd_b[torch.as_tensor(np.asarray(p), dtype=torch.long, device=dev)].mean(0)
                for p in pages])                      # (n_final, d)
            placed_all = torch.from_numpy(
                np.stack([obs_list[j].placed_mask for j in o_idx])).to(dev)   # (O, n_final) bool
            pf = placed_all.unsqueeze(-1).to(page_emb.dtype)                   # (O, n_final, 1)
            cnt = pf.sum(1)                                                    # (O, 1)
            has = (cnt > 0).to(page_emb.dtype)                                 # (O, 1)
            placed_ctx = has * ((page_emb.unsqueeze(0) * pf).sum(1) / cnt.clamp_min(1.0)) \
                + (1.0 - has) * self.empty_placed                             # (O, d)
            place_in = torch.cat(
                [page_emb.unsqueeze(0).expand(O, n_final, d),
                 placed_ctx.unsqueeze(1).expand(O, n_final, d),
                 g.expand(O, n_final, d)], dim=-1)
            logits_all[oi, :n_final] = self.place_mlp(place_in).squeeze(-1)    # (O, n_final)

        logits_all = torch.where(mask_all, logits_all, torch.full_like(logits_all, float("-inf")))
        dist = torch.distributions.Categorical(logits=logits_all)
        if return_h:
            # Same encoder pass the policy just used -- the auxiliary loss must not re-encode.
            return dist.log_prob(actions_t), values, dist.entropy(), h
        return dist.log_prob(actions_t), values, dist.entropy()
