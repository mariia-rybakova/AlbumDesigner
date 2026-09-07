"""Set-Transformer building blocks (Lee et al. 2019) and the gallery encoder.

The encoder is permutation-equivariant over photos (no positional encoding) and maps a
gallery's per-photo features to contextual embeddings ``h`` (N, d) plus a pooled
gallery summary ``g`` (d,). It depends only on *static* photo features, so it is
encoded once per gallery and reused across all timesteps of an episode.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MAB(nn.Module):
    """Multihead Attention Block: X attends to Y."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln0 = nn.LayerNorm(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, y, y)
        h = self.ln0(x + a)
        return self.ln1(h + self.ff(h))


class ISAB(nn.Module):
    """Induced Set Attention Block: O(N*m) attention via m inducing points."""

    def __init__(self, dim: int, n_heads: int, n_induced: int):
        super().__init__()
        self.inducing = nn.Parameter(torch.randn(1, n_induced, dim) * 0.02)
        self.mab0 = MAB(dim, n_heads)
        self.mab1 = MAB(dim, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        i = self.inducing.expand(b, -1, -1)
        h = self.mab0(i, x)     # (b, m, d)
        return self.mab1(x, h)  # (b, N, d)


class PMA(nn.Module):
    """Pooling by Multihead Attention: k seed vectors attend to the set."""

    def __init__(self, dim: int, n_heads: int, n_seeds: int = 1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, dim) * 0.02)
        self.mab = MAB(dim, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        s = self.seeds.expand(b, -1, -1)
        return self.mab(s, x)   # (b, k, d)


class GalleryEncoder(nn.Module):
    def __init__(self, clip_dim: int, face_dim: int, body_dim: int, n_scalar: int,
                 d_model: int, n_isab: int, n_heads: int, n_induced: int,
                 input_skip: bool = False, input_block_norm: bool = False):
        super().__init__()
        # ISAB contextualizes: every output embedding is a function of the whole gallery, which
        # dilutes per-photo identity by design. Measured: raw CLIP puts 80.4% of a photo's
        # nearest neighbour in its own content cluster, `h` only 29.9-37.8% -- worse than a
        # random-init encoder. The skip re-adds the per-photo projection so `h` carries both
        # context and identity. See docs/representation-probe.md.
        self.input_skip = input_skip
        pc, pf, pb = d_model, d_model // 4, d_model // 4
        self.clip_proj = nn.Linear(clip_dim, pc)
        self.face_proj = nn.Linear(face_dim, pf)
        self.body_proj = nn.Linear(body_dim, pb)
        # Per-modality LayerNorm BEFORE concatenation. Measured at init: the 14 raw scalars
        # carry 4.7x the magnitude of the 256-d CLIP projection (2.166 vs 0.464) while being the
        # weakest signal, so cosine neighbourhoods in the concatenated vector are driven by
        # scalars. 1-NN cluster precision drops 0.825 -> 0.598 at exactly this step; equalizing
        # the blocks restores it to ~0.8. Same failure class as flatness=1e6: an unnormalized
        # block swamping everything downstream. See docs/representation-probe.md.
        self.block_norm = input_block_norm
        if input_block_norm:
            self.norm_clip = nn.LayerNorm(pc)
            self.norm_face = nn.LayerNorm(pf)
            self.norm_body = nn.LayerNorm(pb)
            self.norm_scalar = nn.LayerNorm(n_scalar)
        self.input = nn.Sequential(
            nn.Linear(pc + pf + pb + n_scalar, d_model), nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.isab = nn.ModuleList([ISAB(d_model, n_heads, n_induced) for _ in range(n_isab)])
        self.pma = PMA(d_model, n_heads, n_seeds=1)

    def forward(self, clip: torch.Tensor, face: torch.Tensor, body: torch.Tensor,
                scalar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """All inputs (N, ·) for a single gallery. Returns h (N, d), g (d,)."""
        pc, pf, pb = self.clip_proj(clip), self.face_proj(face), self.body_proj(body)
        if self.block_norm:
            pc, pf, pb = self.norm_clip(pc), self.norm_face(pf), self.norm_body(pb)
            scalar = self.norm_scalar(scalar)
        x = torch.cat([pc, pf, pb, scalar], dim=-1)
        x0 = self.input(x)               # (N, d) per-photo, pre-context
        x = x0.unsqueeze(0)              # (1, N, d)
        for blk in self.isab:
            x = blk(x)
        g = self.pma(x).squeeze(0).squeeze(0)  # (d,)
        h = x.squeeze(0)                        # (N, d)
        if self.input_skip:
            h = h + x0
        return h, g
