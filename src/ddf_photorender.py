"""DDFHashGridPhotoRender: hash-grid DDF + RGB head for NeuS-style volume rendering.

Stage 16 — adapts NeuS-style volumetric supervision to DDF. The model produces:
    forward(x, d) -> (t, vis_logit, rgb)

where ``t = DDF(x, d)`` is the directed distance to the surface, ``vis_logit`` is
the standard hit logit, and ``rgb`` is a sigmoid 3-channel colour conditioned on
both position and direction (view-dependent shading).

Compared to :class:`DDFHashGridPhotometric` (Stage 13), the RGB head is the
same shape but it is *trained differently*: it is composited with a NeuS-style
density derived from the DDF distance via ``density(x) propto exp(-beta * t^2)``
and L1-supervised against ground-truth pixel colours from multi-view renders.

The hope is that volumetric rendering (N samples per ray, alpha-composited) gives
a much stronger gradient to the geometry head than the per-hit-point colour
supervision of Stage 13 (which is what Stage 13 found regressed by ~+0.007 CD).

Architecturally:
- Position encoded by the existing ``HashGridEncoder`` (16 levels × 2 feats by
  default = 32 dim).
- Direction encoded by sinusoidal Fourier features (4 freqs = 27 dim including
  the raw 3-D dir).
- Concatenated input -> 2 × 64 hidden trunk -> three heads:
    dist_head:  Linear(hidden, 1), softplus.
    vis_head:   Linear(hidden, 1).
    rgb_head:   Linear(hidden, 3), sigmoid.
- ``beta`` is a learnable scalar parameter (init 10.0) that sharpens the
  distance->density curve (NeuS uses 1/s where s is learnable; same idea).

This module is API-compatible with both ``DDFHashGrid`` (geometry-only eval via
``forward_geom``) and ``DDFHashGridPhotometric`` (3-tuple forward).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ddf_hashgrid import HashGridEncoder
from .ddf_model import sinusoidal_encoding


class DDFHashGridPhotoRender(nn.Module):
    """Hash-grid DDF with RGB head + learnable beta for NeuS-style volume rendering."""

    def __init__(
        self,
        dir_freqs: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        n_levels: int = 16,
        feat_dim: int = 2,
        log2_table_size: int = 19,
        base_res: int = 16,
        growth: float = 1.5,
        bbox_half: float = 1.2,
        beta_init: float = 10.0,
    ):
        super().__init__()
        self.dir_freqs = dir_freqs
        self.bbox_half = bbox_half

        self.pos_encoder = HashGridEncoder(
            n_levels=n_levels,
            feat_dim=feat_dim,
            log2_table_size=log2_table_size,
            base_res=base_res,
            growth=growth,
            bbox_half=bbox_half,
        )

        pos_dim = self.pos_encoder.out_dim
        dir_dim = 3 + 3 * 2 * dir_freqs
        in_dim = pos_dim + dir_dim

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(*layers)
        self.dist_head = nn.Linear(hidden_dim, 1)
        self.vis_head = nn.Linear(hidden_dim, 1)
        self.rgb_head = nn.Linear(hidden_dim, 3)

        # Learnable density sharpness. We parametrise log_beta so beta stays > 0.
        self.log_beta = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(beta_init)))))

    @property
    def beta(self) -> torch.Tensor:
        return self.log_beta.exp()

    def forward(self, points: torch.Tensor, dirs: torch.Tensor):
        p = self.pos_encoder(points)
        d = sinusoidal_encoding(dirs, self.dir_freqs)
        h = self.trunk(torch.cat([p, d], dim=-1))
        dist = torch.nn.functional.softplus(self.dist_head(h)).squeeze(-1)
        vis_logit = self.vis_head(h).squeeze(-1)
        rgb = torch.sigmoid(self.rgb_head(h))
        return dist, vis_logit, rgb

    def forward_geom(self, points: torch.Tensor, dirs: torch.Tensor):
        """2-tuple geometry-only path, for downstream eval (UDF->MC, latency)."""
        dist, vis_logit, _ = self.forward(points, dirs)
        return dist, vis_logit
