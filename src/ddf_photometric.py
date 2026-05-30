"""DDFHashGridPhotometric: hash-grid DDF + parallel RGB head.

Adds a 3-output sigmoid RGB head sharing the trunk with the existing
(dist, vis_logit) heads. The intent is for Stage 13 photometric supervision:
at hit points, the RGB head is regressed against the GS-rendered ground-truth
colour observed along the ray.

Architecturally identical to :class:`DDFHashGrid` (same encoder, trunk, hidden
dim, dir encoding) plus one extra linear (hidden_dim -> 3) head. No view-
dependent bells and whistles — just an MLP that consumes the same trunk
activations as the geometry heads. The hope is that geometry-correlated
appearance acts as an auxiliary signal that sharpens the hash grid features,
without slowing the trunk.

Forward returns (t, vis_logit, rgb). Keep backward-compat by also exposing the
2-tuple via :py:meth:`forward_geom`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ddf_hashgrid import HashGridEncoder
from .ddf_model import sinusoidal_encoding


class DDFHashGridPhotometric(nn.Module):
    """Hash-grid DDF with an extra RGB head."""

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

    def forward(self, points: torch.Tensor, dirs: torch.Tensor):
        p = self.pos_encoder(points)
        d = sinusoidal_encoding(dirs, self.dir_freqs)
        h = self.trunk(torch.cat([p, d], dim=-1))
        dist = torch.nn.functional.softplus(self.dist_head(h)).squeeze(-1)
        vis_logit = self.vis_head(h).squeeze(-1)
        rgb = torch.sigmoid(self.rgb_head(h))
        return dist, vis_logit, rgb

    def forward_geom(self, points: torch.Tensor, dirs: torch.Tensor):
        """2-tuple geometry-only path, for downstream eval scripts that don't need RGB."""
        dist, vis_logit, _ = self.forward(points, dirs)
        return dist, vis_logit
