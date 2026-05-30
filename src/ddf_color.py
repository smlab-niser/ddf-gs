"""DDFHashGridColor: hash-grid DDF with separate geometry and color branches.

The geometry branch is identical to DDFHashGrid (2×64 MLP → dist + vis).
The color branch is a separate 3×64 MLP that takes:
  - Hash-grid positional features (shared with geometry)
  - Sinusoidal direction encoding (shared)
  - Detached geometry trunk features (prevents color gradients from corrupting geometry)

This separation is critical: Stage 16b showed that a shared trunk lets the RGB
head fit color independently of geometry when the density link is weak.

Training:
  - Geometry: GS depth supervision (proven kitchen-sink recipe)
  - Color: sphere-trace to surface (no_grad), predict RGB, L1 vs GT image
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .ddf_hashgrid import HashGridEncoder
from .ddf_model import sinusoidal_encoding


class DDFHashGridColor(nn.Module):

    def __init__(
        self,
        dir_freqs: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        color_hidden_dim: int = 64,
        color_num_layers: int = 3,
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
            n_levels=n_levels, feat_dim=feat_dim,
            log2_table_size=log2_table_size, base_res=base_res,
            growth=growth, bbox_half=bbox_half,
        )

        pos_dim = self.pos_encoder.out_dim
        dir_dim = 3 + 3 * 2 * dir_freqs
        in_dim = pos_dim + dir_dim

        # Geometry branch (same as DDFHashGrid)
        geo_layers = []
        for i in range(num_layers):
            geo_layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            geo_layers.append(nn.ReLU(inplace=True))
        self.geo_trunk = nn.Sequential(*geo_layers)
        self.dist_head = nn.Linear(hidden_dim, 1)
        self.vis_head = nn.Linear(hidden_dim, 1)

        # Color branch (separate trunk, receives detached geo features)
        color_in_dim = pos_dim + dir_dim + hidden_dim
        color_layers = []
        for i in range(color_num_layers):
            color_layers.append(nn.Linear(color_in_dim if i == 0 else color_hidden_dim, color_hidden_dim))
            color_layers.append(nn.ReLU(inplace=True))
        self.color_trunk = nn.Sequential(*color_layers)
        self.rgb_head = nn.Linear(color_hidden_dim, 3)

    def forward(self, points: torch.Tensor, dirs: torch.Tensor):
        """Full forward: returns (dist, vis_logit, rgb)."""
        p = self.pos_encoder(points)
        d = sinusoidal_encoding(dirs, self.dir_freqs)
        pd = torch.cat([p, d], dim=-1)

        geo_feat = self.geo_trunk(pd)
        dist = nn.functional.softplus(self.dist_head(geo_feat)).squeeze(-1)
        vis_logit = self.vis_head(geo_feat).squeeze(-1)

        color_in = torch.cat([pd, geo_feat.detach()], dim=-1)
        color_feat = self.color_trunk(color_in)
        rgb = torch.sigmoid(self.rgb_head(color_feat))

        return dist, vis_logit, rgb

    def forward_geom(self, points: torch.Tensor, dirs: torch.Tensor):
        """Geometry-only forward (for depth supervision, UDF extraction, latency)."""
        p = self.pos_encoder(points)
        d = sinusoidal_encoding(dirs, self.dir_freqs)
        geo_feat = self.geo_trunk(torch.cat([p, d], dim=-1))
        dist = nn.functional.softplus(self.dist_head(geo_feat)).squeeze(-1)
        vis_logit = self.vis_head(geo_feat).squeeze(-1)
        return dist, vis_logit

    def forward_color(self, points: torch.Tensor, dirs: torch.Tensor):
        """Color-only forward at known surface points (for rendering)."""
        p = self.pos_encoder(points)
        d = sinusoidal_encoding(dirs, self.dir_freqs)
        pd = torch.cat([p, d], dim=-1)
        geo_feat = self.geo_trunk(pd).detach()
        color_in = torch.cat([pd, geo_feat], dim=-1)
        color_feat = self.color_trunk(color_in)
        return torch.sigmoid(self.rgb_head(color_feat))
