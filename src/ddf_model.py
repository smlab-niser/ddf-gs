"""DDF: maps (point, direction) -> (distance, hit_probability).

Architecture is a plain MLP with sinusoidal positional encoding on both
the 3D point and the unit direction. Two heads:
  - distance head: softplus output, regresses distance-to-surface along ray
  - visibility head: sigmoid output, predicts whether the ray hits a surface
"""

import torch
import torch.nn as nn


def sinusoidal_encoding(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)
    xb = x.unsqueeze(-1) * freqs
    return torch.cat([x, torch.sin(xb).flatten(-2), torch.cos(xb).flatten(-2)], dim=-1)


class DDF(nn.Module):
    def __init__(
        self,
        pos_freqs: int = 10,
        dir_freqs: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 6,
    ):
        super().__init__()
        self.pos_freqs = pos_freqs
        self.dir_freqs = dir_freqs

        pos_dim = 3 + 3 * 2 * pos_freqs
        dir_dim = 3 + 3 * 2 * dir_freqs
        in_dim = pos_dim + dir_dim

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(*layers)
        self.dist_head = nn.Linear(hidden_dim, 1)
        self.vis_head = nn.Linear(hidden_dim, 1)

    def forward(self, points: torch.Tensor, dirs: torch.Tensor):
        p = sinusoidal_encoding(points, self.pos_freqs)
        d = sinusoidal_encoding(dirs, self.dir_freqs)
        h = self.trunk(torch.cat([p, d], dim=-1))
        dist = torch.nn.functional.softplus(self.dist_head(h)).squeeze(-1)
        vis_logit = self.vis_head(h).squeeze(-1)
        return dist, vis_logit
