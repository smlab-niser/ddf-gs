"""DDFKMix2: K=2 mixture DDF with **per-mode** distance + visibility heads.

Motivation
----------
Stage 14 (``ddf_mixture.py``) tried to predict K distances + scalar mixture
weights and a *shared* visibility, then meshed the argmax-mode UDF. The
extra distance head was free to drift far from the surface (the weighted
"primary" mode it tracked at any voxel kept the loss happy), and the iso
band spread badly → bull CD regressed 0.117 → 0.268.

This module takes the **opposite** approach:

* Two **independent** distance heads ``d_1, d_2`` (softplus).
* Two **independent** visibility heads ``xi_1, xi_2`` (sigmoid).
* **No mixture weight.** The two modes are not competing components of one
  distribution; they are two separate surfaces along the same ray.
* Surfaces are extracted **independently** from each mode and unioned in
  post (see ``scripts/stage3_chamfer_kmix.py``).
* Mode ordering ``d_1 < d_2`` (first/near vs second/far surface) is enforced
  at loss time by swapping if needed; downstream code can treat ``d_1`` as
  the "near hit" (the only surface a GS depth supervisor can see) and
  ``d_2`` as the "back wall" of hollow/through-object rays.

Target geometry
---------------
Mug-like hollow topology — outside-in ray pierces outer wall (``d_1``),
travels through the interior, then re-enters and hits the inner wall
(``d_2``). The KS-only single-distance model averages these two and
collapses the mug to a fat solid (CD 0.193); the hope is that giving the
model a dedicated head for the second surface lets it carve the cavity.

API
---
``forward(points, dirs)`` returns ``(d1, d2, xi1_logit, xi2_logit)``.
``forward_legacy(points, dirs)`` returns ``(d1, xi1_logit)`` so the
existing ``query_udf`` / training paths that expect a single ``(t, vis)``
pair can still call this model and get the primary surface only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ddf_hashgrid import HashGridEncoder
from .ddf_model import sinusoidal_encoding


class DDFKMix2(nn.Module):
    """DDF with hash-grid encoder + two independent (distance, visibility) heads.

    Parameters mirror :class:`~src.ddf_hashgrid.DDFHashGrid`. Architecture:
    shared encoder + trunk, then a single ``Linear(hidden_dim, 4)`` head
    producing ``(d1_raw, d2_raw, xi1_logit, xi2_logit)``.
    """

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

        # 4 outputs: (d1_raw, d2_raw, xi1_logit, xi2_logit).
        # Two distance heads kept as a single Linear for compactness; the only
        # difference vs two-Linear is shared bias init. Both d outputs go
        # through softplus, both xi outputs are returned as logits (BCE-with-
        # logits in the loss).
        self.head = nn.Linear(hidden_dim, 4)

    def _trunk(self, points: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        p = self.pos_encoder(points)
        d = sinusoidal_encoding(dirs, self.dir_freqs)
        return self.trunk(torch.cat([p, d], dim=-1))

    def forward(self, points: torch.Tensor, dirs: torch.Tensor):
        """Returns (d1, d2, xi1_logit, xi2_logit), all (...,) tensors."""
        h = self._trunk(points, dirs)
        out = self.head(h)
        d1 = F.softplus(out[..., 0])
        d2 = F.softplus(out[..., 1])
        xi1_logit = out[..., 2]
        xi2_logit = out[..., 3]
        return d1, d2, xi1_logit, xi2_logit

    def forward_legacy(self, points: torch.Tensor, dirs: torch.Tensor):
        """Drop-in for the single-mode (DDFHashGrid) interface: returns ``(d1, xi1_logit)``.

        Useful for code paths that only care about the primary surface (e.g.
        the existing UDF→MC pipeline if you want to mesh only the near
        surface without unioning d_2).
        """
        d1, _d2, xi1_logit, _xi2_logit = self.forward(points, dirs)
        return d1, xi1_logit
