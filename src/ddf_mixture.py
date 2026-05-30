"""DDFMixture: K-mixture distance head + hash-grid encoding.

Background: the single-output distance head (DDFHashGrid) collapses ambiguous
rays where two valid hit-distances exist (e.g. a ray grazing a thin feature
like a horn or wing — the ray either hits the thin feature at t1 or passes
through to a farther surface at t2). Predicting a single d forces the network
to average the two, producing a blurred / lost feature.

The PDDF recipe (Aumentado-Armstrong 2022, arXiv:2112.05300) predicts K
candidate depths {d_i} with mixture weights {w_i} that sum to 1, plus a
shared visibility logit ξ. Final depth is argmax over w_i. K=2 is the
common default — enough capacity for the bimodal ambiguity, no more.

Implementation
--------------
* Same hash-grid encoder + MLP trunk as :class:`DDFHashGrid`.
* Head produces ``2K + 1`` outputs:
    - K softplus-activated depths   ``d_1..d_K`` (non-negative)
    - (K-1) sigmoid-activated weights for K=2 case ``w_1``  (so ``w_2 = 1 - w_1``)
    - 1 visibility logit ``ξ`` (no activation — caller does BCE-with-logits)
* :meth:`forward` returns ``(d_best, vis_logit)`` where ``d_best`` is the depth
  with the highest mixture weight on each ray. Drop-in compatible with the
  existing training / extraction code paths.
* :meth:`forward_all` exposes ``(d_all, w_all, vis_logit)`` for debugging
  or for losses that want to weight all components.

Notes
-----
The general K > 2 case would use a softmax over K weight logits; with K = 2
a single sigmoid keeps the parameter count minimal (``w_2 = 1 - w_1``) and
matches the PDDF default.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ddf_hashgrid import HashGridEncoder
from .ddf_model import sinusoidal_encoding


class DDFMixture(nn.Module):
    """DDF with hash-grid encoder + K-component mixture distance head.

    Parameters
    ----------
    K : int
        Number of mixture components. K=2 is the recommended default.
    All other args mirror :class:`DDFHashGrid`.
    """

    def __init__(
        self,
        K: int = 2,
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
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        self.K = K
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

        # K depths + (K-1) weight logits (because they sum to 1) + 1 visibility
        # For K=2 we use a single sigmoid for w_1; w_2 = 1 - w_1.
        # For K>=3 we'd use softmax over K weight logits — supported here too.
        self.n_weight_logits = 1 if K == 2 else K
        head_dim = K + self.n_weight_logits + 1
        self.head = nn.Linear(hidden_dim, head_dim)

    def _trunk(self, points: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        p = self.pos_encoder(points)
        d = sinusoidal_encoding(dirs, self.dir_freqs)
        return self.trunk(torch.cat([p, d], dim=-1))

    def _split_head(self, out: torch.Tensor):
        """Returns (d_all (..., K), w_all (..., K), vis_logit (...))."""
        K = self.K
        d_raw = out[..., :K]
        w_raw = out[..., K:K + self.n_weight_logits]
        vis_logit = out[..., -1]

        d_all = F.softplus(d_raw)
        if K == 1:
            w_all = torch.ones_like(d_all)
        elif K == 2:
            w1 = torch.sigmoid(w_raw[..., 0:1])
            w_all = torch.cat([w1, 1.0 - w1], dim=-1)
        else:
            w_all = F.softmax(w_raw, dim=-1)
        return d_all, w_all, vis_logit

    def forward_all(self, points: torch.Tensor, dirs: torch.Tensor):
        """Full mixture output: (d_all, w_all, vis_logit). Useful for debug."""
        h = self._trunk(points, dirs)
        out = self.head(h)
        return self._split_head(out)

    def forward(self, points: torch.Tensor, dirs: torch.Tensor):
        """Drop-in API: returns (d_best, vis_logit) where d_best = d_{argmax_i w_i}.

        Argmax is non-differentiable, so during training the loss path should
        prefer :meth:`forward_all` and choose the best-d / lowest-error
        component explicitly (see train_ddf.py mixture loss).
        """
        d_all, w_all, vis_logit = self.forward_all(points, dirs)
        # Gather d along the argmax-weight index (non-diff, fine for inference).
        if self.K == 1:
            d_best = d_all.squeeze(-1)
        else:
            idx = w_all.argmax(dim=-1, keepdim=True)  # (..., 1)
            d_best = d_all.gather(-1, idx).squeeze(-1)
        return d_best, vis_logit
