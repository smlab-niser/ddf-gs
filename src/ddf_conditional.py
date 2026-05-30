"""Conditional DDF: a single MLP modulated by a per-object latent ``z``.

This is the multi-object generalisation of :class:`src.ddf_model.DDF`. Instead
of training one MLP per object we maintain ONE MLP plus a learned ``z`` (a
small dense vector) per object. The MLP is conditioned on ``z`` through
**FiLM** (Feature-wise Linear Modulation, Perez et al. 2017): each hidden
layer's pre-activation is rescaled and shifted by ``(gamma_l, beta_l)`` that
are themselves produced by a small linear projection from ``z``.

Auto-decoder paradigm (DeepSDF, Park et al. 2019): the per-object ``z`` is
NOT produced by an encoder. It is a learnable embedding optimised jointly
with the MLP at train time; at test time on a new object we keep the MLP
frozen and optimise a fresh ``z`` against a small supervision budget.

Inputs/outputs match :class:`DDF`:
    forward(points, dirs, z)
      points: (..., 3), dirs: (..., 3), z: (..., z_dim) — usually one
        z per object, broadcast/gathered over batched rays.
      returns: (dist, vis_logit) each (...,).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ddf_model import sinusoidal_encoding


class FiLMLinear(nn.Module):
    """Linear layer with FiLM modulation: ``y = (1 + gamma) * (W x + b) + beta``.

    ``gamma`` and ``beta`` come from a per-layer linear projection of the
    latent ``z``. We add a +1 to ``gamma`` so that ``gamma=0`` reduces to the
    identity modulation (helpful at init).
    """

    def __init__(self, in_dim: int, out_dim: int, z_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.film = nn.Linear(z_dim, 2 * out_dim)
        # Init FiLM projection so that (gamma, beta) start near zero -> identity.
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        gb = self.film(z)
        gamma, beta = gb.chunk(2, dim=-1)
        return (1.0 + gamma) * h + beta


class DDFCond(nn.Module):
    """Conditional DDF (shared MLP + per-object latent z, FiLM-modulated).

    Same sinusoidal encodings as the baseline DDF; latent code is broadcast
    into every hidden layer via FiLM. Defaults chosen to roughly match the
    baseline DDF in trunk capacity (256x6) so the comparison is fair.
    """

    def __init__(
        self,
        pos_freqs: int = 10,
        dir_freqs: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 6,
        z_dim: int = 64,
    ):
        super().__init__()
        self.pos_freqs = pos_freqs
        self.dir_freqs = dir_freqs
        self.z_dim = z_dim

        pos_dim = 3 + 3 * 2 * pos_freqs
        dir_dim = 3 + 3 * 2 * dir_freqs
        in_dim = pos_dim + dir_dim

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                FiLMLinear(in_dim if i == 0 else hidden_dim, hidden_dim, z_dim)
            )
        self.dist_head = nn.Linear(hidden_dim, 1)
        self.vis_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, points: torch.Tensor, dirs: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p = sinusoidal_encoding(points, self.pos_freqs)
        d = sinusoidal_encoding(dirs, self.dir_freqs)
        h = torch.cat([p, d], dim=-1)
        # If z is per-batch (B, z_dim) and inputs are (N, ...), allow broadcast.
        if z.dim() == 1:
            z = z.unsqueeze(0).expand(h.shape[0], -1)
        elif z.shape[0] == 1 and h.shape[0] != 1:
            z = z.expand(h.shape[0], -1)
        for layer in self.layers:
            h = F.relu(layer(h, z), inplace=False)
        dist = F.softplus(self.dist_head(h)).squeeze(-1)
        vis_logit = self.vis_head(h).squeeze(-1)
        return dist, vis_logit
