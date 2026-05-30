"""DDF inference variants for closing the latency gap vs gsplat.

All variants are *inference-only* tweaks on top of the baseline DDF in
``src.ddf_model.DDF``. Construction signatures are kept compatible so each
variant can be loaded from the same checkpoint as the baseline whenever
weights are bit-compatible (compiled, bf16). Variants that change shape
(smaller MLP) require retraining.

Variants:

- ``DDFCompiled``       : baseline DDF wrapped with ``torch.compile``.
- ``DDFBF16``           : baseline DDF with bf16 autocast inference.
- ``DDFSmall``          : smaller MLP (hidden=128 or 192, layers=4 or 5);
                          requires retraining from scratch.
- ``DDFSmallCompiled``  : ``DDFSmall`` wrapped with ``torch.compile``.
- ``DDFSmallBF16``      : ``DDFSmall`` with bf16 autocast.

The :func:`build_variant` factory returns an ``(callable, name)`` pair where
``callable(points, dirs) -> (dist, vis_logit)`` matches the baseline API.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from src.ddf_model import DDF, sinusoidal_encoding
from src.ddf_hashgrid import DDFHashGrid


# ---------------------------------------------------------------------------
# Smaller MLP DDF (needs retraining)
# ---------------------------------------------------------------------------

class DDFSmall(nn.Module):
    """Same architecture as ``DDF`` but the trunk width / depth are tunable.

    Keeping the encoder/heads identical lets us reuse the training loop verbatim;
    only the YAML config needs to change.
    """

    def __init__(
        self,
        pos_freqs: int = 10,
        dir_freqs: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 4,
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


# ---------------------------------------------------------------------------
# Inference wrappers
# ---------------------------------------------------------------------------

def _make_compiled(model: nn.Module, mode: str = "reduce-overhead") -> Callable:
    """Return a callable that wraps ``model`` in ``torch.compile``.

    ``torch.compile`` is picky about no_grad / inference_mode contexts and about
    consistent tensor shapes across calls; the bench script controls those.
    """
    model.eval()
    compiled = torch.compile(model, mode=mode, dynamic=False, fullgraph=False)
    return compiled


class BF16Wrapper(nn.Module):
    """Run the wrapped fp32 model under bf16 autocast for inference.

    Weights remain fp32 (preserving training-time accuracy); the autocast
    context casts activations + matmuls to bf16. Outputs are upcast back to
    fp32 so downstream code is unaffected.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, points: torch.Tensor, dirs: torch.Tensor):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            dist, vis = self.model(points, dirs)
        return dist.float(), vis.float()


# ---------------------------------------------------------------------------
# Variant factory
# ---------------------------------------------------------------------------

def _load_state(model: nn.Module, ckpt_path: str, device: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return ckpt


def build_baseline(ckpt_path: str, device: str = "cuda") -> tuple[nn.Module, dict]:
    """Construct a baseline DDF from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    m_cfg = cfg["model"]
    model = DDF(
        pos_freqs=m_cfg["pos_freqs"], dir_freqs=m_cfg["dir_freqs"],
        hidden_dim=m_cfg["hidden_dim"], num_layers=m_cfg["num_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def build_small(ckpt_path: str, device: str = "cuda") -> tuple[nn.Module, dict]:
    """Construct a DDFSmall from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    m_cfg = cfg["model"]
    model = DDFSmall(
        pos_freqs=m_cfg["pos_freqs"], dir_freqs=m_cfg["dir_freqs"],
        hidden_dim=m_cfg["hidden_dim"], num_layers=m_cfg["num_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def build_hashgrid(ckpt_path: str, device: str = "cuda") -> tuple[nn.Module, dict]:
    """Construct a DDFHashGrid from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    m_cfg = cfg["model"]
    model = DDFHashGrid(
        dir_freqs=m_cfg.get("dir_freqs", 4),
        hidden_dim=m_cfg.get("hidden_dim", 64),
        num_layers=m_cfg.get("num_layers", 2),
        n_levels=m_cfg.get("n_levels", 16),
        feat_dim=m_cfg.get("feat_dim", 2),
        log2_table_size=m_cfg.get("log2_table_size", 19),
        base_res=m_cfg.get("base_res", 16),
        growth=m_cfg.get("growth", 1.5),
        bbox_half=m_cfg.get("bbox_half", 1.2),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def build_variant(
    name: str,
    ckpt_path: str,
    device: str = "cuda",
    small_ckpt_path: Optional[str] = None,
    hashgrid_ckpt_path: Optional[str] = None,
) -> tuple[Callable, dict]:
    """Return (callable, ckpt-meta).

    ``name`` selects from:
      baseline, compiled, bf16, compiled_bf16,
      small, small_compiled, small_bf16, small_compiled_bf16,
      hashgrid, hashgrid_compiled, hashgrid_bf16, hashgrid_compiled_bf16
    """
    name = name.lower()

    if name.startswith("hashgrid"):
        if hashgrid_ckpt_path is None:
            raise ValueError("hashgrid_ckpt_path is required for hashgrid* variants")
        model, ckpt = build_hashgrid(hashgrid_ckpt_path, device)
    elif name.startswith("small"):
        if small_ckpt_path is None:
            raise ValueError("small_ckpt_path is required for small* variants")
        model, ckpt = build_small(small_ckpt_path, device)
    else:
        model, ckpt = build_baseline(ckpt_path, device)

    if name in ("baseline", "small", "hashgrid"):
        return model, ckpt
    if name in ("compiled", "small_compiled", "hashgrid_compiled"):
        return _make_compiled(model), ckpt
    if name in ("bf16", "small_bf16", "hashgrid_bf16"):
        return BF16Wrapper(model).to(device).eval(), ckpt
    if name in ("compiled_bf16", "small_compiled_bf16", "hashgrid_compiled_bf16"):
        wrapped = BF16Wrapper(model).to(device).eval()
        return _make_compiled(wrapped), ckpt
    raise ValueError(f"unknown variant: {name}")
