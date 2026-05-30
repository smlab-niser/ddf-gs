"""MultiObjectSupervisor: picks a random object per call and yields rays + obj idx.

Wraps a list of :class:`GSSupervisor` instances (one per object). Each ``sample()``
call selects a uniformly-random training object, draws a ray batch from its
supervisor, and returns ``(origins, dirs, t_gt, hit_gt, obj_idx)`` where
``obj_idx`` is a scalar identifying which object the batch came from. The
training loop uses ``obj_idx`` to look up the per-object latent ``z``.

Memory note: each GSSupervisor holds ~5k Gaussians (5k * (3+4+3+1+3) * 4 B ~=
0.3 MB raw + bbox tensors). 29 objects = ~10 MB GS state on GPU, negligible.
The expensive thing per call is the gsplat raster pass (one per view, two
views per call), which is exactly what a per-object supervisor pays anyway.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

from .gs_supervisor import GSSupervisor


class MultiObjectSupervisor:
    def __init__(
        self,
        gs_paths: list[str | Path],
        image_size: int = 64,
        device: str = "cuda",
        surface_n_ratio: float | None = None,
        seed: int | None = None,
    ):
        self.gs_paths = [str(p) for p in gs_paths]
        self.device = device
        self.rng = random.Random(seed)
        self.supervisors: list[GSSupervisor] = []
        for p in self.gs_paths:
            self.supervisors.append(
                GSSupervisor(
                    gs_path=p,
                    image_size=image_size,
                    device=device,
                    surface_n_ratio=surface_n_ratio,
                )
            )
        self.num_objects = len(self.supervisors)

    def sample(self, batch_size: int):
        idx = self.rng.randrange(self.num_objects)
        sup = self.supervisors[idx]
        origins, dirs, t_gt, hit_gt = sup.sample(batch_size)
        return origins, dirs, t_gt, hit_gt, idx

    def sample_from(self, obj_idx: int, batch_size: int):
        """Targeted sample (used for per-object eval / test-time z optimisation)."""
        sup = self.supervisors[obj_idx]
        origins, dirs, t_gt, hit_gt = sup.sample(batch_size)
        return origins, dirs, t_gt, hit_gt
