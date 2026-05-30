"""Cached supervisor: loads pre-generated (origin, dir, t_gt, hit) data from
a .pt file and samples batches from it. No model queries during training —
as fast as the GS supervisor, with NeuS-quality supervision.

Ray-march self-consistency is applied on-the-fly (same as GSSupervisor).
"""

import torch
from pathlib import Path


class CachedSupervisor:

    def __init__(
        self,
        cache_path: str,
        device: str = "cuda",
        march_ratio: float = 0.5,
    ):
        self.device = device
        self.march_ratio = float(march_ratio)

        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.origins = data["origins"].to(device)
        self.dirs = data["dirs"].to(device)
        self.t_gt = data["t_gt"].to(device)
        self.hit_gt = data["hit_gt"].to(device)

        n = self.origins.shape[0]
        n_hit = self.hit_gt.sum().item()
        print(f"CachedSupervisor: {n:,} rays, {n_hit:,} hits ({n_hit/n:.1%}), "
              f"march_ratio={self.march_ratio}")

        self.hit_indices = self.hit_gt.nonzero(as_tuple=True)[0]

    def sample(
        self, batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.origins.shape[0], (batch_size,), device=self.device)
        origins = self.origins[idx].clone()
        dirs = self.dirs[idx].clone()
        t_gt = self.t_gt[idx].clone()
        hit_gt = self.hit_gt[idx].clone()
        t_gt = torch.where(hit_gt, t_gt, torch.zeros_like(t_gt))

        if self.march_ratio > 0 and self.hit_indices.numel() > 0:
            n_march = int(round(batch_size * self.march_ratio))
            if n_march > 0:
                src_idx = self.hit_indices[
                    torch.randint(0, self.hit_indices.numel(), (n_march,), device=self.device)]
                src_o = self.origins[src_idx]
                src_d = self.dirs[src_idx]
                src_t = self.t_gt[src_idx]
                s_frac = torch.empty(n_march, device=self.device).uniform_(0.05, 0.9)
                s = s_frac * src_t
                origins[-n_march:] = src_o + s.unsqueeze(-1) * src_d
                dirs[-n_march:] = src_d
                t_gt[-n_march:] = src_t - s
                hit_gt[-n_march:] = True

        return origins, dirs, t_gt, hit_gt
