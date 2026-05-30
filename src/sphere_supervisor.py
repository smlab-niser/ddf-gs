"""Stage-0 supervisor: analytical ray-sphere intersection.

Used to sanity-check the DDF model + training loop without GS in the loop.
Replace with `gs_supervisor.py` (Stage 1) once GS fitting is wired up.
"""

import torch


def sample_rays(
    batch_size: int,
    bbox_half: float = 1.5,
    device: str = "cuda",
):
    """Sample ray origins uniformly in a cube around origin, dirs uniformly on S^2."""
    origins = (torch.rand(batch_size, 3, device=device) * 2 - 1) * bbox_half
    dirs = torch.randn(batch_size, 3, device=device)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return origins, dirs


def ray_sphere_intersect(
    origins: torch.Tensor,
    dirs: torch.Tensor,
    radius: float = 1.0,
):
    """Returns (t, hit_mask). t is distance to first hit (0 if miss)."""
    b = (origins * dirs).sum(-1)
    c = (origins * origins).sum(-1) - radius * radius
    disc = b * b - c
    hit = disc > 0
    sqrt_disc = torch.sqrt(disc.clamp_min(0))
    t1 = -b - sqrt_disc
    t2 = -b + sqrt_disc
    t = torch.where(t1 > 0, t1, t2)
    hit = hit & (t > 0)
    t = torch.where(hit, t, torch.zeros_like(t))
    return t, hit
