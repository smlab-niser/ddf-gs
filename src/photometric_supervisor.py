"""Stage 13 supervisor: depth + per-pixel ground-truth RGB from the GS.

Wraps the Stage-1 GSSupervisor logic but additionally returns the GS-rendered
RGB at each ray's hit pixel as a target colour for a parallel RGB head on the
DDF. Cheap construction: the existing 64x64 ``render_mode='RGB+ED'`` call
already produces RGB in channels [0:3] alongside ED depth in channel [3]. We
just stop discarding the RGB.

Per-surface-point RGB is a much weaker signal than NeuS volume rendering, but
it's almost free given we already pay for the rasterization. The DDF learns
a per-(point, dir) colour that should match the GS at hit points; the gradient
back through the hash grid is an additional structural cue on geometry-correlated
appearance.

API:
    PhotometricSupervisor.sample(batch_size) -> (origins, dirs, t_gt, hit_gt, rgb_gt)

``rgb_gt`` is (B, 3) in [0, 1]. For non-hits it is set to zero; the training
loop must mask the RGB loss by ``hit_gt``.
"""

from __future__ import annotations

import math

import torch
from gsplat import rasterization

from .gs_supervisor import (
    GSSupervisor,
    _pixels_to_world_rays,
    _spherical_camera,
)


class PhotometricSupervisor(GSSupervisor):
    """GSSupervisor + RGB target per ray (channels [0:3] of the same render)."""

    @torch.no_grad()
    def _render_one_view_rgb(
        self,
        elev_deg: float,
        azim_deg: float,
        radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like parent's _render_one_view but additionally returns RGB (H,W,3)."""
        w2c, K, c2w = _spherical_camera(
            elev_deg, azim_deg, radius, self.image_size, device=self.device
        )
        renders, alphas, _ = rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.colors,
            viewmats=w2c.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=self.image_size,
            height=self.image_size,
            sh_degree=None,
            render_mode="RGB+ED",
        )
        rgb = renders[0, ..., :3].clamp(0.0, 1.0)  # (H, W, 3)
        depth = renders[0, ..., 3]                  # (H, W)
        alpha = alphas[0, ..., 0]                   # (H, W)
        return rgb, depth, alpha, K, c2w

    @torch.no_grad()
    def _view_to_rays_rgb(self, elev_deg: float, azim_deg: float, radius: float):
        """Render one view; flat (origins, dirs, t_gt, hit_gt, rgb_gt) over all pixels."""
        rgb, depth, alpha, K, c2w = self._render_one_view_rgb(elev_deg, azim_deg, radius)
        origins_grid, dirs_grid = _pixels_to_world_rays(
            K, c2w, self.image_size, device=self.device
        )
        forward_world = c2w[:3, 2]
        cos_theta = (dirs_grid * forward_world.view(1, 1, 3)).sum(-1).clamp_min(1e-4)
        t_grid = depth / cos_theta
        hit_grid = alpha > 0.5
        n_pix = self.image_size * self.image_size
        return (
            origins_grid.reshape(n_pix, 3),
            dirs_grid.reshape(n_pix, 3),
            t_grid.reshape(n_pix),
            hit_grid.reshape(n_pix),
            rgb.reshape(n_pix, 3),
        )

    @torch.no_grad()
    def _surface_anchored_rays_rgb(self, n: int, jitter: float = 0.03):
        """Surface-anchored rays + per-ray ground-truth RGB from a 1x1 GS render."""
        idx = torch.randint(0, self.means.shape[0], (n,), device=self.device)
        origins = self.means[idx] + jitter * torch.randn(n, 3, device=self.device)

        dirs = torch.randn(n, 3, device=self.device)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        world_up = torch.zeros(n, 3, device=self.device)
        world_up[:, 1] = 1.0
        parallel = (dirs * world_up).sum(-1).abs() > 0.99
        fallback = torch.zeros(n, 3, device=self.device)
        fallback[:, 0] = 1.0
        world_up = torch.where(parallel.unsqueeze(-1), fallback, world_up)

        right = torch.linalg.cross(dirs, world_up, dim=-1)
        right = right / right.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        down = torch.linalg.cross(dirs, right, dim=-1)
        down = down / down.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        c2w = torch.zeros(n, 4, 4, device=self.device)
        c2w[:, :3, 0] = right
        c2w[:, :3, 1] = down
        c2w[:, :3, 2] = dirs
        c2w[:, :3, 3] = origins
        c2w[:, 3, 3] = 1.0
        w2c = torch.linalg.inv(c2w)

        fx = fy = 0.5 / math.tan(math.radians(60.0) / 2.0)
        K = torch.tensor(
            [[fx, 0.0, 0.5], [0.0, fy, 0.5], [0.0, 0.0, 1.0]],
            dtype=torch.float32, device=self.device,
        )
        Ks = K.unsqueeze(0).expand(n, -1, -1).contiguous()

        chunk = self.surface_chunk if self.surface_chunk and self.surface_chunk > 0 else n
        t_gt_list = []
        hit_list = []
        rgb_list = []
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            renders, alphas, _ = rasterization(
                means=self.means, quats=self.quats, scales=self.scales,
                opacities=self.opacities, colors=self.colors,
                viewmats=w2c[s:e], Ks=Ks[s:e], width=1, height=1,
                sh_degree=None, render_mode="RGB+ED",
            )
            rgb_list.append(renders[:, 0, 0, :3].clamp(0.0, 1.0))
            t_gt_list.append(renders[:, 0, 0, 3])
            hit_list.append(alphas[:, 0, 0, 0] > 0.5)
        t_gt = torch.cat(t_gt_list, dim=0)
        hit_gt = torch.cat(hit_list, dim=0)
        rgb_gt = torch.cat(rgb_list, dim=0)
        t_gt = torch.where(hit_gt, t_gt, torch.zeros_like(t_gt))
        return origins, dirs, t_gt, hit_gt, rgb_gt

    @torch.no_grad()
    def sample(self, batch_size: int):
        azim1 = float(torch.empty(()).uniform_(0.0, 360.0).item())
        elev1 = float(torch.empty(()).uniform_(-30.0, 60.0).item())
        r1 = float(torch.empty(()).uniform_(2.0, 3.0).item())
        o1, d1, t1, h1, c1 = self._view_to_rays_rgb(elev1, azim1, r1)

        azim2 = float(torch.empty(()).uniform_(0.0, 360.0).item())
        elev2 = float(torch.empty(()).uniform_(-60.0, 75.0).item())
        r2 = float(torch.empty(()).uniform_(0.6, 1.2).item())
        o2, d2, t2, h2, c2 = self._view_to_rays_rgb(elev2, azim2, r2)

        n_view = o1.shape[0] + o2.shape[0]

        if self.surface_n_ratio is None:
            n_surf = max(batch_size // 2, 1024)
        else:
            r = max(0.0, min(0.999, float(self.surface_n_ratio)))
            n_surf = 0 if r <= 0.0 else int(round(n_view * r / max(1e-6, 1.0 - r)))

        if n_surf > 0:
            o3, d3, t3, h3, c3 = self._surface_anchored_rays_rgb(n_surf)
            origins_flat = torch.cat([o1, o2, o3], dim=0)
            dirs_flat = torch.cat([d1, d2, d3], dim=0)
            t_flat = torch.cat([t1, t2, t3], dim=0)
            hit_flat = torch.cat([h1, h2, h3], dim=0)
            rgb_flat = torch.cat([c1, c2, c3], dim=0)
        else:
            origins_flat = torch.cat([o1, o2], dim=0)
            dirs_flat = torch.cat([d1, d2], dim=0)
            t_flat = torch.cat([t1, t2], dim=0)
            hit_flat = torch.cat([h1, h2], dim=0)
            rgb_flat = torch.cat([c1, c2], dim=0)

        n_pix = origins_flat.shape[0]
        if batch_size >= n_pix:
            idx = torch.randint(0, n_pix, (batch_size,), device=self.device)
        else:
            idx = torch.randperm(n_pix, device=self.device)[:batch_size]

        origins = origins_flat[idx].contiguous()
        dirs = dirs_flat[idx].contiguous()
        t_gt = t_flat[idx].contiguous()
        hit_gt = hit_flat[idx].contiguous()
        rgb_gt = rgb_flat[idx].contiguous()
        t_gt = torch.where(hit_gt, t_gt, torch.zeros_like(t_gt))
        rgb_gt = torch.where(hit_gt.unsqueeze(-1), rgb_gt, torch.zeros_like(rgb_gt))
        return origins, dirs, t_gt, hit_gt, rgb_gt
