"""NeuS SDF supervisor: sphere-trace structured camera frustum rays through a
trained NeuS-facto SDF for clean DDF supervision.

Uses the same camera sampling strategy as GSSupervisor (spherical cameras,
pixel-grid rays, surface-anchored marching) but replaces GS-rendered depth
with sphere-traced NeuS SDF distances. This gives:
- Clean, noise-free distances (NeuS SDF is smooth)
- Same proven training distribution as the kitchen-sink recipe
- Ray-march self-consistency (23ddf Eq. 11) built in
"""

import math
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore", category=FutureWarning)

from .gs_supervisor import _spherical_camera, _pixels_to_world_rays


class NeuSSupervisor:

    def __init__(
        self,
        neus_config_path: str,
        device: str = "cuda",
        image_size: int = 128,
        bbox_half: float = 1.2,
        march_ratio: float = 0.5,
        surface_ratio: float = 0.3,
        st_max_iter: int = 64,
        st_eps: float = 1e-3,
        st_t_far: float = 5.0,
    ):
        self.device = device
        self.image_size = image_size
        self.bbox_half = bbox_half
        self.march_ratio = float(march_ratio)
        self.surface_ratio = float(surface_ratio)
        self.st_max_iter = st_max_iter
        self.st_eps = st_eps
        self.st_t_far = st_t_far

        from nerfstudio.utils.eval_utils import eval_setup
        cfg, pipeline, _, step = eval_setup(Path(neus_config_path))
        self.field = pipeline.model.field
        self.field.eval()
        for p in self.field.parameters():
            p.requires_grad_(False)
        print(f"NeuSSupervisor: loaded NeuS step={step}")

        self._precompute_surface_points()

    @torch.no_grad()
    def _query_sdf(self, points: torch.Tensor) -> torch.Tensor:
        return self.field.forward_geonetwork(points)[:, 0]

    @torch.no_grad()
    def _sphere_trace_batch(
        self, origins: torch.Tensor, dirs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N = origins.shape[0]
        t = torch.zeros(N, device=self.device)
        alive = torch.ones(N, dtype=torch.bool, device=self.device)

        for _ in range(self.st_max_iter):
            if not alive.any():
                break
            pts = origins[alive] + t[alive].unsqueeze(-1) * dirs[alive]
            sdf = self._query_sdf(pts).abs()
            t[alive] += sdf
            converged = sdf < self.st_eps
            escaped = t[alive] > self.st_t_far
            alive_idx = alive.nonzero(as_tuple=True)[0]
            alive[alive_idx[converged]] = False
            alive[alive_idx[escaped]] = False

        hit = t < self.st_t_far
        pts_final = origins + t.unsqueeze(-1) * dirs
        sdf_final = self._query_sdf(pts_final).abs()
        hit = hit & (sdf_final < self.st_eps * 10)
        return t, hit

    @torch.no_grad()
    def _precompute_surface_points(self, grid_res: int = 64):
        lin = torch.linspace(-self.bbox_half, self.bbox_half, grid_res, device=self.device)
        gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
        pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
        chunk = 65536
        sdf_all = []
        for i in range(0, pts.shape[0], chunk):
            sdf_all.append(self._query_sdf(pts[i:i+chunk]))
        sdf = torch.cat(sdf_all)
        near_surface = sdf.abs() < 0.05
        self.surface_pts = pts[near_surface]
        if self.surface_pts.shape[0] < 1000:
            near_surface = sdf.abs() < 0.1
            self.surface_pts = pts[near_surface]
        # Estimate normals via SDF gradient for oriented surface rays
        self.surface_normals = self._estimate_normals(self.surface_pts)
        print(f"NeuSSupervisor: {self.surface_pts.shape[0]} surface points, "
              f"image_size={self.image_size}")

    @torch.no_grad()
    def _estimate_normals(self, pts: torch.Tensor, eps: float = 0.005) -> torch.Tensor:
        normals = []
        chunk = 8192
        for i in range(0, pts.shape[0], chunk):
            p = pts[i:i+chunk]
            dx = torch.zeros_like(p); dx[:, 0] = eps
            dy = torch.zeros_like(p); dy[:, 1] = eps
            dz = torch.zeros_like(p); dz[:, 2] = eps
            nx = self._query_sdf(p + dx) - self._query_sdf(p - dx)
            ny = self._query_sdf(p + dy) - self._query_sdf(p - dy)
            nz = self._query_sdf(p + dz) - self._query_sdf(p - dz)
            n = torch.stack([nx, ny, nz], dim=-1)
            n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            normals.append(n)
        return torch.cat(normals)

    @torch.no_grad()
    def _view_to_rays(
        self, elev_deg: float, azim_deg: float, radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render one frustum view via sphere-tracing. Returns (origins, dirs, t_gt, hit)."""
        w2c, K, c2w = _spherical_camera(
            elev_deg, azim_deg, radius, self.image_size, device=self.device)
        origins_grid, dirs_grid = _pixels_to_world_rays(K, c2w, self.image_size, self.device)
        origins_flat = origins_grid.reshape(-1, 3)
        dirs_flat = dirs_grid.reshape(-1, 3)

        # Sphere-trace in chunks to avoid OOM on large images
        chunk = 16384
        t_parts, h_parts = [], []
        for i in range(0, origins_flat.shape[0], chunk):
            t_c, h_c = self._sphere_trace_batch(
                origins_flat[i:i+chunk], dirs_flat[i:i+chunk])
            t_parts.append(t_c)
            h_parts.append(h_c)
        t_gt = torch.cat(t_parts)
        hit_gt = torch.cat(h_parts)

        return origins_flat, dirs_flat, t_gt, hit_gt

    @torch.no_grad()
    def _surface_anchored_rays(
        self, n_rays: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cast rays from near-surface points in outward-normal + jittered directions."""
        idx = torch.randint(0, self.surface_pts.shape[0], (n_rays,), device=self.device)
        base_pts = self.surface_pts[idx]
        normals = self.surface_normals[idx]

        # Jitter origin slightly off the surface along the normal
        offset = torch.empty(n_rays, device=self.device).uniform_(0.01, 0.05)
        origins = base_pts + offset.unsqueeze(-1) * normals

        # Direction: outward normal + random perturbation
        jitter = torch.randn(n_rays, 3, device=self.device) * 0.5
        dirs = normals + jitter
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        t_gt, hit_gt = self._sphere_trace_batch(origins, dirs)
        return origins, dirs, t_gt, hit_gt

    @torch.no_grad()
    def sample(
        self, batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Outside-in view (like GS supervisor view 1)
        azim1 = float(torch.empty(()).uniform_(0.0, 360.0).item())
        elev1 = float(torch.empty(()).uniform_(-30.0, 60.0).item())
        r1 = float(torch.empty(()).uniform_(2.0, 3.0).item())
        o1, d1, t1, h1 = self._view_to_rays(elev1, azim1, r1)

        # Close-up view (like GS supervisor view 2)
        azim2 = float(torch.empty(()).uniform_(0.0, 360.0).item())
        elev2 = float(torch.empty(()).uniform_(-60.0, 75.0).item())
        r2 = float(torch.empty(()).uniform_(0.6, 1.2).item())
        o2, d2, t2, h2 = self._view_to_rays(elev2, azim2, r2)

        n_view = o1.shape[0] + o2.shape[0]

        # Surface-anchored rays
        r = max(0.0, min(0.999, self.surface_ratio))
        if r > 0:
            n_surf = int(round(n_view * r / max(1e-6, 1.0 - r)))
            o3, d3, t3, h3 = self._surface_anchored_rays(n_surf)
            origins_flat = torch.cat([o1, o2, o3])
            dirs_flat = torch.cat([d1, d2, d3])
            t_flat = torch.cat([t1, t2, t3])
            hit_flat = torch.cat([h1, h2, h3])
        else:
            origins_flat = torch.cat([o1, o2])
            dirs_flat = torch.cat([d1, d2])
            t_flat = torch.cat([t1, t2])
            hit_flat = torch.cat([h1, h2])

        n_pix = origins_flat.shape[0]
        if batch_size >= n_pix:
            idx = torch.randint(0, n_pix, (batch_size,), device=self.device)
        else:
            idx = torch.randperm(n_pix, device=self.device)[:batch_size]

        origins = origins_flat[idx].contiguous()
        dirs = dirs_flat[idx].contiguous()
        t_gt = t_flat[idx].contiguous()
        hit_gt = hit_flat[idx].contiguous()
        t_gt = torch.where(hit_gt, t_gt, torch.zeros_like(t_gt))

        # Ray-march self-consistency
        if self.march_ratio > 0 and hit_gt.any():
            n_march = int(round(batch_size * self.march_ratio))
            n_march = min(n_march, batch_size)
            if n_march > 0:
                hit_indices = hit_gt.nonzero(as_tuple=True)[0]
                source_idx = hit_indices[
                    torch.randint(0, hit_indices.numel(), (n_march,), device=self.device)]
                src_o = origins[source_idx]
                src_d = dirs[source_idx]
                src_t = t_gt[source_idx]
                s_frac = torch.empty(n_march, device=self.device).uniform_(0.05, 0.9)
                s = s_frac * src_t
                origins[-n_march:] = (src_o + s.unsqueeze(-1) * src_d).contiguous()
                dirs[-n_march:] = src_d.contiguous()
                t_gt[-n_march:] = (src_t - s).contiguous()
                hit_gt[-n_march:] = True

        return origins, dirs, t_gt, hit_gt
