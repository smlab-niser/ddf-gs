"""Stage-1 supervisor: query depth from a fitted 3DGS via gsplat.

Each `sample()` call renders ONE low-resolution view at a random spherical
camera placement, then treats each pixel of that view as a single ray sample:
camera center -> ray origin, per-pixel world-space ray dir, expected-depth ->
t_gt, alpha>0.5 -> hit_gt.

Known limitation (deliberate Stage-1 simplification): we do NOT do the U/B/S
surface-anchored mixture. Origins are clustered at camera
centers and dirs are restricted to the view frustum. This biases supervision
toward outside-looking-in rays; rays starting near the surface or pointing
sideways are absent. Worth revisiting if convergence suffers.
"""

import math
from pathlib import Path

import torch
from gsplat import rasterization


def _spherical_camera(
    elev_deg: float,
    azim_deg: float,
    radius: float,
    image_size: int,
    fov_deg: float = 60.0,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one OpenCV camera looking at the origin. Returns (w2c, K, c2w).

    OpenCV convention: image x = right, y = down, camera z = forward into scene.
    """
    fov = math.radians(fov_deg)
    fx = fy = (image_size / 2.0) / math.tan(fov / 2.0)
    cx = cy = image_size / 2.0
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32, device=device,
    )

    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)
    eye = torch.tensor(
        [
            radius * math.cos(elev) * math.sin(azim),
            radius * math.sin(elev),
            radius * math.cos(elev) * math.cos(azim),
        ],
        dtype=torch.float32, device=device,
    )
    world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    f = -eye / eye.norm().clamp_min(1e-8)        # forward: eye -> origin
    r = torch.linalg.cross(f, world_up); r = r / r.norm().clamp_min(1e-8)
    d = torch.linalg.cross(f, r); d = d / d.norm().clamp_min(1e-8)  # image-down

    c2w = torch.eye(4, dtype=torch.float32, device=device)
    c2w[:3, 0] = r
    c2w[:3, 1] = d
    c2w[:3, 2] = f
    c2w[:3, 3] = eye

    w2c = torch.linalg.inv(c2w)
    return w2c, K, c2w


def _pixels_to_world_rays(
    K: torch.Tensor,
    c2w: torch.Tensor,
    image_size: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel ray origins (all = camera center) and unit dirs in world space.

    Convention: OpenCV pinhole (z forward, y down in image, x right).
    """
    ys, xs = torch.meshgrid(
        torch.arange(image_size, device=device, dtype=torch.float32),
        torch.arange(image_size, device=device, dtype=torch.float32),
        indexing="ij",
    )
    px = (xs + 0.5 - K[0, 2]) / K[0, 0]
    py = (ys + 0.5 - K[1, 2]) / K[1, 1]
    dirs_cam = torch.stack([px, py, torch.ones_like(px)], dim=-1)  # (H, W, 3)
    dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    R = c2w[:3, :3]
    t = c2w[:3, 3]
    dirs_world = dirs_cam @ R.T                       # (H, W, 3)
    origins = t.view(1, 1, 3).expand_as(dirs_world)   # (H, W, 3)
    return origins, dirs_world


class GSSupervisor:
    """Render-based depth supervisor for DDF distillation."""

    def __init__(
        self,
        gs_path: str,
        image_size: int = 64,
        device: str = "cuda",
        surface_n_ratio: float | None = None,
        surface_chunk: int | None = None,
        march_ratio: float = 0.0,
    ):
        """
        surface_n_ratio: target fraction of the *final batch* coming from
            surface-anchored rays. None reproduces the original behaviour
            (n_surf = max(batch_size // 2, 1024) -> ~20% of the pool with
            image_size=64 and batch=4096). 0.0 disables surface rays. Any
            value in [0, 1] picks the surface count so that, after the
            uniform subsample over the (view1 + view2 + surface) pool,
            surface rays make up ~surface_n_ratio of the batch.
        surface_chunk: when set, chunk the surface-anchored 1x1 batched
            rasterization into groups of this many cameras. Needed at scene
            scale (hundreds of thousands of Gaussians) because gsplat's
            tile-intersection buffer scales as O(N_cam * N_gauss) and OOMs
            otherwise. None = single call (legacy behaviour for small scenes).
        """
        self.device = device
        self.image_size = image_size
        self.surface_n_ratio = surface_n_ratio
        self.surface_chunk = surface_chunk
        # Ray-march self-consistency: replace march_ratio fraction of the batch with
        # marched-along-ray copies of hit rays. For hit ray (o, d, t_gt), sample s in
        # U(0, 0.9*t_gt) and emit (o + s*d, d, t_gt - s, hit=True). Enforces the inherent
        # DDF identity DDF(o + s*d, d) = DDF(o, d) - s along rays. 23ddf-style.
        self.march_ratio = float(march_ratio)

        ckpt = torch.load(Path(gs_path), map_location=device, weights_only=False)

        quats = ckpt["quats"].to(device)
        quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        self.means = ckpt["means"].to(device).contiguous()
        self.quats = quats.contiguous()
        self.scales = ckpt["scales"].to(device).exp().contiguous()
        self.opacities = ckpt["opacities"].to(device).sigmoid().contiguous()
        self.colors = ckpt["colors"].to(device).sigmoid().contiguous()

        self.bbox_min = ckpt["bbox_min"].to(device)
        self.bbox_max = ckpt["bbox_max"].to(device)

    @torch.no_grad()
    def _render_one_view(
        self,
        elev_deg: float,
        azim_deg: float,
        radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        depth = renders[0, ..., 3]      # (H, W) expected z-depth
        alpha = alphas[0, ..., 0]       # (H, W)
        return depth, alpha, K, c2w

    @torch.no_grad()
    def _surface_anchored_rays(self, n: int, jitter: float = 0.03):
        """Anchor n rays near GS means and query depth via batched 1x1 gsplat renders.

        Each ray = (origin near a Gaussian center, random unit dir). Build n viewmats
        (one per ray) and rasterize a 1x1 image per camera; the single pixel's ED depth
        is t_gt along that ray. With cx=cy=0.5 the center pixel coincides with the
        principal point, so the per-pixel ray direction is exactly the camera forward
        (= our chosen dir), making the t_gt-along-ray identity exact (no cos correction).
        """
        idx = torch.randint(0, self.means.shape[0], (n,), device=self.device)
        origins = self.means[idx] + jitter * torch.randn(n, 3, device=self.device)

        dirs = torch.randn(n, 3, device=self.device)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        world_up = torch.zeros(n, 3, device=self.device)
        world_up[:, 1] = 1.0
        parallel = (dirs * world_up).sum(-1).abs() > 0.99
        fallback = torch.zeros(n, 3, device=self.device); fallback[:, 0] = 1.0
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
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            renders, alphas, _ = rasterization(
                means=self.means, quats=self.quats, scales=self.scales,
                opacities=self.opacities, colors=self.colors,
                viewmats=w2c[s:e], Ks=Ks[s:e], width=1, height=1,
                sh_degree=None, render_mode="RGB+ED",
            )
            t_gt_list.append(renders[:, 0, 0, 3])
            hit_list.append(alphas[:, 0, 0, 0] > 0.5)
        t_gt = torch.cat(t_gt_list, dim=0)
        hit_gt = torch.cat(hit_list, dim=0)
        t_gt = torch.where(hit_gt, t_gt, torch.zeros_like(t_gt))
        return origins, dirs, t_gt, hit_gt

    @torch.no_grad()
    def _view_to_rays(self, elev_deg: float, azim_deg: float, radius: float):
        """Render one view; return flat (origins, dirs, t_gt, hit_gt) over all pixels."""
        depth, alpha, K, c2w = self._render_one_view(elev_deg, azim_deg, radius)
        origins_grid, dirs_grid = _pixels_to_world_rays(
            K, c2w, self.image_size, device=self.device
        )
        forward_world = c2w[:3, 2]
        cos_theta = (dirs_grid * forward_world.view(1, 1, 3)).sum(-1).clamp_min(1e-4)
        t_grid = depth / cos_theta
        hit_grid = alpha > 0.5
        n_pix = self.image_size * self.image_size
        o = origins_grid.reshape(n_pix, 3)
        d = dirs_grid.reshape(n_pix, 3)
        t = t_grid.reshape(n_pix)
        h = hit_grid.reshape(n_pix)
        return o, d, t, h

    @torch.no_grad()
    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Three-component ray pool: outside view + close-up view + surface-anchored rays.

        Two-view alone learns the convex hull (no surface-anchored grazing rays). Surface
        rays provide the missing inside-out signal so the DDF can carve concavities.
        """
        azim1 = float(torch.empty(()).uniform_(0.0, 360.0).item())
        elev1 = float(torch.empty(()).uniform_(-30.0, 60.0).item())
        r1 = float(torch.empty(()).uniform_(2.0, 3.0).item())
        o1, d1, t1, h1 = self._view_to_rays(elev1, azim1, r1)

        azim2 = float(torch.empty(()).uniform_(0.0, 360.0).item())
        elev2 = float(torch.empty(()).uniform_(-60.0, 75.0).item())
        r2 = float(torch.empty(()).uniform_(0.6, 1.2).item())
        o2, d2, t2, h2 = self._view_to_rays(elev2, azim2, r2)

        n_view = o1.shape[0] + o2.shape[0]  # 2 * image_size^2

        if self.surface_n_ratio is None:
            # Back-compat default.
            n_surf = max(batch_size // 2, 1024)
        else:
            r = float(self.surface_n_ratio)
            r = max(0.0, min(0.999, r))
            # After uniform subsample over (view1 + view2 + surface), the
            # surface fraction equals n_surf / (n_view + n_surf). Solve for
            # n_surf so that fraction == r.
            if r <= 0.0:
                n_surf = 0
            else:
                n_surf = int(round(n_view * r / max(1e-6, 1.0 - r)))

        if n_surf > 0:
            o3, d3, t3, h3 = self._surface_anchored_rays(n_surf)
            origins_flat = torch.cat([o1, o2, o3], dim=0)
            dirs_flat = torch.cat([d1, d2, d3], dim=0)
            t_flat = torch.cat([t1, t2, t3], dim=0)
            hit_flat = torch.cat([h1, h2, h3], dim=0)
        else:
            origins_flat = torch.cat([o1, o2], dim=0)
            dirs_flat = torch.cat([d1, d2], dim=0)
            t_flat = torch.cat([t1, t2], dim=0)
            hit_flat = torch.cat([h1, h2], dim=0)
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

        # Ray-march self-consistency: replace march_ratio fraction with marched
        # copies of hit rays. Forces DDF(o + s*d, d) = t_gt - s along each ray.
        if self.march_ratio > 0 and hit_gt.any():
            n_march = int(round(batch_size * self.march_ratio))
            n_march = min(n_march, batch_size)
            if n_march > 0:
                hit_indices = hit_gt.nonzero(as_tuple=True)[0]
                source_idx = hit_indices[torch.randint(0, hit_indices.numel(),
                                                       (n_march,), device=self.device)]
                src_o = origins[source_idx]
                src_d = dirs[source_idx]
                src_t = t_gt[source_idx]
                # Random fraction in [0.05, 0.9] of the way to the surface — avoid
                # exact 0 (trivial) and exact t_gt (degenerate vis).
                s_frac = torch.empty(n_march, device=self.device).uniform_(0.05, 0.9)
                s = s_frac * src_t
                new_o = src_o + s.unsqueeze(-1) * src_d
                new_t = src_t - s
                # Overwrite the last n_march positions in the batch.
                origins[-n_march:] = new_o.contiguous()
                dirs[-n_march:] = src_d.contiguous()
                t_gt[-n_march:] = new_t.contiguous()
                hit_gt[-n_march:] = True

        return origins, dirs, t_gt, hit_gt
