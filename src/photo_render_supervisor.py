"""Stage 16 supervisor: multi-view RGB images + cameras for NeuS-style volume rendering.

Loads the pre-rendered multi-view RGB images at ``<run>/views/rgb_*.png`` and
``cameras.npz`` (the same dataset NeuS-facto used in Stage 9/9b). Per ``sample``
call, picks a random subset of views, samples random pixels in each, and returns
the per-pixel rays + GT colours.

Additionally combines this with the standard depth supervision from
:class:`GSSupervisor` so the DDF distance head still gets the strong v3-style
ray-march signal during training. The volume-rendering signal is *extra* — it
provides RGB supervision via density compositing, not a replacement.

API:
    PhotoRenderSupervisor.sample(batch_size) ->
        depth_dict: (origins, dirs, t_gt, hit_gt)   # GS depth supervision
        rgb_dict:   (origins, dirs, rgb_gt, t_near, t_far)   # photo VR supervision

For rgb-side rays, ``t_near``/``t_far`` bracket the ray inside a bounding sphere
of radius ``bbox_half`` so the per-ray N=32-64 sample points land inside the
hash-grid bbox.

Background note: the GT renders are PNG (uint8 RGB, white background ≈
[1, 1, 1]). The volume rendering loop composites foreground colour over a
white background, so non-hit pixels naturally drive the integrated transmittance
to 1 and the rendered colour to white — matching the GT pixel.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .gs_supervisor import GSSupervisor


def _load_views(views_dir: Path, device: str):
    """Return (rgbs (V, H, W, 3) in [0,1], c2w (V, 4, 4), K (3, 3), H, W)."""
    cam = np.load(views_dir / "cameras.npz")
    c2w = torch.from_numpy(cam["c2w"]).float().to(device)        # (V, 4, 4)
    K = torch.from_numpy(cam["K"]).float().to(device)            # (3, 3)
    image_size = int(cam["image_size"])
    V = c2w.shape[0]

    rgbs = []
    for v in range(V):
        p = views_dir / f"rgb_{v:03d}.png"
        im = Image.open(p).convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
        rgbs.append(torch.from_numpy(arr))
    rgbs = torch.stack(rgbs, dim=0).to(device)                   # (V, H, W, 3)
    H, W = rgbs.shape[1], rgbs.shape[2]
    assert H == image_size and W == image_size, f"image_size {image_size} != PNG ({H}, {W})"
    return rgbs, c2w, K, H, W


def _ray_box_t_interval(
    origins: torch.Tensor, dirs: torch.Tensor, bbox_half: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Slab method: per-ray (t_near, t_far) clipping against an AABB [-bbox_half, bbox_half]^3.

    Returns (t_near, t_far). For rays that miss the box, t_far < t_near; caller
    should clamp t_near = max(t_near, 0) and gate with the (t_far > t_near + eps)
    mask. We don't filter here — return both and let the rendering composite over
    a no-op interval.
    """
    safe_d = torch.where(dirs.abs() > 1e-8, dirs, torch.full_like(dirs, 1e-8))
    inv_d = 1.0 / safe_d
    t1 = (-bbox_half - origins) * inv_d   # (B, 3)
    t2 = (bbox_half - origins) * inv_d    # (B, 3)
    tmin = torch.minimum(t1, t2).max(dim=-1).values   # (B,)
    tmax = torch.maximum(t1, t2).min(dim=-1).values   # (B,)
    return tmin.clamp_min(0.0), tmax


class PhotoRenderSupervisor:
    """Combined depth + multi-view RGB supervisor for NeuS-style VR training.

    Wraps a :class:`GSSupervisor` for the depth side (yields the same
    ``(origins, dirs, t_gt, hit_gt)`` tuple as Stage 1) and adds a separate
    sampler for multi-view RGB pixel rays.
    """

    def __init__(
        self,
        gs_path: str,
        views_dir: str,
        image_size: int = 256,
        device: str = "cuda",
        surface_n_ratio: float | None = None,
        surface_chunk: int | None = None,
        march_ratio: float = 0.0,
        bbox_half: float = 1.2,
        n_rgb_rays: int = 1024,
    ):
        """
        bbox_half: half-side of the AABB the hash grid covers. Ray samples
            outside this box have no defined DDF, so we clip rays to it.
        n_rgb_rays: number of multi-view RGB pixel rays per ``sample()``. Each
            ray will be sampled at N points along [t_near, t_far] for volume
            rendering (handled in the training loop, not here).
        """
        self.device = device
        self.bbox_half = bbox_half
        self.n_rgb_rays = n_rgb_rays

        # Depth-supervision branch — identical to GS path used by hashgrid KS.
        self.depth_sup = GSSupervisor(
            gs_path=gs_path,
            image_size=image_size,
            device=device,
            surface_n_ratio=surface_n_ratio,
            surface_chunk=surface_chunk,
            march_ratio=march_ratio,
        )

        # Photometric branch — pre-render dataset on disk.
        v_dir = Path(views_dir)
        if not v_dir.exists():
            raise FileNotFoundError(f"views_dir not found: {v_dir}")
        rgbs, c2w, K, H, W = _load_views(v_dir, device=device)
        self.rgbs = rgbs            # (V, H, W, 3) float in [0,1]
        self.c2w = c2w              # (V, 4, 4)
        self.K = K                  # (3, 3)
        self.V = rgbs.shape[0]
        self.Hpx = H
        self.Wpx = W

        # Cameras.npz already stores c2w in the SAME normalised frame as the GS
        # (eye norms ≈ stored radius = 2.5). mesh_center / mesh_scale are only
        # used elsewhere to map the *GT mesh* back into this frame. So we use
        # the c2w as-is.
        self.c2w_norm = self.c2w.clone()

        # Pre-build per-pixel camera-frame ray directions (OpenCV pinhole), shape
        # (H, W, 3) — same for every view; only c2w changes.
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        px = (xs + 0.5 - K[0, 2]) / K[0, 0]
        py = (ys + 0.5 - K[1, 2]) / K[1, 1]
        dirs_cam = torch.stack([px, py, torch.ones_like(px)], dim=-1)
        dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        self.dirs_cam = dirs_cam   # (H, W, 3)

    @torch.no_grad()
    def sample_depth(self, batch_size: int):
        return self.depth_sup.sample(batch_size)

    @torch.no_grad()
    def sample_rgb_rays(self, n: int | None = None):
        """Return (origins, dirs, rgb_gt, t_near, t_far) for n random multi-view rays."""
        n = self.n_rgb_rays if n is None else n
        # Random view + pixel per ray.
        view_idx = torch.randint(0, self.V, (n,), device=self.device)
        py_idx = torch.randint(0, self.Hpx, (n,), device=self.device)
        px_idx = torch.randint(0, self.Wpx, (n,), device=self.device)

        # GT colours.
        rgb_gt = self.rgbs[view_idx, py_idx, px_idx]            # (n, 3)

        # Direction in camera frame -> world.
        dirs_cam = self.dirs_cam[py_idx, px_idx]                # (n, 3)
        c2w_sel = self.c2w_norm[view_idx]                       # (n, 4, 4)
        R = c2w_sel[:, :3, :3]                                  # (n, 3, 3)
        t = c2w_sel[:, :3, 3]                                   # (n, 3)
        dirs_world = torch.bmm(dirs_cam.unsqueeze(1), R.transpose(1, 2)).squeeze(1)
        dirs_world = dirs_world / dirs_world.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        origins = t                                             # (n, 3)

        # AABB-clip the rays so the N sample points lie inside the hash-grid bbox.
        t_near, t_far = _ray_box_t_interval(origins, dirs_world, self.bbox_half)
        # Clamp to a sensible minimum interval (rays barely grazing the box get
        # near==far; the rendering loop will produce alpha=0 on those, which
        # composites to white background — which matches the GT for backgrounds).
        return origins, dirs_world, rgb_gt, t_near, t_far

    @torch.no_grad()
    def sample(self, batch_size: int):
        """Convenience: depth-side sample of size batch_size + RGB rays.

        Returns ``(depth_origins, depth_dirs, t_gt, hit_gt,
                   rgb_origins, rgb_dirs, rgb_gt, t_near, t_far)``.
        """
        o, d, t, h = self.sample_depth(batch_size)
        ro, rd, rgb, tn, tf = self.sample_rgb_rays()
        return o, d, t, h, ro, rd, rgb, tn, tf
