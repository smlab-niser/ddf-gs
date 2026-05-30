"""Sanity-render a fitted/converted 3DGS via gsplat RGB+ED.

Renders a few orbit cameras and writes a 2xN grid of RGB and ED-depth panes
so we can verify the .ply was loaded correctly (no obviously broken
geometry, opacities, color channels).
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Pin GPU before torch CUDA init (script can be run with CUDA_VISIBLE_DEVICES
# already set; this is a no-op in that case).
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from gsplat import rasterization  # noqa: E402

# Re-use the same OpenCV spherical-camera builder used by the GSSupervisor so
# the convention matches everywhere in the repo.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.gs_supervisor import _spherical_camera  # noqa: E402


@torch.no_grad()
def render_views(gs_ckpt: Path, out_path: Path, image_size: int = 256,
                 radius: float = 2.5, n_views: int = 4, device: str = "cuda"):
    ckpt = torch.load(gs_ckpt, map_location=device, weights_only=False)
    quats = ckpt["quats"].to(device)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    means = ckpt["means"].to(device).contiguous()
    scales = ckpt["scales"].to(device).exp().contiguous()
    opacities = ckpt["opacities"].to(device).sigmoid().contiguous()
    colors = ckpt["colors"].to(device).sigmoid().contiguous()
    print(f"loaded {means.shape[0]} Gaussians")

    rows_rgb = []
    rows_depth = []
    for i in range(n_views):
        azim = 360.0 * i / n_views
        elev = 15.0 if i % 2 == 0 else -10.0
        w2c, K, _ = _spherical_camera(elev, azim, radius, image_size, device=device)
        renders, alphas, _ = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=w2c.unsqueeze(0), Ks=K.unsqueeze(0),
            width=image_size, height=image_size, sh_degree=None,
            render_mode="RGB+ED",
        )
        rgb = renders[0, ..., :3].clamp(0, 1).cpu().numpy()
        depth = renders[0, ..., 3].cpu().numpy()
        alpha = alphas[0, ..., 0].cpu().numpy()
        # White-bg compose:
        rgb = rgb + (1.0 - alpha[..., None]) * 1.0
        rgb = np.clip(rgb, 0, 1)
        # Visualize depth as inferno-like by inverse normalization on hit pixels.
        d_vis = np.zeros_like(depth)
        hit = alpha > 0.1
        if hit.any():
            d_hit = depth[hit]
            lo, hi = float(np.percentile(d_hit, 2)), float(np.percentile(d_hit, 98))
            if hi > lo:
                d_vis[hit] = np.clip((depth[hit] - lo) / (hi - lo), 0, 1)
        rows_rgb.append(rgb)
        rows_depth.append(np.stack([d_vis, d_vis, d_vis], axis=-1))
        print(f"  view {i}: azim={azim:.0f}° elev={elev:.0f}° hit_frac={hit.mean():.3f} "
              f"depth_range=[{float(depth[hit].min()) if hit.any() else 0:.3f}, "
              f"{float(depth[hit].max()) if hit.any() else 0:.3f}]")

    rgb_grid = np.concatenate(rows_rgb, axis=1)
    depth_grid = np.concatenate(rows_depth, axis=1)
    img = np.concatenate([rgb_grid, depth_grid], axis=0)
    img8 = (img * 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img8).save(out_path)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--radius", type=float, default=2.5)
    ap.add_argument("--n_views", type=int, default=4)
    args = ap.parse_args()
    render_views(args.gs, args.out, image_size=args.image_size,
                 radius=args.radius, n_views=args.n_views)


if __name__ == "__main__":
    main()
