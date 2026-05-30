"""Ray-query latency: DDF MLP forward vs gsplat depth render.

For each N in a sweep, time how long each method takes to produce depth at N
queries on the bull (5000 Gaussians). gsplat is given an HxW viewport so total
pixels >= N; we report per-ray timings. DDF accepts arbitrary ray bundles, so
N pairs of (origin, dir) are sampled and forwarded.
"""

import argparse
import math
import time
from pathlib import Path

import torch

from src.ddf_model import DDF
from src.gs_supervisor import GSSupervisor, _spherical_camera
from gsplat import rasterization


@torch.no_grad()
def time_ddf(model, n_rays: int, device: str, repeats: int = 5) -> float:
    o = torch.randn(n_rays, 3, device=device) * 0.8
    d = torch.randn(n_rays, 3, device=device)
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    for _ in range(2):
        model(o, d)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        model(o, d)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


@torch.no_grad()
def time_gsplat(sup: GSSupervisor, n_rays: int, device: str, repeats: int = 5) -> tuple[float, int]:
    side = int(math.ceil(math.sqrt(n_rays)))
    w2c, K, _ = _spherical_camera(20.0, 45.0, 2.5, side, device=device)
    for _ in range(2):
        rasterization(
            means=sup.means, quats=sup.quats, scales=sup.scales,
            opacities=sup.opacities, colors=sup.colors,
            viewmats=w2c.unsqueeze(0), Ks=K.unsqueeze(0),
            width=side, height=side, sh_degree=None, render_mode="RGB+ED",
        )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        rasterization(
            means=sup.means, quats=sup.quats, scales=sup.scales,
            opacities=sup.opacities, colors=sup.colors,
            viewmats=w2c.unsqueeze(0), Ks=K.unsqueeze(0),
            width=side, height=side, sh_degree=None, render_mode="RGB+ED",
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats, side * side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddf_ckpt", default="runs/bull_ddf_v3/ddf_final.pt")
    ap.add_argument("--gs_path", default="runs/bull/gaussians.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ckpt = torch.load(args.ddf_ckpt, map_location=args.device, weights_only=False)
    cfg = ckpt["cfg"]
    model = DDF(
        pos_freqs=cfg["model"]["pos_freqs"], dir_freqs=cfg["model"]["dir_freqs"],
        hidden_dim=cfg["model"]["hidden_dim"], num_layers=cfg["model"]["num_layers"],
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sup = GSSupervisor(args.gs_path, image_size=64, device=args.device)

    print(f"{'N':>10s}  {'DDF (ms)':>10s}  {'gsplat (ms)':>13s}  "
          f"{'DDF Mray/s':>11s}  {'gsplat Mray/s':>14s}  {'speedup':>9s}")

    for n in [1024, 10240, 102400, 1048576]:
        ddf_t = time_ddf(model, n, args.device)
        gs_t, gs_n = time_gsplat(sup, n, args.device)
        ddf_mrs = n / ddf_t / 1e6
        gs_mrs = gs_n / gs_t / 1e6
        speedup = (gs_t / gs_n) / (ddf_t / n)
        print(f"{n:>10d}  {ddf_t*1e3:>10.3f}  {gs_t*1e3:>13.3f}  "
              f"{ddf_mrs:>11.2f}  {gs_mrs:>14.2f}  {speedup:>8.2f}x")


if __name__ == "__main__":
    main()
