"""Pre-generate DDF training data by sphere-tracing through NeuS SDF.

Generates millions of (origin, dir, t_gt, hit) tuples from diverse camera
views and surface-anchored rays, saves to a .pt file. Training then loads
this cache — no NeuS queries during training, same speed as GS-supervised.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/pregenerate_neus_data.py \
      --obj bull --n_views 500 --image_size 128 --out runs/bull_ddf_neus_v3/cache.pt
"""
import argparse
import math
import sys
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gs_supervisor import _spherical_camera, _pixels_to_world_rays


def _get_neus_configs():
    configs = {}
    base = Path("runs/sota_comparison_30k")
    if base.exists():
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            latest = d / "neus" / "latest_run.txt"
            if latest.exists():
                configs[d.name] = latest.read_text().strip()
    return configs

NEUS_CONFIGS = _get_neus_configs()


@torch.no_grad()
def sphere_trace_batch(field, origins, dirs, max_iter=64, eps=1e-3, t_far=5.0):
    N = origins.shape[0]
    t = torch.zeros(N, device=origins.device)
    alive = torch.ones(N, dtype=torch.bool, device=origins.device)
    for _ in range(max_iter):
        if not alive.any():
            break
        pts = origins[alive] + t[alive].unsqueeze(-1) * dirs[alive]
        sdf = field.forward_geonetwork(pts)[:, 0].abs()
        t[alive] += sdf
        converged = sdf < eps
        escaped = t[alive] > t_far
        idx = alive.nonzero(as_tuple=True)[0]
        alive[idx[converged]] = False
        alive[idx[escaped]] = False
    hit = t < t_far
    pts_final = origins + t.unsqueeze(-1) * dirs
    sdf_final = field.forward_geonetwork(pts_final)[:, 0].abs()
    hit = hit & (sdf_final < eps * 10)
    return t, hit


def generate_view_rays(field, image_size, elev, azim, radius, device, chunk=16384):
    w2c, K, c2w = _spherical_camera(elev, azim, radius, image_size, device=device)
    origins, dirs = _pixels_to_world_rays(K, c2w, image_size, device)
    o = origins.reshape(-1, 3)
    d = dirs.reshape(-1, 3)
    t_parts, h_parts = [], []
    for i in range(0, o.shape[0], chunk):
        t_c, h_c = sphere_trace_batch(field, o[i:i+chunk], d[i:i+chunk])
        t_parts.append(t_c)
        h_parts.append(h_c)
    return o, d, torch.cat(t_parts), torch.cat(h_parts)


def generate_surface_rays(field, surface_pts, surface_normals, n_rays, device):
    idx = torch.randint(0, surface_pts.shape[0], (n_rays,), device=device)
    pts = surface_pts[idx]
    normals = surface_normals[idx]
    offset = torch.empty(n_rays, device=device).uniform_(0.01, 0.05)
    origins = pts + offset.unsqueeze(-1) * normals
    jitter = torch.randn(n_rays, 3, device=device) * 0.5
    dirs = normals + jitter
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    t, hit = sphere_trace_batch(field, origins, dirs)
    return origins, dirs, t, hit


def find_surface_points(field, bbox_half, device, grid_res=64):
    lin = torch.linspace(-bbox_half, bbox_half, grid_res, device=device)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    sdf_all = []
    for i in range(0, pts.shape[0], 65536):
        sdf_all.append(field.forward_geonetwork(pts[i:i+65536])[:, 0])
    sdf = torch.cat(sdf_all)
    mask = sdf.abs() < 0.05
    surf_pts = pts[mask]
    # Estimate normals
    eps = 0.005
    normals = []
    for i in range(0, surf_pts.shape[0], 8192):
        p = surf_pts[i:i+8192]
        dx = torch.zeros_like(p); dx[:, 0] = eps
        dy = torch.zeros_like(p); dy[:, 1] = eps
        dz = torch.zeros_like(p); dz[:, 2] = eps
        nx = field.forward_geonetwork(p + dx)[:, 0] - field.forward_geonetwork(p - dx)[:, 0]
        ny = field.forward_geonetwork(p + dy)[:, 0] - field.forward_geonetwork(p - dy)[:, 0]
        nz = field.forward_geonetwork(p + dz)[:, 0] - field.forward_geonetwork(p - dz)[:, 0]
        n = torch.stack([nx, ny, nz], dim=-1)
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        normals.append(n)
    return surf_pts, torch.cat(normals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--neus_config", default=None)
    ap.add_argument("--n_views", type=int, default=500)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--surface_rays_per_view", type=int, default=2000)
    ap.add_argument("--bbox_half", type=float, default=1.2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    neus_cfg = args.neus_config or NEUS_CONFIGS.get(args.obj)
    if neus_cfg is None:
        raise SystemExit(f"No NeuS config for {args.obj}. Pass --neus_config.")

    from nerfstudio.utils.eval_utils import eval_setup
    cfg, pipeline, _, step = eval_setup(Path(neus_cfg))
    field = pipeline.model.field
    field.eval()
    for p in field.parameters():
        p.requires_grad_(False)
    print(f"Loaded NeuS step={step}")

    surf_pts, surf_normals = find_surface_points(field, args.bbox_half, args.device)
    print(f"Found {surf_pts.shape[0]} surface points")

    all_o, all_d, all_t, all_h = [], [], [], []
    t0 = time.time()

    for i in range(args.n_views):
        # Random camera (mix of outside-in and close-up)
        if i % 3 == 0:
            elev = float(torch.empty(()).uniform_(-60, 75).item())
            radius = float(torch.empty(()).uniform_(0.6, 1.2).item())
        else:
            elev = float(torch.empty(()).uniform_(-30, 60).item())
            radius = float(torch.empty(()).uniform_(2.0, 3.0).item())
        azim = float(torch.empty(()).uniform_(0, 360).item())

        o, d, t, h = generate_view_rays(field, args.image_size, elev, azim, radius, args.device)
        all_o.append(o.cpu())
        all_d.append(d.cpu())
        all_t.append(t.cpu())
        all_h.append(h.cpu())

        # Surface rays
        if args.surface_rays_per_view > 0:
            so, sd, st, sh = generate_surface_rays(
                field, surf_pts, surf_normals, args.surface_rays_per_view, args.device)
            all_o.append(so.cpu())
            all_d.append(sd.cpu())
            all_t.append(st.cpu())
            all_h.append(sh.cpu())

        if (i + 1) % 50 == 0:
            n_total = sum(o.shape[0] for o in all_o)
            n_hits = sum(h.sum().item() for h in all_h)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (args.n_views - i - 1)
            print(f"view {i+1}/{args.n_views}: {n_total:,} rays, "
                  f"{n_hits:,} hits ({n_hits/n_total:.1%}), "
                  f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)

    origins = torch.cat(all_o)
    dirs = torch.cat(all_d)
    t_gt = torch.cat(all_t)
    hit_gt = torch.cat(all_h)

    n_total = origins.shape[0]
    n_hits = hit_gt.sum().item()
    print(f"\nTotal: {n_total:,} rays, {n_hits:,} hits ({n_hits/n_total:.1%})")
    print(f"Wall time: {time.time()-t0:.0f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "origins": origins,
        "dirs": dirs,
        "t_gt": t_gt,
        "hit_gt": hit_gt,
        "obj": args.obj,
        "n_views": args.n_views,
        "image_size": args.image_size,
    }, args.out)
    sz = Path(args.out).stat().st_size / 1e6
    print(f"Saved {args.out} ({sz:.0f} MB)")


if __name__ == "__main__":
    main()
