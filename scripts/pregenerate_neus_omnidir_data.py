"""Pre-generate a ray-cast cache from a trained NeuS SDF (image-trained, NO mesh)
using OMNIDIRECTIONAL surface-anchored rays.

A DDF trained on this cache becomes a *mesh-free* secondary-ray (shadow/AO)
oracle. This closes the circularity of the GT-mesh oracle
(`src/gtmesh_supervisor.py`, 95% shadow agreement): real GS scenes have no
mesh, but NeuS comes from images, so a NeuS-omnidir-trained DDF is mesh-free.

The OMNIDIR ray distribution mirrors `scripts/pregenerate_gtmesh_cache.py`
(the recipe that gave 95% shadow agreement):
  - surface-anchored rays: near-surface pt p, fully-uniform random unit dir u
    (= normalize(randn(3))), origin = p + uniform(0.01,0.05)*u, dir = u
  - outside-in rays: origin on sphere r ~ uniform(1.5,3.0), dir = -radial +
    0.2*randn, normalized
plus frustum-primary rays (reused from `generate_view_rays`) so the SAME DDF
also traces primary rays cleanly.

CRITICAL self-hit fix: surface-anchored origins sit only 0.01-0.05 off the
surface, so the SDF sphere-trace would immediately "converge" on the origin
itself and report a spurious near-zero occluder (over-shadowing). We seed the
trace with t_min = t_self (~0.06) so the march starts already skipped past the
origin and reports the NEXT surface. The printed hit-distance histogram must
NOT spike near 0 if this fix is working.

Saved cache has EXACTLY the keys CachedSupervisor expects:
  origins (N,3) float32, dirs (N,3) float32, t_gt (N,) float32 (0 on miss),
  hit_gt (N,) bool.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/pregenerate_neus_omnidir_data.py \
      --obj bull --n_rays 8000000 --out runs/bull_ddf_neus_omnidir/cache.pt
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
def _sdf_chunked(field, pts, chunk=65536):
    """Signed (not abs) SDF over pts, chunked."""
    out = []
    for i in range(0, pts.shape[0], chunk):
        out.append(field.forward_geonetwork(pts[i:i + chunk])[:, 0])
    return torch.cat(out)


@torch.no_grad()
def sphere_trace_batch(field, origins, dirs, max_iter=96, eps=1e-3, t_far=5.0,
                       t_min=None):
    """Sphere-trace rays through a NeuS SDF.

    t_min (optional, shape (N,) or scalar): seed the marched distance, so the
    trace starts already past the origin. Used for surface-anchored rays to
    skip the origin's own surface (the self-hit fix).
    """
    N = origins.shape[0]
    t = torch.zeros(N, device=origins.device)
    if t_min is not None:
        if not torch.is_tensor(t_min):
            t_min = torch.full((N,), float(t_min), device=origins.device)
        t = t + t_min
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


def find_surface_points(field, bbox_half, device, grid_res=128):
    """Find near-zero-set points on a grid, then ONE Newton snap onto the
    zero-set: p <- p - SDF(p)*grad/|grad|, grad via finite differences."""
    lin = torch.linspace(-bbox_half, bbox_half, grid_res, device=device)
    gx, gy, gz = torch.meshgrid(lin, lin, lin, indexing="ij")
    pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    sdf = _sdf_chunked(field, pts)

    mask = sdf.abs() < 0.05
    if int(mask.sum()) < 1000:
        mask = sdf.abs() < 0.1
    surf_pts = pts[mask]

    # ONE Newton snap onto the zero-set using finite-difference gradient.
    eps = 0.005
    snapped = []
    for i in range(0, surf_pts.shape[0], 8192):
        p = surf_pts[i:i + 8192]
        dx = torch.zeros_like(p); dx[:, 0] = eps
        dy = torch.zeros_like(p); dy[:, 1] = eps
        dz = torch.zeros_like(p); dz[:, 2] = eps
        s = field.forward_geonetwork(p)[:, 0]
        gx_ = field.forward_geonetwork(p + dx)[:, 0] - field.forward_geonetwork(p - dx)[:, 0]
        gy_ = field.forward_geonetwork(p + dy)[:, 0] - field.forward_geonetwork(p - dy)[:, 0]
        gz_ = field.forward_geonetwork(p + dz)[:, 0] - field.forward_geonetwork(p - dz)[:, 0]
        grad = torch.stack([gx_, gy_, gz_], dim=-1) / (2 * eps)
        gnorm = grad.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        p_snap = p - s.unsqueeze(-1) * grad / gnorm
        snapped.append(p_snap)
    return torch.cat(snapped) if snapped else surf_pts


def generate_view_rays(field, image_size, elev, azim, radius, device, chunk=65536):
    """Frustum primary rays from a spherical camera, traced through the SDF."""
    w2c, K, c2w = _spherical_camera(elev, azim, radius, image_size, device=device)
    origins, dirs = _pixels_to_world_rays(K, c2w, image_size, device)
    o = origins.reshape(-1, 3)
    d = dirs.reshape(-1, 3)
    t_parts, h_parts = [], []
    for i in range(0, o.shape[0], chunk):
        t_c, h_c = sphere_trace_batch(field, o[i:i + chunk], d[i:i + chunk])
        t_parts.append(t_c)
        h_parts.append(h_c)
    return o, d, torch.cat(t_parts), torch.cat(h_parts)


def gen_surface_anchored(field, surf_pts, n, t_self, device, chunk=65536):
    """OMNIDIR surface-anchored rays (mirror of gtmesh recipe + self-hit fix).

    p = random near-surface pt; u = normalize(randn(3)) fully-uniform dir;
    origin = p + uniform(0.01,0.05)*u; dir = u. Trace seeded with t_min=t_self.
    """
    idx = torch.randint(0, surf_pts.shape[0], (n,), device=device)
    p = surf_pts[idx]
    u = torch.randn(n, 3, device=device)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    offset = torch.empty(n, device=device).uniform_(0.01, 0.05)
    origins = p + offset.unsqueeze(-1) * u
    dirs = u
    t_parts, h_parts = [], []
    for i in range(0, n, chunk):
        t_c, h_c = sphere_trace_batch(
            field, origins[i:i + chunk], dirs[i:i + chunk], t_min=t_self)
        t_parts.append(t_c)
        h_parts.append(h_c)
    return origins, dirs, torch.cat(t_parts), torch.cat(h_parts)


def gen_outside_in(field, n, device, chunk=65536):
    """Outside-in rays: origin on sphere r~uniform(1.5,3.0), dir=-radial+0.2*randn."""
    u = torch.randn(n, 3, device=device)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    radii = torch.empty(n, 1, device=device).uniform_(1.5, 3.0)
    origins = u * radii
    dirs = -u + torch.randn(n, 3, device=device) * 0.2
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    t_parts, h_parts = [], []
    for i in range(0, n, chunk):
        t_c, h_c = sphere_trace_batch(field, origins[i:i + chunk], dirs[i:i + chunk])
        t_parts.append(t_c)
        h_parts.append(h_c)
    return origins, dirs, torch.cat(t_parts), torch.cat(h_parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--neus_config", default=None)
    ap.add_argument("--n_rays", type=int, default=8_000_000)
    ap.add_argument("--bbox_half", type=float, default=1.2)
    ap.add_argument("--t_self", type=float, default=0.06)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_path = args.out or f"runs/{args.obj}_ddf_neus_omnidir/cache.pt"

    # Idempotent: skip if cache already exists.
    if Path(out_path).exists():
        print(f"exists: {out_path}")
        return

    neus_cfg = args.neus_config or NEUS_CONFIGS.get(args.obj)
    if neus_cfg is None:
        raise SystemExit(
            f"No NeuS config for {args.obj}. Available: {sorted(NEUS_CONFIGS)}. "
            f"Pass --neus_config.")

    from nerfstudio.utils.eval_utils import eval_setup
    cfg, pipeline, _, step = eval_setup(Path(neus_cfg))
    field = pipeline.model.field
    field.eval()
    for p in field.parameters():
        p.requires_grad_(False)
    print(f"Loaded NeuS step={step}")

    surf_pts = find_surface_points(field, args.bbox_half, args.device)
    print(f"Found {surf_pts.shape[0]:,} surface points (Newton-snapped)")

    # Ray mix: ~45% surface-anchored omnidir, ~15% outside-in, ~40% frustum.
    n_total = int(args.n_rays)
    n_surf = int(round(n_total * 0.45))
    n_out = int(round(n_total * 0.15))
    n_frust = n_total - n_surf - n_out

    all_o, all_d, all_t, all_h = [], [], [], []
    t0 = time.time()

    # --- Surface-anchored omnidir rays (with self-hit fix) ---
    gen_chunk = 1_000_000
    done = 0
    while done < n_surf:
        m = min(gen_chunk, n_surf - done)
        o, d, t, h = gen_surface_anchored(field, surf_pts, m, args.t_self, args.device)
        all_o.append(o.cpu()); all_d.append(d.cpu())
        all_t.append(t.cpu()); all_h.append(h.cpu())
        done += m
        print(f"  surface {done:,}/{n_surf:,}  hit={h.float().mean().item():.3f}",
              flush=True)

    # --- Outside-in rays ---
    done = 0
    while done < n_out:
        m = min(gen_chunk, n_out - done)
        o, d, t, h = gen_outside_in(field, m, args.device)
        all_o.append(o.cpu()); all_d.append(d.cpu())
        all_t.append(t.cpu()); all_h.append(h.cpu())
        done += m
        print(f"  outside-in {done:,}/{n_out:,}  hit={h.float().mean().item():.3f}",
              flush=True)

    # --- Frustum-primary rays (random cameras) ---
    rays_per_view = args.image_size * args.image_size
    done = 0
    while done < n_frust:
        elev = float(torch.empty(()).uniform_(-30, 60).item())
        azim = float(torch.empty(()).uniform_(0, 360).item())
        radius = float(torch.empty(()).uniform_(1.5, 3.0).item())
        o, d, t, h = generate_view_rays(
            field, args.image_size, elev, azim, radius, args.device)
        take = min(rays_per_view, n_frust - done)
        all_o.append(o[:take].cpu()); all_d.append(d[:take].cpu())
        all_t.append(t[:take].cpu()); all_h.append(h[:take].cpu())
        done += take
        if (done // rays_per_view) % 25 == 0 or done >= n_frust:
            print(f"  frustum {done:,}/{n_frust:,}", flush=True)

    origins = torch.cat(all_o)
    dirs = torch.cat(all_d)
    t_gt = torch.cat(all_t)
    hit_gt = torch.cat(all_h)

    # t_gt = 0 on miss (CachedSupervisor convention).
    t_gt = torch.where(hit_gt, t_gt, torch.zeros_like(t_gt)).float()
    origins = origins.float()
    dirs = dirs.float()
    hit_gt = hit_gt.bool()

    n = origins.shape[0]
    n_hits = int(hit_gt.sum())
    print(f"\nTotal: {n:,} rays, {n_hits:,} hits ({n_hits / n:.1%})")

    # Hit-distance histogram summary — verify NO spike at ~0 (self-hit fix).
    hit_t = t_gt[hit_gt]
    if hit_t.numel() > 0:
        qs = torch.tensor([0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95, 1.0])
        qvals = torch.quantile(hit_t, qs.to(hit_t.dtype))
        print("Hit-distance quantiles:")
        for q, v in zip(qs.tolist(), qvals.tolist()):
            print(f"  q{q:0.2f} = {v:.4f}")
        frac_lt_self = float((hit_t < args.t_self).float().mean())
        print(f"  median hit distance = {qvals[5].item():.4f}")
        print(f"  fraction of hits below t_self ({args.t_self}) = {frac_lt_self:.4f}")
        if qvals[5].item() < 0.1:
            print("  WARNING: median hit distance < 0.1 — possible self-hit spike!")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "origins": origins,   # (N,3) float32
        "dirs": dirs,         # (N,3) float32
        "t_gt": t_gt,         # (N,) float32, 0 on miss
        "hit_gt": hit_gt,     # (N,) bool
        "obj": args.obj,
        "t_self": args.t_self,
        "bbox_half": args.bbox_half,
    }, out_path)
    sz = Path(out_path).stat().st_size / 1e6
    print(f"\nSaved {out_path} ({sz:.0f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
