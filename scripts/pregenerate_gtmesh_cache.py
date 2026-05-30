"""Pre-generate embree ray-cast supervision data for a DDF (CPU-only).

Casts a large pool (~8-10M) of OMNIDIRECTIONAL surface-anchored rays against
the frame-correct GT mesh and saves the exact nearest-hit distances. This is
the OMNIDIR oracle used for secondary/shadow rays. Training then reads from
this cache via CachedSupervisor (GPU-bound), instead of raycasting every step
(CPU-bound).

Reuses GTMeshSupervisor for frame-correct mesh loading + the embree
intersector. The saved cache has EXACTLY the keys CachedSupervisor expects:
  origins (N,3) float32, dirs (N,3) float32, t_gt (N,) float32, hit_gt (N,) bool.

CPU-only: never touches a GPU.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gtmesh_supervisor import GTMeshSupervisor


def _str2bool(s: str) -> bool:
    return str(s).lower() in ("true", "1", "yes", "y", "t")


def _gen_omnidir_rays(mesh, n, outside_ratio=0.30):
    """Generate omnidirectional surface-anchored + outside-in rays.

    Matches GTMeshSupervisor's default (frustum_ratio=0) sampling:
      - surface rays: surface pt + uniform(0.01,0.05)*unit_dir, dir = unit_dir
      - outside-in rays: origin on sphere r in [1.5,3.0], dir = -radial + jitter
    """
    import trimesh

    n_out = int(round(n * outside_ratio))
    n_surf = n - n_out

    # Surface-anchored rays
    surf_pts, _ = trimesh.sample.sample_surface(mesh, n_surf)
    surf_pts = np.asarray(surf_pts, dtype=np.float32)
    offset = np.random.uniform(0.01, 0.05, (n_surf, 1)).astype(np.float32)
    surf_dirs = np.random.randn(n_surf, 3).astype(np.float32)
    surf_dirs /= (np.linalg.norm(surf_dirs, axis=1, keepdims=True) + 1e-8)
    surf_origins = surf_pts + offset * surf_dirs

    # Outside-in rays
    out_dirs_unit = np.random.randn(n_out, 3).astype(np.float32)
    out_dirs_unit /= (np.linalg.norm(out_dirs_unit, axis=1, keepdims=True) + 1e-8)
    radii = np.random.uniform(1.5, 3.0, (n_out, 1)).astype(np.float32)
    out_origins = out_dirs_unit * radii
    out_dirs = -out_dirs_unit + np.random.randn(n_out, 3).astype(np.float32) * 0.2
    out_dirs /= (np.linalg.norm(out_dirs, axis=1, keepdims=True) + 1e-8)

    origins = np.concatenate([surf_origins, out_origins], axis=0)
    dirs = np.concatenate([surf_dirs, out_dirs], axis=0)
    return origins, dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--gt_mesh", required=True)
    ap.add_argument("--gs_path", required=True)
    ap.add_argument("--gso_rotate", type=_str2bool, default=False)
    ap.add_argument("--n_rays", type=int, default=8_000_000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk", type=int, default=500_000)
    args = ap.parse_args()

    out_path = args.out or f"runs/{args.obj}_gtmesh_cache/cache.pt"

    # Idempotent: out file acts as a done-marker for the watchdog.
    if os.path.exists(out_path):
        print(f"exists: {out_path}")
        return

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # mesh_center / mesh_scale from the object's gaussians file
    gs = torch.load(args.gs_path, map_location="cpu", weights_only=False)
    mc = gs["mesh_center"].numpy().astype(np.float32)
    ms = float(gs["mesh_scale"].item())
    print(f"mesh_center={mc.tolist()} mesh_scale={ms}")

    # Reuse GTMeshSupervisor for frame-correct mesh loading + embree intersector.
    # frustum_ratio=0 -> omnidirectional sampling path. device='cpu' (no GPU).
    sup = GTMeshSupervisor(
        gt_mesh_path=args.gt_mesh,
        mesh_center=mc,
        mesh_scale=ms,
        device="cpu",
        frustum_ratio=0.0,
        gso_rotate=args.gso_rotate,
    )

    n_total = int(args.n_rays)
    chunk = int(args.chunk)

    origins_all = np.empty((n_total, 3), dtype=np.float32)
    dirs_all = np.empty((n_total, 3), dtype=np.float32)
    t_all = np.zeros(n_total, dtype=np.float32)
    hit_all = np.zeros(n_total, dtype=bool)

    done = 0
    next_report = 1_000_000
    n_hit_running = 0
    while done < n_total:
        m = min(chunk, n_total - done)
        o, d = _gen_omnidir_rays(sup.mesh, m)
        t_np, hit_np = sup._cast(o, d)

        origins_all[done:done + m] = o
        dirs_all[done:done + m] = d
        t_all[done:done + m] = np.where(hit_np, t_np, 0.0)
        hit_all[done:done + m] = hit_np

        n_hit_running += int(hit_np.sum())
        done += m

        if done >= next_report or done >= n_total:
            print(f"  {done:,}/{n_total:,} rays  hit_rate={n_hit_running/done:.3f}")
            next_report += 1_000_000

    cache = {
        "origins": torch.from_numpy(origins_all),          # (N,3) float32
        "dirs": torch.from_numpy(dirs_all),                # (N,3) float32
        "t_gt": torch.from_numpy(t_all),                   # (N,) float32
        "hit_gt": torch.from_numpy(hit_all),               # (N,) bool
    }
    torch.save(cache, out_path)

    sz_mb = os.path.getsize(out_path) / 1e6
    print(f"saved {out_path}  rays={n_total:,}  "
          f"hit_rate={n_hit_running/n_total:.3f}  size={sz_mb:.1f} MB")


if __name__ == "__main__":
    main()
