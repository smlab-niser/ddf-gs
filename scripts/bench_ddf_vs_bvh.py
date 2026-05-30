"""DECISIVE BENCHMARK: DDF (1 network eval/ray) vs BVH-over-Gaussians.

Both answer the same query: "find the first surface hit along this ray."
  - DDF:  ONE forward pass -> (distance, hit). Cost is FLAT in N_gaussians
          (constant params, constant memory): the network never sees the
          scene's Gaussian count.
  - BVH:  build an explicit acceleration structure over per-Gaussian ellipsoid
          proxies, then traverse it per ray. Cost RISES with N_gaussians
          (more primitives -> deeper/wider tree -> more node + triangle tests).

The BVH baseline is the algorithmic stand-in for ray-traced 3DGS pipelines —
3DGRT (Moenne-Loccoz et al., arXiv:2407.07090) and RaySplats
(arXiv:2501.19196) — which build per-Gaussian convex proxies and trace them on
RT cores. We measure the *hardware-agnostic O(N)-traversal cost* the DDF avoids;
we are NOT claiming to beat a tuned RT-core kernel in wall-clock. Embree (CPU,
all 96 cores) is the portable proxy. The DDF runs on its native GPU. To stay
hardware-invariant we ALSO report each method NORMALIZED to its own 5k value —
the SLOPE (DDF-flat vs BVH-rising) is the headline and is hardware-independent.

The proxy per Gaussian is an OCTAHEDRON hull (6 verts, 8 tris) sized to the
3-sigma anisotropic extent, rotated by the Gaussian's quaternion. One big
trimesh is assembled by direct offset-indexed array construction (NOT an
O(N^2) concatenate loop).

Usage:
  source activate auto-gs
  PYTHONPATH=. python scripts/bench_ddf_vs_bvh.py                 # full sweep
  PYTHONPATH=. python scripts/bench_ddf_vs_bvh.py --smoke         # 2x2 smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# --- pick the least-busy GPU BEFORE importing torch ------------------------
def _pick_gpu() -> str:
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        rows = []
        for line in out.strip().splitlines():
            idx, mem, util = [x.strip() for x in line.split(",")]
            rows.append((int(idx), int(mem), int(util)))
        rows.sort(key=lambda r: (r[2], r[1]))  # lowest util, tie-break lowest mem
        return str(rows[0][0])
    except Exception:
        return "0"


os.environ.setdefault("CUDA_VISIBLE_DEVICES", _pick_gpu())

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

# Reuse the EXACT DDF-column primitives from the SDF benchmark so the numbers
# line up with prior tables. `scripts/` is not a package, so load by path.
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location(
    "_bench_ddf_vs_sdf", str(Path(__file__).resolve().parent / "bench_ddf_vs_sdf.py"))
_bvs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_bvs)
make_rays, time_fn = _bvs.make_rays, _bvs.time_fn


N_GAUSSIANS_SWEEP = [5_000, 50_000, 200_000, 1_000_000]
N_RAYS_SWEEP = [1024, 10_240, 102_400, 1_048_576]

# Unit octahedron: 6 verts on the axes, 8 triangular faces. Vertices at +-1 on
# each axis so that scaling by (sx,sy,sz) gives a 3-sigma-extent diamond hull.
_OCTA_VERTS = np.array([
    [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
], dtype=np.float64)
_OCTA_FACES = np.array([
    [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
    [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
], dtype=np.int64)


def _quat_to_rotmat(quats: np.ndarray) -> np.ndarray:
    """(N,4) scalar-first [w,x,y,z] (gsplat convention) -> (N,3,3) rotation."""
    q = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_proxy_mesh(means, quats, log_scales, sigma=3.0):
    """Build ONE trimesh of octahedron proxies for all Gaussians.

    Per Gaussian: octahedron verts scaled by sigma*exp(log_scale) along each
    axis, rotated by the quaternion, translated to the mean. Vertex/face arrays
    are constructed directly with offset indexing — O(N), no concatenate loop.
    """
    means = np.asarray(means, dtype=np.float64)
    quats = np.asarray(quats, dtype=np.float64)
    scales = sigma * np.exp(np.asarray(log_scales, dtype=np.float64))  # 3-sigma std
    N = means.shape[0]
    R = _quat_to_rotmat(quats)  # (N,3,3)

    # Scale unit octahedron per Gaussian:  v_scaled[g,i,:] = octa[i] * scales[g]
    v = _OCTA_VERTS[None, :, :] * scales[:, None, :]          # (N, 6, 3)
    # Rotate:  v_rot[g,i,:] = R[g] @ v_scaled[g,i,:]
    v = np.einsum("gij,gkj->gki", R, v)                        # (N, 6, 3)
    v = v + means[:, None, :]                                  # translate
    verts = v.reshape(N * 6, 3)

    # Faces: tile the 8 base faces N times, offset by 6 per Gaussian.
    face_off = (np.arange(N) * 6)[:, None, None]               # (N,1,1)
    faces = (_OCTA_FACES[None, :, :] + face_off).reshape(N * 8, 3)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return mesh


def replicate_gaussians(g, n_target, jitter_scale=0.05, seed=0):
    """Tile base Gaussians with positional jitter to reach n_target (numpy).

    Mirrors bench_scene_scale._replicate_with_jitter: means jittered, the rest
    broadcast-tiled. First n_src kept exact. Returns (means, quats, log_scales).
    """
    means = np.asarray(g["means"], dtype=np.float64)
    quats = np.asarray(g["quats"], dtype=np.float64)
    log_scales = np.asarray(g["scales"], dtype=np.float64)
    n_src = means.shape[0]
    if n_target == n_src:
        return means, quats, log_scales

    reps = math.ceil(n_target / n_src)
    bbox_min = np.asarray(g["bbox_min"], dtype=np.float64)
    bbox_max = np.asarray(g["bbox_max"], dtype=np.float64)
    extent = float(np.linalg.norm(bbox_max - bbox_min))

    m = np.tile(means, (reps, 1))[:n_target].copy()
    q = np.tile(quats, (reps, 1))[:n_target].copy()
    s = np.tile(log_scales, (reps, 1))[:n_target].copy()

    rng = np.random.default_rng(seed)
    jitter = rng.standard_normal((n_target, 3)) * (jitter_scale * extent / math.sqrt(3))
    jitter[:n_src] = 0.0
    m = m + jitter
    return m, q, s


def time_embree_first(intersector, origins_np, dirs_np, reps=10):
    """Wall-clock median (ms) of intersects_first over `reps`. Also returns the
    hit-fraction from one call for a sanity check."""
    res = intersector.intersects_first(origins_np, dirs_np)
    hit_frac = float((res != -1).mean())
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        intersector.intersects_first(origins_np, dirs_np)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000.0, hit_frac


@torch.no_grad()
def load_ddf(device, ckpt_path="runs/bull_ddf_gtmesh/ddf_final.pt"):
    from src.ddf_hashgrid import DDFHashGrid
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    mc = ck["cfg"]["model"]
    ddf = DDFHashGrid(
        dir_freqs=mc.get("dir_freqs", 4), hidden_dim=mc.get("hidden_dim", 64),
        num_layers=mc.get("num_layers", 2), n_levels=mc.get("n_levels", 16),
        feat_dim=mc.get("feat_dim", 2), log2_table_size=mc.get("log2_table_size", 19),
        base_res=mc.get("base_res", 16), growth=mc.get("growth", 1.5),
        bbox_half=mc.get("bbox_half", 1.2),
    ).to(device).eval()
    ddf.load_state_dict(ck["model"])
    return ddf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs_path", default="runs/bull/gaussians.pt")
    ap.add_argument("--ddf_ckpt", default="runs/bull_ddf_gtmesh/ddf_final.pt")
    ap.add_argument("--out_json", default="runs/bench_ddf_vs_bvh.json")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--smoke", action="store_true",
                    help="small 2x2 sweep: N_gauss {5k,50k} x N_rays {1k,10k}")
    args = ap.parse_args()

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.smoke:
        n_gauss_sweep = [5_000, 50_000]
        n_rays_sweep = [1024, 10_240]
    else:
        n_gauss_sweep = list(N_GAUSSIANS_SWEEP)
        n_rays_sweep = list(N_RAYS_SWEEP)

    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"GPU: {torch.cuda.get_device_name(0)}  | CPU cores: {os.cpu_count()}")
    print(f"smoke={args.smoke}  N_gaussians={n_gauss_sweep}  N_rays={n_rays_sweep}\n")

    # ---- load base scene + DDF ----------------------------------------------
    g = torch.load(args.gs_path, map_location="cpu", weights_only=False)
    n_base = g["means"].shape[0]
    print(f"Base scene: {n_base} Gaussians from {args.gs_path}")

    ddf = load_ddf(device, args.ddf_ckpt)
    ddf_params = sum(p.numel() for p in ddf.parameters())
    ddf_bytes = sum(p.numel() * p.element_size() for p in ddf.parameters())
    print(f"DDF: DDFHashGrid {ddf_params:,} params, {ddf_bytes/1e6:.1f} MB "
          f"(SCENE-INDEPENDENT)\n")
    ddf_c = torch.compile(ddf, mode="reduce-overhead")

    # ---- DDF column: time once per N_rays (independent of N_gaussians) ------
    print("Timing DDF column (compiled+bf16, independent of N_gaussians)...")
    ddf_times = {}  # n_rays -> per-ray us
    ddf_ms = {}
    for n in n_rays_sweep:
        o, d = make_rays(n, device)

        def _run():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ddf_c(o, d)
        ms = time_fn(_run, reps=args.reps, warmup=5)
        ddf_times[n] = ms * 1e3 / n  # per-ray microseconds
        ddf_ms[n] = ms
        print(f"  N_rays={n:>9,}: DDF {ms:8.3f} ms total  ->  {ddf_times[n]:8.4f} us/ray")

    # ---- BVH column: per N_gaussians ----------------------------------------
    rows = []
    bvh_mem = {}    # n_gauss -> bytes
    build_times = {}
    bvh_per_ray = {}  # n_gauss -> {n_rays: us/ray}
    OCTA_BYTES_PER_GAUSS = 6 * 3 * 8 + 8 * 3 * 8  # 6 verts f64 + 8 faces i64 = 336 B

    for n_g in n_gauss_sweep:
        print(f"\n=== N_gaussians = {n_g:,} ===")
        means, quats, lscales = replicate_gaussians(g, n_g)

        t0 = time.perf_counter()
        mesh = build_proxy_mesh(means, quats, lscales, sigma=3.0)
        t_mesh = time.perf_counter() - t0
        # mem of the explicit geometry (verts f64 + faces i64), O(N).
        geom_bytes = mesh.vertices.nbytes + mesh.faces.nbytes
        print(f"  proxy mesh: {len(mesh.vertices):,} verts, {len(mesh.faces):,} tris "
              f"({t_mesh:.2f}s build, geom {geom_bytes/1e6:.1f} MB)")

        t0 = time.perf_counter()
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
        t_build = time.perf_counter() - t0
        build_times[n_g] = t_build
        print(f"  BVH build (embree): {t_build:.2f}s  [OUTSIDE timed region]")
        bvh_mem[n_g] = geom_bytes

        bvh_per_ray[n_g] = {}
        print(f"  {'N_rays':>10} | {'BVH ms':>10} | {'BVH us/ray':>11} | {'hit%':>6} "
              f"| {'DDF us/ray':>11}")
        print("  " + "-" * 60)
        for n in n_rays_sweep:
            o, d = make_rays(n, device)
            o_np = o.detach().cpu().numpy().astype(np.float64)
            d_np = d.detach().cpu().numpy().astype(np.float64)
            bvh_ms, hit_frac = time_embree_first(intersector, o_np, d_np, reps=args.reps)
            us_ray = bvh_ms * 1e3 / n
            bvh_per_ray[n_g][n] = us_ray
            print(f"  {n:>10,} | {bvh_ms:>9.2f} | {us_ray:>10.4f} | {hit_frac*100:>5.1f} "
                  f"| {ddf_times[n]:>10.4f}")
            rows.append({
                "n_gaussians": n_g, "n_rays": n,
                "bvh_ms": bvh_ms, "bvh_us_per_ray": us_ray, "bvh_hit_frac": hit_frac,
                "ddf_ms": ddf_ms[n], "ddf_us_per_ray": ddf_times[n],
            })
        del intersector, mesh
        import gc
        gc.collect()

    # ---- normalize to each method's own 5k value (hardware-invariant slope) --
    ref_g = n_gauss_sweep[0]
    norm = {}
    for n_g in n_gauss_sweep:
        norm[n_g] = {}
        for n in n_rays_sweep:
            bvh_rel = bvh_per_ray[n_g][n] / bvh_per_ray[ref_g][n]
            norm[n_g][n] = {"bvh_norm": bvh_rel, "ddf_norm": 1.0}  # DDF flat by construction

    # ---- persist JSON --------------------------------------------------------
    out = {
        "device": torch.cuda.get_device_name(0),
        "cpu_cores": os.cpu_count(),
        "smoke": args.smoke,
        "gs_path": args.gs_path,
        "ddf_ckpt": args.ddf_ckpt,
        "n_gaussians_sweep": n_gauss_sweep,
        "n_rays_sweep": n_rays_sweep,
        "ddf": {
            "params": ddf_params,
            "bytes": ddf_bytes,
            "mb": ddf_bytes / 1e6,
            "scene_independent": True,
            "us_per_ray": {str(n): ddf_times[n] for n in n_rays_sweep},
            "ms_total": {str(n): ddf_ms[n] for n in n_rays_sweep},
        },
        "bvh": {
            "proxy": "octahedron-6vert-8tri @ 3-sigma",
            "build_time_s": {str(k): v for k, v in build_times.items()},
            "geom_bytes": {str(k): v for k, v in bvh_mem.items()},
            "bytes_per_gaussian": OCTA_BYTES_PER_GAUSS,
            "us_per_ray": {str(k): {str(n): bvh_per_ray[k][n] for n in n_rays_sweep}
                           for k in n_gauss_sweep},
        },
        "normalized_to_5k": {str(k): {str(n): norm[k][n] for n in n_rays_sweep}
                             for k in n_gauss_sweep},
        "rows": rows,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_json}")

    # ---- markdown report -----------------------------------------------------
    print("\n## Per-ray latency (us/ray) — DDF (GPU) vs BVH/embree (CPU)\n")
    print("| N_gauss \\ N_rays | " + " | ".join(f"{n:,}" for n in n_rays_sweep) + " |")
    print("|---:|" + "|".join([":---:"] * len(n_rays_sweep)) + "|")
    print("| **DDF (any N_gauss)** | " +
          " | ".join(f"{ddf_times[n]:.4f}" for n in n_rays_sweep) + " |")
    for n_g in n_gauss_sweep:
        print(f"| BVH @ {n_g:,} | " +
              " | ".join(f"{bvh_per_ray[n_g][n]:.4f}" for n in n_rays_sweep) + " |")

    print("\n## Slope (normalized to own 5k value) — hardware-invariant\n")
    print("| N_gauss | " + " | ".join(f"BVH @ {n:,}rays" for n in n_rays_sweep) + " |")
    print("|---:|" + "|".join([":---:"] * len(n_rays_sweep)) + "|")
    for n_g in n_gauss_sweep:
        print(f"| {n_g:,} | " +
              " | ".join(f"{norm[n_g][n]['bvh_norm']:.2f}x" for n in n_rays_sweep) + " |")
    print("\n(DDF normalized slope is 1.00x at every N_gauss by construction — "
          "the network never sees the Gaussian count.)")

    print("\n## Memory\n")
    print(f"DDF: {ddf_bytes/1e6:.1f} MB FIXED, scene-independent.")
    print("| N_gauss | BVH geom (MB) | bytes/Gauss |")
    print("|---:|---:|---:|")
    for n_g in n_gauss_sweep:
        print(f"| {n_g:,} | {bvh_mem[n_g]/1e6:.1f} | {OCTA_BYTES_PER_GAUSS} |")

    # ---- verdict -------------------------------------------------------------
    print("\n=== DIRECTIONALITY CHECK ===")
    for n in n_rays_sweep:
        first, last = n_gauss_sweep[0], n_gauss_sweep[-1]
        bvh_growth = bvh_per_ray[last][n] / bvh_per_ray[first][n]
        print(f"  N_rays={n:>9,}: BVH grows {bvh_growth:.2f}x "
              f"({first:,}->{last:,} Gauss) ; DDF flat (1.00x)")


if __name__ == "__main__":
    main()
