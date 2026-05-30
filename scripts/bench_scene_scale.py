"""Scene-scale latency benchmark: gsplat vs best DDF variant.

For each ``N_gaussians`` we synthesize a denser Gaussian cloud by replicating
the bull's 5k means with positional jitter (quats/scales/opacities/colors are
just broadcast-replicated). This is not a photometrically meaningful scene,
but gsplat's runtime is dominated by Gaussian count + per-tile load, so the
benchmark is a fair proxy for "what happens when the scene is dense?".

For each ``N_rays`` we time both backends like ``scripts/bench_variants.py``:
  - gsplat: ``rasterization(..., render_mode='RGB+ED')`` on a sqrt(N) x sqrt(N)
    spherical view.
  - DDF: ``small + compile + bf16`` variant from ``src.ddf_variants``.

Outputs:
  - Markdown table to stdout
  - JSON timings to runs/scene_scale_bench.json
  - matplotlib plot to runs/scene_scale_speedup.png (speedup vs N_gaussians
    at fixed N_rays = 1M).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable

# Pick the least-loaded GPU before importing torch. Siblings are on 0-3 but
# A100s have headroom. We probe via nvidia-smi here.
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
        # Prefer lowest util, tie-break on lowest memory.
        rows.sort(key=lambda r: (r[2], r[1]))
        return str(rows[0][0])
    except Exception:
        return "0"


os.environ.setdefault("CUDA_VISIBLE_DEVICES", _pick_gpu())

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from src.ddf_variants import build_variant  # noqa: E402
from src.gs_supervisor import GSSupervisor, _spherical_camera  # noqa: E402
from gsplat import rasterization  # noqa: E402


N_GAUSSIANS_SWEEP = [5_000, 50_000, 200_000, 1_000_000]
N_RAYS_SWEEP = [1024, 10_240, 102_400, 1_048_576]


def _replicate_with_jitter(
    sup: GSSupervisor,
    n_target: int,
    jitter_scale: float = 0.05,
    device: str = "cuda",
    seed: int = 0,
) -> GSSupervisor:
    """Return a new GSSupervisor-like object with ``n_target`` Gaussians.

    The original 5k means are tiled ``ceil(n_target/5k)`` times with Gaussian
    positional jitter; quats/scales/opacities/colors are tiled identically.
    Bounding box is the original (jitter is small relative to scene extent so
    means rarely escape).
    """
    n_src = sup.means.shape[0]
    if n_target == n_src:
        return sup

    reps = math.ceil(n_target / n_src)
    # Build a deep view: copy fields, then resample.
    g = torch.Generator(device=device).manual_seed(seed)

    means = sup.means.repeat(reps, 1)[:n_target].contiguous()
    quats = sup.quats.repeat(reps, 1)[:n_target].contiguous()
    scales = sup.scales.repeat(reps, 1)[:n_target].contiguous()
    opacities = sup.opacities.repeat(reps)[:n_target].contiguous()
    colors = sup.colors.repeat(reps, 1)[:n_target].contiguous()

    # Scene extent ~ 2 in y, ~ 1 in x/z. Jitter that fraction.
    extent = (sup.bbox_max - sup.bbox_min).norm().item()
    jitter = torch.randn(n_target, 3, generator=g, device=device) * (jitter_scale * extent / math.sqrt(3))
    # Skip jittering the first n_src so original 5k are preserved exactly
    jitter[:n_src] = 0.0
    means = means + jitter

    # Return a lightweight stand-in matching GSSupervisor's rasterization API.
    new = sup.__class__.__new__(sup.__class__)
    new.device = sup.device
    new.image_size = sup.image_size
    new.means = means
    new.quats = quats
    new.scales = scales
    new.opacities = opacities
    new.colors = colors
    new.bbox_min = sup.bbox_min
    new.bbox_max = sup.bbox_max
    return new


@torch.no_grad()
def time_callable(
    fn: Callable,
    n_rays: int,
    device: str,
    repeats: int = 5,
    warmups: int = 2,
) -> float:
    o = torch.randn(n_rays, 3, device=device) * 0.8
    d = torch.randn(n_rays, 3, device=device)
    d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    for _ in range(warmups):
        fn(o, d)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(o, d)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


@torch.no_grad()
def time_gsplat(sup, n_rays: int, device: str, repeats: int = 5) -> tuple[float, int]:
    side = int(math.ceil(math.sqrt(n_rays)))
    w2c, K, _ = _spherical_camera(20.0, 45.0, 2.5, side, device=device)
    kwargs = dict(
        means=sup.means, quats=sup.quats, scales=sup.scales,
        opacities=sup.opacities, colors=sup.colors,
        viewmats=w2c.unsqueeze(0), Ks=K.unsqueeze(0),
        width=side, height=side, sh_degree=None, render_mode="RGB+ED",
    )
    for _ in range(2):
        rasterization(**kwargs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        rasterization(**kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats, side * side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs_path", default="runs/bull/gaussians.pt")
    ap.add_argument("--small_ckpt", default="runs/bull_ddf_small/ddf_final.pt")
    ap.add_argument("--ddf_ckpt", default="runs/bull_ddf_v3/ddf_final.pt")
    ap.add_argument("--variant", default="small_compiled_bf16")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out_json", default="runs/scene_scale_bench.json")
    ap.add_argument("--out_plot", default="runs/scene_scale_speedup.png")
    args = ap.parse_args()

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load base supervisor.
    base_sup = GSSupervisor(args.gs_path, image_size=64, device=device)
    print(f"Base bull Gaussians: {base_sup.means.shape[0]}")

    # Build DDF once (same model used for all scenes — DDF cost is independent
    # of N_gaussians, that's the whole point of the claim).
    fn, _ = build_variant(
        args.variant, args.ddf_ckpt, device=device, small_ckpt_path=args.small_ckpt,
    )
    print(f"DDF variant: {args.variant}")

    # Pre-time DDF once per N_rays (independent of N_gaussians).
    print("\nTiming DDF (independent of N_gaussians)...")
    ddf_times: dict[int, float] = {}
    for n in N_RAYS_SWEEP:
        t = time_callable(fn, n, device, repeats=args.repeats)
        ddf_times[n] = t
        print(f"  N_rays={n:>8d}: DDF {t*1e3:.3f} ms")

    results: list[dict] = []

    for n_g in N_GAUSSIANS_SWEEP:
        print(f"\n=== N_gaussians = {n_g:,} ===")
        sup = _replicate_with_jitter(base_sup, n_g, device=device)
        # Memory sanity print.
        gb = (sup.means.numel() + sup.quats.numel() + sup.scales.numel()
              + sup.opacities.numel() + sup.colors.numel()) * 4 / 1e9
        print(f"  GS params memory ~{gb*1e3:.1f} MB")

        print(f"  {'N_rays':>10s}  {'gsplat (ms)':>13s}  {'DDF (ms)':>10s}  "
              f"{'gsplat Mray/s':>14s}  {'DDF Mray/s':>11s}  {'speedup':>14s}")

        for n in N_RAYS_SWEEP:
            try:
                gs_t, gs_n = time_gsplat(sup, n, device, repeats=args.repeats)
            except Exception as e:
                print(f"  N_rays={n} gsplat ERROR: {e}")
                results.append({"n_gaussians": n_g, "n_rays": n, "error": str(e)})
                continue

            ddf_t = ddf_times[n]
            gs_mrs = gs_n / gs_t / 1e6
            ddf_mrs = n / ddf_t / 1e6
            speedup = (gs_t / gs_n) / (ddf_t / n)
            tag = f"DDF {speedup:.2f}x" if speedup > 1.0 else f"gsplat {1.0/speedup:.2f}x"
            print(f"  {n:>10d}  {gs_t*1e3:>13.3f}  {ddf_t*1e3:>10.3f}  "
                  f"{gs_mrs:>14.2f}  {ddf_mrs:>11.2f}  {tag:>14s}")

            results.append({
                "n_gaussians": n_g,
                "n_rays": n,
                "gsplat_ms": gs_t * 1e3,
                "ddf_ms": ddf_t * 1e3,
                "gsplat_pixels": gs_n,
                "gsplat_mray_s": gs_mrs,
                "ddf_mray_s": ddf_mrs,
                "per_ray_speedup_ddf_over_gsplat": speedup,
            })

        # Free the synthetic supervisor's tensors before next size.
        del sup
        torch.cuda.empty_cache()

    # Persist JSON.
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "gs_path": args.gs_path,
            "ddf_variant": args.variant,
            "ddf_ckpt": args.ddf_ckpt,
            "small_ckpt": args.small_ckpt,
            "repeats": args.repeats,
            "n_gaussians_sweep": N_GAUSSIANS_SWEEP,
            "n_rays_sweep": N_RAYS_SWEEP,
            "ddf_times_ms": {n: ddf_times[n] * 1e3 for n in N_RAYS_SWEEP},
            "rows": results,
        }, f, indent=2)
    print(f"\nwrote raw timings to {out_json}")

    # Markdown table.
    print("\n## Markdown table (gsplat ms / DDF ms / speedup)\n")
    print("| N_gaussians | N_rays | gsplat (ms) | DDF (ms) | per-ray speedup (DDF / gsplat) |")
    print("|---:|---:|---:|---:|:---:|")
    for r in results:
        if "error" in r:
            print(f"| {r['n_gaussians']:,} | {r['n_rays']:,} | ERROR | ERROR | — |")
            continue
        s = r["per_ray_speedup_ddf_over_gsplat"]
        tag = f"**DDF {s:.2f}x**" if s > 1.0 else f"gsplat {1.0/s:.2f}x"
        print(f"| {r['n_gaussians']:,} | {r['n_rays']:,} | {r['gsplat_ms']:.3f} | "
              f"{r['ddf_ms']:.3f} | {tag} |")

    # Plot: speedup vs N_gaussians at fixed N_rays = 1M (and also full set).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for n_rays in N_RAYS_SWEEP:
            xs, ys = [], []
            for r in results:
                if r.get("n_rays") == n_rays and "per_ray_speedup_ddf_over_gsplat" in r:
                    xs.append(r["n_gaussians"])
                    ys.append(r["per_ray_speedup_ddf_over_gsplat"])
            if xs:
                ax.plot(xs, ys, "-o", label=f"N_rays = {n_rays:,}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8,
                   label="parity (1.0×)")
        ax.set_xlabel("N_gaussians")
        ax.set_ylabel("speedup (DDF over gsplat, per ray)")
        ax.set_title("Scene-scale ray-query speedup: DDF (small+compile+bf16) vs gsplat depth")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
        out_plot = Path(args.out_plot)
        out_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_plot, dpi=140)
        print(f"wrote plot to {out_plot}")
    except Exception as e:
        print(f"plotting failed: {e}")


if __name__ == "__main__":
    main()
