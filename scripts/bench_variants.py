"""Latency benchmark for DDF inference variants vs gsplat reference.

Reuses ``scripts.stage3_latency.time_gsplat`` and the gs supervisor setup, but
swaps the DDF callable for any variant defined in ``src.ddf_variants``.

Run a single variant or compare several side-by-side:

    python scripts/bench_variants.py --variant baseline
    python scripts/bench_variants.py --variant compiled,bf16,compiled_bf16

The script prints a markdown-style table and (optionally) appends a JSON
summary line to ``--out_json`` for later aggregation.
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

# Pin to GPU 0 (sibling agent reserved GPUs 1-3). Must precede ``import torch``.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Ensure project root is on sys.path so `from src...` works when run as
# `python scripts/bench_variants.py` (mirrors how stage3_latency.py is used).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from src.ddf_variants import build_variant
from src.gs_supervisor import GSSupervisor, _spherical_camera
from gsplat import rasterization


N_SWEEP = [1024, 10240, 102400, 1048576]


@torch.no_grad()
def time_callable(
    fn: Callable,
    n_rays: int,
    device: str,
    repeats: int = 5,
    warmups: int = 2,
) -> float:
    """Time a DDF-like callable on ``n_rays`` random (origin, dir) pairs."""
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
def time_gsplat(sup: GSSupervisor, n_rays: int, device: str, repeats: int = 5) -> tuple[float, int]:
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


def bench_one_variant(
    variant: str,
    ddf_ckpt: str,
    gs_path: str,
    device: str,
    small_ckpt: str | None,
    sup: GSSupervisor,
    gs_times: dict[int, tuple[float, int]],
    repeats: int,
    hashgrid_ckpt: str | None = None,
) -> dict:
    fn, ckpt_meta = build_variant(
        variant, ddf_ckpt, device=device,
        small_ckpt_path=small_ckpt,
        hashgrid_ckpt_path=hashgrid_ckpt,
    )

    print(f"\n=== variant: {variant} ===")
    print(f"{'N':>10s}  {'DDF (ms)':>10s}  {'gsplat (ms)':>13s}  "
          f"{'DDF Mray/s':>11s}  {'gsplat Mray/s':>14s}  {'speedup':>10s}")

    rows = []
    for n in N_SWEEP:
        try:
            ddf_t = time_callable(fn, n, device, repeats=repeats)
        except Exception as e:
            print(f"{n:>10d}  ERROR: {e}")
            rows.append({"n": n, "error": str(e)})
            continue
        gs_t, gs_n = gs_times[n]
        ddf_mrs = n / ddf_t / 1e6
        gs_mrs = gs_n / gs_t / 1e6
        speedup = (gs_t / gs_n) / (ddf_t / n)
        tag = f"DDF {speedup:.2f}x" if speedup > 1.0 else f"gsplat {1.0/speedup:.2f}x"
        print(f"{n:>10d}  {ddf_t*1e3:>10.3f}  {gs_t*1e3:>13.3f}  "
              f"{ddf_mrs:>11.2f}  {gs_mrs:>14.2f}  {tag:>10s}")
        rows.append({
            "n": n,
            "ddf_ms": ddf_t * 1e3,
            "gsplat_ms": gs_t * 1e3,
            "ddf_mray_s": ddf_mrs,
            "gsplat_mray_s": gs_mrs,
            "per_ray_speedup_ddf_over_gsplat": speedup,
        })

    return {"variant": variant, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline",
                    help="Comma-separated list, e.g. baseline,compiled,bf16")
    ap.add_argument("--ddf_ckpt", default="runs/bull_ddf_v3/ddf_final.pt")
    ap.add_argument("--gs_path", default="runs/bull/gaussians.pt")
    ap.add_argument(
        "--small_ckpt",
        default="runs/bull_ddf_small/ddf_final.pt",
        help="Checkpoint for small* variants. Ignored otherwise.",
    )
    ap.add_argument(
        "--hashgrid_ckpt",
        default="runs/bull_ddf_hashgrid/ddf_final.pt",
        help="Checkpoint for hashgrid* variants. Ignored otherwise.",
    )
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_json", default=None,
                    help="If set, write summary JSON to this path.")
    args = ap.parse_args()

    device = args.device
    # Speed knobs that benefit baseline + variants alike.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    sup = GSSupervisor(args.gs_path, image_size=64, device=device)

    # Precompute gsplat timings once — they don't depend on the DDF variant.
    print("Timing gsplat reference once for the N-sweep...")
    gs_times: dict[int, tuple[float, int]] = {}
    for n in N_SWEEP:
        gs_t, gs_n = time_gsplat(sup, n, device, repeats=args.repeats)
        gs_times[n] = (gs_t, gs_n)
        print(f"  N={n:>8d}: gsplat {gs_t*1e3:.3f} ms (pixels rendered: {gs_n})")

    summaries = []
    for variant in [v.strip() for v in args.variant.split(",") if v.strip()]:
        try:
            summaries.append(bench_one_variant(
                variant=variant,
                ddf_ckpt=args.ddf_ckpt,
                gs_path=args.gs_path,
                device=device,
                small_ckpt=args.small_ckpt,
                sup=sup,
                gs_times=gs_times,
                repeats=args.repeats,
                hashgrid_ckpt=args.hashgrid_ckpt,
            ))
        except Exception as e:
            print(f"variant {variant} failed: {e}")
            summaries.append({"variant": variant, "error": str(e)})

    if args.out_json is not None:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({
                "ddf_ckpt": args.ddf_ckpt,
                "small_ckpt": args.small_ckpt,
                "gs_path": args.gs_path,
                "gsplat_times_ms": {n: gs_times[n][0] * 1e3 for n in N_SWEEP},
                "gsplat_pixels": {n: gs_times[n][1] for n in N_SWEEP},
                "variants": summaries,
            }, f, indent=2)
        print(f"\nwrote summary to {args.out_json}")


if __name__ == "__main__":
    main()
