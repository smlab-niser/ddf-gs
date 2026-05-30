"""Latency benchmark for the real-scene 3DGS vs the trained DDFs.

Reuses ``scripts.bench_variants.time_callable`` / ``time_gsplat`` but pins the
GS path and DDF ckpt to the realscene_<name> directory; lets us emit a single
markdown / JSON summary specific to the real scene without recomputing the
gsplat reference each run.

Default variant list mirrors the headline trio (see paper): hashgrid
(quality default), small_compiled_bf16 (speed default), and the bf16+compile
hashgrid which is the sweet spot.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# Pick a GPU index BEFORE importing torch.
def _maybe_pin_cuda():
    for i, a in enumerate(sys.argv):
        if a == "--cuda_device" and i + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
            return
        if a.startswith("--cuda_device="):
            os.environ["CUDA_VISIBLE_DEVICES"] = a.split("=", 1)[1]
            return
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")


_maybe_pin_cuda()

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import torch  # noqa: E402

from src.ddf_variants import build_variant  # noqa: E402
from src.gs_supervisor import GSSupervisor  # noqa: E402
import bench_variants  # noqa: E402
time_callable = bench_variants.time_callable
time_gsplat = bench_variants.time_gsplat
N_SWEEP = bench_variants.N_SWEEP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs_path", required=True)
    ap.add_argument("--small_ckpt", required=True)
    ap.add_argument("--hashgrid_ckpt", required=True)
    ap.add_argument("--variants", default="small_compiled_bf16,hashgrid_compiled_bf16")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--cuda_device", default=None, help="handled pre-import")
    args = ap.parse_args()

    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    sup = GSSupervisor(args.gs_path, image_size=64, device=device)
    print(f"N_gaussians = {sup.means.shape[0]:,}")

    print("Timing gsplat reference for the N-sweep...")
    gs_times: dict[int, tuple[float, int]] = {}
    for n in N_SWEEP:
        gs_t, gs_n = time_gsplat(sup, n, device, repeats=args.repeats)
        gs_times[n] = (gs_t, gs_n)
        print(f"  N={n:>8d}: gsplat {gs_t*1e3:>7.3f} ms (rendered px {gs_n})")

    summaries: list[dict] = []
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for variant in variants:
        try:
            fn, _ = build_variant(
                variant, args.small_ckpt, device=device,
                small_ckpt_path=args.small_ckpt,
                hashgrid_ckpt_path=args.hashgrid_ckpt,
            )
        except Exception as e:
            print(f"variant {variant} failed to build: {e}")
            summaries.append({"variant": variant, "error": str(e)})
            continue

        print(f"\n=== variant: {variant} ===")
        print(f"{'N':>10s}  {'DDF (ms)':>10s}  {'gsplat (ms)':>13s}  {'speedup':>14s}")
        rows = []
        for n in N_SWEEP:
            try:
                ddf_t = time_callable(fn, n, device, repeats=args.repeats)
            except Exception as e:
                print(f"  N={n} ERROR: {e}")
                rows.append({"n": n, "error": str(e)})
                continue
            gs_t, _ = gs_times[n]
            speedup = gs_t / ddf_t  # DDF wins if >1
            tag = f"DDF {speedup:.2f}x" if speedup > 1 else f"gsplat {1/speedup:.2f}x"
            print(f"  {n:>10d}  {ddf_t*1e3:>10.3f}  {gs_t*1e3:>13.3f}  {tag:>14s}")
            rows.append({
                "n": n, "ddf_ms": ddf_t * 1e3, "gsplat_ms": gs_t * 1e3,
                "speedup": speedup,
            })
        summaries.append({"variant": variant, "rows": rows})

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({
            "gs_path": args.gs_path,
            "n_gaussians": int(sup.means.shape[0]),
            "small_ckpt": args.small_ckpt,
            "hashgrid_ckpt": args.hashgrid_ckpt,
            "repeats": args.repeats,
            "gsplat_times_ms": {n: gs_times[n][0] * 1e3 for n in N_SWEEP},
            "gsplat_pixels": {n: gs_times[n][1] for n in N_SWEEP},
            "variants": summaries,
        }, f, indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
