"""DECISIVE BENCHMARK: DDF (1 eval/ray) vs NeuS-SDF sphere-trace (N evals/ray).

Both answer the same query: "where is the surface along this ray?"
- DDF: single forward pass -> (distance, hit)
- NeuS SDF: sphere-trace loop, K iterations of sdf(o + t*d) until |sdf| < eps

Wall-clock timed, same GPU, same rays. This determines whether the DDF's
O(1) ray-query premise holds against the obvious SDF-sphere-trace alternative.

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/bench_ddf_vs_sdf.py
"""
import os, sys, math, time, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")


def make_rays(n, device, seed=0):
    """Random rays: origins on a sphere r in [1.5,3], dirs toward origin + jitter."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    d_unit = torch.randn(n, 3, generator=g)
    d_unit = d_unit / d_unit.norm(dim=-1, keepdim=True)
    radii = torch.empty(n, 1).uniform_(1.5, 3.0, generator=g)
    origins = (d_unit * radii).to(device)
    dirs = -d_unit + torch.randn(n, 3, generator=g) * 0.15
    dirs = (dirs / dirs.norm(dim=-1, keepdim=True)).to(device)
    return origins, dirs


def time_fn(fn, reps=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.time()
        fn()
        torch.cuda.synchronize(); ts.append(time.time() - t0)
    return float(np.median(ts)) * 1000.0  # ms


@torch.no_grad()
def main():
    device = "cuda"
    Ns = [1024, 10240, 102400, 1048576]

    # ---- Load DDF (GT-mesh trained hash-grid) ----
    from src.ddf_hashgrid import DDFHashGrid
    ck = torch.load("runs/bull_ddf_gtmesh/ddf_final.pt", map_location=device, weights_only=False)
    mc = ck["cfg"]["model"]
    ddf = DDFHashGrid(
        dir_freqs=mc.get("dir_freqs", 4), hidden_dim=mc.get("hidden_dim", 64),
        num_layers=mc.get("num_layers", 2), n_levels=mc.get("n_levels", 16),
        feat_dim=mc.get("feat_dim", 2), log2_table_size=mc.get("log2_table_size", 19),
        base_res=mc.get("base_res", 16), growth=mc.get("growth", 1.5),
        bbox_half=mc.get("bbox_half", 1.2),
    ).to(device).eval()
    ddf.load_state_dict(ck["model"])
    ddf_params = sum(p.numel() for p in ddf.parameters())
    print(f"DDF: DDFHashGrid, {ddf_params:,} params")

    # ---- Load NeuS SDF field ----
    from nerfstudio.utils.eval_utils import eval_setup
    neus_cfg = Path("runs/sota_comparison_30k/bull/neus/latest_run.txt").read_text().strip()
    _, pipeline, _, step = eval_setup(Path(neus_cfg))
    field = pipeline.model.field
    field.eval()
    for p in field.parameters():
        p.requires_grad_(False)
    neus_params = sum(p.numel() for p in field.parameters())
    print(f"NeuS SDF field: {neus_params:,} params (step {step})")

    def neus_sdf(pts):
        return field.forward_geonetwork(pts)[:, 0]

    # ---- count sphere-trace iterations for NeuS ----
    def neus_sphere_trace(origins, dirs, max_iter=48, eps=1e-3, t_far=5.0, count=False):
        N = origins.shape[0]
        t = torch.zeros(N, device=device)
        alive = torch.ones(N, dtype=torch.bool, device=device)
        iters = 0
        for _ in range(max_iter):
            if not alive.any():
                break
            iters += 1
            idx = alive.nonzero(as_tuple=True)[0]
            pts = origins[idx] + t[idx].unsqueeze(-1) * dirs[idx]
            s = neus_sdf(pts).abs()
            t[idx] += s
            conv = s < eps
            esc = t[idx] > t_far
            alive[idx[conv]] = False
            alive[idx[esc]] = False
        if count:
            return t, iters
        return t

    def ddf_query(origins, dirs):
        return ddf(origins, dirs)  # (dist, vis)

    # Report avg sphere-trace iterations
    o_probe, d_probe = make_rays(10240, device)
    _, n_iters = neus_sphere_trace(o_probe, d_probe, count=True)
    print(f"\nNeuS sphere-trace uses up to {n_iters} iterations (max 48) to converge a 10k batch\n")

    # ---- bf16 + compile DDF variant (best inference stack) ----
    ddf_c = torch.compile(ddf, mode="reduce-overhead")

    results = {}
    print(f"{'N rays':>10} | {'DDF 1-eval':>12} | {'DDF c+bf16':>12} | {'NeuS strace':>12} | {'speedup(c)':>10}")
    print("-" * 70)
    for N in Ns:
        o, d = make_rays(N, device)

        t_ddf = time_fn(lambda: ddf_query(o, d))

        def ddf_compiled_bf16():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ddf_c(o, d)
        t_ddf_c = time_fn(ddf_compiled_bf16, reps=10, warmup=5)

        t_neus = time_fn(lambda: neus_sphere_trace(o, d))

        speedup = t_neus / t_ddf_c
        results[N] = {"ddf_ms": t_ddf, "ddf_compiled_bf16_ms": t_ddf_c,
                      "neus_strace_ms": t_neus, "speedup_vs_neus": speedup}
        print(f"{N:>10,} | {t_ddf:>10.3f}ms | {t_ddf_c:>10.3f}ms | {t_neus:>10.3f}ms | {speedup:>8.1f}x")

    out = Path("runs/bench_ddf_vs_sdf.json")
    out.write_text(json.dumps({
        "ddf_params": ddf_params, "neus_params": neus_params,
        "neus_strace_iters": n_iters, "results": results,
    }, indent=2))
    print(f"\nsaved {out}")
    print("\n=== VERDICT ===")
    for N in Ns:
        r = results[N]
        winner = "DDF" if r["speedup_vs_neus"] > 1 else "NeuS"
        print(f"  N={N:>9,}: {winner} faster by {max(r['speedup_vs_neus'], 1/r['speedup_vs_neus']):.1f}x")


if __name__ == "__main__":
    main()
