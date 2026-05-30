"""Train hash-grid DDF on the 7 objects that currently only have sinusoidal
results so the NeuS-vs-DDF comparison is apples-to-apples (hash-grid is the
headline DDF variant per Stage 3 / bull = 0.117 mean CD).

For each obj: invokes `src.train_ddf` with `configs/bull_hashgrid.yaml` (the
same hash-grid recipe used to produce bull's 0.117 / 0.063 result), then
`scripts/stage3_chamfer.py` to score it. Results land at
`runs/<obj>_ddf_hg/{ddf_final.pt, stage3/metrics.json}` -- the same layout
the rest of the repo expects.

GPUs 0 and 1 only (sibling agent owns 2/3); 2-way sharded sequential.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
import sys
PY = sys.executable

DEFAULT_OBJS = [
    # 4 GSO without hash-grid DDF yet
    "lion", "spino", "mug", "turtle",
    # 3 ShapeNet without hash-grid DDF yet
    "airplane_94c4ade3", "chair_5f1b4529", "car_9ee32f51",
]


def run_one(obj: str, gpu: int, config: str, suffix: str = "_ddf_hg") -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    out_dir = REPO / "runs" / f"{obj}{suffix}"
    log_path = REPO / "runs" / "sota_comparison" / obj / "hashgrid.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_path.open("w") as logf:
        ckpt = out_dir / "ddf_final.pt"
        if not ckpt.exists():
            subprocess.run(
                [PY, "-u", "-m", "src.train_ddf",
                 "--config", config,
                 "--gs_path", f"runs/{obj}/gaussians.pt",
                 "--out_dir", f"runs/{obj}{suffix}"],
                cwd=str(REPO), env=env, stdout=logf, stderr=subprocess.STDOUT,
                check=True,
            )
        # Chamfer
        chamf_dir = out_dir / "stage3"
        metrics = chamf_dir / "metrics.json"
        if not metrics.exists():
            subprocess.run(
                [PY, "scripts/stage3_chamfer.py",
                 "--obj", obj,
                 "--ddf_ckpt", str(ckpt),
                 "--out_dir", str(chamf_dir)],
                cwd=str(REPO), env=env, stdout=logf, stderr=subprocess.STDOUT,
                check=True,
            )
    elapsed = time.time() - t0

    m = json.loads((out_dir / "stage3" / "metrics.json").read_text())
    return {
        "obj": obj, "gpu": gpu,
        "wall_seconds": elapsed,
        "chamfer_mean": m["chamfer_mean"],
        "chamfer_median": m["chamfer_median"],
        "n_verts": m.get("n_verts"),
    }


def _run_shard(objs, gpu, config, suffix):
    rows = []
    for obj in objs:
        try:
            rows.append(run_one(obj, gpu, config, suffix))
            print(f"[gpu {gpu}] {obj}: CD={rows[-1]['chamfer_mean']:.4f} wall={rows[-1]['wall_seconds']:.1f}s")
        except subprocess.CalledProcessError as e:
            rows.append({"obj": obj, "gpu": gpu, "error": str(e)})
            print(f"[gpu {gpu}] {obj}: FAILED ({e})")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objs", nargs="*", default=DEFAULT_OBJS)
    ap.add_argument("--gpus", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--config", default="configs/bull_hashgrid.yaml")
    ap.add_argument("--suffix", default="_ddf_hg")
    ap.add_argument("--out_summary", default="runs/sota_comparison/hashgrid_results.json")
    args = ap.parse_args()

    objs = list(args.objs); gpus = list(args.gpus)
    shards = [[] for _ in gpus]
    for i, obj in enumerate(objs):
        shards[i % len(gpus)].append(obj)
    print(f"[hg] {len(objs)} objs on {len(gpus)} gpus:")
    for gi, sh in enumerate(shards):
        print(f"  gpu {gpus[gi]} -> {sh}")

    rows = []
    with ProcessPoolExecutor(max_workers=len(gpus)) as ex:
        futs = {ex.submit(_run_shard, shard, gpus[gi], args.config, args.suffix): gpus[gi]
                for gi, shard in enumerate(shards)}
        for fut in as_completed(futs):
            try:
                rows.extend(fut.result())
            except Exception as e:
                print(f"[hg] shard failed: {e}", file=sys.stderr)

    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary).write_text(json.dumps(rows, indent=2))
    print(f"[hg] wrote {args.out_summary} ({len(rows)} objects)")


if __name__ == "__main__":
    main()
