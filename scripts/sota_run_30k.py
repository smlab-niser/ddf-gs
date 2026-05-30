"""Stage 9b — Re-run NeuS-facto at the paper-recommended 30k iters on the
same 10 objects from Stage 9, then compute Chamfer + F-score @ {0.05, 0.10,
0.20} in the same coordinate frame.

Reuses `runs/sota_comparison/<obj>/sdf_data/` (built by
`sota_make_sdfstudio_dataset.py` in Stage 9). Writes all new outputs to
`runs/sota_comparison_30k/<obj>/neus/` -- does NOT overwrite Stage 9.

Per-object outputs:
    runs/sota_comparison_30k/<obj>/neus/<run-dir>/  (nerfstudio output)
    runs/sota_comparison_30k/<obj>/neus/sdf_mesh.ply
    runs/sota_comparison_30k/<obj>/neus/chamfer.json
    runs/sota_comparison_30k/<obj>/neus/wall.json
    runs/sota_comparison_30k/<obj>/neus/pointclouds.npz   (gt + pred samples)
    runs/sota_comparison_30k/<obj>/neus/fscore.json
    runs/sota_comparison_30k/<obj>/neus/latest_run.txt

GPUs 0 and 1 only (sibling agent owns 2/3 for Poisson+TSDF baselines).
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
    # 5 GSO
    "bull", "lion", "spino", "mug", "turtle",
    # 5 ShapeNet
    "airplane_94c4ade3", "chair_5f1b4529", "car_9ee32f51",
    "bottle_59d7b4e7", "sofa_145bd097",
]


def run_one(obj: str, gpu: int, max_iters: int, rays: int, resolution: int) -> dict:
    """Train NeuS @ max_iters, export mesh, run Chamfer + F-score. Crashes propagate."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    # Reuse Stage 9's SDFStudio dataset.
    data_dir = REPO / "runs" / "sota_comparison" / obj / "sdf_data"
    if not (data_dir / "meta_data.json").exists():
        raise FileNotFoundError(f"sdf_data missing for {obj}: {data_dir}")

    # New, separate output tree.
    out_root = REPO / "runs" / "sota_comparison_30k" / obj
    neus_dir = out_root / "neus"
    out_root.mkdir(parents=True, exist_ok=True)
    neus_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "pipeline.log"
    t0 = time.time()
    with log_path.open("w") as logf:
        # 1) Train neus-facto @ 30k + export mesh
        subprocess.run(
            [PY, "scripts/sota_train_neus.py",
             "--obj", obj,
             "--data_dir", str(data_dir),
             "--out_dir", str(neus_dir),
             "--max_iters", str(max_iters),
             "--rays_per_batch", str(rays),
             "--resolution", str(resolution)],
            cwd=str(REPO), env=env, stdout=logf, stderr=subprocess.STDOUT,
            check=True,
        )
        # 2) Chamfer (writes chamfer.json AND pointclouds.npz via our updated step)
        subprocess.run(
            [PY, "scripts/sota_chamfer_fscore.py",
             "--obj", obj,
             "--pred_mesh", str(neus_dir / "sdf_mesh.ply"),
             "--out_dir", str(neus_dir)],
            cwd=str(REPO), env=env, stdout=logf, stderr=subprocess.STDOUT,
            check=True,
        )
    elapsed = time.time() - t0

    wall = json.loads((neus_dir / "wall.json").read_text())
    cd = json.loads((neus_dir / "chamfer.json").read_text())
    fs = json.loads((neus_dir / "fscore.json").read_text())
    return {
        "obj": obj, "gpu": gpu,
        "wall_seconds": elapsed,
        "train_seconds": wall.get("train_seconds"),
        "export_seconds": wall.get("export_seconds"),
        "chamfer_mean": cd["chamfer_mean"],
        "chamfer_median": cd["chamfer_median"],
        "n_pred_verts": cd["n_pred_verts"],
        "fscore": fs["fscore"],
    }


def _run_shard(objs, gpu, max_iters, rays, resolution):
    rows = []
    for obj in objs:
        try:
            row = run_one(obj, gpu, max_iters, rays, resolution)
            rows.append(row)
            f005 = row["fscore"].get("0.05", {}).get("f1", 0.0)
            print(f"[gpu {gpu}] {obj}: CD={row['chamfer_mean']:.4f} "
                  f"med={row['chamfer_median']:.4f} F1@0.05={f005:.3f} "
                  f"train={row['train_seconds']:.1f}s wall={row['wall_seconds']:.1f}s",
                  flush=True)
        except subprocess.CalledProcessError as e:
            rows.append({"obj": obj, "gpu": gpu, "error": str(e)})
            print(f"[gpu {gpu}] {obj}: FAILED ({e})", flush=True)
        except Exception as e:
            rows.append({"obj": obj, "gpu": gpu, "error": repr(e)})
            print(f"[gpu {gpu}] {obj}: FAILED ({e!r})", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objs", nargs="*", default=DEFAULT_OBJS)
    ap.add_argument("--gpus", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--max_iters", type=int, default=30000)
    ap.add_argument("--rays_per_batch", type=int, default=2048)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--out_summary", default="runs/sota_comparison_30k/all_results.json")
    args = ap.parse_args()

    objs = list(args.objs); gpus = list(args.gpus)
    shards = [[] for _ in gpus]
    for i, obj in enumerate(objs):
        shards[i % len(gpus)].append(obj)
    print(f"[30k] {len(objs)} objs on {len(gpus)} gpus, max_iters={args.max_iters}:")
    for gi, sh in enumerate(shards):
        print(f"  gpu {gpus[gi]} -> {sh}")

    summary_rows = []
    with ProcessPoolExecutor(max_workers=len(gpus)) as ex:
        futs = {ex.submit(_run_shard, shard, gpus[gi], args.max_iters,
                          args.rays_per_batch, args.resolution): gpus[gi]
                for gi, shard in enumerate(shards)}
        for fut in as_completed(futs):
            try:
                summary_rows.extend(fut.result())
            except Exception as e:
                print(f"[30k] shard failed: {e}", file=sys.stderr)

    out = Path(args.out_summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary_rows, indent=2))
    print(f"[30k] wrote {out} ({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
