"""Orchestrate the full SOTA-baseline comparison on 10 objects, sharded
across GPUs 0 and 1 (sibling agent owns 2/3). For each object:
    1) build SDFStudio dataset from runs/<obj>/views/
    2) train neus-facto for --max_iters iterations
    3) export marching-cubes mesh
    4) compute Chamfer vs same GT mesh used in stage3_chamfer

Per-object outputs:
    runs/sota_comparison/<obj>/sdf_data/        (dataset)
    runs/sota_comparison/<obj>/neus/<run-dir>/  (nerfstudio output)
    runs/sota_comparison/<obj>/neus/sdf_mesh.ply
    runs/sota_comparison/<obj>/neus/chamfer.json
    runs/sota_comparison/<obj>/neus/wall.json
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
    # 5 GSO (short -> via NAME_MAP in stage3_chamfer.py)
    "bull", "lion", "spino", "mug", "turtle",
    # 5 ShapeNet (via SHAPENET_MAP)
    "airplane_94c4ade3", "chair_5f1b4529", "car_9ee32f51",
    "bottle_59d7b4e7", "sofa_145bd097",
]


def run_one(obj: str, gpu: int, max_iters: int, rays: int, resolution: int) -> dict:
    """Returns a dict with timings + chamfer for this object. Crashes propagate."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    out_dir = REPO / "runs" / "sota_comparison" / obj
    log_path = out_dir / "pipeline.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_path.open("w") as logf:
        # 1) dataset
        subprocess.run(
            [PY, "scripts/sota_make_sdfstudio_dataset.py", "--obj", obj],
            cwd=str(REPO), env=env, stdout=logf, stderr=subprocess.STDOUT,
            check=True,
        )
        # 2,3) train + export
        subprocess.run(
            [PY, "scripts/sota_train_neus.py",
             "--obj", obj,
             "--max_iters", str(max_iters),
             "--rays_per_batch", str(rays),
             "--resolution", str(resolution)],
            cwd=str(REPO), env=env, stdout=logf, stderr=subprocess.STDOUT,
            check=True,
        )
        # 4) chamfer
        subprocess.run(
            [PY, "scripts/sota_chamfer_mesh.py",
             "--obj", obj,
             "--pred_mesh", str(out_dir / "neus" / "sdf_mesh.ply"),
             "--out", str(out_dir / "neus" / "chamfer.json")],
            cwd=str(REPO), env=env, stdout=logf, stderr=subprocess.STDOUT,
            check=True,
        )
    elapsed = time.time() - t0

    wall = json.loads((out_dir / "neus" / "wall.json").read_text())
    cd = json.loads((out_dir / "neus" / "chamfer.json").read_text())
    return {
        "obj": obj, "gpu": gpu,
        "wall_seconds": elapsed,
        "train_seconds": wall.get("train_seconds"),
        "export_seconds": wall.get("export_seconds"),
        "chamfer_mean": cd["chamfer_mean"],
        "chamfer_median": cd["chamfer_median"],
        "n_pred_verts": cd["n_pred_verts"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objs", nargs="*", default=DEFAULT_OBJS,
                    help="object short ids; default = 10 paper objects")
    ap.add_argument("--gpus", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--max_iters", type=int, default=5000)
    ap.add_argument("--rays_per_batch", type=int, default=2048)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--out_summary", default="runs/sota_comparison/all_results.json")
    args = ap.parse_args()

    objs = list(args.objs)
    gpus = list(args.gpus)
    print(f"[orchestrate] {len(objs)} objs across {len(gpus)} gpus")
    print(f"[orchestrate] objs: {objs}")
    print(f"[orchestrate] gpus: {gpus}")

    # Round-robin assign each object to a GPU; run all in parallel.
    # ProcessPoolExecutor with max_workers=len(gpus) ensures only K jobs run
    # at a time, but we want each GPU to do many jobs serially: better to
    # split objs into K shards and run each shard sequentially in its own
    # subprocess.
    shards = [[] for _ in gpus]
    for i, obj in enumerate(objs):
        shards[i % len(gpus)].append(obj)

    futures = {}
    summary_rows = []
    with ProcessPoolExecutor(max_workers=len(gpus)) as ex:
        # Submit shard runners
        for gi, shard in enumerate(shards):
            futures[ex.submit(_run_shard, shard, gpus[gi], args.max_iters,
                              args.rays_per_batch, args.resolution)] = gpus[gi]
        for fut in as_completed(futures):
            gpu = futures[fut]
            try:
                rows = fut.result()
            except Exception as e:
                print(f"[gpu {gpu}] shard FAILED: {e}", file=sys.stderr)
                continue
            for r in rows:
                print(f"[gpu {gpu}] {r['obj']}: CD={r['chamfer_mean']:.4f} med={r['chamfer_median']:.4f} train={r['train_seconds']:.1f}s wall={r['wall_seconds']:.1f}s")
                summary_rows.append(r)

    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary).write_text(json.dumps(summary_rows, indent=2))
    print(f"[done] wrote {args.out_summary}  ({len(summary_rows)} objects)")


def _run_shard(objs, gpu, max_iters, rays, resolution):
    rows = []
    for obj in objs:
        try:
            row = run_one(obj, gpu, max_iters, rays, resolution)
            rows.append(row)
        except subprocess.CalledProcessError as e:
            rows.append({"obj": obj, "gpu": gpu, "error": str(e)})
    return rows


if __name__ == "__main__":
    main()
