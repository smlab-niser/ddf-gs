"""Chamfer + F-score for an externally-extracted mesh (NeuS @ 30k), scored in
the same normalised frame as stage3_chamfer.py.

Loads GT mesh via the same `mesh_center` / `mesh_scale` from
`runs/<obj>/gaussians.pt`, samples 20k points from GT and 20k points from the
predicted mesh, writes:
    <out_dir>/chamfer.json
    <out_dir>/fscore.json
    <out_dir>/pointclouds.npz   # gt + pred, for re-scoring

F1 at multiple thresholds (default tau in {0.05, 0.10, 0.20}):
    precision = mean(min_dist(pred -> GT) < tau)
    recall    = mean(min_dist(GT   -> pred) < tau)
    F1 = 2 P R / (P + R)
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_spec = _ilu.spec_from_file_location("stage3_chamfer", str(_REPO_ROOT / "scripts" / "stage3_chamfer.py"))
_stage3 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_stage3)
chamfer = _stage3.chamfer
load_gt_points = _stage3.load_gt_points
NAME_MAP = _stage3.NAME_MAP
SHAPENET_MAP = _stage3.SHAPENET_MAP


def fscore(pred_pts: np.ndarray, gt_pts: np.ndarray, taus):
    tpred = cKDTree(pred_pts)
    tgt = cKDTree(gt_pts)
    d_pred_to_gt = tgt.query(pred_pts, k=1)[0]
    d_gt_to_pred = tpred.query(gt_pts, k=1)[0]
    out = {}
    for tau in taus:
        p = float((d_pred_to_gt < tau).mean())
        r = float((d_gt_to_pred < tau).mean())
        f1 = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0
        out[f"{tau:.2f}"] = {"precision": p, "recall": r, "f1": f1}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--pred_mesh", required=True)
    ap.add_argument("--gs_path", default=None)
    ap.add_argument("--gt_mesh", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=20000)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--method", default="neus_30k")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)

    pred_path = Path(args.pred_mesh)
    if not pred_path.exists():
        raise SystemExit(f"missing pred_mesh: {pred_path}")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    gs_path = Path(args.gs_path) if args.gs_path else Path(f"runs/{args.obj}/gaussians.pt")
    gs = torch.load(gs_path, map_location="cpu", weights_only=False)
    mesh_center = gs["mesh_center"].numpy().astype(np.float32)
    mesh_scale = float(gs["mesh_scale"].item())

    if args.gt_mesh is not None:
        gt_path = Path(args.gt_mesh)
    elif args.obj in SHAPENET_MAP:
        gt_path = Path(f"data/shapenet/{args.obj}/model.glb")
    elif args.obj in NAME_MAP:
        gt_path = Path(f"data/gso/{NAME_MAP[args.obj]}/meshes/model.obj")
    else:
        raise SystemExit(f"unknown obj {args.obj!r}")

    gt_pts = load_gt_points(gt_path, mesh_center, mesh_scale, args.n_samples)

    pred_mesh = trimesh.load(pred_path, force="mesh", process=False)
    if isinstance(pred_mesh, trimesh.Scene):
        geoms = [g for g in pred_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise SystemExit(f"no mesh in {pred_path}")
        pred_mesh = trimesh.util.concatenate(geoms)
    if len(pred_mesh.faces) == 0:
        raise SystemExit(f"empty mesh: {pred_path}")
    pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, args.n_samples)
    pred_pts = np.asarray(pred_pts, dtype=np.float32)

    cd_mean, cd_med = chamfer(gt_pts, pred_pts)
    fs = fscore(pred_pts, gt_pts, args.taus)

    np.savez(out_dir / "pointclouds.npz", gt=gt_pts, pred=pred_pts)

    (out_dir / "chamfer.json").write_text(json.dumps({
        "obj": args.obj,
        "method": args.method,
        "pred_mesh": str(pred_path),
        "gt_mesh": str(gt_path),
        "n_pred_verts": int(len(pred_mesh.vertices)),
        "n_pred_faces": int(len(pred_mesh.faces)),
        "chamfer_mean": cd_mean,
        "chamfer_median": cd_med,
    }, indent=2))

    (out_dir / "fscore.json").write_text(json.dumps({
        "obj": args.obj,
        "method": args.method,
        "taus": list(args.taus),
        "n_samples": args.n_samples,
        "fscore": fs,
    }, indent=2))

    f005 = fs.get("0.05", {}).get("f1", 0.0)
    f010 = fs.get("0.10", {}).get("f1", 0.0)
    f020 = fs.get("0.20", {}).get("f1", 0.0)
    print(f"[{args.obj}] {args.method} CD={cd_mean:.4f} med={cd_med:.4f} "
          f"F1@0.05={f005:.3f} F1@0.10={f010:.3f} F1@0.20={f020:.3f}")


if __name__ == "__main__":
    main()
