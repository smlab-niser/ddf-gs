"""Compute Chamfer distance between an externally-extracted mesh (e.g. NeuS,
a fallback SDF-from-GS, etc.) and the GT GSO/ShapeNet mesh, in the same
normalized coordinate frame as scripts/stage3_chamfer.py.

Coordinate frame: GT mesh is normalized via the per-object `mesh_center` and
`mesh_scale` stored in `runs/<obj>/gaussians.pt`. Predicted mesh is assumed to
already be in that normalized frame (which is what the rest of the pipeline
uses for views/, GS, and DDF training -- the NeuS run keeps the same poses
because we run nerfstudio's SDFStudio parser with `auto_orient=False`).

Usage:
    python scripts/sota_chamfer_mesh.py \
        --obj bull --pred_mesh runs/sota_comparison/bull/neus/sdf_mesh.ply \
        --out runs/sota_comparison/bull/neus/chamfer.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import importlib.util as _ilu

_stage3_spec = _ilu.spec_from_file_location(
    "stage3_chamfer", str(_REPO_ROOT / "scripts" / "stage3_chamfer.py"),
)
_stage3_mod = _ilu.module_from_spec(_stage3_spec)
_stage3_spec.loader.exec_module(_stage3_mod)
chamfer = _stage3_mod.chamfer
load_gt_points = _stage3_mod.load_gt_points
NAME_MAP = _stage3_mod.NAME_MAP
SHAPENET_MAP = _stage3_mod.SHAPENET_MAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--pred_mesh", required=True)
    ap.add_argument("--gt_mesh", default=None,
                    help="override path; default uses obj_manifest.json / shapenet")
    ap.add_argument("--gs_path", default=None,
                    help="default runs/<obj>/gaussians.pt (gives mesh_center, mesh_scale)")
    ap.add_argument("--out", default=None,
                    help="default <pred_mesh_dir>/chamfer.json")
    ap.add_argument("--n_samples", type=int, default=20000)
    ap.add_argument("--method", default="neus", help="tag to write into the json output")
    args = ap.parse_args()

    pred_path = Path(args.pred_mesh)
    if not pred_path.exists():
        raise SystemExit(f"missing pred_mesh: {pred_path}")

    out = Path(args.out) if args.out else pred_path.parent / "chamfer.json"

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
        raise SystemExit(f"unknown obj {args.obj!r}; pass --gt_mesh explicitly")

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

    res = {
        "obj": args.obj,
        "method": args.method,
        "pred_mesh": str(pred_path),
        "gt_mesh": str(gt_path),
        "n_pred_verts": int(len(pred_mesh.vertices)),
        "n_pred_faces": int(len(pred_mesh.faces)),
        "chamfer_mean": cd_mean,
        "chamfer_median": cd_med,
    }
    out.write_text(json.dumps(res, indent=2))
    print(f"[{args.obj}] {args.method} CD mean={cd_mean:.4f} median={cd_med:.4f} -> {out}")


if __name__ == "__main__":
    main()
