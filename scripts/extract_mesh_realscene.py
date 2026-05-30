"""Extract a UDF -> marching-cubes mesh from a trained DDF for a real scene.

Same pipeline as ``scripts/stage3_chamfer.py``'s extraction half, but no GT
mesh is expected (none exists for the real scene). Saves the mesh to
``<out_dir>/pred_mesh.ply`` and writes a small ``mesh_info.json`` with
vertex/face counts and UDF statistics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Pin GPU before torch is imported. We accept --cuda_device on argv (matching
# train_ddf.py), or fall back to CUDA_VISIBLE_DEVICES env var, or default to 2.
def _maybe_pin_cuda():
    for i, a in enumerate(sys.argv):
        if a == "--cuda_device" and i + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
            return
        if a.startswith("--cuda_device="):
            os.environ["CUDA_VISIBLE_DEVICES"] = a.split("=", 1)[1]
            return
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")


_maybe_pin_cuda()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import trimesh
from skimage.measure import marching_cubes

from src.ddf_model import DDF
from src.ddf_hashgrid import DDFHashGrid


def _build_model_from_cfg(cfg, device):
    m_cfg = cfg["model"]
    kind = str(m_cfg.get("type", "ddf")).lower()
    if kind == "hashgrid":
        return DDFHashGrid(
            dir_freqs=m_cfg.get("dir_freqs", 4),
            hidden_dim=m_cfg.get("hidden_dim", 64),
            num_layers=m_cfg.get("num_layers", 2),
            n_levels=m_cfg.get("n_levels", 16),
            feat_dim=m_cfg.get("feat_dim", 2),
            log2_table_size=m_cfg.get("log2_table_size", 19),
            base_res=m_cfg.get("base_res", 16),
            growth=m_cfg.get("growth", 1.5),
            bbox_half=m_cfg.get("bbox_half", 1.2),
        ).to(device)
    return DDF(
        pos_freqs=m_cfg["pos_freqs"], dir_freqs=m_cfg["dir_freqs"],
        hidden_dim=m_cfg["hidden_dim"], num_layers=m_cfg["num_layers"],
    ).to(device)


@torch.no_grad()
def query_udf(model, points: torch.Tensor, n_dirs: int = 32, vis_thresh: float = 0.3):
    n = points.shape[0]
    dirs = torch.randn(n * n_dirs, 3, device=points.device)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    pts = points.repeat_interleave(n_dirs, dim=0)
    t, vis_logit = model(pts, dirs)
    visible = vis_logit.sigmoid() > vis_thresh
    t = torch.where(visible, t, torch.full_like(t, 1e3))
    return t.view(n, n_dirs).min(dim=-1).values


@torch.no_grad()
def extract_mesh(model, grid_size: int, bbox_half: float, n_dirs: int,
                 iso: float, device: str, chunk: int = 65536,
                 vis_thresh: float = 0.3):
    coords = torch.linspace(-bbox_half, bbox_half, grid_size, device=device)
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing="ij")
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    udf_vals = []
    for i in range(0, pts.shape[0], chunk):
        udf_vals.append(query_udf(model, pts[i:i + chunk], n_dirs=n_dirs,
                                  vis_thresh=vis_thresh).cpu())
    udf = torch.cat(udf_vals).view(grid_size, grid_size, grid_size).numpy()
    spacing = (2 * bbox_half) / (grid_size - 1)
    try:
        verts, faces, _, _ = marching_cubes(udf, level=iso, spacing=(spacing,) * 3)
    except (ValueError, RuntimeError) as e:
        return None, None, udf, str(e)
    verts = verts + np.array([-bbox_half, -bbox_half, -bbox_half], dtype=np.float32)
    return verts, faces, udf, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddf_ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--grid_size", type=int, default=128)
    ap.add_argument("--bbox_half", type=float, default=1.2)
    ap.add_argument("--n_dirs", type=int, default=32)
    ap.add_argument("--iso", type=float, default=0.05)
    ap.add_argument("--vis_thresh", type=float, default=0.3,
                    help="visibility sigmoid threshold for hit masking")
    ap.add_argument("--cuda_device", default=None, help="handled pre-import")
    args = ap.parse_args()

    device = "cuda"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ddf_ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = _build_model_from_cfg(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"extracting mesh: grid={args.grid_size}, n_dirs={args.n_dirs}, iso={args.iso}")
    verts, faces, udf, err = extract_mesh(
        model, args.grid_size, args.bbox_half, args.n_dirs, args.iso, device,
        vis_thresh=args.vis_thresh,
    )
    if verts is None:
        print(f"marching cubes failed: {err}")
        print(f"UDF stats: min={udf.min():.4f} max={udf.max():.4f} "
              f"mean={udf.mean():.4f} p1={np.percentile(udf,1):.4f}")
        new_iso = float(np.percentile(udf, 1))
        spacing = (2 * args.bbox_half) / (args.grid_size - 1)
        try:
            verts, faces, _, _ = marching_cubes(udf, level=new_iso, spacing=(spacing,) * 3)
            verts = verts + np.array([-args.bbox_half] * 3, dtype=np.float32)
            print(f"retry with iso={new_iso:.4f} succeeded")
        except Exception as e2:
            print(f"retry also failed: {e2}")
            return

    pred_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    pred_mesh_path = args.out_dir / "pred_mesh.ply"
    pred_mesh.export(pred_mesh_path)
    print(f"wrote {pred_mesh_path}  ({len(verts)} verts, {len(faces)} faces)")

    info = {
        "ddf_ckpt": str(args.ddf_ckpt),
        "grid_size": args.grid_size,
        "iso": args.iso,
        "n_dirs": args.n_dirs,
        "bbox_half": args.bbox_half,
        "n_verts": int(len(verts)),
        "n_faces": int(len(faces)),
        "udf_min": float(udf.min()),
        "udf_max": float(udf.max()),
        "udf_p1": float(np.percentile(udf, 1)),
        "udf_p50": float(np.percentile(udf, 50)),
    }
    with (args.out_dir / "mesh_info.json").open("w") as f:
        json.dump(info, f, indent=2)
    print(f"info -> {args.out_dir / 'mesh_info.json'}")


if __name__ == "__main__":
    main()
