"""Evaluate a trained conditional DDF.

Two modes:
  --mode trained
      For each object in the training set, query the model with its trained
      ``z`` and compute Chamfer the usual way (UDF -> marching cubes -> sample
      surface -> Chamfer vs GT mesh). Output: per-object metrics + mean.

  --mode heldout
      For each object listed in --heldout_ids (must have GS + GT mesh on
      disk), try two strategies and report Chamfer for each:
        (a) z = mean of trained zs (no test-time optimisation).
        (b) z optimised against the held-out object's GS supervisor for
            --opt_steps steps, MLP frozen (DeepSDF inference). Initialised
            from the mean z.

Reuses ``stage3_chamfer`` utilities (UDF -> marching cubes -> mesh sampling
-> kd-tree symmetric Chamfer).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _pin_cuda():
    for i, a in enumerate(sys.argv):
        if a == "--cuda_device" and i + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
            return
        if a.startswith("--cuda_device="):
            os.environ["CUDA_VISIBLE_DEVICES"] = a.split("=", 1)[1]
            return


_pin_cuda()

# Make scripts/ importable as a package even though there's no __init__.py.
_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh

from stage3_chamfer import (  # type: ignore
    NAME_MAP,
    SHAPENET_MAP,
    chamfer,
    load_gt_points,
)
from src.conditional_supervisor import MultiObjectSupervisor
from src.ddf_conditional import DDFCond
from src.gs_supervisor import GSSupervisor
from skimage.measure import marching_cubes


# ----- mesh extraction for the conditional DDF (z-aware) -----

@torch.no_grad()
def query_udf_cond(
    model: DDFCond, z: torch.Tensor, points: torch.Tensor,
    n_dirs: int = 32, vis_thresh: float = 0.3,
) -> torch.Tensor:
    n = points.shape[0]
    dirs = torch.randn(n * n_dirs, 3, device=points.device)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    pts = points.repeat_interleave(n_dirs, dim=0)
    z_b = z.expand(pts.shape[0], -1)
    t, vis_logit = model(pts, dirs, z_b)
    visible = vis_logit.sigmoid() > vis_thresh
    t = torch.where(visible, t, torch.full_like(t, 1e3))
    return t.view(n, n_dirs).min(dim=-1).values


@torch.no_grad()
def extract_mesh_cond(
    model: DDFCond, z: torch.Tensor, grid_size: int, bbox_half: float,
    n_dirs: int, iso: float, device: str, chunk: int = 65536,
):
    coords = torch.linspace(-bbox_half, bbox_half, grid_size, device=device)
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing="ij")
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    udf_vals = []
    for i in range(0, pts.shape[0], chunk):
        udf_vals.append(query_udf_cond(model, z, pts[i:i + chunk], n_dirs).cpu())
    udf = torch.cat(udf_vals).view(grid_size, grid_size, grid_size).numpy()
    spacing = (2 * bbox_half) / (grid_size - 1)
    try:
        verts, faces, _, _ = marching_cubes(udf, level=iso, spacing=(spacing,) * 3)
    except (ValueError, RuntimeError) as e:
        return None, None, udf, str(e)
    verts = verts + np.array([-bbox_half, -bbox_half, -bbox_half], dtype=np.float32)
    return verts, faces, udf, None


def compute_chamfer_for(
    model: DDFCond, z: torch.Tensor, obj_id: str, gs_path: Path, gt_mesh_path: Path,
    grid_size: int, bbox_half: float, n_dirs: int, iso: float, n_samples: int,
    device: str,
):
    verts, faces, udf, err = extract_mesh_cond(
        model, z, grid_size, bbox_half, n_dirs, iso, device,
    )
    if verts is None:
        # Retry at 1st percentile of UDF.
        new_iso = float(np.percentile(udf, 1))
        spacing = (2 * bbox_half) / (grid_size - 1)
        try:
            verts, faces, _, _ = marching_cubes(udf, level=new_iso, spacing=(spacing,) * 3)
            verts = verts + np.array([-bbox_half] * 3, dtype=np.float32)
            print(f"[{obj_id}] retry iso={new_iso:.4f} succeeded")
        except Exception as e2:
            print(f"[{obj_id}] marching cubes failed even at retry: {e2}")
            return {
                "obj": obj_id, "n_verts": 0, "n_faces": 0,
                "chamfer_mean": float("nan"), "chamfer_median": float("nan"),
            }

    gs = torch.load(gs_path, map_location="cpu", weights_only=False)
    mesh_center = gs["mesh_center"].numpy().astype(np.float32)
    mesh_scale = float(gs["mesh_scale"].item())
    gt_pts = load_gt_points(gt_mesh_path, mesh_center, mesh_scale, n_samples)
    pred_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, n_samples)
    pred_pts = np.asarray(pred_pts, dtype=np.float32)
    cd_mean, cd_med = chamfer(gt_pts, pred_pts)
    return {
        "obj": obj_id,
        "n_verts": int(len(verts)),
        "n_faces": int(len(faces)),
        "chamfer_mean": cd_mean,
        "chamfer_median": cd_med,
    }


def load_model_and_z(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    model = DDFCond(
        pos_freqs=a["pos_freqs"], dir_freqs=a["dir_freqs"],
        hidden_dim=a["hidden_dim"], num_layers=a["num_layers"],
        z_dim=a["z_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    z_table = nn.Embedding(len(ckpt["objects"]), a["z_dim"]).to(device)
    z_table.load_state_dict(ckpt["z_table"])
    return model, z_table, ckpt["objects"], a


def gt_mesh_path_for(obj_id: str) -> Path:
    if obj_id in SHAPENET_MAP:
        return Path(f"data/shapenet/{obj_id}/model.glb")
    if obj_id in NAME_MAP:
        return Path(f"data/gso/{NAME_MAP[obj_id]}/meshes/model.obj")
    raise ValueError(f"unknown obj {obj_id}")


def gs_path_for(obj_id: str) -> Path:
    return Path(f"runs/{obj_id}/gaussians.pt")


def optim_z_for(
    model: DDFCond, z_init: torch.Tensor, gs_path: Path, opt_steps: int,
    lr_z: float, batch_size: int, lambda_vis: float, image_size: int,
    surface_n_ratio: float | None, device: str, log_every: int = 50,
):
    z = z_init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr_z)
    # Freeze the MLP.
    for p in model.parameters():
        p.requires_grad_(False)
    sup = GSSupervisor(
        gs_path=str(gs_path), image_size=image_size, device=device,
        surface_n_ratio=surface_n_ratio,
    )
    losses = []
    for step in range(1, opt_steps + 1):
        origins, dirs, t_gt, hit_gt = sup.sample(batch_size)
        t_pred, vis_logit = model(origins, dirs, z.unsqueeze(0))
        vis_target = hit_gt.float()
        loss_vis = F.binary_cross_entropy_with_logits(vis_logit, vis_target)
        if hit_gt.any():
            loss_dist = F.l1_loss(t_pred[hit_gt], t_gt[hit_gt])
        else:
            loss_dist = torch.zeros((), device=device)
        loss = loss_dist + lambda_vis * loss_vis
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == opt_steps:
            losses.append((step, loss.item(), loss_dist.item(), loss_vis.item()))
            print(
                f"  z-opt step {step:4d} | loss {loss.item():.4f} | "
                f"dist {loss_dist.item():.4f} | vis {loss_vis.item():.4f}"
            )
    # Re-enable MLP grads for safety (caller may resume something else).
    for p in model.parameters():
        p.requires_grad_(True)
    return z.detach(), losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=["trained", "heldout"], required=True)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--grid_size", type=int, default=128)
    ap.add_argument("--bbox_half", type=float, default=1.2)
    ap.add_argument("--n_dirs", type=int, default=32)
    ap.add_argument("--iso", type=float, default=0.05)
    ap.add_argument("--n_samples", type=int, default=20000)
    # heldout-specific
    ap.add_argument("--heldout_ids", nargs="+", default=None)
    ap.add_argument("--opt_steps", type=int, default=500)
    ap.add_argument("--lr_z", type=float, default=1.0e-2)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lambda_vis", type=float, default=0.1)
    ap.add_argument("--image_size", type=int, default=64)
    ap.add_argument("--surface_n_ratio", type=float, default=None)
    ap.add_argument("--cuda_device", default=None)
    args = ap.parse_args()

    device = "cuda"
    model, z_table, objects, train_args = load_model_and_z(Path(args.ckpt), device)
    print(f"loaded model trained on {len(objects)} objects ({objects})")

    results: dict = {"mode": args.mode, "ckpt": args.ckpt, "objects": objects}

    if args.mode == "trained":
        per_obj = []
        for i, obj in enumerate(objects):
            t0 = time.time()
            z = z_table.weight[i].detach().unsqueeze(0)
            gs_p = gs_path_for(obj)
            gt_p = gt_mesh_path_for(obj)
            print(f"\n--- [{i+1}/{len(objects)}] {obj} ---")
            m = compute_chamfer_for(
                model, z, obj, gs_p, gt_p,
                args.grid_size, args.bbox_half, args.n_dirs, args.iso,
                args.n_samples, device,
            )
            m["sec"] = time.time() - t0
            print(
                f"[{obj}] CD mean={m['chamfer_mean']:.4f} "
                f"median={m['chamfer_median']:.4f}  ({m['sec']:.1f}s)"
            )
            per_obj.append(m)
        valid = [m for m in per_obj if not (m["chamfer_mean"] != m["chamfer_mean"])]
        if valid:
            results["mean_cd_mean"] = float(np.mean([m["chamfer_mean"] for m in valid]))
            results["mean_cd_median"] = float(np.mean([m["chamfer_median"] for m in valid]))
        results["per_object"] = per_obj
    else:
        assert args.heldout_ids, "--heldout_ids required in heldout mode"
        z_mean = z_table.weight.mean(0).detach()
        print(f"z_mean stats: norm={z_mean.norm().item():.3f}")
        heldout: list[dict] = []
        for obj in args.heldout_ids:
            print(f"\n=== held-out: {obj} ===")
            gs_p = gs_path_for(obj)
            gt_p = gt_mesh_path_for(obj)
            if not gs_p.exists():
                print(f"[skip] {gs_p} missing")
                heldout.append({"obj": obj, "skipped": True})
                continue

            # (a) zero-shot z=mean
            print(f"  (a) zero-shot z=mean")
            m_a = compute_chamfer_for(
                model, z_mean.unsqueeze(0), obj, gs_p, gt_p,
                args.grid_size, args.bbox_half, args.n_dirs, args.iso,
                args.n_samples, device,
            )
            m_a["strategy"] = "zero_shot_mean"
            print(
                f"  [{obj}] zero-shot CD mean={m_a['chamfer_mean']:.4f} "
                f"median={m_a['chamfer_median']:.4f}"
            )

            # (b) test-time z optim
            print(f"  (b) z-optim, MLP frozen, {args.opt_steps} steps")
            t0 = time.time()
            z_opt, opt_log = optim_z_for(
                model, z_mean, gs_p, args.opt_steps, args.lr_z,
                args.batch_size, args.lambda_vis, args.image_size,
                args.surface_n_ratio, device,
            )
            z_opt_sec = time.time() - t0
            m_b = compute_chamfer_for(
                model, z_opt.unsqueeze(0), obj, gs_p, gt_p,
                args.grid_size, args.bbox_half, args.n_dirs, args.iso,
                args.n_samples, device,
            )
            m_b["strategy"] = "z_optim"
            m_b["opt_steps"] = args.opt_steps
            m_b["opt_sec"] = z_opt_sec
            print(
                f"  [{obj}] z-optim CD mean={m_b['chamfer_mean']:.4f} "
                f"median={m_b['chamfer_median']:.4f} ({z_opt_sec:.1f}s)"
            )
            heldout.append({
                "obj": obj,
                "zero_shot": m_a,
                "z_optim": m_b,
            })
        results["heldout"] = heldout

    out_json = Path(args.out_json) if args.out_json else (
        Path(args.ckpt).parent / f"eval_{args.mode}.json"
    )
    with out_json.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
