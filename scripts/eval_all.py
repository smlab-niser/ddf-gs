"""Evaluate every trained DDF/GS pair under runs/ and produce a summary table + grid.

For each `<obj>_ddf/ddf_final.pt` paired with `<obj>/gaussians.pt`:
  - Render the GS at a fixed held-out view, query the DDF on the same rays.
  - Compute depth L1 (on hits) and mask IoU.
  - Save a 2x2 per-object grid; stitch all per-object grids into one figure.
"""

import argparse
from pathlib import Path

import torch
import torchvision

from src.ddf_model import DDF
from src.gs_supervisor import GSSupervisor, _pixels_to_world_rays, _spherical_camera
from gsplat import rasterization


def eval_one(ddf_ckpt: Path, gs_path: Path, image_size: int, elev: float, azim: float,
             radius: float, device: str):
    ckpt = torch.load(ddf_ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = DDF(
        pos_freqs=cfg["model"]["pos_freqs"], dir_freqs=cfg["model"]["dir_freqs"],
        hidden_dim=cfg["model"]["hidden_dim"], num_layers=cfg["model"]["num_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sup = GSSupervisor(str(gs_path), image_size=image_size, device=device)
    H = W = image_size
    w2c, K, c2w = _spherical_camera(elev, azim, radius, H, device=device)

    with torch.no_grad():
        renders, alphas, _ = rasterization(
            means=sup.means, quats=sup.quats, scales=sup.scales,
            opacities=sup.opacities, colors=sup.colors,
            viewmats=w2c.unsqueeze(0), Ks=K.unsqueeze(0),
            width=W, height=H, sh_degree=None, render_mode="RGB+ED",
        )
        depth_gs = renders[0, ..., 3]
        alpha_gs = alphas[0, ..., 0]

        origins, dirs = _pixels_to_world_rays(K, c2w, H, device=device)
        forward_world = c2w[:3, 2]
        cos_theta = (dirs * forward_world.view(1, 1, 3)).sum(-1).clamp_min(1e-4)
        t_gt = depth_gs / cos_theta
        hit_gt = alpha_gs > 0.5

        flat_o = origins.reshape(-1, 3)
        flat_d = dirs.reshape(-1, 3)
        t_pred, vis_logit = model(flat_o, flat_d)
        t_pred = t_pred.reshape(H, W)
        vis_prob = vis_logit.sigmoid().reshape(H, W)

    abs_err = (t_pred[hit_gt] - t_gt[hit_gt]).abs() if hit_gt.any() else torch.zeros(0)
    iou_num = ((vis_prob > 0.5) & hit_gt).sum().item()
    iou_den = ((vis_prob > 0.5) | hit_gt).sum().item()
    iou = iou_num / max(iou_den, 1)

    def norm_depth(d, mask):
        d = d.clone()
        if mask.any():
            vmin = d[mask].min(); vmax = d[mask].max()
            d = (d - vmin) / (vmax - vmin).clamp_min(1e-6)
        d[~mask] = 0
        return d.clamp(0, 1)

    gt_depth_img = norm_depth(t_gt, hit_gt).unsqueeze(0).repeat(3, 1, 1)
    pred_depth_img = norm_depth(t_pred, vis_prob > 0.5).unsqueeze(0).repeat(3, 1, 1)
    gt_mask_img = hit_gt.float().unsqueeze(0).repeat(3, 1, 1)
    pred_mask_img = vis_prob.unsqueeze(0).repeat(3, 1, 1)

    grid = torchvision.utils.make_grid(
        torch.stack([gt_depth_img, pred_depth_img, gt_mask_img, pred_mask_img]),
        nrow=2, padding=4,
    )
    return {
        "depth_l1_mean": float(abs_err.mean()) if len(abs_err) else float("nan"),
        "depth_l1_median": float(abs_err.median()) if len(abs_err) else float("nan"),
        "mask_iou": float(iou),
        "n_hits": int(hit_gt.sum().item()),
        "n_pixels": int(hit_gt.numel()),
        "grid": grid,
    }


def find_pairs(runs_dir: Path):
    pairs = []
    for ddf_dir in sorted(runs_dir.glob("*_ddf*")):
        ddf_ckpt = ddf_dir / "ddf_final.pt"
        if not ddf_ckpt.exists():
            continue
        name = ddf_dir.name.replace("_ddf_v3", "").replace("_ddf", "")
        gs = runs_dir / name / "gaussians.pt"
        if gs.exists():
            pairs.append((ddf_dir.name, ddf_ckpt, gs))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/eval_summary.png")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=45.0)
    ap.add_argument("--radius", type=float, default=2.5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    runs_dir = Path(args.runs)
    pairs = find_pairs(runs_dir)
    print(f"found {len(pairs)} DDF/GS pairs under {runs_dir}")
    print(f"{'name':<20s}  {'L1 mean':>9s}  {'L1 med':>9s}  {'IoU':>6s}  {'hits':>10s}")

    grids = []
    rows = []
    for name, ddf, gs in pairs:
        try:
            r = eval_one(ddf, gs, args.image_size, args.elev, args.azim, args.radius, args.device)
        except Exception as e:
            print(f"{name:<20s}  FAILED: {e}")
            continue
        print(f"{name:<20s}  {r['depth_l1_mean']:>9.4f}  {r['depth_l1_median']:>9.4f}"
              f"  {r['mask_iou']:>6.3f}  {r['n_hits']:>5d}/{r['n_pixels']}")
        torchvision.utils.save_image(r["grid"], ddf.parent / "eval_view.png")
        grids.append(r["grid"])
        rows.append((name, r))

    if grids:
        mega = torchvision.utils.make_grid(torch.stack(grids), nrow=2, padding=8)
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        torchvision.utils.save_image(mega, out)
        print(f"\nmega-grid saved to {out}")


if __name__ == "__main__":
    main()
