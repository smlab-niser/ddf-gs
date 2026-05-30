"""Multi-view eval averaged over 24 viewpoints, 29 objects.

For each obj, run eval_ddf-style evaluation at 24 cameras
  azim = linspace(0, 360, 24, endpoint=False), elev=20, radius=2.5
Average depth-L1 mean/median + mask-IoU per object. Compare to single-view
numbers in runs/eval_summary.png.

Writes runs/cheap_eval/multiview/multiview.json + .md.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.ddf_model import DDF
from src.gs_supervisor import GSSupervisor, _pixels_to_world_rays, _spherical_camera
from gsplat import rasterization


OBJECTS = [
    "airplane", "allo", "bagel", "blueMug", "boatShoe", "bowl", "bull",
    "bundtPan", "bus", "car", "chocoBox", "clock", "eagle", "hammer",
    "horse", "lion", "mug", "orca", "panda", "rhino", "sausage", "shoe",
    "spino", "teapot", "teddy", "thomas", "torch", "triceratop", "turtle",
]


def ddf_ckpt_for(obj: str) -> Path:
    if obj == "bull":
        p = Path("runs/bull_ddf_v3/ddf_final.pt")
        if p.exists():
            return p
    return Path(f"runs/{obj}_ddf/ddf_final.pt")


def gs_path_for(obj: str) -> Path:
    return Path(f"runs/{obj}/gaussians.pt")


@torch.no_grad()
def eval_view(model, sup, elev, azim, radius, image_size, device):
    H = W = image_size
    w2c, K, c2w = _spherical_camera(elev, azim, radius, H, device=device)
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

    if hit_gt.any():
        abs_err = (t_pred[hit_gt] - t_gt[hit_gt]).abs()
        l1_mean = float(abs_err.mean())
        l1_med = float(abs_err.median())
    else:
        l1_mean = float("nan")
        l1_med = float("nan")

    iou_num = ((vis_prob > 0.5) & hit_gt).sum().item()
    iou_den = ((vis_prob > 0.5) | hit_gt).sum().item()
    iou = iou_num / max(iou_den, 1)
    return l1_mean, l1_med, iou


def eval_obj(obj: str, n_views: int, elev: float, radius: float,
             image_size: int, device: str):
    ckpt_p = ddf_ckpt_for(obj); gs_p = gs_path_for(obj)
    if not (ckpt_p.exists() and gs_p.exists()):
        print(f"[{obj}] missing ckpt or gs: {ckpt_p} / {gs_p}")
        return None
    ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = DDF(
        pos_freqs=cfg["model"]["pos_freqs"], dir_freqs=cfg["model"]["dir_freqs"],
        hidden_dim=cfg["model"]["hidden_dim"], num_layers=cfg["model"]["num_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    sup = GSSupervisor(str(gs_p), image_size=image_size, device=device)

    azims = np.linspace(0.0, 360.0, n_views, endpoint=False)
    l1m, l1med, ious = [], [], []
    for az in azims:
        a, b, c = eval_view(model, sup, elev, float(az), radius, image_size, device)
        l1m.append(a); l1med.append(b); ious.append(c)
    return {
        "obj": obj,
        "l1_mean_avg": float(np.mean(l1m)),
        "l1_median_avg": float(np.mean(l1med)),
        "iou_avg": float(np.mean(ious)),
        "l1_mean_std": float(np.std(l1m)),
        "iou_std": float(np.std(ious)),
        "n_views": n_views,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_views", type=int, default=24)
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--radius", type=float, default=2.5)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="runs/cheap_eval/multiview")
    ap.add_argument("--objs", nargs="*", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    objs = args.objs if args.objs else OBJECTS

    rows = []
    for obj in objs:
        try:
            r = eval_obj(obj, args.n_views, args.elev, args.radius,
                         args.image_size, args.device)
        except Exception as e:
            print(f"[{obj}] FAILED: {e}")
            continue
        if r is None:
            continue
        rows.append(r)
        print(f"[{obj:<12s}] L1mean={r['l1_mean_avg']:.4f} "
              f"L1med={r['l1_median_avg']:.4f} IoU={r['iou_avg']:.3f}")

    if rows:
        mean = {
            "obj": "MEAN",
            "l1_mean_avg": float(np.mean([r["l1_mean_avg"] for r in rows])),
            "l1_median_avg": float(np.mean([r["l1_median_avg"] for r in rows])),
            "iou_avg": float(np.mean([r["iou_avg"] for r in rows])),
        }
        rows.append(mean)

    (out_dir / "multiview.json").write_text(json.dumps(rows, indent=2))

    cols = ["obj", "l1_mean_avg", "l1_median_avg", "iou_avg"]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for r in rows:
        vals = [r["obj"]]
        for c in cols[1:]:
            v = r.get(c, float("nan"))
            vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    (out_dir / "multiview.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_dir}/multiview.json + multiview.md")


if __name__ == "__main__":
    main()
