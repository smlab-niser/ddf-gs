"""Compare DDF predictions vs GS-rendered ground truth over a held-out view.

Renders the GS at a fixed camera, queries the DDF at the same per-pixel rays,
saves a 2x2 grid: [GT depth | pred depth] / [GT mask | pred vis].
"""

import argparse
from pathlib import Path

import torch
import torchvision

from src.ddf_model import DDF
from src.gs_supervisor import GSSupervisor, _pixels_to_world_rays, _spherical_camera
from gsplat import rasterization


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddf_ckpt", required=True)
    ap.add_argument("--gs_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=45.0)
    ap.add_argument("--radius", type=float, default=2.5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ckpt = torch.load(args.ddf_ckpt, map_location=args.device, weights_only=False)
    cfg = ckpt["cfg"]
    model = DDF(
        pos_freqs=cfg["model"]["pos_freqs"], dir_freqs=cfg["model"]["dir_freqs"],
        hidden_dim=cfg["model"]["hidden_dim"], num_layers=cfg["model"]["num_layers"],
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    sup = GSSupervisor(args.gs_path, image_size=args.image_size, device=args.device)

    H = W = args.image_size
    w2c, K, c2w = _spherical_camera(args.elev, args.azim, args.radius, H, device=args.device)

    with torch.no_grad():
        renders, alphas, _ = rasterization(
            means=sup.means, quats=sup.quats, scales=sup.scales,
            opacities=sup.opacities, colors=sup.colors,
            viewmats=w2c.unsqueeze(0), Ks=K.unsqueeze(0),
            width=W, height=H, sh_degree=None, render_mode="RGB+ED",
        )
        depth_gs = renders[0, ..., 3]
        alpha_gs = alphas[0, ..., 0]

        origins, dirs = _pixels_to_world_rays(K, c2w, H, device=args.device)
        forward_world = c2w[:3, 2]
        cos_theta = (dirs * forward_world.view(1, 1, 3)).sum(-1).clamp_min(1e-4)
        t_gt = depth_gs / cos_theta
        hit_gt = alpha_gs > 0.5

        flat_o = origins.reshape(-1, 3)
        flat_d = dirs.reshape(-1, 3)
        t_pred, vis_logit = model(flat_o, flat_d)
        t_pred = t_pred.reshape(H, W)
        vis_prob = vis_logit.sigmoid().reshape(H, W)

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
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(grid, out)

    hit_mask = hit_gt
    abs_err = (t_pred[hit_mask] - t_gt[hit_mask]).abs()
    iou_num = ((vis_prob > 0.5) & hit_gt).sum().item()
    iou_den = ((vis_prob > 0.5) | hit_gt).sum().item()
    print(f"saved {out}")
    print(f"depth L1 on hits: mean={abs_err.mean():.4f} median={abs_err.median():.4f} "
          f"n_hits={hit_mask.sum().item()}/{H*W}")
    print(f"mask IoU: {iou_num/max(iou_den,1):.3f}")


if __name__ == "__main__":
    main()
