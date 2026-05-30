"""Render a DDF photo-render checkpoint to images and compute PSNR vs GT views.

Loads a `DDFHashGridPhotoRender` ckpt, renders K held-out cameras from
`<run>/views/` using the same NeuS-style volume rendering as training, and
computes PSNR against the corresponding GT pngs.

Outputs:
    <out_dir>/render_<view_idx>.png      side-by-side GT | DDF
    <out_dir>/psnr.json                  per-view + mean PSNR
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

# Pin CUDA device before torch CUDA init.
def _pin():
    import sys
    for i, a in enumerate(sys.argv):
        if a == "--cuda_device" and i + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[i + 1]
            return
        if a.startswith("--cuda_device="):
            os.environ["CUDA_VISIBLE_DEVICES"] = a.split("=", 1)[1]
            return
_pin()

from src.ddf_photorender import DDFHashGridPhotoRender  # noqa: E402
from src.photo_render_supervisor import _ray_box_t_interval  # noqa: E402
from src.train_ddf import _volume_render  # noqa: E402


def _build_model(cfg, device):
    m_cfg = cfg["model"]
    return DDFHashGridPhotoRender(
        dir_freqs=m_cfg.get("dir_freqs", 4),
        hidden_dim=m_cfg.get("hidden_dim", 64),
        num_layers=m_cfg.get("num_layers", 2),
        n_levels=m_cfg.get("n_levels", 16),
        feat_dim=m_cfg.get("feat_dim", 2),
        log2_table_size=m_cfg.get("log2_table_size", 19),
        base_res=m_cfg.get("base_res", 16),
        growth=m_cfg.get("growth", 1.5),
        bbox_half=m_cfg.get("bbox_half", 1.2),
        beta_init=m_cfg.get("beta_init", 10.0),
    ).to(device)


@torch.no_grad()
def render_view(
    model, c2w, K, H, W, bbox_half, n_samples, bg_color, device,
    chunk: int = 16384,
):
    """Volume-render one camera. Returns (H, W, 3) float in [0,1]."""
    # Per-pixel camera-frame dirs.
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    px = (xs + 0.5 - K[0, 2]) / K[0, 0]
    py = (ys + 0.5 - K[1, 2]) / K[1, 1]
    dirs_cam = torch.stack([px, py, torch.ones_like(px)], dim=-1).reshape(-1, 3)
    dirs_cam = dirs_cam / dirs_cam.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    R = c2w[:3, :3]
    t = c2w[:3, 3]
    dirs_world = dirs_cam @ R.T
    origins = t.view(1, 3).expand_as(dirs_world)

    t_near, t_far = _ray_box_t_interval(origins, dirs_world, bbox_half)
    out = torch.zeros(origins.shape[0], 3, device=device)
    for s in range(0, origins.shape[0], chunk):
        e = min(s + chunk, origins.shape[0])
        rgb, _, _ = _volume_render(
            model, origins[s:e].contiguous(), dirs_world[s:e].contiguous(),
            t_near[s:e].contiguous(), t_far[s:e].contiguous(),
            n_samples=n_samples, bg_color=bg_color, jitter=False,
        )
        out[s:e] = rgb
    return out.view(H, W, 3).clamp(0.0, 1.0)


def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= 0:
        return 99.0
    return -10.0 * math.log10(mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--views_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=64)
    ap.add_argument("--n_views", type=int, default=8,
                    help="render this many views (even strides through the 50)")
    ap.add_argument("--cuda_device", default=None)
    args = ap.parse_args()

    device = "cuda"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = _build_model(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cam = np.load(Path(args.views_dir) / "cameras.npz")
    c2w_all = torch.from_numpy(cam["c2w"]).float().to(device)
    K = torch.from_numpy(cam["K"]).float().to(device)
    image_size = int(cam["image_size"])
    V = c2w_all.shape[0]
    bbox_half = float(cfg["model"].get("bbox_half", 1.2))
    bg = torch.tensor([1.0, 1.0, 1.0], device=device)

    view_idxs = np.linspace(0, V - 1, args.n_views).astype(int).tolist()
    psnrs = {}
    for v in view_idxs:
        gt = np.asarray(Image.open(Path(args.views_dir) / f"rgb_{v:03d}.png").convert("RGB"),
                        dtype=np.float32) / 255.0
        pred = render_view(
            model, c2w_all[v], K, image_size, image_size,
            bbox_half=bbox_half, n_samples=args.n_samples, bg_color=bg, device=device,
        ).cpu().numpy()
        p = psnr(pred, gt)
        psnrs[int(v)] = p
        side = np.concatenate([gt, pred], axis=1)
        Image.fromarray((side * 255).clip(0, 255).astype(np.uint8)).save(
            out_dir / f"render_{v:03d}.png",
        )
        print(f"view {v:3d}: PSNR {p:.2f}")

    mean_psnr = float(np.mean(list(psnrs.values())))
    (out_dir / "psnr.json").write_text(json.dumps({
        "per_view": psnrs, "mean": mean_psnr,
        "ckpt": str(args.ckpt), "n_samples": args.n_samples,
    }, indent=2))
    print(f"mean PSNR: {mean_psnr:.2f}  ({len(psnrs)} views)")


if __name__ == "__main__":
    main()
