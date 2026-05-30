"""Image supervisor: provides pixel rays + GT RGB + foreground mask from
pre-rendered multi-view images. No GS, no NeuS — pure image supervision.

Each sample() call picks a random view, samples random foreground + background
pixels, and returns per-pixel rays with GT color and mask.
"""

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class ImageSupervisor:

    def __init__(
        self,
        views_dir: str,
        device: str = "cuda",
        fg_ratio: float = 0.7,
        bg_threshold: float = 0.98,
    ):
        self.device = device
        self.fg_ratio = fg_ratio

        vdir = Path(views_dir)
        cam = np.load(vdir / "cameras.npz")
        self.c2ws = torch.from_numpy(cam["c2w"]).float().to(device)
        self.K = torch.from_numpy(cam["K"]).float().to(device)
        self.image_size = int(cam["image_size"])
        self.n_views = self.c2ws.shape[0]

        imgs = []
        for p in sorted(vdir.glob("rgb_*.png")):
            im = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            imgs.append(torch.from_numpy(im).to(device))
        self.images = torch.stack(imgs)

        # Precompute foreground masks (non-white pixels)
        self.fg_masks = (self.images.sum(-1) / 3.0) < bg_threshold

        # Precompute per-pixel ray directions in camera space (shared across views)
        isz = self.image_size
        ys, xs = torch.meshgrid(
            torch.arange(isz, device=device, dtype=torch.float32),
            torch.arange(isz, device=device, dtype=torch.float32),
            indexing="ij",
        )
        K = self.K
        px = (xs + 0.5 - K[0, 2]) / K[0, 0]
        py = (ys + 0.5 - K[1, 2]) / K[1, 1]
        self.dirs_cam = torch.stack([px, py, torch.ones_like(px)], dim=-1)
        self.dirs_cam = self.dirs_cam / self.dirs_cam.norm(dim=-1, keepdim=True)

        n_fg = sum(m.sum().item() for m in self.fg_masks)
        n_total = self.n_views * isz * isz
        print(f"ImageSupervisor: {self.n_views} views, {isz}px, "
              f"fg={n_fg/n_total:.1%}")

    def sample(
        self, batch_size: int,
    ) -> dict:
        """Sample pixel rays from a random view.

        Returns dict with:
            origins: (B, 3) ray origins in world space
            dirs: (B, 3) ray directions in world space (unit)
            rgb_gt: (B, 3) ground-truth pixel colors
            fg_gt: (B,) bool — True if foreground pixel
            view_idx: int — which view was sampled
        """
        vi = torch.randint(0, self.n_views, ()).item()
        c2w = self.c2ws[vi]
        R = c2w[:3, :3]
        t = c2w[:3, 3]

        img = self.images[vi]
        fg_mask = self.fg_masks[vi]

        n_fg = int(batch_size * self.fg_ratio)
        n_bg = batch_size - n_fg

        # Sample foreground pixels
        fg_idx = fg_mask.nonzero()
        if fg_idx.shape[0] > n_fg:
            sel_fg = fg_idx[torch.randperm(fg_idx.shape[0], device=self.device)[:n_fg]]
        else:
            sel_fg = fg_idx
            n_fg = sel_fg.shape[0]

        # Sample background pixels
        bg_mask = ~fg_mask
        bg_idx = bg_mask.nonzero()
        if bg_idx.shape[0] > n_bg:
            sel_bg = bg_idx[torch.randperm(bg_idx.shape[0], device=self.device)[:n_bg]]
        else:
            sel_bg = bg_idx
            n_bg = sel_bg.shape[0]

        sel = torch.cat([sel_fg, sel_bg], dim=0)
        pixel_y = sel[:, 0]
        pixel_x = sel[:, 1]

        dirs_cam_sel = self.dirs_cam[pixel_y, pixel_x]
        dirs_world = dirs_cam_sel @ R.T
        origins = t.unsqueeze(0).expand(sel.shape[0], -1)

        rgb_gt = img[pixel_y, pixel_x]
        fg_gt = torch.cat([
            torch.ones(n_fg, dtype=torch.bool, device=self.device),
            torch.zeros(n_bg, dtype=torch.bool, device=self.device),
        ])

        return {
            "origins": origins,
            "dirs": dirs_world,
            "rgb_gt": rgb_gt,
            "fg_gt": fg_gt,
            "view_idx": vi,
        }
