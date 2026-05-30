"""Convert our `runs/<obj>/views/{rgb_*.png, cameras.npz}` into a nerfstudio
dataset (`transforms.json` + image dir). Camera convention is converted from
OpenCV (our internal convention) -> OpenGL (nerfstudio).

Output layout (writes to <out_dir>/):
    transforms.json
    images/rgb_000.png  (symlink to source)
    images/...

We also drop a tight `aabb_scale=1.0` and `applied_transform=identity` so the
scene stays in our normalized coordinate frame, since this is render-from-known-
geometry data with cameras already on a sphere of radius ~2.5 centered at the
origin. Disables nerfstudio's auto orientation/centering on the consumer side
via the dataparser flags `--orientation-method none --center-method none
--auto-scale-poses False`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def opencv_c2w_to_opengl(c2w: np.ndarray) -> np.ndarray:
    """OpenCV (right, down, forward) -> OpenGL (right, up, back).

    Negate the Y and Z columns of the rotation block. Translation unchanged.
    """
    out = c2w.copy().astype(np.float32)
    out[..., :3, 1] *= -1.0
    out[..., :3, 2] *= -1.0
    return out


def make_dataset(views_dir: Path, out_dir: Path, eval_every: int = 8) -> dict:
    """Build transforms.json in out_dir. Returns the dict that was written."""
    data = np.load(views_dir / "cameras.npz")
    c2w_cv = data["c2w"]  # (N, 4, 4) OpenCV
    K = data["K"]         # (3, 3)
    H = int(data["image_size"])
    W = int(data["image_size"])

    c2w_gl = opencv_c2w_to_opengl(c2w_cv)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    # Symlink images (faster than copy + identical for nerfstudio).
    n = c2w_gl.shape[0]
    frames = []
    for i in range(n):
        src = (views_dir / f"rgb_{i:03d}.png").resolve()
        dst = img_dir / f"rgb_{i:03d}.png"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.symlink(src, dst)
        except OSError:
            # fallback: read+write
            import shutil
            shutil.copy(src, dst)
        frames.append({
            "file_path": f"./images/rgb_{i:03d}.png",
            "transform_matrix": c2w_gl[i].tolist(),
        })

    meta = {
        "camera_model": "OPENCV",  # pinhole (no distortion)
        "fl_x": fx, "fl_y": fy,
        "cx": cx, "cy": cy,
        "w": W, "h": H,
        "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        # Identity applied_transform so 'dataparser_transform' = identity x scale.
        "applied_transform": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        "frames": frames,
    }
    (out_dir / "transforms.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True,
                    help="short id; will read runs/<obj>/views/")
    ap.add_argument("--out_dir", default=None,
                    help="default runs/sota_comparison/<obj>/ns_data")
    ap.add_argument("--views_dir", default=None,
                    help="override; default runs/<obj>/views")
    args = ap.parse_args()

    views_dir = Path(args.views_dir) if args.views_dir else Path(f"runs/{args.obj}/views")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"runs/sota_comparison/{args.obj}/ns_data")

    if not views_dir.exists():
        raise SystemExit(f"views_dir not found: {views_dir}")

    meta = make_dataset(views_dir, out_dir)
    print(f"[{args.obj}] wrote {out_dir}/transforms.json ({len(meta['frames'])} frames)")


if __name__ == "__main__":
    main()
