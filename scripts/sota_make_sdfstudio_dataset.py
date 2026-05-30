"""Convert `runs/<obj>/views/{rgb_*.png, cameras.npz}` into an SDFStudio
dataset (`meta_data.json` + images dir, as consumed by
nerfstudio.data.dataparsers.sdfstudio_dataparser.SDFStudio).

Spec:
    meta_data.json:
        {
          "height": H,
          "width": W,
          "scene_box": { "aabb": [[xmin,ymin,zmin],[xmax,ymax,zmax]],
                         "near": 0.5, "far": 4.5, "radius": 1.0,
                         "collider_type": "box" },
          "frames": [ { "rgb_path": "images/000.png",
                        "intrinsics": [[fx,0,cx,0],[0,fy,cy,0],[0,0,1,0],[0,0,0,1]],
                        "camtoworld": [[..4x4..]] }, ... ]
        }

Camera convention: OpenCV (the dataparser flips Y,Z internally) -- our c2w is
already OpenCV.

Foreground masks are also supported but we skip them; the renders are RGB
over white background; the scene box [-1.2, 1.2]^3 covers the normalized
mesh (mesh_scale normalizes longest extent to ~1.0).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def make_dataset(views_dir: Path, out_dir: Path,
                 scene_radius: float = 1.0,
                 near: float = 0.5,
                 far: float = 4.5) -> dict:
    data = np.load(views_dir / "cameras.npz")
    c2w_cv = data["c2w"].astype(np.float32)  # (N, 4, 4) OpenCV
    K = data["K"].astype(np.float32)         # (3, 3)
    H = int(data["image_size"])
    W = int(data["image_size"])

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    intrinsics_4x4 = [
        [fx, 0.0, cx, 0.0],
        [0.0, fy, cy, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    frames = []
    n = c2w_cv.shape[0]
    for i in range(n):
        src = (views_dir / f"rgb_{i:03d}.png").resolve()
        dst_name = f"rgb_{i:03d}.png"
        dst = img_dir / dst_name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.symlink(src, dst)
        except OSError:
            import shutil
            shutil.copy(src, dst)
        frames.append({
            "rgb_path": f"images/{dst_name}",
            "intrinsics": intrinsics_4x4,
            "camtoworld": c2w_cv[i].tolist(),
        })

    # Use a bbox slightly larger than our normalized mesh extent; the mesh
    # has been scaled so its longest extent is ~1 (mesh_scale), so [-1, 1]
    # tightly bounds it; we leave room to [-scene_radius, scene_radius]^3.
    r = float(scene_radius)
    meta = {
        "height": H,
        "width": W,
        "has_mono_prior": False,
        "scene_box": {
            "aabb": [[-r, -r, -r], [r, r, r]],
            "near": near,
            "far": far,
            "radius": r,
            "collider_type": "near_far",
        },
        "frames": frames,
    }
    (out_dir / "meta_data.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--views_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--scene_radius", type=float, default=1.0)
    ap.add_argument("--near", type=float, default=0.5)
    ap.add_argument("--far", type=float, default=4.5)
    args = ap.parse_args()

    views_dir = Path(args.views_dir) if args.views_dir else Path(f"runs/{args.obj}/views")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"runs/sota_comparison/{args.obj}/sdf_data")
    if not views_dir.exists():
        raise SystemExit(f"views_dir not found: {views_dir}")
    meta = make_dataset(views_dir, out_dir, args.scene_radius, args.near, args.far)
    print(f"[{args.obj}] wrote {out_dir}/meta_data.json ({len(meta['frames'])} frames, aabb=+-{args.scene_radius})")


if __name__ == "__main__":
    main()
