"""Side-by-side turntable renders: GS via gsplat vs DDF-extracted mesh.

Renders N orbit cameras of the GS scene (using gsplat) and the same camera
poses against the DDF-extracted .ply mesh (rasterized with trimesh's CPU-side
OpenGL via pyrender, then composed white-bg). Writes a single contact-sheet
PNG plus the individual per-view PNGs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


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

# Headless rendering for pyrender / trimesh.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from PIL import Image
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gsplat import rasterization
from src.gs_supervisor import _spherical_camera


def _load_gs(gs_path: Path, device: str):
    ckpt = torch.load(gs_path, map_location=device, weights_only=False)
    quats = ckpt["quats"].to(device)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return dict(
        means=ckpt["means"].to(device).contiguous(),
        quats=quats.contiguous(),
        scales=ckpt["scales"].to(device).exp().contiguous(),
        opacities=ckpt["opacities"].to(device).sigmoid().contiguous(),
        colors=ckpt["colors"].to(device).sigmoid().contiguous(),
    )


@torch.no_grad()
def render_gs_one(gs, elev_deg, azim_deg, radius, image_size, device):
    w2c, K, c2w = _spherical_camera(elev_deg, azim_deg, radius, image_size, device=device)
    renders, alphas, _ = rasterization(
        means=gs["means"], quats=gs["quats"], scales=gs["scales"],
        opacities=gs["opacities"], colors=gs["colors"],
        viewmats=w2c.unsqueeze(0), Ks=K.unsqueeze(0),
        width=image_size, height=image_size, sh_degree=None,
        render_mode="RGB+ED",
    )
    rgb = renders[0, ..., :3].clamp(0, 1).cpu().numpy()
    a = alphas[0, ..., 0].cpu().numpy()
    rgb = rgb + (1 - a[..., None])  # white bg
    return np.clip(rgb, 0, 1), c2w.cpu().numpy(), K.cpu().numpy()


def render_mesh_one(mesh: trimesh.Trimesh, c2w_opencv: np.ndarray, K: np.ndarray,
                    image_size: int):
    """Render the mesh with pyrender using the OpenCV-convention c2w.

    pyrender wants OpenGL camera convention (x-right, y-up, z-out of screen),
    so we flip y and z of the OpenCV c2w when handing it over.
    """
    import pyrender
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.4, 0.4, 0.4])
    # The mesh has no texture; give it a soft gray material.
    mesh_render = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh_render)

    # Camera: convert OpenCV c2w to OpenGL c2w.
    c2w_gl = c2w_opencv.copy()
    c2w_gl[:3, 1] *= -1
    c2w_gl[:3, 2] *= -1

    fy = K[1, 1]
    fx = K[0, 0]
    cy = K[1, 2]
    cx = K[0, 2]
    cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=10.0)
    scene.add(cam, pose=c2w_gl)

    # Diffuse fill light at camera position.
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.5)
    scene.add(light, pose=c2w_gl)

    r = pyrender.OffscreenRenderer(viewport_width=image_size,
                                   viewport_height=image_size)
    color, _ = r.render(scene)
    r.delete()
    return color.astype(np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs", type=Path, required=True)
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_views", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--radius", type=float, default=2.5)
    ap.add_argument("--elev", type=float, default=15.0)
    ap.add_argument("--cuda_device", default=None, help="handled pre-import")
    args = ap.parse_args()

    device = "cuda"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading GS from {args.gs}")
    gs = _load_gs(args.gs, device)
    print(f"  {gs['means'].shape[0]} Gaussians")

    print(f"loading mesh from {args.mesh}")
    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    cols_gs, cols_mesh = [], []
    for i in range(args.n_views):
        azim = 360.0 * i / args.n_views
        rgb_gs, c2w, K = render_gs_one(gs, args.elev, azim, args.radius,
                                       args.image_size, device)
        rgb_mesh = render_mesh_one(mesh, c2w, K, args.image_size)
        # Save individual pair.
        pair = np.concatenate([rgb_gs, rgb_mesh], axis=1)
        Image.fromarray((pair * 255).astype(np.uint8)).save(
            args.out_dir / f"view_{i:02d}.png"
        )
        cols_gs.append(rgb_gs)
        cols_mesh.append(rgb_mesh)
        print(f"  view {i}: azim={azim:.1f}")

    # Contact sheet: 2 rows (top GS, bottom mesh), n_views columns.
    row_gs = np.concatenate(cols_gs, axis=1)
    row_mesh = np.concatenate(cols_mesh, axis=1)
    sheet = np.concatenate([row_gs, row_mesh], axis=0)
    Image.fromarray((sheet * 255).astype(np.uint8)).save(
        args.out_dir / "turntable_sheet.png"
    )
    print(f"saved contact sheet -> {args.out_dir / 'turntable_sheet.png'}")


if __name__ == "__main__":
    main()
