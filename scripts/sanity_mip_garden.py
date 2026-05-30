"""Sanity check + camera export for the Mip-NeRF360 garden real scene.

Confirms the reused GS (runs/garden/gaussians.pt, normalized) and the COLMAP
cameras line up in ONE frame: renders the GS from a few COLMAP poses (transformed
into the normalized frame) and saves them side-by-side with the real downsampled
photos. If they match, cameras are correct and the GI pipeline can proceed.

Camera transform: normalization is a similarity (uniform scale s=mesh_scale +
translation mesh_center), so a camera's ORIENTATION is unchanged and only its
CENTER moves: C_norm = (C_world - mesh_center) * mesh_scale. Projection is then
identical (both numerator and denominator scale by s). COLMAP is OpenCV
convention (x right, y down, z forward) = our gsplat viewmat convention.

Also writes runs/garden/cameras.npz: per-image w2c (normalized frame), K (scaled
to the images_4 resolution), image name, W, H.

Run: PYTHONPATH=. CUDA_VISIBLE_DEVICES=<g> python scripts/sanity_mip_garden.py
"""
import os, sys, argparse
import numpy as np, torch
from PIL import Image
sys.path.insert(0, ".")
from gsplat import rasterization
from nerfstudio.data.utils.colmap_parsing_utils import (
    read_cameras_binary, read_images_binary, qvec2rotmat)

SCENE = "data/mip_nerf/garden"
GS = "runs/garden/gaussians.pt"
IMG_DIR = f"{SCENE}/images_4"
COLMAP = f"{SCENE}/sparse/0"


def cam_K(cam, sx, sy):
    """COLMAP camera -> 3x3 K scaled by (sx,sy) for the downsampled image."""
    m, p = cam.model, cam.params
    if m in ("PINHOLE", "OPENCV"):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    elif m in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = p[0]; cx, cy = p[1], p[2]
    else:
        raise ValueError(f"unhandled COLMAP model {m}")
    return np.array([[fx*sx, 0, cx*sx], [0, fy*sy, cy*sy], [0, 0, 1]], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sanity", type=int, default=3)
    ap.add_argument("--out", default="renders/mip_garden_sanity.png")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    gs = torch.load(GS, map_location=dev, weights_only=False)
    mc = gs["mesh_center"].cpu().numpy().astype(np.float64)
    ms = float(gs["mesh_scale"].item())
    means = gs["means"].to(dev); quats = gs["quats"].to(dev)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = gs["scales"].to(dev).exp(); opac = gs["opacities"].to(dev).sigmoid()
    colors = gs["colors"].to(dev).sigmoid()
    print(f"GS: {means.shape[0]} Gaussians, mesh_scale={ms}")

    cams = read_cameras_binary(f"{COLMAP}/cameras.bin")
    imgs = read_images_binary(f"{COLMAP}/images.bin")
    cam0 = cams[list(cams.keys())[0]]
    print(f"COLMAP: {len(imgs)} images, camera model={cam0.model}, "
          f"full-res {cam0.width}x{cam0.height}")

    # downsample factor from a real images_4 file
    names = {im.name for im in imgs.values()}
    sample = sorted(os.listdir(IMG_DIR))[0]
    W4, H4 = Image.open(f"{IMG_DIR}/{sample}").size
    sx, sy = W4 / cam0.width, H4 / cam0.height
    print(f"images_4: {W4}x{H4}  (scale {sx:.4f},{sy:.4f})")

    recs = []
    for im in imgs.values():
        R = qvec2rotmat(im.qvec)              # world->cam rotation (OpenCV)
        t = im.tvec.reshape(3)
        C_world = -R.T @ t                    # camera center in world
        C_norm = (C_world - mc) * ms          # normalized frame
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R                       # rotation unchanged
        w2c[:3, 3] = -R @ C_norm              # t_norm = -R C_norm
        K = cam_K(cams[im.camera_id], sx, sy)
        recs.append({"name": im.name, "w2c": w2c, "K": K})
    recs.sort(key=lambda r: r["name"])

    np.savez(f"runs/garden/cameras.npz",
             names=np.array([r["name"] for r in recs]),
             w2c=np.stack([r["w2c"] for r in recs]),
             K=np.stack([r["K"] for r in recs]), W=W4, H=H4)
    print(f"wrote runs/garden/cameras.npz ({len(recs)} cams)")

    # sanity render a few, side by side with the real photo
    idxs = np.linspace(0, len(recs)-1, args.n_sanity).astype(int)
    rows = []
    for i in idxs:
        r = recs[i]
        w2c = torch.from_numpy(r["w2c"]).to(dev).unsqueeze(0)
        K = torch.from_numpy(r["K"]).to(dev).unsqueeze(0)
        ren, alpha, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opac, colors=colors,
            viewmats=w2c, Ks=K, width=W4, height=H4, sh_degree=None, render_mode="RGB")
        rgb = (ren[0, ..., :3] + (1 - alpha[0]) * 1.0).clamp(0, 1).cpu().numpy()
        real = np.asarray(Image.open(f"{IMG_DIR}/{r['name']}").convert("RGB")) / 255.0
        if real.shape[:2] != (H4, W4):
            real = np.asarray(Image.open(f"{IMG_DIR}/{r['name']}").convert("RGB").resize((W4, H4)))/255.0
        rows.append(np.concatenate([real, rgb], axis=1))
        print(f"  [{r['name']}] rendered")
    sheet = (np.concatenate(rows, axis=0) * 255).astype(np.uint8)
    os.makedirs("renders", exist_ok=True)
    Image.fromarray(sheet).save(args.out)
    print(f"wrote {args.out}  (left=real photo | right=GS render, per row)")


if __name__ == "__main__":
    main()
