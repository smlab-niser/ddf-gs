"""Stage 10 — classical mesh-extraction baselines.

Two methods, same 10 objects as Stage 9 NeuS comparison:

  A. Screened Poisson Surface Reconstruction (Kazhdan 2013) over the GS means.
  B. TSDF fusion over 50 multi-view ED depth renders from gsplat.

For each, we extract a mesh in the same unit-normalised object frame as
`stage3_chamfer.py`, then score Chamfer + F-score @ τ∈{0.05,0.10,0.20} against
the same GT mesh used everywhere else. Outputs land in
`runs/baselines/<short>/{poisson,tsdf}/` with per-object metrics and meshes.

GPUs: this script uses CUDA only for the gsplat rendering pass in (B). Set
the visible device with `CUDA_VISIBLE_DEVICES=2` (or 3). Poisson + TSDF
itself is CPU-bound (open3d).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh

# This file lives in scripts/ which has no __init__.py — load sibling scripts
# via importlib (same pattern as bull_sweeps.py).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, str(_REPO_ROOT / "scripts" / f"{name}.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_s3 = _load_script("stage3_chamfer")
_fs = _load_script("eval_fscore")
NAME_MAP = _s3.NAME_MAP
SHAPENET_MAP = _s3.SHAPENET_MAP
chamfer = _s3.chamfer
load_gt_points = _s3.load_gt_points
fscore = _fs.fscore


# Same 10 objects as Stage 9.
OBJECTS = [
    "bull",
    "lion",
    "spino",
    "mug",
    "turtle",
    "airplane_94c4ade3",
    "chair_5f1b4529",
    "car_9ee32f51",
    "bottle_59d7b4e7",
    "sofa_145bd097",
]


def gt_mesh_path(obj: str) -> Path:
    if obj in SHAPENET_MAP:
        return Path(f"data/shapenet/{obj}/model.glb")
    if obj in NAME_MAP:
        return Path(f"data/gso/{NAME_MAP[obj]}/meshes/model.obj")
    raise SystemExit(f"unknown obj {obj!r}")


# -------------------- Poisson --------------------

def run_poisson(
    means: np.ndarray,
    depth: int = 9,
    density_quantile: float = 0.01,
    knn: int = 30,
) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Screened Poisson on Gaussian centers; trims by density quantile.

    Normals estimated with k-NN (k=knn=30) and oriented via tangent-plane
    propagation (Hoppe et al.). Density trim removes the bottom `density_quantile`
    fraction of vertices (typical 1%).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pcd.orient_normals_consistent_tangent_plane(k=max(knn // 2, 10))

    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=1.1, linear_fit=False,
    )
    density = np.asarray(density)
    n_verts_raw = len(mesh.vertices)
    n_faces_raw = len(mesh.triangles)

    if density_quantile > 0 and len(density) > 0:
        thresh = float(np.quantile(density, density_quantile))
        keep = density >= thresh
        mesh.remove_vertices_by_mask(~keep)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    info = {
        "n_verts_raw": int(n_verts_raw),
        "n_faces_raw": int(n_faces_raw),
        "n_verts": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.triangles)),
        "density_quantile": density_quantile,
        "depth": depth,
        "knn": knn,
    }
    return mesh, info


# -------------------- TSDF --------------------

def _spherical_cameras(
    n_views: int,
    radius: float,
    image_size: int,
    fov_deg: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate `n_views` OpenCV-convention (w2c, K) pairs on a sphere shell.

    Half on a low elevation ring, half on a higher elevation ring (Fibonacci-ish
    distribution would also work; this is simpler and sufficient for TSDF fusion).
    """
    fov = math.radians(fov_deg)
    fx = fy = (image_size / 2.0) / math.tan(fov / 2.0)
    cx = cy = image_size / 2.0
    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32, device=device,
    )

    # Spread azimuths uniformly; sweep elevations across multiple rings so we
    # cover top + bottom + sides.
    n_rings = 5
    per_ring = n_views // n_rings
    extra = n_views - per_ring * n_rings
    elevs_deg = np.linspace(-50.0, 70.0, n_rings)

    w2cs = []
    azim_offsets = np.linspace(0.0, 360.0 / max(per_ring, 1), n_rings, endpoint=False)
    for ri, elev_deg in enumerate(elevs_deg):
        n_here = per_ring + (1 if ri < extra else 0)
        azims = (
            np.linspace(0.0, 360.0, n_here, endpoint=False)
            + azim_offsets[ri]
        )
        elev = math.radians(float(elev_deg))
        for azim_deg in azims:
            azim = math.radians(float(azim_deg))
            eye = np.array([
                radius * math.cos(elev) * math.sin(azim),
                radius * math.sin(elev),
                radius * math.cos(elev) * math.cos(azim),
            ], dtype=np.float32)
            eye_t = torch.from_numpy(eye).to(device)
            world_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
            f = -eye_t / eye_t.norm().clamp_min(1e-8)
            r = torch.linalg.cross(f, world_up); r = r / r.norm().clamp_min(1e-8)
            d = torch.linalg.cross(f, r); d = d / d.norm().clamp_min(1e-8)
            c2w = torch.eye(4, dtype=torch.float32, device=device)
            c2w[:3, 0] = r
            c2w[:3, 1] = d
            c2w[:3, 2] = f
            c2w[:3, 3] = eye_t
            w2cs.append(torch.linalg.inv(c2w))
    w2c = torch.stack(w2cs, dim=0)
    return w2c, K


@torch.no_grad()
def render_depth_views(
    gs: dict,
    n_views: int,
    image_size: int,
    radius: float,
    fov_deg: float,
    device: str,
    chunk: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render `n_views` ED-depth + alpha images from the fitted GS.

    Returns (depths, alphas, c2ws, K) — all numpy, shapes:
        depths: (V, H, W) float32      expected z-depth (camera-frame z, m)
        alphas: (V, H, W) float32      pixel coverage
        c2ws:   (V, 4, 4) float32      OpenCV camera-to-world
        K:      (3, 3)    float32
    """
    from gsplat import rasterization

    quats = gs["quats"].to(device)
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    means = gs["means"].to(device).contiguous()
    scales = gs["scales"].to(device).exp().contiguous()
    opacities = gs["opacities"].to(device).sigmoid().contiguous()
    colors = gs["colors"].to(device).sigmoid().contiguous()

    w2c, K = _spherical_cameras(n_views, radius, image_size, fov_deg, device)
    c2w = torch.linalg.inv(w2c)
    Ks = K.unsqueeze(0).expand(w2c.shape[0], -1, -1).contiguous()

    depths = []
    alphas = []
    for s in range(0, w2c.shape[0], chunk):
        e = min(s + chunk, w2c.shape[0])
        renders, alph, _ = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=w2c[s:e], Ks=Ks[s:e],
            width=image_size, height=image_size,
            sh_degree=None, render_mode="RGB+ED",
        )
        depths.append(renders[..., 3].cpu())  # (B, H, W)
        alphas.append(alph[..., 0].cpu())
    depths = torch.cat(depths, dim=0).numpy().astype(np.float32)
    alphas = torch.cat(alphas, dim=0).numpy().astype(np.float32)
    return depths, alphas, c2w.cpu().numpy().astype(np.float32), K.cpu().numpy().astype(np.float32)


def run_tsdf(
    depths: np.ndarray,
    alphas: np.ndarray,
    c2ws: np.ndarray,
    K: np.ndarray,
    voxel_length: float = 0.01,
    sdf_trunc: float = 0.04,
    alpha_thresh: float = 0.5,
    max_depth: float = 6.0,
) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Integrate the ED depth maps into a ScalableTSDFVolume; extract mesh."""
    V, H, W = depths.shape
    fx = float(K[0, 0]); fy = float(K[1, 1])
    cx = float(K[0, 2]); cy = float(K[1, 2])
    intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
    )

    n_integrated = 0
    for i in range(V):
        d = depths[i].copy()
        a = alphas[i]
        d[a < alpha_thresh] = 0.0
        d[d > max_depth] = 0.0
        if not np.any(d > 0):
            continue
        # ED is z-depth in camera frame, in meters.
        # Open3D expects a uint16 depth image with depth_scale, OR a float
        # image with depth_scale=1.0.
        depth_o3d = o3d.geometry.Image(d.astype(np.float32))
        # Provide a dummy color image (we set NoColor mode so it's unused).
        dummy_color = np.zeros((H, W, 3), dtype=np.uint8)
        color_o3d = o3d.geometry.Image(dummy_color)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0, depth_trunc=max_depth,
            convert_rgb_to_intensity=False,
        )
        # Open3D's TSDFVolume expects an *extrinsic* = world->cam matrix
        # (OpenCV convention, which is exactly our w2c).
        c2w = c2ws[i]
        extrinsic = np.linalg.inv(c2w).astype(np.float64)
        vol.integrate(rgbd, intr, extrinsic)
        n_integrated += 1

    mesh = vol.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    info = {
        "voxel_length": voxel_length,
        "sdf_trunc": sdf_trunc,
        "alpha_thresh": alpha_thresh,
        "n_views_integrated": int(n_integrated),
        "n_verts": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.triangles)),
    }
    return mesh, info


# -------------------- evaluation --------------------

def sample_mesh_points(mesh: o3d.geometry.TriangleMesh, n: int) -> np.ndarray | None:
    if len(mesh.triangles) == 0:
        return None
    tm = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.triangles, dtype=np.int64),
        process=False,
    )
    pts, _ = trimesh.sample.sample_surface(tm, n)
    return np.asarray(pts, dtype=np.float32)


def eval_against_gt(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    taus: list[float],
) -> dict:
    cd_mean, cd_med = chamfer(gt_pts, pred_pts)
    fs = fscore(pred_pts, gt_pts, taus)
    out = {"chamfer_mean": cd_mean, "chamfer_median": cd_med}
    for tau, v in fs.items():
        out[f"P@{tau}"] = v["precision"]
        out[f"R@{tau}"] = v["recall"]
        out[f"F1@{tau}"] = v["f1"]
    return out


# -------------------- orchestration --------------------

def process_object(
    obj: str,
    out_root: Path,
    device: str,
    n_views: int,
    image_size: int,
    radius: float,
    fov_deg: float,
    poisson_depth: int,
    voxel_length: float,
    sdf_trunc: float,
    n_eval_samples: int,
    taus: list[float],
) -> dict:
    print(f"\n=== {obj} ===")
    gs_path = Path(f"runs/{obj}/gaussians.pt")
    gs = torch.load(gs_path, map_location="cpu", weights_only=False)
    means = gs["means"].numpy().astype(np.float32)
    mesh_center = gs["mesh_center"].numpy().astype(np.float32)
    mesh_scale = float(gs["mesh_scale"].item())

    gt_pts = load_gt_points(gt_mesh_path(obj), mesh_center, mesh_scale, n_eval_samples)

    obj_out = out_root / obj
    obj_out.mkdir(parents=True, exist_ok=True)
    row: dict = {"obj": obj}

    # ---- Poisson ----
    pois_dir = obj_out / "poisson"
    pois_dir.mkdir(exist_ok=True)
    t0 = time.time()
    pmesh, pinfo = run_poisson(means, depth=poisson_depth)
    poisson_wall = time.time() - t0
    o3d.io.write_triangle_mesh(str(pois_dir / "poisson_mesh.ply"), pmesh)
    pred_pts = sample_mesh_points(pmesh, n_eval_samples)
    if pred_pts is None:
        print(f"[{obj}] Poisson produced empty mesh")
        pres = {"chamfer_mean": float("nan"), "chamfer_median": float("nan")}
    else:
        pres = eval_against_gt(pred_pts, gt_pts, taus)
    pres.update({"wall_s": poisson_wall, **pinfo})
    (pois_dir / "metrics.json").write_text(json.dumps(pres, indent=2))
    print(
        f"[{obj}] Poisson: CD mean={pres['chamfer_mean']:.4f} "
        f"med={pres['chamfer_median']:.4f}  verts={pinfo['n_verts']}  "
        f"wall={poisson_wall:.1f}s"
    )
    row["poisson"] = pres

    # ---- TSDF ----
    tsdf_dir = obj_out / "tsdf"
    tsdf_dir.mkdir(exist_ok=True)
    t0 = time.time()
    depths, alphas, c2ws, K = render_depth_views(
        gs, n_views=n_views, image_size=image_size, radius=radius,
        fov_deg=fov_deg, device=device,
    )
    render_s = time.time() - t0
    t0 = time.time()
    tmesh, tinfo = run_tsdf(
        depths, alphas, c2ws, K,
        voxel_length=voxel_length, sdf_trunc=sdf_trunc,
    )
    fuse_s = time.time() - t0
    tsdf_wall = render_s + fuse_s
    o3d.io.write_triangle_mesh(str(tsdf_dir / "tsdf_mesh.ply"), tmesh)
    pred_pts = sample_mesh_points(tmesh, n_eval_samples)
    if pred_pts is None:
        print(f"[{obj}] TSDF produced empty mesh")
        tres = {"chamfer_mean": float("nan"), "chamfer_median": float("nan")}
    else:
        tres = eval_against_gt(pred_pts, gt_pts, taus)
    tres.update({
        "wall_s": tsdf_wall, "render_s": render_s, "fuse_s": fuse_s,
        "n_views": n_views, "image_size": image_size, "radius": radius,
        "fov_deg": fov_deg, **tinfo,
    })
    (tsdf_dir / "metrics.json").write_text(json.dumps(tres, indent=2))
    print(
        f"[{obj}] TSDF:    CD mean={tres['chamfer_mean']:.4f} "
        f"med={tres['chamfer_median']:.4f}  verts={tinfo['n_verts']}  "
        f"wall={tsdf_wall:.1f}s (render {render_s:.1f}s + fuse {fuse_s:.1f}s)"
    )
    row["tsdf"] = tres

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", default=OBJECTS)
    ap.add_argument("--out_root", default="runs/baselines")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_views", type=int, default=50)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--radius", type=float, default=2.5)
    ap.add_argument("--fov_deg", type=float, default=60.0)
    ap.add_argument("--poisson_depth", type=int, default=9)
    ap.add_argument("--voxel_length", type=float, default=0.01)
    ap.add_argument("--sdf_trunc", type=float, default=0.04)
    ap.add_argument("--n_eval_samples", type=int, default=20000)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for obj in args.objects:
        try:
            row = process_object(
                obj, out_root, args.device,
                n_views=args.n_views, image_size=args.image_size,
                radius=args.radius, fov_deg=args.fov_deg,
                poisson_depth=args.poisson_depth,
                voxel_length=args.voxel_length, sdf_trunc=args.sdf_trunc,
                n_eval_samples=args.n_eval_samples, taus=args.taus,
            )
            rows.append(row)
        except Exception as e:
            print(f"[{obj}] FAILED: {e!r}")
            rows.append({"obj": obj, "error": repr(e)})

    (out_root / "all_metrics.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out_root}/all_metrics.json with {len(rows)} rows")


if __name__ == "__main__":
    main()
