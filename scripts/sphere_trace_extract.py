"""Sphere-trace mesh extraction from a trained DDF.

Direct alternative to UDF -> marching cubes (`scripts/stage3_chamfer.py`). We
sphere-trace per-camera per-pixel rays through the DDF, collect surface hits
across many cameras, filter by visibility, and Poisson-reconstruct the
aggregate point cloud.

Pipeline per object:
  1. Place ~200 cameras around the object on a Fibonacci sphere at varied radii
     (1.5-3.0). FOV 60 deg, image 256x256 (~65k rays each).
  2. For each camera, sphere-trace rays:
        t = 0
        for k in range(max_iter):
            x = o + t * d
            dist, vis_logit = DDF(x, d)
            if dist < eps: HIT
            t += dist
            if t > t_far: MISS
  3. Aggregate hit points + per-ray direction (used to orient normals + as
     the visibility query direction).
  4. Filter by visibility(x_final, d).sigmoid() > 0.5.
  5. Subsample to ~500k via farthest-point on a uniform-random downsample
     (we use random subsample here; FPS over millions is slow).
  6. Estimate normals via Open3D k-NN + orient against -d_hit.
  7. Poisson reconstruction (depth=10). Trim bottom 5% by density.
  8. Sample mesh, compute Chamfer + F-score vs GT.

GPU: pass --device cuda; sphere-trace is GPU-bound. Set
``CUDA_VISIBLE_DEVICES=2`` externally.
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
build_model_from_cfg = _s3._build_model_from_cfg
fscore = _fs.fscore


# Same 10 objects as Stage 9 SOTA / kitchen-sink suite.
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


# When set (via --ckpt_suffix), force ckpt/out-dir to runs/<obj><suffix>/...,
# overriding the default ks/hg search and the bull kitchen-sink special case.
# Used to extract from the GT-mesh-supervised DDFs (suffix "_ddf_gtmesh").
CKPT_SUFFIX_OVERRIDE: str | None = None


def ckpt_for(obj: str) -> Path:
    if CKPT_SUFFIX_OVERRIDE:
        return Path(f"runs/{obj}{CKPT_SUFFIX_OVERRIDE}/ddf_final.pt")
    if obj == "bull":
        return Path("runs/bull_ddf_hashgrid_kitchensink/ddf_final.pt")
    for suffix in ("_ddf_ks", "_ddf_hg", "_ddf_hashgrid", "_ddf_v3", "_ddf"):
        p = Path(f"runs/{obj}{suffix}/ddf_final.pt")
        if p.exists():
            return p
    raise SystemExit(f"no DDF ckpt found for {obj!r}")


def out_dir_for(obj: str) -> Path:
    if CKPT_SUFFIX_OVERRIDE:
        return Path(f"runs/{obj}{CKPT_SUFFIX_OVERRIDE}/stage3_spheretrace")
    if obj == "bull":
        return Path("runs/bull_ddf_hashgrid_kitchensink/stage3_spheretrace")
    return ckpt_for(obj).parent / "stage3_spheretrace"


def gt_mesh_path(obj: str) -> Path:
    if obj in SHAPENET_MAP:
        return Path(f"data/shapenet/{obj}/model.glb")
    if obj in NAME_MAP:
        return Path(f"data/gso/{NAME_MAP[obj]}/meshes/model.obj")
    raise SystemExit(f"unknown obj {obj!r}")


def stage3_metrics_for(obj: str) -> Path:
    if obj == "bull":
        return Path("runs/bull_ddf_hashgrid_kitchensink/stage3/metrics.json")
    return ckpt_for(obj).parent / "stage3" / "metrics.json"


def stage3_mesh_for(obj: str) -> Path:
    if obj == "bull":
        return Path("runs/bull_ddf_hashgrid_kitchensink/stage3/pred_mesh.ply")
    return ckpt_for(obj).parent / "stage3" / "pred_mesh.ply"


# ---------------- cameras ----------------

def fibonacci_sphere(n: int) -> np.ndarray:
    """Return n unit vectors uniformly distributed on the sphere (golden-angle)."""
    i = np.arange(n, dtype=np.float64)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = 2.0 * math.pi * i / phi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def look_at_c2w(eye: np.ndarray) -> np.ndarray:
    """Build an OpenCV camera-to-world for a camera at `eye` pointing to origin.

    Convention (this repo): c2w[:3, 0]=right, c2w[:3, 1]=down, c2w[:3, 2]=forward.
    """
    target = np.zeros(3, dtype=np.float32)
    up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-8)
    # If forward is nearly parallel to up_hint, switch the hint.
    if abs(float(np.dot(f, up_hint))) > 0.95:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    # OpenCV: right = up_hint x forward (right-handed). Then down = forward x right.
    right = np.cross(up_hint, f)
    right = right / (np.linalg.norm(right) + 1e-8)
    down = np.cross(f, right)
    down = down / (np.linalg.norm(down) + 1e-8)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = f
    c2w[:3, 3] = eye
    return c2w


def build_camera_set(n_cams: int, radii: tuple[float, float], seed: int = 0
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Return (eyes, c2ws) arrays of shape (N, 3) and (N, 4, 4).

    Camera positions are Fibonacci-sphere directions scaled by random radii in
    [r_min, r_max]. Always look at origin.
    """
    rng = np.random.default_rng(seed)
    dirs = fibonacci_sphere(n_cams)
    radii_arr = rng.uniform(radii[0], radii[1], size=n_cams).astype(np.float32)
    eyes = dirs * radii_arr[:, None]
    c2ws = np.stack([look_at_c2w(eye) for eye in eyes], axis=0)
    return eyes, c2ws


def pixel_rays(c2w: torch.Tensor, image_size: int, fov_deg: float,
               device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (origins, dirs) in world frame, each (H*W, 3)."""
    fov = math.radians(fov_deg)
    fx = fy = (image_size / 2.0) / math.tan(fov / 2.0)
    cx = cy = image_size / 2.0
    ys, xs = torch.meshgrid(
        torch.arange(image_size, device=device, dtype=torch.float32),
        torch.arange(image_size, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # OpenCV pixel -> camera direction: (x-cx)/fx, (y-cy)/fy, 1.
    cam_dirs = torch.stack([
        (xs - cx) / fx, (ys - cy) / fy, torch.ones_like(xs),
    ], dim=-1).reshape(-1, 3)
    # Rotate into world via c2w[:3, :3].
    R = c2w[:3, :3]
    dirs_w = cam_dirs @ R.T
    dirs_w = dirs_w / dirs_w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    origin = c2w[:3, 3].unsqueeze(0).expand_as(dirs_w).contiguous()
    return origin, dirs_w


# ---------------- sphere trace ----------------

@torch.no_grad()
def sphere_trace(
    model,
    origins: torch.Tensor,  # (N, 3)
    dirs: torch.Tensor,     # (N, 3)
    eps: float,
    max_iter: int,
    t_min: float,
    t_far: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sphere-trace N rays through the DDF.

    Returns:
        hit:        (N,) bool       — true if a hit was found
        x_final:    (N, 3) float    — surface point at hit (or last point traced)
        t_final:    (N,) float      — distance from origin
        dist_final: (N,) float      — DDF residual at termination
        vis_final:  (N,) float      — DDF visibility logit at termination
    """
    device = origins.device
    N = origins.shape[0]
    # Bring rays close to the bbox before tracing. The DDF predicts distance
    # along d to the next surface; for very large t_init the prediction is
    # extrapolated (since training origins were within bbox). We jump-start the
    # ray with sphere-bbox intersection to the unit-ish sphere of the data.
    t = torch.full((N,), t_min, device=device, dtype=torch.float32)
    active = torch.ones(N, dtype=torch.bool, device=device)
    last_dist = torch.full((N,), 1.0, device=device, dtype=torch.float32)
    last_vis = torch.full((N,), -10.0, device=device, dtype=torch.float32)
    x_final = origins + dirs * t.unsqueeze(-1)
    hit = torch.zeros(N, dtype=torch.bool, device=device)

    for k in range(max_iter):
        if not active.any():
            break
        idx = active.nonzero(as_tuple=False).squeeze(-1)
        x = origins[idx] + dirs[idx] * t[idx].unsqueeze(-1)
        out = model(x, dirs[idx])
        dist = out[0]
        vis = out[1]
        last_dist[idx] = dist
        last_vis[idx] = vis
        # Hits: DDF residual below eps. Mark and freeze.
        is_hit = dist < eps
        if is_hit.any():
            hit_idx = idx[is_hit]
            hit[hit_idx] = True
            x_final[hit_idx] = x[is_hit]
            active[hit_idx] = False
        # Advance the remaining active rays.
        not_hit = ~is_hit
        if not_hit.any():
            adv_idx = idx[not_hit]
            # Cap step so we don't shoot past t_far in a single jump.
            step = dist[not_hit].clamp(min=eps * 0.5, max=t_far)
            t[adv_idx] = t[adv_idx] + step
            # Mark misses beyond t_far.
            miss = t[adv_idx] > t_far
            if miss.any():
                miss_idx = adv_idx[miss]
                # Save last x at the (still-active-but-past-far) location for
                # diagnostics; mark inactive.
                x_final[miss_idx] = origins[miss_idx] + dirs[miss_idx] * t[miss_idx].unsqueeze(-1)
                active[miss_idx] = False

    # For rays still active at iter limit, record their last-traced point.
    if active.any():
        idx = active.nonzero(as_tuple=False).squeeze(-1)
        x_final[idx] = origins[idx] + dirs[idx] * t[idx].unsqueeze(-1)

    return hit, x_final, t, last_dist, last_vis


def intersect_sphere(origins: torch.Tensor, dirs: torch.Tensor, radius: float
                     ) -> torch.Tensor:
    """Per-ray entry t into a sphere of given radius centered at origin.

    Returns clamp(t_enter, 0). If the ray misses, returns 0 (we fall back to
    starting at the origin and letting sphere-trace handle the miss).
    """
    # Solve |o + t d|^2 = r^2 -> a t^2 + b t + c = 0 with a=1 (dirs unit).
    b = 2.0 * (origins * dirs).sum(dim=-1)
    c = (origins * origins).sum(dim=-1) - radius * radius
    disc = b * b - 4.0 * c
    valid = disc > 0
    sq = torch.sqrt(disc.clamp(min=0.0))
    t_enter = torch.where(valid, (-b - sq) * 0.5, torch.zeros_like(b))
    return t_enter.clamp(min=0.0)


# ---------------- per-object pipeline ----------------

@torch.no_grad()
def collect_hits(
    model,
    c2ws: np.ndarray,
    image_size: int,
    fov_deg: float,
    eps: float,
    max_iter: int,
    t_far: float,
    vis_thresh: float,
    bbox_half: float,
    device: str,
    ray_chunk: int = 200_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Sphere-trace all rays across all cameras; return surface points, normals
    (placeholder zeros; estimated downstream), and per-ray hit direction.

    Returns (points, dirs, vis_logits, stats).
    """
    all_pts = []
    all_dirs = []
    all_vis = []
    n_total_rays = 0
    n_hits = 0
    n_vis_pass = 0
    # Pad bbox a touch so we trace from the boundary.
    sphere_r = bbox_half + 0.05

    for i, c2w_np in enumerate(c2ws):
        c2w = torch.from_numpy(c2w_np).to(device)
        origins, dirs = pixel_rays(c2w, image_size, fov_deg, device)
        # Optimization: jump-start each ray to where it enters a unit-ish
        # bounding sphere — saves wasted DDF queries far outside the model's
        # training support.
        t_enter = intersect_sphere(origins, dirs, sphere_r)
        origins = origins + dirs * t_enter.unsqueeze(-1)
        # Sphere-trace in chunks for memory.
        n_rays = origins.shape[0]
        hits_ = []
        x_ = []
        vis_ = []
        for s in range(0, n_rays, ray_chunk):
            e = min(s + ray_chunk, n_rays)
            hit, x_final, t_final, last_dist, last_vis = sphere_trace(
                model,
                origins[s:e].contiguous(),
                dirs[s:e].contiguous(),
                eps=eps,
                max_iter=max_iter,
                t_min=0.0,
                t_far=t_far,
            )
            hits_.append(hit)
            x_.append(x_final)
            vis_.append(last_vis)
        hit = torch.cat(hits_, dim=0)
        x_final = torch.cat(x_, dim=0)
        last_vis = torch.cat(vis_, dim=0)

        n_total_rays += n_rays
        if hit.any():
            n_hits += int(hit.sum().item())
            pts_h = x_final[hit]
            dirs_h = dirs[hit]
            vis_h = last_vis[hit]
            # Visibility filter on hit rays.
            keep = vis_h.sigmoid() > vis_thresh
            n_vis_pass += int(keep.sum().item())
            if keep.any():
                all_pts.append(pts_h[keep].cpu().numpy())
                all_dirs.append(dirs_h[keep].cpu().numpy())
                all_vis.append(vis_h[keep].cpu().numpy())

    if not all_pts:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            {
                "n_cameras": len(c2ws),
                "image_size": image_size,
                "n_total_rays": n_total_rays,
                "n_hits": n_hits,
                "n_kept_after_vis": n_vis_pass,
            },
        )

    pts = np.concatenate(all_pts, axis=0).astype(np.float32)
    dirs_arr = np.concatenate(all_dirs, axis=0).astype(np.float32)
    vis_arr = np.concatenate(all_vis, axis=0).astype(np.float32)

    # Drop hits outside the bbox (sometimes the trace lands far away when the
    # DDF predicts a near-zero residual at a far-out point, especially in
    # under-trained regions of the model).
    inside = np.max(np.abs(pts), axis=1) <= bbox_half + 0.05
    pts = pts[inside]
    dirs_arr = dirs_arr[inside]
    vis_arr = vis_arr[inside]

    stats = {
        "n_cameras": len(c2ws),
        "image_size": image_size,
        "n_total_rays": int(n_total_rays),
        "n_hits": int(n_hits),
        "n_kept_after_vis": int(n_vis_pass),
        "n_kept_after_bbox": int(pts.shape[0]),
    }
    return pts, dirs_arr, vis_arr, stats


def subsample_random(pts: np.ndarray, dirs: np.ndarray, n: int,
                     rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if pts.shape[0] <= n:
        return pts, dirs
    idx = rng.choice(pts.shape[0], size=n, replace=False)
    return pts[idx], dirs[idx]


def poisson_reconstruct(
    pts: np.ndarray,
    hit_dirs: np.ndarray,
    depth: int = 10,
    knn: int = 30,
    density_quantile: float = 0.05,
) -> tuple[o3d.geometry.TriangleMesh, dict]:
    """Estimate normals (k-NN) + orient against -d_hit, run Poisson, trim."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    # Orient normals to point away from the surface (outward). For a sphere-
    # traced hit, the outward normal n satisfies n . (-d_hit) > 0 (i.e. n
    # points back toward the camera). Flip per-point normals that don't.
    normals = np.asarray(pcd.normals)
    minus_d = -hit_dirs.astype(np.float64)
    dot = (normals * minus_d).sum(axis=1)
    flip = dot < 0
    normals[flip] = -normals[flip]
    pcd.normals = o3d.utility.Vector3dVector(normals)

    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=1.1, linear_fit=False,
    )
    density = np.asarray(density)
    n_verts_raw = len(mesh.vertices)
    n_faces_raw = len(mesh.triangles)

    if density_quantile > 0 and len(density) > 0:
        thresh = float(np.quantile(density, density_quantile))
        mesh.remove_vertices_by_mask(density < thresh)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    info = {
        "n_verts_raw": int(n_verts_raw),
        "n_faces_raw": int(n_faces_raw),
        "n_verts": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.triangles)),
        "depth": depth,
        "knn": knn,
        "density_quantile": density_quantile,
    }
    return mesh, info


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


def load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model_from_cfg(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def process_object(
    obj: str,
    device: str,
    n_cams: int,
    radii: tuple[float, float],
    image_size: int,
    fov_deg: float,
    eps: float,
    max_iter: int,
    t_far: float,
    vis_thresh: float,
    subsample_n: int,
    poisson_depth: int,
    poisson_knn: int,
    density_quantile: float,
    n_eval_samples: int,
    taus: list[float],
    bbox_half: float,
    seed: int,
) -> dict:
    t_total = time.time()
    print(f"\n=== {obj} ===")
    ckpt_path = ckpt_for(obj)
    out_dir = out_dir_for(obj)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(ckpt_path, device)
    if hasattr(model, "bbox_half"):
        try:
            bbox_half_eff = float(model.bbox_half)
        except (TypeError, ValueError):
            bbox_half_eff = bbox_half
    else:
        bbox_half_eff = bbox_half

    eyes, c2ws = build_camera_set(n_cams, radii, seed=seed)

    t0 = time.time()
    pts, hit_dirs, vis_logits, stats = collect_hits(
        model, c2ws,
        image_size=image_size, fov_deg=fov_deg,
        eps=eps, max_iter=max_iter, t_far=t_far,
        vis_thresh=vis_thresh, bbox_half=bbox_half_eff, device=device,
    )
    trace_s = time.time() - t0
    print(
        f"[{obj}] traced {stats['n_total_rays']:,} rays from {stats['n_cameras']} "
        f"cams -> {stats['n_hits']:,} hits, {stats['n_kept_after_vis']:,} pass vis, "
        f"{stats.get('n_kept_after_bbox', stats['n_kept_after_vis']):,} in bbox  "
        f"({trace_s:.1f}s)"
    )

    if pts.shape[0] < 1000:
        print(f"[{obj}] too few points ({pts.shape[0]}) for Poisson; aborting")
        return {
            "obj": obj,
            "error": "too_few_points",
            "stats": stats,
            "trace_s": trace_s,
        }

    rng = np.random.default_rng(seed)
    pts_sub, dirs_sub = subsample_random(pts, hit_dirs, subsample_n, rng)
    print(f"[{obj}] subsampled to {pts_sub.shape[0]:,} for Poisson")

    t0 = time.time()
    mesh, mesh_info = poisson_reconstruct(
        pts_sub, dirs_sub,
        depth=poisson_depth, knn=poisson_knn,
        density_quantile=density_quantile,
    )
    poisson_s = time.time() - t0
    print(
        f"[{obj}] Poisson depth={poisson_depth}: "
        f"{mesh_info['n_verts_raw']} -> {mesh_info['n_verts']} verts  "
        f"({poisson_s:.1f}s)"
    )

    pred_mesh_path = out_dir / "pred_mesh.ply"
    o3d.io.write_triangle_mesh(str(pred_mesh_path), mesh)

    # Save the raw + subsampled point clouds for reproducibility.
    np.savez(
        out_dir / "spheretrace_points.npz",
        pts_all=pts, dirs_all=hit_dirs, vis_logits=vis_logits,
        pts_sub=pts_sub, dirs_sub=dirs_sub,
    )

    # Eval Chamfer + F-score.
    pred_pts = sample_mesh_points(mesh, n_eval_samples)
    if pred_pts is None:
        print(f"[{obj}] Poisson produced empty mesh")
        return {
            "obj": obj,
            "error": "empty_poisson_mesh",
            "stats": stats,
            "trace_s": trace_s,
            "poisson_s": poisson_s,
            **mesh_info,
        }

    gs_path = Path(f"runs/{obj}/gaussians.pt")
    gs = torch.load(gs_path, map_location="cpu", weights_only=False)
    mesh_center = gs["mesh_center"].numpy().astype(np.float32)
    mesh_scale = float(gs["mesh_scale"].item())
    gt_pts = load_gt_points(gt_mesh_path(obj), mesh_center, mesh_scale, n_eval_samples)

    cd_mean, cd_med = chamfer(gt_pts, pred_pts)
    fs = fscore(pred_pts, gt_pts, taus)
    np.savez(out_dir / "pointclouds.npz", gt=gt_pts, pred=pred_pts)

    result = {
        "obj": obj,
        "ddf_ckpt": str(ckpt_path),
        "n_cameras": stats["n_cameras"],
        "image_size": image_size,
        "fov_deg": fov_deg,
        "eps": eps,
        "max_iter": max_iter,
        "t_far": t_far,
        "vis_thresh": vis_thresh,
        "subsample_n": subsample_n,
        "poisson_depth": poisson_depth,
        "poisson_knn": poisson_knn,
        "density_quantile": density_quantile,
        "bbox_half": bbox_half_eff,
        "stats": stats,
        "trace_s": trace_s,
        "poisson_s": poisson_s,
        "wall_s": time.time() - t_total,
        "n_verts": mesh_info["n_verts"],
        "n_faces": mesh_info["n_faces"],
        "chamfer_mean": cd_mean,
        "chamfer_median": cd_med,
    }
    for tau, v in fs.items():
        result[f"P@{tau}"] = v["precision"]
        result[f"R@{tau}"] = v["recall"]
        result[f"F1@{tau}"] = v["f1"]
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    print(
        f"[{obj}] Chamfer mean={cd_mean:.4f}  median={cd_med:.4f}  "
        f"F1@0.05={fs[0.05]['f1']:.3f}  wall={result['wall_s']:.1f}s"
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="+", default=OBJECTS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_cams", type=int, default=200)
    ap.add_argument("--radii", type=float, nargs=2, default=[1.5, 3.0])
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--fov_deg", type=float, default=60.0)
    ap.add_argument("--eps", type=float, default=0.05,
                    help="ray-hit threshold; ~0.05 matches the DDF's surface "
                         "noise floor (saturated softplus residual). MC iso=0.05.")
    ap.add_argument("--max_iter", type=int, default=48)
    ap.add_argument("--t_far", type=float, default=5.0)
    ap.add_argument("--vis_thresh", type=float, default=0.5)
    ap.add_argument("--subsample_n", type=int, default=500_000)
    ap.add_argument("--poisson_depth", type=int, default=10)
    ap.add_argument("--poisson_knn", type=int, default=30)
    ap.add_argument("--density_quantile", type=float, default=0.05)
    ap.add_argument("--n_eval_samples", type=int, default=20_000)
    ap.add_argument("--taus", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--bbox_half", type=float, default=1.2,
                    help="fallback when the ckpt does not advertise its bbox_half")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt_suffix", default=None,
                    help="force ckpt/out at runs/<obj><suffix>/ (e.g. _ddf_gtmesh)")
    ap.add_argument("--summary_out", default="runs/ddf_spheretrace/summary.json")
    args = ap.parse_args()

    if args.ckpt_suffix:
        global CKPT_SUFFIX_OVERRIDE
        CKPT_SUFFIX_OVERRIDE = args.ckpt_suffix

    rows = []
    for obj in args.objects:
        try:
            row = process_object(
                obj, args.device,
                n_cams=args.n_cams, radii=tuple(args.radii),
                image_size=args.image_size, fov_deg=args.fov_deg,
                eps=args.eps, max_iter=args.max_iter, t_far=args.t_far,
                vis_thresh=args.vis_thresh,
                subsample_n=args.subsample_n,
                poisson_depth=args.poisson_depth, poisson_knn=args.poisson_knn,
                density_quantile=args.density_quantile,
                n_eval_samples=args.n_eval_samples, taus=args.taus,
                bbox_half=args.bbox_half, seed=args.seed,
            )
            rows.append(row)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{obj}] FAILED: {e!r}")
            rows.append({"obj": obj, "error": repr(e)})

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {summary_path} with {len(rows)} rows")


if __name__ == "__main__":
    main()
