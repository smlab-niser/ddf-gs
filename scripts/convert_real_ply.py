"""Convert a standard 3DGS .ply checkpoint (graphdeco-inria/Inria format) to
our internal gaussians.pt format used by GSSupervisor / fit_gs.

Standard 3DGS .ply layout (sh_degree=3):
  x,y,z              -- means (world meters)
  nx,ny,nz           -- normals (ignored)
  f_dc_0,f_dc_1,f_dc_2  -- SH DC term (one per RGB channel, in SH units;
                        -- view-independent RGB = 0.5 + 0.28209479 * f_dc)
  f_rest_0...f_rest_44  -- higher-order SH (3 channels x 15 = 45 coefs); we
                        -- drop these (view-dependent; DDF is geometry-only)
  opacity            -- logit (sigmoid for activation)
  scale_0,1,2        -- log-scale (exp for activation)
  rot_0,1,2,3        -- raw quaternion (wxyz convention in standard 3DGS;
                        -- our pipeline normalizes via q/|q|)

Our gaussians.pt format (see runs/bull/gaussians.pt):
  means(N,3), quats(N,4), scales(N,3), opacities(N,), colors(N,3),
  bbox_min(3), bbox_max(3), mesh_center(3), mesh_scale(()).

Convention conversions:
  - means: center on robust median, scale so foreground fits in unit-sphere
    region (target r_max ~= 1.0). DDF training bbox_half is then 1.2.
  - colors: convert SH DC -> RGB via 0.5 + 0.28209479*f_dc, then convert
    that [0,1] RGB into the *pre-sigmoid logit* expected by our supervisor
    (which applies sigmoid at load). Clip to safe range so logit is finite.
  - opacity / scales / quats: stored exactly as the source (raw pre-activation)
    so GSSupervisor's sigmoid/exp/normalize works unchanged.
  - mesh_center / mesh_scale: set to the affine transform we applied so that
    a downstream consumer (e.g. mesh extraction) can map back to world.

The script also writes a cropping mask: Gaussians beyond ``crop_radius`` (in
world meters) from the foreground centroid are dropped, otherwise the
background sky/buildings dominate the bbox and the foreground compresses to
near-zero size in the normalized frame.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

SH0 = 0.28209479177387814  # SH band 0 coefficient


def _logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--crop_radius", type=float, default=8.0,
        help="Keep Gaussians within this many world-meters of the foreground "
             "centroid; drops sky/distant background.",
    )
    ap.add_argument(
        "--target_r", type=float, default=1.0,
        help="After centering, divide world-meters by (crop_radius/target_r) "
             "so the foreground roughly fills the unit sphere.",
    )
    ap.add_argument(
        "--op_filter", type=float, default=0.05,
        help="Drop Gaussians with sigmoid(opacity) below this (very low-confidence ghosts).",
    )
    args = ap.parse_args()

    print(f"loading {args.ply}...")
    p = PlyData.read(args.ply)
    e = p["vertex"]
    n_in = len(e.data)
    print(f"  {n_in} Gaussians")

    xyz = np.stack([e["x"], e["y"], e["z"]], axis=-1).astype(np.float32)
    op_logit = np.asarray(e["opacity"], dtype=np.float32)
    op = 1.0 / (1.0 + np.exp(-op_logit))

    # Foreground centroid: median of high-opacity Gaussians.
    fg_mask_init = op > 0.1
    centroid = np.median(xyz[fg_mask_init], axis=0).astype(np.float32)
    print(f"  foreground centroid (world m): {centroid}")

    # Crop + opacity filter.
    rel = xyz - centroid
    r = np.linalg.norm(rel, axis=-1)
    keep_mask = (r < args.crop_radius) & (op > args.op_filter)
    n_keep = int(keep_mask.sum())
    print(f"  kept {n_keep}/{n_in} after r<{args.crop_radius}m and op>{args.op_filter} filter")

    # World scale: divide by (crop_radius / target_r) so the kept region's
    # outer envelope sits at radius `target_r`.
    world_scale = args.target_r / args.crop_radius  # multiplicative
    print(f"  normalization: subtract {centroid.tolist()}, multiply by {world_scale:.6f}")

    means = rel[keep_mask] * world_scale
    # Scales (log space): scaling positions by `world_scale` scales lengths by
    # the same factor, which becomes an additive shift in log space.
    log_scale = np.stack([
        e["scale_0"], e["scale_1"], e["scale_2"]
    ], axis=-1).astype(np.float32)[keep_mask]
    log_scale = log_scale + float(np.log(world_scale))

    # Quaternions: source convention (Inria 3DGS) is (w, x, y, z); gsplat's
    # `rasterization` accepts that order with normalization done internally
    # (matches our GSSupervisor.load: it divides by norm). Just keep as-is.
    quats = np.stack([e["rot_0"], e["rot_1"], e["rot_2"], e["rot_3"]],
                     axis=-1).astype(np.float32)[keep_mask]

    opac_logit_kept = op_logit[keep_mask].astype(np.float32)

    # Colors: SH DC -> view-independent RGB, then convert to pre-sigmoid
    # (logit) expected by our load path.
    f_dc = np.stack([e["f_dc_0"], e["f_dc_1"], e["f_dc_2"]],
                    axis=-1).astype(np.float32)[keep_mask]
    rgb01 = 0.5 + SH0 * f_dc
    colors_logit = _logit(rgb01).astype(np.float32)

    # bbox in our normalized frame.
    bbox_min = means.min(axis=0).astype(np.float32)
    bbox_max = means.max(axis=0).astype(np.float32)
    print(f"  normalized bbox_min={bbox_min}, bbox_max={bbox_max}")
    print(f"  radial p99 after norm: {np.percentile(np.linalg.norm(means, axis=-1), 99):.3f}")

    # mesh_center / mesh_scale store the affine that converted world -> norm.
    # (means_norm = (means_world - mesh_center) * mesh_scale)
    mesh_center = centroid.astype(np.float32)
    mesh_scale = np.float32(world_scale)

    ckpt = {
        "means": torch.from_numpy(means),
        "quats": torch.from_numpy(quats),
        "scales": torch.from_numpy(log_scale),
        "opacities": torch.from_numpy(opac_logit_kept),
        "colors": torch.from_numpy(colors_logit),
        "bbox_min": torch.from_numpy(bbox_min),
        "bbox_max": torch.from_numpy(bbox_max),
        "mesh_center": torch.from_numpy(mesh_center),
        "mesh_scale": torch.tensor(float(mesh_scale), dtype=torch.float32),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"saved {args.out}  ({n_keep} Gaussians)")


if __name__ == "__main__":
    main()
