"""Multi-view GI evaluation: DDF as the secondary-ray oracle vs embree GT.

Headline TVCG-GI claim: a DDF distilled from a GS/implicit scene answers
SECONDARY-ray visibility queries (shadows, ambient occlusion) — which Gaussian
Splatting rasterization cannot produce — in ONE network eval per ray. This
script quantifies how good that oracle is against an embree ray-traced
ground-truth, over multiple views, with clean aggregate metrics.

----------------------------------------------------------------------------
CRITICAL design decision — isolate the secondary-ray oracle
----------------------------------------------------------------------------
We use the EMBREE MESH for the PRIMARY visible surface in BOTH the DDF-GI and
the reference renders. That means *identical* primary hit points x and
*identical* primary normals (mesh face normals, oriented toward the camera) in
both pipelines. ONLY the SECONDARY (shadow / AO) occlusion query differs:

    DDF-occluded (one network eval per ray)  vs  embree-occluded (ground truth)

This isolates exactly the quantity the paper measures — the DDF secondary-ray
oracle — and removes confounders from DDF *primary*-trace noise and DDF *normal*
noise, both of which are known/independent issues (see the paper's discussion).
The shading math is identical (callback injection through render_shadows /
render_ao), so any pixel difference comes purely from the occlusion oracle.

Usage:
  PYOPENGL_PLATFORM=egl PYTHONPATH=$REPO_ROOT \
  python scripts/gi_render_eval.py --obj bull \
      --ckpt runs/bull_ddf_gtmesh/ddf_final.pt \
      --gt_mesh data/gso/Schleich_Hereford_Bull/meshes/model.obj \
      --gs_path runs/bull/gaussians.pt --gso_rotate false \
      --n_views 8 --image_size 512
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from src.train_ddf import build_model
from src.ddf_gi_render import (
    ddf_occluded, MeshOracle, render_shadows, render_ao,
)
from src.gtmesh_supervisor import GTMeshSupervisor
from sphere_trace_extract import look_at_c2w, pixel_rays  # noqa: E402

try:
    from skimage.metrics import structural_similarity as _ssim_fn
    _HAVE_SSIM = True
except Exception:  # pragma: no cover
    _HAVE_SSIM = False


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR between two [0,1] image tensors."""
    mse = ((a - b) ** 2).mean().item()
    return 50.0 if mse < 1e-10 else -10.0 * math.log10(mse)


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """SSIM between two (H,W,3) [0,1] image tensors. NaN if skimage missing."""
    if not _HAVE_SSIM:
        return float("nan")
    an = a.detach().cpu().numpy().astype(np.float64)
    bn = b.detach().cpu().numpy().astype(np.float64)
    return float(_ssim_fn(an, bn, channel_axis=-1, data_range=1.0))


def mask_iou(pred: torch.Tensor, gt: torch.Tensor, fg: torch.Tensor) -> float:
    """IoU of two boolean masks restricted to the foreground."""
    inter = (pred & gt & fg).sum().item()
    union = ((pred | gt) & fg).sum().item()
    return inter / max(union, 1)


# ---------------------------------------------------------------------------
# Camera generation (turntable: fixed elev/radius, sweep azimuth)
# ---------------------------------------------------------------------------

def turntable_c2ws(n_views: int, elev: float, radius: float) -> np.ndarray:
    """Return (n_views, 4, 4) OpenCV c2w matrices on an azimuth turntable."""
    c2ws = []
    for i in range(n_views):
        azim = 360.0 * i / n_views
        e = math.radians(elev)
        a = math.radians(azim)
        eye = np.array([
            radius * math.cos(e) * math.sin(a),
            radius * math.sin(e),
            radius * math.cos(e) * math.cos(a),
        ], dtype=np.float32)
        c2ws.append(look_at_c2w(eye))
    return np.stack(c2ws, axis=0)


# ---------------------------------------------------------------------------
# Per-view render: DDF-oracle vs embree-oracle, IDENTICAL primary surface
# ---------------------------------------------------------------------------

def _save_triptych(ddf_img, ref_img, out_path: Path):
    """Save [DDF | embree-ref | abs-error] side by side (error x3 gain)."""
    err = (ddf_img - ref_img).abs().mean(-1, keepdim=True)
    err3 = (err * 3.0).clamp(0, 1).expand(-1, -1, 3)
    sbs = torch.cat([ddf_img.clamp(0, 1), ref_img.clamp(0, 1), err3], dim=1)
    Image.fromarray((sbs.cpu().numpy() * 255).astype(np.uint8)).save(out_path)


@torch.no_grad()
def eval_view(
    view_idx: int,
    c2w_np: np.ndarray,
    mesh_oracle: MeshOracle,
    ddf_occ_fn,
    light_dir: torch.Tensor,
    shadow_offset: float,
    image_size: int,
    fov: float,
    n_ao: int,
    ao_radius: float,
    t_self: float,
    save_imgs: bool,
    out_dir: Path,
    obj: str,
    device: str,
) -> dict:
    """Render one view in shadow + AO mode, DDF-oracle vs embree-oracle.

    Primary surface is the EMBREE MESH in BOTH renders (mesh intersect callback,
    mesh face normals via normal_fn=None). Only the occlude callback differs.
    """
    c2w = torch.from_numpy(c2w_np).to(device)
    origins, dirs = pixel_rays(c2w, image_size, fov, device)
    origins = origins.reshape(-1, 3).contiguous()
    dirs = dirs.reshape(-1, 3).contiguous()
    isz = image_size

    # Primary surface oracle: ALWAYS embree mesh, in both pipelines. The mesh
    # occlusion query uses the SAME t_self as the DDF so the comparison is fair
    # (both ignore occluders nearer than t_self — self-intersection guard).
    def mesh_intersect(o, d):
        return mesh_oracle.intersect(o, d)

    def mesh_occ(o, d, md):
        return mesh_oracle.occluded(o, d, md, t_self=t_self)

    # ---- SHADOWS ----
    # DDF-GI: embree primary + mesh face normals + DDF shadow occlusion.
    sh_ddf = render_shadows(mesh_intersect, ddf_occ_fn, None,
                            origins, dirs, light_dir,
                            shadow_offset=shadow_offset, device=device)
    # Reference: embree primary + mesh face normals + embree shadow occlusion.
    sh_ref = render_shadows(mesh_intersect, mesh_occ, None,
                            origins, dirs, light_dir,
                            shadow_offset=shadow_offset, device=device)

    # ---- AO ----
    ao_ddf = render_ao(mesh_intersect, ddf_occ_fn, None, origins, dirs,
                       n_ao=n_ao, ao_radius=ao_radius, shadow_offset=shadow_offset,
                       seed=view_idx, device=device)
    ao_ref = render_ao(mesh_intersect, mesh_occ, None, origins, dirs,
                       n_ao=n_ao, ao_radius=ao_radius, shadow_offset=shadow_offset,
                       seed=view_idx, device=device)

    # Foreground = embree primary hit (identical in both pipelines).
    fg = sh_ref["hit"].reshape(isz, isz)

    sh_ddf_img = sh_ddf["rgb"].reshape(isz, isz, 3)
    sh_ref_img = sh_ref["rgb"].reshape(isz, isz, 3)
    ao_ddf_img = ao_ddf["rgb"].reshape(isz, isz, 3)
    ao_ref_img = ao_ref["rgb"].reshape(isz, isz, 3)

    # --- shadow metrics ---
    sd_ddf = sh_ddf["shadow"].reshape(isz, isz)
    sd_ref = sh_ref["shadow"].reshape(isz, isz)
    shadow_iou = mask_iou(sd_ddf, sd_ref, fg)
    shadow_psnr = psnr(sh_ddf_img, sh_ref_img)
    shadow_ssim = ssim(sh_ddf_img, sh_ref_img)
    # Fraction of foreground that is truly in shadow (embree). IoU is only
    # meaningful where this is non-trivial: a frontally-lit view has almost no
    # true shadow, so the IoU denominator (union) is tiny and the metric is
    # dominated by a handful of edge pixels regardless of oracle quality.
    ref_shadow_frac = (sd_ref & fg).sum().item() / max(fg.sum().item(), 1)

    # --- AO metrics ---
    ao_d = ao_ddf["ao"].reshape(isz, isz)
    ao_r = ao_ref["ao"].reshape(isz, isz)
    if fg.any():
        ao_mae = (ao_d[fg] - ao_r[fg]).abs().mean().item()
    else:
        ao_mae = float("nan")
    ao_psnr = psnr(ao_ddf_img, ao_ref_img)
    ao_ssim = ssim(ao_ddf_img, ao_ref_img)

    if save_imgs:
        _save_triptych(sh_ddf_img, sh_ref_img,
                       out_dir / f"{obj}_shadow_view{view_idx:02d}.png")
        _save_triptych(ao_ddf_img, ao_ref_img,
                       out_dir / f"{obj}_ao_view{view_idx:02d}.png")

    return {
        "view": view_idx,
        "fg_px": int(fg.sum().item()),
        "ref_shadow_frac": ref_shadow_frac,
        "shadow_iou": shadow_iou,
        "shadow_psnr": shadow_psnr,
        "shadow_ssim": shadow_ssim,
        "ao_mae": ao_mae,
        "ao_psnr": ao_psnr,
        "ao_ssim": ao_ssim,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obj", required=True)
    ap.add_argument("--ckpt", default=None,
                    help="default runs/<obj>_ddf_gtmesh/ddf_final.pt")
    ap.add_argument("--gt_mesh", required=True)
    ap.add_argument("--gs_path", default=None,
                    help="default runs/<obj>/gaussians.pt (mesh_center/scale)")
    ap.add_argument("--gso_rotate", type=lambda s: s.lower() in ("1", "true", "yes"),
                    default=False,
                    help="match the DDF training supervisor.gso_rotate")
    ap.add_argument("--n_views", type=int, default=8)
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--radius", type=float, default=2.5)
    ap.add_argument("--fov", type=float, default=50.0)
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--n_ao", type=int, default=32)
    ap.add_argument("--ao_radius", type=float, default=0.5)
    ap.add_argument("--shadow_offset", type=float, default=0.08,
                    help="surface offset for secondary-ray origins (anti self-hit)")
    ap.add_argument("--t_self", type=float, default=0.10,
                    help="ignore occluders nearer than this along secondary rays. "
                         "Must exceed the DDF surface noise floor (~0.05) — at the "
                         "noise floor the DDF reports a spurious near-surface "
                         "residual along EVERY direction from a near-surface point, "
                         "so a too-small t_self makes the DDF over-shadow massively "
                         "(occ-frac ~0.8 vs embree ~0.12 on bull AO). t_self=0.10 "
                         "(~2x the noise floor) matches embree. Applied to BOTH "
                         "oracles so the comparison is fair.")
    ap.add_argument("--n_save", type=int, default=3,
                    help="save side-by-side PNGs for the first n_save views")
    ap.add_argument("--out_dir", default=None,
                    help="default runs/<obj>_ddf_gtmesh/gi")
    ap.add_argument("--vis_thresh", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if metrics.json exists")
    args = ap.parse_args()

    obj = args.obj
    ckpt = args.ckpt or f"runs/{obj}_ddf_gtmesh/ddf_final.pt"
    gs_path = args.gs_path or f"runs/{obj}/gaussians.pt"
    out_dir = Path(args.out_dir or f"runs/{obj}_ddf_gtmesh/gi")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    dev = args.device

    if metrics_path.exists() and not args.force:
        print(f"[{obj}] {metrics_path} exists; skipping (use --force to recompute).")
        print(metrics_path.read_text())
        return

    # ---- load DDF ----
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    model = build_model(ck["cfg"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{obj}] DDF loaded: {type(model).__name__}, {n_params:,} params")

    # ---- load embree mesh in the SAME world frame as the DDF ----
    gs = torch.load(gs_path, map_location="cpu", weights_only=False)
    mc = gs["mesh_center"].numpy().astype(np.float32)
    ms = float(gs["mesh_scale"].item())
    sup = GTMeshSupervisor(args.gt_mesh, mc, ms, device=dev,
                           march_ratio=0.0, gso_rotate=args.gso_rotate)
    mesh_oracle = MeshOracle(sup.mesh, device=dev)

    # ---- DDF secondary-ray occlusion oracle (ONE eval per ray) ----
    # max_dist for the directional light is large (10.0) so any occluder along
    # the shadow ray counts; vis_thresh / t_self carried through. AO clamps to
    # ao_radius internally via render_ao's per-ray max_dist tensor.
    def ddf_occ_fn(o, d, md):
        return ddf_occluded(model, o, d, md, vis_thresh=args.vis_thresh,
                            t_self=args.t_self)

    # ---- directional light ----
    light_dir = torch.tensor([0.4, -0.8, -0.3], device=dev)
    light_dir = light_dir / light_dir.norm()

    c2ws = turntable_c2ws(args.n_views, args.elev, args.radius)

    rows = []
    for i, c2w_np in enumerate(c2ws):
        r = eval_view(
            i, c2w_np, mesh_oracle, ddf_occ_fn, light_dir, args.shadow_offset,
            image_size=args.image_size, fov=args.fov,
            n_ao=args.n_ao, ao_radius=args.ao_radius, t_self=args.t_self,
            save_imgs=(i < args.n_save), out_dir=out_dir, obj=obj, device=dev,
        )
        rows.append(r)
        print(f"[{obj}] view {i:02d}: shadow_iou={r['shadow_iou']:.3f} "
              f"shadow_psnr={r['shadow_psnr']:.2f} ao_mae={r['ao_mae']:.4f} "
              f"ao_psnr={r['ao_psnr']:.2f} (fg={r['fg_px']}px)")

    def _mean(key):
        vals = [r[key] for r in rows if not (isinstance(r[key], float) and math.isnan(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    # IoU restricted to views with a meaningful real-shadow mask (>=5% of fg).
    SHADOW_FRAC_MIN = 0.05
    shadowed = [r for r in rows if r["ref_shadow_frac"] >= SHADOW_FRAC_MIN]
    shadow_iou_shaded = (float(np.mean([r["shadow_iou"] for r in shadowed]))
                         if shadowed else float("nan"))

    summary = {
        "obj": obj,
        "ckpt": str(ckpt),
        "gt_mesh": str(args.gt_mesh),
        "gso_rotate": args.gso_rotate,
        "n_views": args.n_views,
        "image_size": args.image_size,
        "fov": args.fov,
        "elev": args.elev,
        "radius": args.radius,
        "n_ao": args.n_ao,
        "ao_radius": args.ao_radius,
        "light_dir": light_dir.cpu().tolist(),
        "vis_thresh": args.vis_thresh,
        "shadow_offset": args.shadow_offset,
        "t_self": args.t_self,
        "shadow_iou": _mean("shadow_iou"),
        "shadow_iou_shaded": shadow_iou_shaded,
        "n_views_shaded": len(shadowed),
        "shadow_psnr": _mean("shadow_psnr"),
        "shadow_ssim": _mean("shadow_ssim"),
        "ao_mae": _mean("ao_mae"),
        "ao_psnr": _mean("ao_psnr"),
        "ao_ssim": _mean("ao_ssim"),
        "per_view": rows,
    }
    metrics_path.write_text(json.dumps(summary, indent=2))

    print(f"\n=== {obj} GI summary ({args.n_views} views) ===")
    print(f"  shadow:  IoU={summary['shadow_iou']:.3f} (all)  "
          f"IoU={summary['shadow_iou_shaded']:.3f} (over {len(shadowed)} shaded views)  "
          f"PSNR={summary['shadow_psnr']:.2f} dB  SSIM={summary['shadow_ssim']:.4f}")
    print(f"  AO:      MAE={summary['ao_mae']:.4f}  "
          f"PSNR={summary['ao_psnr']:.2f} dB  SSIM={summary['ao_ssim']:.4f}")
    print(f"  wrote {metrics_path}")
    print(f"  saved side-by-side PNGs for {min(args.n_save, args.n_views)} views to {out_dir}")


if __name__ == "__main__":
    main()
