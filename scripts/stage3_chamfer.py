"""Stage 3: DDF -> UDF -> marching cubes -> mesh -> Chamfer vs GT GSO mesh.

UDF(x) = min over directions of DDF(x, dir) — sample K random directions per
grid point, mask out non-hits via visibility, take min. Marching cubes at a
small iso recovers a surface mesh, which we compare to the normalized GSO
ground-truth mesh via symmetric Chamfer distance on sampled point clouds.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from src.ddf_model import DDF
from src.ddf_hashgrid import DDFHashGrid
from src.ddf_mixture import DDFMixture
from src.ddf_photometric import DDFHashGridPhotometric
from src.ddf_photorender import DDFHashGridPhotoRender


def _build_model_from_cfg(cfg, device):
    """Dispatch on cfg['model']['type'] to instantiate the right class."""
    m_cfg = cfg["model"]
    kind = str(m_cfg.get("type", "ddf")).lower()
    if kind == "hashgrid":
        return DDFHashGrid(
            dir_freqs=m_cfg.get("dir_freqs", 4),
            hidden_dim=m_cfg.get("hidden_dim", 64),
            num_layers=m_cfg.get("num_layers", 2),
            n_levels=m_cfg.get("n_levels", 16),
            feat_dim=m_cfg.get("feat_dim", 2),
            log2_table_size=m_cfg.get("log2_table_size", 19),
            base_res=m_cfg.get("base_res", 16),
            growth=m_cfg.get("growth", 1.5),
            bbox_half=m_cfg.get("bbox_half", 1.2),
        ).to(device)
    if kind == "mixture":
        return DDFMixture(
            K=m_cfg.get("K", 2),
            dir_freqs=m_cfg.get("dir_freqs", 4),
            hidden_dim=m_cfg.get("hidden_dim", 64),
            num_layers=m_cfg.get("num_layers", 2),
            n_levels=m_cfg.get("n_levels", 16),
            feat_dim=m_cfg.get("feat_dim", 2),
            log2_table_size=m_cfg.get("log2_table_size", 19),
            base_res=m_cfg.get("base_res", 16),
            growth=m_cfg.get("growth", 1.5),
            bbox_half=m_cfg.get("bbox_half", 1.2),
        ).to(device)
    if kind == "hashgrid_photometric":
        return DDFHashGridPhotometric(
            dir_freqs=m_cfg.get("dir_freqs", 4),
            hidden_dim=m_cfg.get("hidden_dim", 64),
            num_layers=m_cfg.get("num_layers", 2),
            n_levels=m_cfg.get("n_levels", 16),
            feat_dim=m_cfg.get("feat_dim", 2),
            log2_table_size=m_cfg.get("log2_table_size", 19),
            base_res=m_cfg.get("base_res", 16),
            growth=m_cfg.get("growth", 1.5),
            bbox_half=m_cfg.get("bbox_half", 1.2),
        ).to(device)
    if kind == "hashgrid_photorender":
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
    return DDF(
        pos_freqs=m_cfg["pos_freqs"], dir_freqs=m_cfg["dir_freqs"],
        hidden_dim=m_cfg["hidden_dim"], num_layers=m_cfg["num_layers"],
    ).to(device)


_MANIFEST_PATH = Path("runs/obj_manifest.json")
_FALLBACK = {
    "bull": "Schleich_Hereford_Bull",
    "lion": "Schleich_Lion_Action_Figure",
    "spino": "Schleich_Spinosaurus_Action_Figure",
    "shoe": "11pro_SL_TRX_FG",
    "turtle": "Vtech_Roll_Learn_Turtle",
    "mug": "ACE_Coffee_Mug_Kristen_16_oz_cup",
}
if _MANIFEST_PATH.exists():
    NAME_MAP = {**_FALLBACK, **json.loads(_MANIFEST_PATH.read_text())}
else:
    NAME_MAP = _FALLBACK

# ShapeNet manifest is separate (short_id -> {category, model_hash, glb_path}).
# Its objects live at data/shapenet/<short>/model.glb and are not in NAME_MAP.
_SHAPENET_MANIFEST_PATH = Path("runs/shapenet_manifest.json")
if _SHAPENET_MANIFEST_PATH.exists():
    SHAPENET_MAP = json.loads(_SHAPENET_MANIFEST_PATH.read_text())
else:
    SHAPENET_MAP = {}


@torch.no_grad()
def query_udf(model, points: torch.Tensor, n_dirs: int = 32, vis_thresh: float = 0.3):
    """For each point, take min DDF over n_dirs random directions; masked by visibility."""
    n = points.shape[0]
    dirs = torch.randn(n * n_dirs, 3, device=points.device)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    pts = points.repeat_interleave(n_dirs, dim=0)
    out = model(pts, dirs)
    # Photometric variant returns (t, vis_logit, rgb); take the first two.
    t, vis_logit = out[0], out[1]
    visible = vis_logit.sigmoid() > vis_thresh
    t = torch.where(visible, t, torch.full_like(t, 1e3))
    return t.view(n, n_dirs).min(dim=-1).values


@torch.no_grad()
def extract_mesh(model, grid_size: int, bbox_half: float, n_dirs: int,
                 iso: float, device: str, chunk: int = 65536):
    coords = torch.linspace(-bbox_half, bbox_half, grid_size, device=device)
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing="ij")
    pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    udf_vals = []
    for i in range(0, pts.shape[0], chunk):
        udf_vals.append(query_udf(model, pts[i:i + chunk], n_dirs=n_dirs).cpu())
    udf = torch.cat(udf_vals).view(grid_size, grid_size, grid_size).numpy()
    spacing = (2 * bbox_half) / (grid_size - 1)
    try:
        verts, faces, _, _ = marching_cubes(udf, level=iso, spacing=(spacing,) * 3)
    except (ValueError, RuntimeError) as e:
        return None, None, udf, str(e)
    verts = verts + np.array([-bbox_half, -bbox_half, -bbox_half], dtype=np.float32)
    return verts, faces, udf, None


def chamfer(p1: np.ndarray, p2: np.ndarray) -> tuple[float, float]:
    """Symmetric Chamfer: sum of one-sided means. Also returns L1 (sum of medians)."""
    t1, t2 = cKDTree(p1), cKDTree(p2)
    d12 = t2.query(p1, k=1)[0]
    d21 = t1.query(p2, k=1)[0]
    return float(d12.mean() + d21.mean()), float(np.median(d12) + np.median(d21))


def load_gt_points(obj_path: Path, mesh_center: np.ndarray, mesh_scale: float,
                   n_samples: int = 20000) -> np.ndarray:
    gt = trimesh.load(obj_path, force="mesh", process=False)
    # .glb may load as a Scene even with force='mesh' (multi-material scenes).
    if isinstance(gt, trimesh.Scene):
        geoms = [g for g in gt.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError(f"No mesh geometry in {obj_path}")
        gt = trimesh.util.concatenate(geoms)
    gt.apply_translation(-mesh_center)
    gt.apply_scale(mesh_scale)
    pts, _ = trimesh.sample.sample_surface(gt, n_samples)
    return np.asarray(pts, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True,
                    help="short id; must be in obj_manifest.json (GSO) or shapenet_manifest.json")
    ap.add_argument("--ddf_ckpt", default=None, help="defaults to runs/<obj>_ddf/ddf_final.pt or _v3")
    ap.add_argument("--gs_path", default=None, help="defaults to runs/<obj>/gaussians.pt")
    ap.add_argument("--gt_mesh", default=None,
                    help="explicit GT mesh path; overrides manifest lookup (use for ShapeNet .glb)")
    ap.add_argument("--grid_size", type=int, default=128)
    ap.add_argument("--bbox_half", type=float, default=1.2)
    ap.add_argument("--n_dirs", type=int, default=32)
    ap.add_argument("--iso", type=float, default=0.05)
    ap.add_argument("--n_samples", type=int, default=20000)
    ap.add_argument("--out_dir", default=None, help="defaults to runs/<obj>_ddf/stage3")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    obj = args.obj
    # Validate obj is known unless an explicit GT mesh was given.
    if args.gt_mesh is None and obj not in NAME_MAP and obj not in SHAPENET_MAP:
        raise SystemExit(
            f"unknown obj {obj!r}: not in GSO manifest and not in ShapeNet manifest; "
            "pass --gt_mesh to bypass manifest lookup"
        )
    ddf_ckpt = Path(args.ddf_ckpt) if args.ddf_ckpt else Path(f"runs/{obj}_ddf/ddf_final.pt")
    if not ddf_ckpt.exists() and obj == "bull":
        ddf_ckpt = Path("runs/bull_ddf_v3/ddf_final.pt")
    gs_path = Path(args.gs_path) if args.gs_path else Path(f"runs/{obj}/gaussians.pt")
    out_dir = Path(args.out_dir) if args.out_dir else ddf_ckpt.parent / "stage3"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device

    ckpt = torch.load(ddf_ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = _build_model_from_cfg(cfg, device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    gs = torch.load(gs_path, map_location="cpu", weights_only=False)
    mesh_center = gs["mesh_center"].numpy().astype(np.float32)
    mesh_scale = float(gs["mesh_scale"].item())

    print(f"[{obj}] extracting mesh @ grid={args.grid_size}, n_dirs={args.n_dirs}, iso={args.iso}...")
    verts, faces, udf, err = extract_mesh(
        model, args.grid_size, args.bbox_half, args.n_dirs, args.iso, device
    )
    if verts is None:
        print(f"[{obj}] marching cubes failed at iso={args.iso}: {err}")
        print(f"[{obj}] UDF stats: min={udf.min():.4f} max={udf.max():.4f} "
              f"mean={udf.mean():.4f} p1={np.percentile(udf,1):.4f}")
        # Retry at the 1st percentile of UDF as a fallback iso.
        new_iso = float(np.percentile(udf, 1))
        spacing = (2 * args.bbox_half) / (args.grid_size - 1)
        try:
            verts, faces, _, _ = marching_cubes(udf, level=new_iso, spacing=(spacing,) * 3)
            verts = verts + np.array([-args.bbox_half] * 3, dtype=np.float32)
            print(f"[{obj}] retry with iso={new_iso:.4f} succeeded")
        except Exception as e2:
            print(f"[{obj}] retry also failed: {e2}")
            return

    pred_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    pred_mesh_path = out_dir / "pred_mesh.ply"
    pred_mesh.export(pred_mesh_path)
    print(f"[{obj}] wrote {pred_mesh_path} ({len(verts)} verts, {len(faces)} faces)")

    if args.gt_mesh is not None:
        obj_path = Path(args.gt_mesh)
    elif obj in SHAPENET_MAP:
        obj_path = Path(f"data/shapenet/{obj}/model.glb")
    else:
        obj_path = Path(f"data/gso/{NAME_MAP[obj]}/meshes/model.obj")
    gt_pts = load_gt_points(obj_path, mesh_center, mesh_scale, args.n_samples)
    pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, args.n_samples)
    pred_pts = np.asarray(pred_pts, dtype=np.float32)
    cd_mean, cd_med = chamfer(gt_pts, pred_pts)

    np.savez(out_dir / "pointclouds.npz", gt=gt_pts, pred=pred_pts)

    result = {
        "obj": obj,
        "ddf_ckpt": str(ddf_ckpt),
        "grid_size": args.grid_size,
        "iso": args.iso,
        "n_verts": int(len(verts)),
        "n_faces": int(len(faces)),
        "chamfer_mean": cd_mean,
        "chamfer_median": cd_med,
    }
    with (out_dir / "metrics.json").open("w") as f:
        json.dump(result, f, indent=2)

    print(f"[{obj}] Chamfer mean={cd_mean:.4f}  median={cd_med:.4f}  -> {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
