"""F-score @ thresholds for all 29 objects.

For each obj:
  - load pred mesh from runs/<obj>_ddf/stage3/pred_mesh.ply
  - load GT pointcloud from runs/<obj>_ddf*/stage3/pointclouds.npz['gt']
  - sample N points from pred mesh
  - for each tau in {0.05, 0.10, 0.20}:
        precision = mean(min_dist(pred->GT) < tau)
        recall    = mean(min_dist(GT->pred) < tau)
        F1 = 2 p r / (p+r)
Writes per-object table + mean to runs/cheap_eval/fscore/.

Coords are unit-normalized object scale (same frame as stage3 pointclouds).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


# Canonical 29 objects (matches the paper's eval list); bull uses v3 ckpt.
OBJECTS = [
    "airplane", "allo", "bagel", "blueMug", "boatShoe", "bowl", "bull",
    "bundtPan", "bus", "car", "chocoBox", "clock", "eagle", "hammer",
    "horse", "lion", "mug", "orca", "panda", "rhino", "sausage", "shoe",
    "spino", "teapot", "teddy", "thomas", "torch", "triceratop", "turtle",
]


def stage3_dir_for(obj: str) -> Path:
    if obj == "bull":
        p = Path(f"runs/bull_ddf_v3/stage3")
        if p.exists():
            return p
    return Path(f"runs/{obj}_ddf/stage3")


def fscore(pred_pts: np.ndarray, gt_pts: np.ndarray,
           taus: list[float]) -> dict[float, dict[str, float]]:
    tpred = cKDTree(pred_pts)
    tgt = cKDTree(gt_pts)
    d_pred_to_gt = tgt.query(pred_pts, k=1)[0]
    d_gt_to_pred = tpred.query(gt_pts, k=1)[0]
    out: dict[float, dict[str, float]] = {}
    for tau in taus:
        p = float((d_pred_to_gt < tau).mean())
        r = float((d_gt_to_pred < tau).mean())
        f1 = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0
        out[tau] = {"precision": p, "recall": r, "f1": f1}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    ap.add_argument("--n_samples", type=int, default=20000)
    ap.add_argument("--out_dir", default="runs/cheap_eval/fscore")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    taus = args.taus
    rng = np.random.default_rng(0)

    rows = []
    for obj in OBJECTS:
        sd = stage3_dir_for(obj)
        mesh_p = sd / "pred_mesh.ply"
        pc_p = sd / "pointclouds.npz"
        if not (mesh_p.exists() and pc_p.exists()):
            print(f"[{obj}] missing inputs: {mesh_p} / {pc_p}")
            continue
        # Use cached pred + gt pointclouds (already sampled at 20k each).
        npz = np.load(pc_p)
        gt_pts = npz["gt"].astype(np.float32)
        pred_pts = npz["pred"].astype(np.float32)
        # Resample pred to n_samples from the mesh for consistency.
        if args.n_samples != pred_pts.shape[0]:
            mesh = trimesh.load(mesh_p, force="mesh", process=False)
            pp, _ = trimesh.sample.sample_surface(mesh, args.n_samples)
            pred_pts = np.asarray(pp, dtype=np.float32)
        result = fscore(pred_pts, gt_pts, taus)
        row = {"obj": obj}
        for tau in taus:
            row[f"P@{tau}"] = result[tau]["precision"]
            row[f"R@{tau}"] = result[tau]["recall"]
            row[f"F1@{tau}"] = result[tau]["f1"]
        rows.append(row)
        msg = "  ".join(f"F1@{tau:.2f}={result[tau]['f1']:.3f}" for tau in taus)
        print(f"[{obj:<12s}] {msg}")

    # Compute mean per metric
    mean_row: dict[str, float] = {"obj": "MEAN"}
    if rows:
        for tau in taus:
            for key in (f"P@{tau}", f"R@{tau}", f"F1@{tau}"):
                mean_row[key] = float(np.mean([r[key] for r in rows]))
    rows.append(mean_row)

    # Write JSON
    (out_dir / "fscore.json").write_text(json.dumps(rows, indent=2))

    # Write markdown table
    cols = ["obj"] + [c for tau in taus for c in (f"P@{tau}", f"R@{tau}", f"F1@{tau}")]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for r in rows:
        vals = [r["obj"]] + [f"{r.get(c, float('nan')):.3f}" if c != "obj" else r["obj"]
                              for c in cols[1:]]
        lines.append("| " + " | ".join(vals) + " |")
    (out_dir / "fscore.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_dir}/fscore.json + fscore.md")


if __name__ == "__main__":
    main()
