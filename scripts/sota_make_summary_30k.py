"""Build runs/sota_comparison_30k/SUMMARY.md.

Pulls numbers from:
- runs/<obj>_ddf_hg/stage3/{metrics.json, pointclouds.npz}  (DDF hash-grid; bull = bull_ddf_hashgrid)
- runs/sota_comparison/<obj>/neus/chamfer.json              (NeuS @ 5k from Stage 9)
- runs/sota_comparison_30k/<obj>/neus/{chamfer.json, fscore.json, wall.json}  (NeuS @ 30k)

Writes:
- runs/sota_comparison_30k/SUMMARY.md      (full report w/ deltas + verdict)
- runs/sota_comparison_30k/all_metrics.json
- runs/sota_comparison_30k/ddf_hg_fscore.json  (per-object F-score for DDF hg, recomputed)

DDF (hg) F-score is computed on the cached `pointclouds.npz` files written by
stage3_chamfer.py (gt + pred at 20k samples). NeuS (30k) F-score is read from
the run's own fscore.json. Both use the same tau grid {0.05, 0.10, 0.20} and
the same normalised coordinate frame from `mesh_center` / `mesh_scale`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parent.parent

OBJ_ORDER = [
    "bull", "lion", "spino", "mug", "turtle",
    "airplane_94c4ade3", "chair_5f1b4529", "car_9ee32f51",
    "bottle_59d7b4e7", "sofa_145bd097",
]

DDF_HG_DIR = {
    "bull":              "bull_ddf_hashgrid",
    "lion":              "lion_ddf_hg",
    "spino":             "spino_ddf_hg",
    "mug":               "mug_ddf_hg",
    "turtle":            "turtle_ddf_hg",
    "airplane_94c4ade3": "airplane_94c4ade3_ddf_hg",
    "chair_5f1b4529":    "chair_5f1b4529_ddf_hg",
    "car_9ee32f51":      "car_9ee32f51_ddf_hg",
    "bottle_59d7b4e7":   "bottle_59d7b4e7_ddf_hg",
    "sofa_145bd097":     "sofa_145bd097_ddf_hg",
}

TAUS = [0.05, 0.10, 0.20]


def fscore(pred_pts: np.ndarray, gt_pts: np.ndarray, taus):
    tpred = cKDTree(pred_pts); tgt = cKDTree(gt_pts)
    d_pred_to_gt = tgt.query(pred_pts, k=1)[0]
    d_gt_to_pred = tpred.query(gt_pts, k=1)[0]
    out = {}
    for tau in taus:
        p = float((d_pred_to_gt < tau).mean())
        r = float((d_gt_to_pred < tau).mean())
        f1 = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0
        out[f"{tau:.2f}"] = {"precision": p, "recall": r, "f1": f1}
    return out


def load_ddf_hg(obj):
    sd = REPO / "runs" / DDF_HG_DIR[obj] / "stage3"
    m = sd / "metrics.json"; pc = sd / "pointclouds.npz"
    if not (m.exists() and pc.exists()):
        return None
    metrics = json.loads(m.read_text())
    npz = np.load(pc)
    gt = npz["gt"].astype(np.float32); pred = npz["pred"].astype(np.float32)
    return metrics, fscore(pred, gt, TAUS)


def load_neus_5k(obj):
    p = REPO / "runs" / "sota_comparison" / obj / "neus" / "chamfer.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_neus_30k(obj):
    base = REPO / "runs" / "sota_comparison_30k" / obj / "neus"
    cd = base / "chamfer.json"
    fs = base / "fscore.json"
    wall = base / "wall.json"
    return (
        json.loads(cd.read_text()) if cd.exists() else None,
        json.loads(fs.read_text()) if fs.exists() else None,
        json.loads(wall.read_text()) if wall.exists() else None,
    )


def fmt(x, digits=4):
    return "-" if x is None else f"{x:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="runs/sota_comparison_30k")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    ddf_hg_fscore_all = {}
    for obj in OBJ_ORDER:
        hg = load_ddf_hg(obj)
        if hg is not None:
            hg_metrics, hg_fs = hg
            ddf_hg_fscore_all[obj] = {"chamfer_mean": hg_metrics["chamfer_mean"],
                                       "fscore": hg_fs}
        else:
            hg_metrics, hg_fs = None, None
        n5 = load_neus_5k(obj)
        n30_cd, n30_fs, n30_wall = load_neus_30k(obj)
        rows.append({
            "obj": obj,
            "ddf_hg_mean": hg_metrics["chamfer_mean"] if hg_metrics else None,
            "ddf_hg_median": hg_metrics["chamfer_median"] if hg_metrics else None,
            "ddf_hg_fscore": hg_fs,
            "neus5k_mean": n5["chamfer_mean"] if n5 else None,
            "neus5k_median": n5["chamfer_median"] if n5 else None,
            "neus30k_mean": n30_cd["chamfer_mean"] if n30_cd else None,
            "neus30k_median": n30_cd["chamfer_median"] if n30_cd else None,
            "neus30k_n_verts": n30_cd.get("n_pred_verts") if n30_cd else None,
            "neus30k_fscore": n30_fs.get("fscore") if n30_fs else None,
            "neus30k_train_s": n30_wall.get("train_seconds") if n30_wall else None,
            "neus30k_export_s": n30_wall.get("export_seconds") if n30_wall else None,
        })

    # Write per-object DDF hg F-score record (regenerated each call).
    (out_dir / "ddf_hg_fscore.json").write_text(json.dumps(ddf_hg_fscore_all, indent=2))

    # Aggregates ------------------------------------------------------------
    have30 = [r for r in rows if r["neus30k_mean"] is not None]
    have_hg_and_30 = [r for r in rows if r["neus30k_mean"] is not None and r["ddf_hg_mean"] is not None]
    have_5_and_30 = [r for r in rows if r["neus30k_mean"] is not None and r["neus5k_mean"] is not None]

    def mean_fscore(rs, key, tau):
        vals = [r[key][f"{tau:.2f}"]["f1"] for r in rs if r.get(key) and f"{tau:.2f}" in r[key]]
        return mean(vals) if vals else None

    lines: list[str] = []
    def L(s: str = ""): lines.append(s)

    L("# Stage 9b — NeuS-facto @ 30k iters vs DDF hash-grid (10 objects)\n")
    L("Re-run of Stage 9 SOTA comparison at the paper-recommended 30k iters.\n"
      "Same 10 objects, same SDFStudio dataset, same `inside_outside=False` /\n"
      "`background_color=white` knobs, same multi-res marching cubes (res 512,\n"
      "iso=0), same normalised coordinate frame for Chamfer + F-score.\n")
    L("Adds F-score @ tau in {0.05, 0.10, 0.20} for both methods so the\n"
      "headline isn't a single noisy CD number. DDF hash-grid F-score is\n"
      "recomputed from the cached `runs/<obj>_ddf_hg/stage3/pointclouds.npz`\n"
      "(20k pred + 20k gt samples) so the comparison is apples-to-apples.\n")

    L("## Per-object Chamfer\n")
    L("`DDF (hg)` = hash-grid DDF (30 k steps, v3 supervisor) — Stage 9 headline DDF.\n")
    L("| Object | DDF (hg) mean / med | NeuS @ 5k mean / med | NeuS @ 30k mean / med | Δ30k-5k | Δ30k-DDFhg |")
    L("|---|---|---|---|---|---|")
    for r in rows:
        d_hg = f"{fmt(r['ddf_hg_mean'])} / {fmt(r['ddf_hg_median'])}"
        d_n5 = f"{fmt(r['neus5k_mean'])} / {fmt(r['neus5k_median'])}"
        d_n30 = f"{fmt(r['neus30k_mean'])} / {fmt(r['neus30k_median'])}"
        delta_n = (r["neus30k_mean"] - r["neus5k_mean"]) if (r["neus30k_mean"] is not None and r["neus5k_mean"] is not None) else None
        delta_d = (r["neus30k_mean"] - r["ddf_hg_mean"]) if (r["neus30k_mean"] is not None and r["ddf_hg_mean"] is not None) else None
        L(f"| {r['obj']} | {d_hg} | {d_n5} | {d_n30} | "
          f"{(f'{delta_n:+.4f}' if delta_n is not None else '-')} | "
          f"{(f'{delta_d:+.4f}' if delta_d is not None else '-')} |")
    L("")

    L("## Per-object F-score @ tau (F1, higher = better)\n")
    L("| Object | DDF (hg) F1@0.05 / .10 / .20 | NeuS @ 30k F1@0.05 / .10 / .20 | ΔF1@0.10 (NeuS - DDFhg) |")
    L("|---|---|---|---|")
    for r in rows:
        hg = r["ddf_hg_fscore"] or {}
        n30 = r["neus30k_fscore"] or {}
        def f(d, tau): return d.get(f"{tau:.2f}", {}).get("f1") if d else None
        hg_s = " / ".join(fmt(f(hg, t), 3) for t in TAUS)
        n_s  = " / ".join(fmt(f(n30, t), 3) for t in TAUS)
        d10 = (f(n30, 0.10) - f(hg, 0.10)) if (f(n30, 0.10) is not None and f(hg, 0.10) is not None) else None
        L(f"| {r['obj']} | {hg_s} | {n_s} | {(f'{d10:+.3f}' if d10 is not None else '-')} |")
    L("")

    L("## Aggregates (n={} objects with NeuS @ 30k)\n".format(len(have30)))
    if have_hg_and_30:
        d_hg = [r["ddf_hg_mean"] for r in have_hg_and_30]
        n30 = [r["neus30k_mean"] for r in have_hg_and_30]
        deltas = [a - b for a, b in zip(n30, d_hg)]
        wins_n = sum(1 for d in deltas if d < 0)
        L(f"- Mean CD: DDF (hg) **{mean(d_hg):.4f}** vs NeuS @ 30k **{mean(n30):.4f}**, "
          f"ΔNeuS = {mean(deltas):+.4f} ({mean(deltas)/mean(d_hg)*100:+.1f}%). "
          f"NeuS wins {wins_n}/{len(have_hg_and_30)} per-object on mean CD.")
    if have_5_and_30:
        n5 = [r["neus5k_mean"] for r in have_5_and_30]
        n30 = [r["neus30k_mean"] for r in have_5_and_30]
        deltas = [a - b for a, b in zip(n30, n5)]
        wins_30 = sum(1 for d in deltas if d < 0)
        L(f"- Mean CD: NeuS @ 5k **{mean(n5):.4f}** vs NeuS @ 30k **{mean(n30):.4f}**, "
          f"Δ30k-5k = {mean(deltas):+.4f}. 30k beats 5k on {wins_30}/{len(have_5_and_30)} objects.")

    for tau in TAUS:
        hg_vals = [r["ddf_hg_fscore"][f"{tau:.2f}"]["f1"] for r in have_hg_and_30
                   if r.get("ddf_hg_fscore") and f"{tau:.2f}" in r["ddf_hg_fscore"]]
        n_vals  = [r["neus30k_fscore"][f"{tau:.2f}"]["f1"] for r in have_hg_and_30
                   if r.get("neus30k_fscore") and f"{tau:.2f}" in r["neus30k_fscore"]]
        if hg_vals and n_vals:
            L(f"- Mean F1@{tau:.2f}: DDF (hg) **{mean(hg_vals):.3f}** vs NeuS @ 30k **{mean(n_vals):.3f}**, "
              f"Δ = {mean(n_vals)-mean(hg_vals):+.3f}.")
    train_wall = [r["neus30k_train_s"] for r in rows if r["neus30k_train_s"]]
    if train_wall:
        L(f"- NeuS @ 30k train wall: mean {mean(train_wall):.0f}s/obj "
          f"({mean(train_wall)/60:.1f} min), range "
          f"{min(train_wall):.0f}-{max(train_wall):.0f}s.")
    L("")

    L("## Verdict\n")
    # Only emit verdict if we have the comparison data.
    if have_hg_and_30:
        d_hg_mean = mean([r["ddf_hg_mean"] for r in have_hg_and_30])
        n30_mean = mean([r["neus30k_mean"] for r in have_hg_and_30])
        delta = n30_mean - d_hg_mean
        rel = delta / d_hg_mean
        wins_n = sum(1 for r in have_hg_and_30 if r["neus30k_mean"] < r["ddf_hg_mean"])
        if abs(delta) <= 0.015:
            verdict = (f"**Tied within noise.** ΔCD = {delta:+.4f} ({rel*100:+.1f}%), "
                       f"inside the ±0.015 single-eval sampling noise floor. "
                       f"Per-object wins split {wins_n}/{len(have_hg_and_30)} for NeuS.")
        elif delta < -0.02:
            verdict = (f"**NeuS @ 30k clearly beats DDF hash-grid on geometry.** "
                       f"ΔCD = {delta:+.4f} ({rel*100:+.1f}%) -- well outside the "
                       f"±0.015 noise floor. NeuS wins {wins_n}/{len(have_hg_and_30)}.\n\n"
                       "Paper framing shifts: 'DDF is the *fast* method even though "
                       "NeuS wins on geometry by ~{:.0%}.'".format(-rel))
        elif delta > 0.02:
            verdict = (f"**DDF hash-grid wins on geometry.** ΔCD = {delta:+.4f} "
                       f"({rel*100:+.1f}%) in DDF's favour, outside noise. DDF wins "
                       f"{len(have_hg_and_30)-wins_n}/{len(have_hg_and_30)}.")
        else:
            verdict = (f"**Noise-flipped from Stage 9.** ΔCD = {delta:+.4f} "
                       f"({rel*100:+.1f}%) -- within ±0.02 but outside ±0.015. "
                       f"NeuS wins {wins_n}/{len(have_hg_and_30)}; the comparison "
                       f"is too close to call definitively.")
        L(verdict + "\n")

    L("Raw: `runs/sota_comparison_30k/all_metrics.json`, "
      "`runs/sota_comparison_30k/ddf_hg_fscore.json`, "
      "per-object `runs/sota_comparison_30k/<obj>/neus/{chamfer,fscore,wall}.json`.")

    (out_dir / "SUMMARY.md").write_text("\n".join(lines))
    (out_dir / "all_metrics.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {out_dir}/SUMMARY.md  +  all_metrics.json  +  ddf_hg_fscore.json")


if __name__ == "__main__":
    main()
