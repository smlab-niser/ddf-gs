"""Aggregate the E4 GI-at-scale study: shadow/AO fidelity of GT-mesh-supervised
DDF oracles vs embree ground truth, across all trained objects.

Each runs/<obj>_ddf_gtmesh/gi/metrics.json holds shadow + AO metrics computed
by gi_render_eval.py (embree primary trace in both; only the secondary-ray
oracle differs — DDF vs embree-mesh). Robust metrics: shadow_psnr, ao_psnr,
ao_mae (PSNR of the rendered shadow/AO maps vs embree GT). shadow_iou_shaded is
reported but flagged: it degenerates (~0) on frontally-lit / thin objects where
the true-shadow foreground is tiny, so mean/median of it is noisy.

Splits GSO (plain names) vs ShapeNet (name_<8hexhash>). Writes
runs/e4_gi_summary.json + prints a markdown report.
"""
import json, re, glob, statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SHAPENET_RE = re.compile(r"_[0-9a-f]{8}$")  # ShapeNet ids end with an 8-hex hash


def collect():
    rows = []
    for f in sorted(glob.glob(str(REPO / "runs/*_ddf_gtmesh/gi/metrics.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        obj = d.get("obj", Path(f).parts[-3].replace("_ddf_gtmesh", ""))
        rows.append({
            "obj": obj,
            "set": "shapenet" if SHAPENET_RE.search(obj) else "gso",
            "shadow_psnr": d.get("shadow_psnr"),
            "ao_psnr": d.get("ao_psnr"),
            "ao_mae": d.get("ao_mae"),
            "shadow_iou_shaded": d.get("shadow_iou_shaded"),
            "shadow_ssim": d.get("shadow_ssim"),
            "ao_ssim": d.get("ao_ssim"),
        })
    return rows


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": round(st.mean(vals), 3), "median": round(st.median(vals), 3),
            "min": round(min(vals), 3), "max": round(max(vals), 3), "n": len(vals)}


def main():
    rows = collect()
    print(f"collected {len(rows)} objects "
          f"({sum(r['set']=='gso' for r in rows)} GSO, "
          f"{sum(r['set']=='shapenet' for r in rows)} ShapeNet)\n")

    summary = {"n_total": len(rows)}
    for subset in ("all", "gso", "shapenet"):
        sel = rows if subset == "all" else [r for r in rows if r["set"] == subset]
        if not sel:
            continue
        summary[subset] = {
            "n": len(sel),
            "shadow_psnr": agg([r["shadow_psnr"] for r in sel]),
            "ao_psnr": agg([r["ao_psnr"] for r in sel]),
            "ao_mae": agg([r["ao_mae"] for r in sel]),
            "shadow_iou_shaded": agg([r["shadow_iou_shaded"] for r in sel]),
            "shadow_ssim": agg([r["shadow_ssim"] for r in sel]),
            "ao_ssim": agg([r["ao_ssim"] for r in sel]),
        }

    # distribution: fraction of objects above PSNR thresholds (robust headline)
    sp = [r["shadow_psnr"] for r in rows if r["shadow_psnr"] is not None]
    ap = [r["ao_psnr"] for r in rows if r["ao_psnr"] is not None]
    summary["shadow_psnr_frac_ge"] = {t: round(sum(v >= t for v in sp) / len(sp), 3)
                                      for t in (20, 25, 30)}
    summary["ao_psnr_frac_ge"] = {t: round(sum(v >= t for v in ap) / len(ap), 3)
                                  for t in (18, 20, 22)}

    out = REPO / "runs/e4_gi_summary.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    # ---- markdown report ----
    print("## E4 — GI-at-scale: DDF secondary-ray oracle vs embree GT "
          f"(n={len(rows)} GT-mesh-supervised oracles)\n")
    print("| subset | n | shadow_psnr (mean/med) | ao_psnr (mean/med) | ao_mae (mean) |")
    print("|---|---:|---:|---:|---:|")
    for subset in ("all", "gso", "shapenet"):
        if subset not in summary:
            continue
        s = summary[subset]
        sp_, ap_, am_ = s["shadow_psnr"], s["ao_psnr"], s["ao_mae"]
        print(f"| {subset} | {s['n']} | {sp_['mean']}/{sp_['median']} dB "
              f"| {ap_['mean']}/{ap_['median']} dB | {am_['mean']} |")
    print(f"\nshadow_psnr ≥ {{20,25,30}} dB: "
          + ", ".join(f"{int(t)}→{summary['shadow_psnr_frac_ge'][t]*100:.0f}%"
                      for t in (20, 25, 30)))
    print(f"ao_psnr ≥ {{18,20,22}} dB: "
          + ", ".join(f"{int(t)}→{summary['ao_psnr_frac_ge'][t]*100:.0f}%"
                      for t in (18, 20, 22)))
    sis = summary["all"]["shadow_iou_shaded"]
    print(f"\nshadow_iou_shaded (NOISY — degenerate on thin/frontal-lit): "
          f"mean {sis['mean']}, median {sis['median']}, range [{sis['min']},{sis['max']}]")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
