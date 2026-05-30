"""Fig (E4): GI-at-scale — distribution of DDF shadow/AO fidelity vs embree over
142 GT-mesh-supervised oracles (29 GSO + 113 ShapeNet).

Two panels: shadow_psnr and ao_psnr histograms, GSO vs ShapeNet stacked, with
mean/median lines. Reads runs/e4_gi_summary.json. Writes renders/fig_e4_dist.png/.pdf.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})

D = json.load(open("runs/e4_gi_summary.json"))
rows = D["rows"]
gso = lambda key: [r[key] for r in rows if r["set"] == "gso" and r[key] is not None]
shp = lambda key: [r[key] for r in rows if r["set"] == "shapenet" and r[key] is not None]

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, key, title in [(axs[0], "shadow_psnr", "Shadow fidelity"),
                       (axs[1], "ao_psnr", "Ambient-occlusion fidelity")]:
    g, s = gso(key), shp(key)
    allv = g + s
    bins = np.linspace(min(allv), max(allv), 24)
    ax.hist([g, s], bins=bins, stacked=True, color=["#1f77b4", "#aec7e8"],
            label=[f"GSO (n={len(g)})", f"ShapeNet (n={len(s)})"], edgecolor="white", lw=0.3)
    m, md = np.mean(allv), np.median(allv)
    ax.axvline(m, color="#d62728", lw=2, label=f"mean {m:.1f} dB")
    ax.axvline(md, color="#d62728", ls="--", lw=1.5, label=f"median {md:.1f} dB")
    ax.set_xlabel("PSNR vs embree GT (dB)")
    ax.set_ylabel("number of objects")
    ax.set_title(f"{title}  (n={len(allv)})")
    ax.legend(fontsize=8.5)
fig.suptitle("DDF secondary-ray oracle reproduces embree GI across 142 objects "
             "(O(1)/ray)", y=1.02, fontsize=12)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"renders/fig_e4_dist.{ext}", bbox_inches="tight")
print("wrote renders/fig_e4_dist.png / .pdf")
