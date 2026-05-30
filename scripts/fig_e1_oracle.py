"""Fig (E1): mesh-free secondary-ray oracle — shadow-ray agreement vs embree GT.

Grouped bars per object + mean: gtmesh-omnidir (mesh-trained upper bound),
NeuS-omnidir (ours, mesh-free), neus_v3 (frustum-only baseline). Reads
runs/e1_shadow_agreement.json. Writes renders/fig_e1_oracle.png/.pdf.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})

D = json.load(open("runs/e1_shadow_agreement.json"))
objs = [k for k in D if not k.startswith("_")]
order = ["bull", "mug", "turtle", "lion"]
objs = [o for o in order if o in objs] + [o for o in objs if o not in order]

def col(o, key):
    return D[o].get(key)

labels = objs + ["mean"]
gt = [col(o, "gtmesh_omnidir") for o in objs] + [D["_mean"]["gtmesh_omnidir"]]
ours = [col(o, "neus_omnidir") for o in objs] + [D["_mean"]["neus_omnidir"]]
base = [col(o, "neus_v3") for o in objs] + [D["_mean"]["neus_v3"]]

x = np.arange(len(labels)); w = 0.26
fig, ax = plt.subplots(figsize=(8.2, 4.4))
ax.bar(x - w, gt, w, label="gtmesh-omnidir (mesh, upper bound)", color="#7f7f7f")
ax.bar(x, ours, w, label="NeuS-omnidir (ours, mesh-free)", color="#1f77b4")
ax.bar(x + w, base, w, label="neus_v3 (frustum-only baseline)", color="#d62728")
for xi, (a, b, c) in enumerate(zip(gt, ours, base)):
    for dx, v in [(-w, a), (0, b), (w, c)]:
        ax.text(xi + dx, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)

ax.axhspan(0, 50, color="0.93", zorder=0)
ax.text(len(labels)-0.5, 47, "≈ trivial-predictor zone", fontsize=8, color="0.5", ha="right", va="top")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("shadow-ray agreement vs embree GT (%)")
ax.set_ylim(0, 105)
ax.set_title("Mesh-free DDF oracle closes the circularity\n"
             "(images → NeuS → DDF, no mesh): approaches the mesh-trained bound, ≫ baseline")
ax.legend(fontsize=8.5, loc="lower center", ncol=1, framealpha=0.9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"renders/fig_e1_oracle.{ext}", bbox_inches="tight")
print("wrote renders/fig_e1_oracle.png / .pdf")
