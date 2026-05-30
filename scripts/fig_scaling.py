"""Fig 3 (headline): DDF vs BVH-over-Gaussians scaling. Two panels:
(a) per-ray query latency vs scene size; (b) ray-oracle memory vs scene size.
Panel descriptions live in the LaTeX caption; one shared legend at the bottom.
Reads runs/bench_ddf_vs_bvh.json. Writes renders/fig_scaling.png/.pdf.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 15, "font.family": "DejaVu Sans",
    "axes.labelsize": 17, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 16, "axes.linewidth": 1.3,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150,
})

D = json.load(open("runs/bench_ddf_vs_bvh.json"))
DRT = json.load(open("runs/bench_ddf_vs_bvh_rt.json"))
DTC = json.load(open("runs/bench_ddf_tcnn.json"))
ng = [int(x) for x in D["n_gaussians_sweep"]]
bvh_us = DRT["bvh_rt"]["us_per_ray"]  # OptiX RT-core on RTX 2080 Ti
ddf_us = D["ddf"]["us_per_ray"]
ddf_tcnn_us = DTC["ddf_tcnn"]["us_per_ray"]
ddf_mb = D["ddf"]["mb"]
bpg = DRT["bvh_rt"]["bytes_per_gaussian"]
DDF_C, BVH_C, TC_C = "#1f77b4", "#d62728", "#2ca02c"
LW, MS = 3.4, 11

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4))

# ---- panel (a): per-ray latency vs N_gaussians (BVH shown at a 1M-ray frame) ----
nr = "1048576"
yb = [bvh_us[str(g)][nr] for g in ng]
lddf, = axL.plot(ng, [ddf_us[nr]] * len(ng), color=DDF_C, lw=LW, ls="--", marker="D", ms=MS, alpha=0.85)
ltc, = axL.plot(ng, [ddf_tcnn_us[nr]] * len(ng), color=TC_C, lw=LW, marker="o", ms=MS)
lbvh, = axL.plot(ng, yb, color=BVH_C, lw=LW, marker="s", ms=MS)
axL.set_xscale("log"); axL.set_yscale("log")
axL.set_xlabel("scene size (N Gaussians)")
axL.set_ylabel("per-ray latency (µs)")
gr = yb[-1] / yb[0]
axL.annotate(f"BVH ×{gr:.1f}", xy=(ng[-1], yb[-1]),
             xytext=(ng[1], yb[-1] * 1.5), fontsize=15, color=BVH_C, ha="center", fontweight="bold")
axL.text(0.04, 0.90, "(a)", transform=axL.transAxes, fontsize=19, fontweight="bold")

# ---- panel (b): ray-oracle memory vs N_gaussians ----
gg = np.array([5000, 50000, 200000, 1_000_000, 4_000_000], dtype=float)
mb_bvh = gg * bpg / 1e6
axR.plot(gg, mb_bvh, color=BVH_C, lw=LW, marker="s", ms=MS)
axR.axhline(ddf_mb, color=DDF_C, lw=LW)
cross = ddf_mb * 1e6 / bpg
axR.axvline(cross, color="gray", ls=":", lw=1.8)
axR.annotate(f"crossover\n~{cross/1e3:.0f}k", xy=(cross, ddf_mb),
             xytext=(cross * 1.18, ddf_mb * 0.22), fontsize=13, color="0.35")
axR.axvspan(5e5, 4e6, color="0.85", alpha=0.5)
axR.text(1.45e6, mb_bvh[-1] * 0.45, "real GS\nscenes", fontsize=13, color="0.4", ha="center")
axR.set_xscale("log"); axR.set_yscale("log")
axR.set_xlabel("scene size (N Gaussians)")
axR.set_ylabel("ray-oracle memory (MB)")
axR.text(0.04, 0.90, "(b)", transform=axR.transAxes, fontsize=19, fontweight="bold")

# ---- one shared legend at the bottom ----
fig.legend([ltc, lddf, lbvh],
           ["DDF (tinycudann, ours)", "DDF (pure PyTorch, ours)", "BVH-over-Gaussians (RT cores)"],
           loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=[0, 0.09, 1, 1])
for ext in ("png", "pdf"):
    fig.savefig(f"renders/fig_scaling.{ext}", bbox_inches="tight")
print("wrote renders/fig_scaling.png / .pdf")
