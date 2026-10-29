"""Figure of Paper 5 draft: declared predictions against the formula fixed after the run (noise wall stage 1,
29 Oct 2026). Usage, from docs/papers: python readable_depth_figures.py PATH_TO_TOQB"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
T = (sys.argv[1] if len(sys.argv) > 1 else "../../../TOQB") + "/results/"
D = {"ibm_marrakesh": (json.load(open(T + "noise-wall-1-marrakesh-2026-10-29/job1/charge_score_selfref.json")),
                       "declared: p from own n=20, L=24 (0.00719)", {20: (2.27, 2.30), 36: (2.25, 2.39)}),
     "ibm_kingston": (json.load(open(T + "noise-wall-1-kingston-2026-10-29/job1/charge_score.json")),
                      "declared: p = 0.0051 (from 10/28)", {20: (2.24, 2.30), 36: (2.22, 2.39)})}
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#6b6b6b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#bbbbbb",
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharex=True, sharey=True)
Lg = np.linspace(0, 105, 200)
for i, (dev, (d, dlab, br)) in enumerate(D.items()):
    p = d["p"]["value"]; kdec = -2 * np.log(1 - 4 * p / 3)
    for j, n in enumerate((20, 36)):
        ax = axes[i, j]
        get = lambda arm: sorted([c for c in d["cells"] if c["arm"] == arm and c["n"] == n], key=lambda c: c["L"])
        R, Z = get("fold:rule"), get("fold:zipcyc")
        kfix = -np.log(R[0]["rho4"]) / 24  # fixed: own L = 24 point of this ring
        # declared (dashed)
        ax.plot(Lg, np.exp(-kdec * Lg), "--", color=BLUE, lw=1.5, alpha=0.9)
        ax.fill_between(Lg, np.exp(-kdec * br[n][1] * Lg), np.exp(-kdec * br[n][0] * Lg), color=ORANGE, alpha=0.12, lw=0)
        ax.plot(Lg, np.exp(-kdec * np.mean(br[n]) * Lg), "--", color=ORANGE, lw=1.5, alpha=0.9)
        # fixed (solid + band)
        ax.fill_between(Lg, np.exp(-1.2 * kfix * Lg), np.exp(-0.8 * kfix * Lg), color=BLUE, alpha=0.15, lw=0)
        ax.plot(Lg, np.exp(-kfix * Lg), "-", color=BLUE, lw=2)
        ax.fill_between(Lg, np.exp(-3.9 * kfix * Lg), np.exp(-3.3 * kfix * Lg), color=ORANGE, alpha=0.18, lw=0)
        ax.plot(Lg, np.exp(-3.6 * kfix * Lg), "-", color=ORANGE, lw=2)
        # measured
        for S, col, lab in ((R, BLUE, "rule"), (Z, ORANGE, "zipper")):
            ax.errorbar([c["L"] for c in S], [c["rho4"] for c in S], yerr=[2 * c["se4"] for c in S], fmt="o",
                        ms=6, color=col, mec="white", mew=1.2, capsize=0, zorder=5)
        ax.axhline(0, color="#dddddd", lw=0.8, zorder=0)
        ax.set_title(f"{dev}, n = {n}", fontsize=10, color=INK, loc="left")
        ax.grid(axis="y", color="#eeeeee", lw=0.8)
        if j == 0:
            ax.text(0.98, 0.95, dlab, transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
        if i == 1: ax.set_xlabel("depth L (steps)", color=INK)
        if j == 0: ax.set_ylabel("charge ratio rho", color=INK)
ax = axes[0, 1]
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
h = [Line2D([], [], marker="o", ls="", color=BLUE, mec="white", ms=7, label="rule, measured (±2 SE)"),
     Line2D([], [], marker="o", ls="", color=ORANGE, mec="white", ms=7, label="zipper, measured (±2 SE)"),
     Line2D([], [], ls="--", color=MUTED, lw=1.5, label="declared prediction (before the run)"),
     Line2D([], [], ls="-", color=MUTED, lw=2, label="fixed after the run: own L=24 point; zipper = rule rate × 3.6"),
     Patch(color=MUTED, alpha=0.2, label="bands: rule ±20 % rate; zipper κ 3.3-3.9 (fixed), structure bracket (declared)")]
fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Noise wall stage 1 (29 Oct 2026): declared predictions against the formula fixed after the run",
             fontsize=11, color=INK, x=0.02, ha="left")
fig.text(0.02, 0.935, "Fixed curves are fitted to the same day's data (post-hoc); they become evidence only if they predict the next run.",
         fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.08, 1, 0.93))
out = "figures/readable_depth_declared_vs_fixed.png"
fig.savefig(out, dpi=150); print(out)
