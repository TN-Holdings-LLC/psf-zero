import csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C = {"none": "#2a78d6", "each": "#eb6834"}  # categorical slots 1-2 (validated palette)
LBL = {"none": "gc: none (automatic only)", "each": "gc: collect() every iteration"}
D = {m: list(csv.DictReader(open(f"longrun_{m}_2026-09-25.csv"))) for m in ("none", "each")}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False})
note = "RunPod pod (RTX 4090 pod, CPU timings), Qiskit 2.5.2 / qiskit-aer 0.17.2. Timings are for this machine only."

# Figure 1: timing box plots, iterations 6-100
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
for ax, key, title in ((axes[0], "w1_compile_ms", "W1: XOR compile (4 circuits, M4 layout)"),
                       (axes[1], "w2_synth_verify_ms", "W2: PSF-Zero synthesis + GPU check")):
    data = [[float(r[key]) for r in D[m][5:]] for m in ("none", "each")]
    bp = ax.boxplot(data, widths=0.45, patch_artist=True, medianprops=dict(color=INK, linewidth=2),
                    whiskerprops=dict(color=INK2), capprops=dict(color=INK2),
                    flierprops=dict(marker="o", markersize=4, markerfacecolor="none", markeredgecolor=INK2))
    for patch, m in zip(bp["boxes"], ("none", "each")):
        patch.set_facecolor(C[m]); patch.set_alpha(0.85); patch.set_edgecolor(SURF); patch.set_linewidth(2)
    ax.set_xticks([1, 2], ["gc: none", "gc: every iteration"])
    for i, d in enumerate(data, start=1):
        v = np.array(d); ax.text(i + 0.3, np.median(v), f"median {np.median(v):.1f}\nCV {v.std(ddof=1)/v.mean():.3f}",
                                 va="center", fontsize=8.5, color=INK2)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.set_ylabel("milliseconds")
    ax.set_xlim(0.5, 2.9)
fig.suptitle("Figure 1. Per-iteration time, iterations 6-100 (warm-up 1-5 excluded)", x=0.01, ha="left", fontsize=11.5, color=INK)
fig.text(0.01, 0.01, note, fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.95)); fig.savefig("longrun_fig1_timing.png", dpi=160); plt.close(fig)

# Figure 2: W1 <Z0> distributions per input with binomial expectation
fig, axes = plt.subplots(1, 4, figsize=(11, 3.6), sharey=True)
for ax, inp in zip(axes, ["00", "01", "10", "11"]):
    allz = np.array([float(r["w1_z_" + inp]) for m in D for r in D[m]])
    bins = np.linspace(allz.min() - 0.004, allz.max() + 0.004, 16)
    for m in ("none", "each"):
        z = [float(r["w1_z_" + inp]) for r in D[m]]
        ax.hist(z, bins=bins, histtype="step", linewidth=2, color=C[m], label=LBL[m])
    mu = allz.mean(); sd = math.sqrt((1 - mu * mu) / 4000); x = np.linspace(bins[0], bins[-1], 200)
    ax.plot(x, 100 * (bins[1] - bins[0]) * np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)),
            color=INK2, linestyle="--", linewidth=1.5, label="binomial expectation (4,000 shots)")
    ax.set_title(f"input {inp}", fontsize=10.5, color=INK, loc="left")
    ax.set_xlabel("noisy <Z0>")
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
axes[0].set_ylabel("iterations (of 100)")
axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.42), ncol=3, frameon=False, fontsize=8.5)
fig.suptitle("Figure 2. W1 noisy <Z0> over 100 iterations per process (new simulator seed each iteration)", x=0.01, ha="left", fontsize=11.5, color=INK)
fig.tight_layout(rect=(0, 0.06, 1, 0.93)); fig.savefig("longrun_fig2_z0.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# Figure 3: RSS vs iteration
fig, ax = plt.subplots(figsize=(9, 4))
for m in ("none", "each"):
    it = [int(r["iteration"]) for r in D[m]]; rss = [float(r["rss_mb"]) for r in D[m]]
    ax.plot(it, rss, color=C[m], linewidth=2, label=LBL[m])
    ax.text(101.5, rss[-1] + (0.8 if m == "none" else -0.8), f"{rss[-1]:.1f} MB", color=INK2, fontsize=8.5, va="center")
ax.axvline(10, color=INK2, linestyle=":", linewidth=1)
ax.text(10.8, ax.get_ylim()[0] + 1, "iteration 10 (L6 reference)", fontsize=8.5, color=INK2)
ax.set_xlabel("iteration"); ax.set_ylabel("resident memory, MB (VmRSS)"); ax.set_xlim(0, 110)
ax.legend(frameon=False, loc="lower right", fontsize=9)
ax.set_title("Figure 3. Process memory per iteration (both workloads)", fontsize=11.5, color=INK, loc="left")
fig.text(0.01, 0.01, "Growth iteration 10 -> 100: +0.4% (none), +0.2% (every iteration). RunPod pod.", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig("longrun_fig3_rss.png", dpi=160); plt.close(fig)
print("ok")