"""
Regenerates charts/compile_time_scaling.png with the real, artifact-controlled
section 4 data (see README.md section 4 for the full writeup of what changed
and why). This REPLACES the old chart, which depicted the retracted
"203x at 15 qubits -> 4.4x at 1000 qubits" curve -- that curve was built
almost entirely out of measurement artifacts (a Qiskit no-op transpile bug,
a ConsolidateBlocks force_consolidate bug, and an unwarmed per-process
transpile() cold-start cost), not real compute time differences.

Data sources:
  - 15/50/100/156 qubits: phase1.py (basis_gates fix + symmetric warm-up
    patch applied), mean +/- stdev over 10 seeds, real Windows machine, real
    psf_zero_core.
  - 300/500/1000 qubits: phase2.py (basis_gates fix, force_consolidate fix,
    symmetric warm-up patch applied), single run each (no seed loop in that
    script) -- same machine and core. Treat these three as indicative, not
    statistically confirmed the way the 10-seed points are.
  - 156 qubits appears in both scripts (0.0345s +/- 0.0065s in phase1.py's
    10-seed run vs. 0.0264s single-run in phase2.py for Qiskit; 0.0636s
    +/- 0.0087s vs. 0.0507s single-run for PSF-Zero) -- close enough to be
    the same effect measured two ways, plotted here using phase1.py's
    10-seed value since it's the better-supported one.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QISKIT = "#4C6EF5"
PSF = "#F59F00"
GRID = "#E1E4E8"
INK = "#1F2937"
MUTED = "#6B7280"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": MUTED,
})

qubits =  [15,     50,     100,    156,    300,    500,    1000]
blocks =  [7,      25,     50,     78,     150,    250,    500]
qiskit =  [0.0103, 0.0158, 0.0220, 0.0345, 0.0450, 0.0713, 0.1319]
qiskit_e = [0.0018, 0.0024, 0.0037, 0.0065, None,   None,   None]
psf =     [0.0068, 0.0257, 0.0360, 0.0636, 0.0893, 0.1593, 0.2957]
psf_e =   [0.0006, 0.0070, 0.0025, 0.0087, None,   None,   None]

qiskit_per_block = [q / b * 1000 for q, b in zip(qiskit, blocks)]
psf_per_block = [p / b * 1000 for p, b in zip(psf, blocks)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=200)

# --- Panel 1: absolute compile time, log-log ---
ax1.errorbar(qubits[:4], qiskit[:4], yerr=[e for e in qiskit_e[:4]],
             marker="o", color=QISKIT, label="Qiskit (opt L3, warmed)",
             capsize=3, linewidth=1.8)
ax1.plot(qubits[3:], qiskit[3:], marker="o", color=QISKIT, linewidth=1.8,
          linestyle="--")
ax1.errorbar(qubits[:4], psf[:4], yerr=[e for e in psf_e[:4]],
             marker="o", color=PSF, label="PSF-Zero (warmed)",
             capsize=3, linewidth=1.8)
ax1.plot(qubits[3:], psf[3:], marker="o", color=PSF, linewidth=1.8,
          linestyle="--")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Qubits (log scale)")
ax1.set_ylabel("Compile time, seconds (log scale)")
ax1.set_title("Compile time: both warmed up, real Rust core\n"
               "(dashed = single run, not yet multi-seed)",
               fontsize=11, pad=10, loc="left")
ax1.spines[["top", "right"]].set_visible(False)
ax1.spines[["left", "bottom"]].set_color(GRID)
ax1.grid(True, which="both", color=GRID, linewidth=0.7, zorder=0)
ax1.set_axisbelow(True)
ax1.legend(frameon=False, loc="upper left", fontsize=9)

# --- Panel 2: per-block cost ---
ax2.plot(blocks[:4], qiskit_per_block[:4], marker="o", color=QISKIT,
          linewidth=1.8, label="Qiskit ms/block")
ax2.plot(blocks[3:], qiskit_per_block[3:], marker="o", color=QISKIT,
          linewidth=1.8, linestyle="--")
ax2.plot(blocks[:4], psf_per_block[:4], marker="o", color=PSF,
          linewidth=1.8, label="PSF-Zero ms/block")
ax2.plot(blocks[3:], psf_per_block[3:], marker="o", color=PSF,
          linewidth=1.8, linestyle="--")
ax2.set_xscale("log")
ax2.set_xlabel("2-qubit blocks (log scale)")
ax2.set_ylabel("Cost per block, ms")
ax2.set_title("Per-block cost: Qiskit's overhead amortizes,\nPSF-Zero's doesn't",
               fontsize=11, pad=10, loc="left")
ax2.set_ylim(0, max(qiskit_per_block + psf_per_block) * 1.15)
ax2.spines[["top", "right"]].set_visible(False)
ax2.spines[["left", "bottom"]].set_color(GRID)
ax2.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
ax2.set_axisbelow(True)
ax2.legend(frameon=False, loc="upper right", fontsize=9)

fig.suptitle(
    "Section 4, corrected: with the no-op transpile bug, the "
    "force_consolidate bug, and the per-process warm-up artifact all "
    "removed, PSF-Zero is faster only at the smallest scale tested (7 "
    "blocks) and slower beyond that.",
    fontsize=9.5, color=MUTED, y=1.04, wrap=True,
)

fig.tight_layout()
fig.savefig("/home/claude/report/charts/compile_time_scaling.png",
            bbox_inches="tight", facecolor="white")
print("wrote charts/compile_time_scaling.png")
