"""Figures of qml_compiler_confounds.tex, computed from TOQB's result records (2026-10-23).

    python docs/papers/qml_confounds_figures.py TOQB_DIR OUT_DIR

TOQB_DIR is a TOQB checkout (https://github.com/TN-Holdings-LLC/TOQB) at or after commit fe488cb with the results of
2026-10-23. Every number drawn is read from those records; nothing is typed in by hand.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARMS = [("zipper", "zipper (formula)", "#2a78d6", "o"),
        ("item31", "item 31 (formula)", "#eb6834", "s"),
        ("qiskit", "Qiskit (search)", "#1baf7a", "^")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def far_rows(toqb):
    """{(teacher, arm, loop): row} from the two far-teacher scores (zipper and item 31; zipper and Qiskit)."""
    rows = {}
    for f in ("far-teacher-2026-10-23/score.json", "far-qiskit-2026-10-23/score.json"):
        for r in json.load(open(os.path.join(toqb, "results", f), encoding="utf-8"))["runs"]:
            rows[(r["teacher"], r["arm"], r["loop"])] = r
    return rows


def fig_far(rows, out):
    teachers = sorted({k[0] for k in rows})
    loops = [0, 500, 2000]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, sharey="row")
    for j, t in enumerate(teachers):
        for i, (key, title) in enumerate((("acc", "stand-in accuracy, no shots"),
                                          ("noiseless_acc", "trained model, noiseless"))):
            ax = axes[i][j]
            style(ax)
            for arm, label, color, mk in ARMS:
                ys = [rows[(t, arm, L)][key] for L in loops]
                ax.plot(range(3), ys, color=color, marker=mk, markersize=6, linewidth=2, label=label,
                        markeredgecolor="#fcfcfb", markeredgewidth=1)
            ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
            ax.set_xticks(range(3), ["0", "500", "2,000"])
            ax.set_ylim(0.45, 0.85)
            if i == 0:
                ax.set_title(f"teacher {t}", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(title, fontsize=8, color=INK)
            if i == 1:
                ax.set_xlabel("training loops", fontsize=8, color=INK)
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"qml_confounds_far.{ext}"), dpi=200)
    plt.close(fig)


def fig_plus(rows, out):
    teachers = sorted({k[0] for k in rows})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    w = 0.26
    for a, (loop, title) in enumerate(((0, r"untrained model ($\theta = 0$)"), (2000, "after 2,000 loops"))):
        ax = axes[a]
        style(ax)
        for k, (arm, label, color, _) in enumerate(ARMS):
            xs = [i + (k - 1) * (w + 0.02) for i in range(len(teachers))]
            ys = [rows[(t, arm, loop)]["plus_share"] for t in teachers]
            ax.bar(xs, ys, width=w, color=color, label=label, edgecolor="#fcfcfb", linewidth=1)
        ax.axhline(0.5, color=INK, linewidth=1, linestyle="--")
        ax.set_xticks(range(len(teachers)), [f"teacher {t}" for t in teachers])
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=9, color=INK)
        if a == 0:
            ax.set_ylabel("share of +1 answers (stand-in)", fontsize=8, color=INK)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"qml_confounds_plus.{ext}"), dpi=200)
    plt.close(fig)


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 1
    toqb, out = argv
    os.makedirs(out, exist_ok=True)
    rows = far_rows(toqb)
    fig_far(rows, out)
    fig_plus(rows, out)
    print("written", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
