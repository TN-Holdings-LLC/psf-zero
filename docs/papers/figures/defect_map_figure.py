"""Figure for Paper 5 (draft 0.3): per-site echo signal along the rule's ring on ibm_kingston, 30 October 2026, in the
three runs of the same circuit (n = 40) and of the control ring (n = 20). Reads TOQB's committed records.

    python defect_map_figure.py TOQB_DIR OUT.png
"""
import gzip
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INPUTS = ("zero", "one", "neel", "antineel")


def input_bits(name, n):
    return {"zero": [0] * n, "one": [1] * n, "neel": [q % 2 for q in range(n)],
            "antineel": [1 - q % 2 for q in range(n)]}[name]


def site_signal(job, runs, name):
    """Raw per-site echo signal: mean over the 4 inputs of <s_j Z_j>, s = -z_in (1 perfect, 0 random)."""
    man = json.load(open(os.path.join(job, "manifest.json")))
    rows = json.load(gzip.open(os.path.join(job, "raw_counts.json.gz")))["rows"]
    rec = json.load(open(os.path.join(runs, name + ".json")))
    cid = [k for k, c in enumerate(man["cells"]) if c["name"] == name][0]
    n = rec["task"]["n"]
    a = np.zeros(n)
    for k, p in enumerate(man["pubs"]):
        if p["cell"] != cid:
            continue
        pt = rec["points"][p["point"]]
        if pt["setting"] != "echo":
            continue
        zin = 1 - 2 * np.array(input_bits(pt["input"], n))
        c = rows[k]["counts"]
        N = float(sum(c.values()))
        m = np.zeros(n)
        for key, v in c.items():
            b = key.replace(" ", "")
            m += v * np.array([1 - 2 * int(b[n - 1 - j]) for j in range(n)])
        a += -zin * m / N / 4
    mq = rec["echo"]["measured_qubits"]
    return a, [int(mq[str(j)]) for j in range(n)]


def main(tq, out):
    R = os.path.join(tq, "results", "device-2026-10-30")
    runs = [("12:22 CET (test 3E / test 2)", None),
            ("14:12 CET (rule-only rerun)", ("rerun-rule-only/job", "rerun-rule-only/runs")),
            ("15:24 CET (options test, off)", ("options-test/job_off", "options-test/runs"))]
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    markers = ["o", "s", "^"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharey=True)
    for ax, n, morning in ((axes[0], 40, "test3E/kingston"), (axes[1], 20, "test2/kingston")):
        name = f"kecho_n{n}_D24_hz0_h0.6-fold_rule"
        phys = None
        for (lab, src), col, mk in zip(runs, colors, markers):
            job, rr = (os.path.join(R, morning, "job"), os.path.join(R, morning, "runs")) if src is None else \
                (os.path.join(R, src[0]), os.path.join(R, src[1]))
            a, phys = site_signal(job, rr, name)
            ax.plot(range(n), a, color=col, lw=2, marker=mk, ms=6, label=lab)
        if n == 40:
            dead = {3, 4, 5, 6, 7, 17, 27, 28, 29, 38, 49, 50, 51, 52, 53, 54, 55, 59}
            idx = [j for j, q in enumerate(phys) if q in dead]
            ax.axvspan(min(idx) - 0.5, max(idx) + 0.5, color="#d9d8d2", alpha=0.5, lw=0)
            ax.text((min(idx) + max(idx)) / 2, 0.62, "18 sites, the same in all three runs", ha="center", fontsize=9,
                    color="#3d3d3a")
        ax.axhline(0, color="#8a8984", lw=1)
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(q) for q in phys], fontsize=7, rotation=90)
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylim(-0.4, 0.75)
        ax.set_ylabel("per-site echo signal")
        ax.set_title(f"n = {n}" + (" (control: no dead sites)" if n == 20 else ""), fontsize=10, loc="left")
        ax.grid(axis="y", color="#e6e5df", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].set_xlabel("physical qubit, in ring order (D = 24, hz = 0, rule placement)")
    axes[0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
