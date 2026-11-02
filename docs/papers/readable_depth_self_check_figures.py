"""Figures and derived numbers of Paper 6 (draft): scan, register, avoid and repair on three IBM heavy-hex devices.

Every number is read from TOQB's device records (results/); nothing is typed in. Per-site echo signals are computed
from the raw counts with TOQB's own reader (toqb.defect_list.job_profiles), the same code that made the registry.
Usage, from docs/papers:  python readable_depth_self_check_figures.py PATH_TO_TOQB
Writes figures/*.png and readable_depth_self_check_numbers.json (the derived numbers quoted in the text).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TQ = sys.argv[1] if len(sys.argv) > 1 else "../../../TOQB"
R = os.path.join(TQ, "results")
sys.path.insert(0, TQ)
from toqb import defect_list as DL  # noqa: E402

BLUE, ORANGE, GREEN, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#2e9e6a", "#222222", "#6b6b6b", "#eeeeee"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#bbbbbb",
                     "axes.spines.top": False, "axes.spines.right": False})
os.makedirs("figures", exist_ok=True)
NUM = {}


def rp(*p):
    return os.path.join(R, *p)


def profile(job, runs, n):
    prof, job_id, backend, when = DL.job_profiles(rp(job), rp(runs))
    return prof[n]


# ---------------------------------------------------------------- Figure 1: kingston n = 40, per site, three days
days = [("30 Oct (test 3E)", "device-2026-10-30/test3E/kingston/job", "device-2026-10-30/test3E/kingston/runs"),
        ("31 Oct (coverage day 1, A)", "coverage-days-2026-10-31-day1/job_A", "coverage-days-2026-10-31-day1/runs_A"),
        ("1 Nov (coverage day 2, A)", "coverage-days-2026-11-01-day2/job_A", "coverage-days-2026-11-01-day2/runs_A")]
avoid = [("31 Oct (coverage day 1, B)", "coverage-days-2026-10-31-day1/job_B", "coverage-days-2026-10-31-day1/runs_B"),
         ("1 Nov (coverage day 2, B)", "coverage-days-2026-11-01-day2/job_B", "coverage-days-2026-11-01-day2/runs_B")]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.2), sharey=True)
ring0 = None
cols = [MUTED, BLUE, ORANGE]
NUM["fig1"] = {}
for (lab, job, runs), c in zip(days, cols):
    ring, a = profile(job, runs, 40)
    ring0 = ring0 or ring
    assert ring == ring0, "the standard n = 40 ring changed between days"
    dead = DL.dead_runs(a)
    NUM["fig1"][lab] = dict(ring=list(ring), usable=len(DL.usable(a)), dead=[[ring[(s + t) % 40] for t in range(L)]
                                                                            for s, L in dead])
    ax1.plot(range(40), a, "-o", ms=3.5, lw=1.2, color=c, label=lab)
ax1.axhline(0.12, color="#c0392b", lw=0.9, ls="--")
ax1.text(16.5, 0.15, "dead line 0.12", ha="center", fontsize=8, color="#c0392b")
ax1.axhline(0, color="#dddddd", lw=0.8)
ax1.set_xticks(range(40))
ax1.set_xticklabels([str(q) for q in ring0], rotation=90, fontsize=6.5)
ax1.set_title("ibm_kingston, standard n = 40 ring (byte-identical circuit), echo signal per site at D = 24",
              loc="left", fontsize=10, color=INK)
ax1.set_ylabel("site signal a_j")
ax1.legend(frameon=False, fontsize=8, loc="lower right")
ax1.grid(axis="y", color=GRID)
ringb = None
for (lab, job, runs), c in zip(avoid, [BLUE, ORANGE]):
    ring, a = profile(job, runs, 40)
    ringb = ringb or ring
    assert ring == ringb
    NUM["fig1"][lab] = dict(ring=list(ring), usable=len(DL.usable(a)))
    ax2.plot(range(40), a, "-o", ms=3.5, lw=1.2, color=c, label=lab)
ax2.axhline(0.12, color="#c0392b", lw=0.9, ls="--")
ax2.axhline(0, color="#dddddd", lw=0.8)
ax2.set_xticks(range(40))
ax2.set_xticklabels([str(q) for q in ringb], rotation=90, fontsize=6.5)
ax2.set_title("the ring placed around the registry (masked couplers (9, 10), (29, 38), (85, 86))", loc="left",
              fontsize=10, color=INK)
ax2.set_xlabel("physical qubit, in ring order")
ax2.set_ylabel("site signal a_j")
ax2.legend(frameon=False, fontsize=8, loc="lower right")
ax2.grid(axis="y", color=GRID)
fig.tight_layout()
fig.savefig("figures/self_check_kingston_n40_sites.png", dpi=150)

# ---------------------------------------------------------------- Figure 2: usable sites, standard against avoiding
cov = [("kingston\n31 Oct", "coverage-days-2026-10-31-day1/coverage.json"),
       ("kingston\n1 Nov", "coverage-days-2026-11-01-day2/coverage.json"),
       ("fez\n1 Nov", "two-devices-2026-11-01/fez/avoid/coverage.json"),
       ("marrakesh\n1 Nov", "two-devices-2026-11-01/marrakesh/avoid/coverage.json")]
NUM["width"] = {}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=False)
for ax, n in zip(axes, (40, 72)):
    A, B, labs = [], [], []
    for lab, f in cov:
        s = json.load(open(rp(f)))["sizes"][str(n)]
        A.append(s["A"]["usable"])
        B.append(s["B"]["usable"])
        labs.append(lab)
        NUM["width"][f"{lab.replace(chr(10), ' ')} n{n}"] = dict(
            standard=s["A"]["usable"], avoiding=s["B"]["usable"], verdict=s.get("verdict"),
            mean_usable=(s["A"]["mean_usable"], s["B"]["mean_usable"]), mean_all=(s["A"]["mean_all"], s["B"]["mean_all"]))
    x = np.arange(len(labs))
    ax.bar(x - 0.19, A, 0.36, color=MUTED, label="standard ring")
    ax.bar(x + 0.19, B, 0.36, color=BLUE, label="ring placed around the registry")
    for xi, a_, b_ in zip(x, A, B):
        ax.text(xi - 0.19, a_ + 0.5, str(a_), ha="center", fontsize=8, color=INK)
        ax.text(xi + 0.19, b_ + 0.5, str(b_), ha="center", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylim(0, n * 1.12)
    ax.set_title(f"usable sites out of n = {n}", loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID)
axes[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig("figures/self_check_width.png", dpi=150)

# ---------------------------------------------------------------- Figure 3: M3 second form, per site
S = json.load(open(rp("qca-m3b-2026-11-01/score_S.json")))
V = json.load(open(rp("qca-m3b-2026-11-01/score_V.json")))
fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharey=True)
NUM["m3b"] = {}
for ax, D in zip(axes, (12, 24)):
    s = S[f"kqsh_n72_D{D}-fold_rule"]
    v = V[f"kqsh_n72_D{D}-fold_rule"]
    only_s = [q for q in s["phys"] if q not in v["phys"]]
    only_v = [q for q in v["phys"] if q not in s["phys"]]
    common = [q for q in s["phys"] if q in v["phys"]]
    sa = dict(zip(s["phys"], s["a"]))
    va = dict(zip(v["phys"], v["a"]))
    NUM["m3b"][f"D{D}"] = dict(mean_standard=s["mean"], mean_avoid=v["mean"], per_bit=(s["per_bit"], v["per_bit"]),
                               usable=(s["usable"], v["usable"]), only_standard=only_s, only_avoid=only_v,
                               common_mean=(float(np.mean([sa[q] for q in common])),
                                            float(np.mean([va[q] for q in common]))),
                               swapped_mean=(float(np.mean([sa[q] for q in only_s])),
                                             float(np.mean([va[q] for q in only_v]))), n_common=len(common))
    ax.plot(range(72), s["a"], "-o", ms=3, lw=1, color=MUTED, label="standard ring")
    ax.plot(range(72), v["a"], "-o", ms=3, lw=1, color=BLUE, label="ring placed around the registry")
    for j, q in enumerate(s["phys"]):
        if q in only_s:
            ax.plot(j, s["a"][j], "o", ms=6, mfc="none", mec=ORANGE, mew=1.4)
    ax.axhline(0.12, color="#c0392b", lw=0.9, ls="--")
    ax.set_title(f"ibm_marrakesh, forward identity computation (period 6), n = 72, D = {D}: site signal",
                 loc="left", fontsize=10, color=INK)
    ax.set_ylabel("site signal a_j")
    ax.grid(axis="y", color=GRID)
axes[1].set_xlabel("ring position (circles: the 10 qubits only on the standard ring)")
axes[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig("figures/self_check_m3b_sites.png", dpi=150)

# ---------------------------------------------------------------- Figure 4: repair (frame flip), block mean
fixes = [("31 Oct", "black-fix-2026-10-31/ff_score.json"), ("1 Nov", "fix-repeat-2026-11-01/ff_score.json")]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
NUM["repair"] = {}
for ax, (lab, f) in zip(axes, fixes):
    d = json.load(open(rp(f)))
    arms = [("off", "off", MUTED), ("global flip", "ff", ORANGE), ("flip at 27, 29", "lfB3", GREEN)]
    x = np.arange(2)
    for k, (name, key, c) in enumerate(arms):
        vals = []
        for D in (12, 24):
            cell = d["cells"][f"n40_D{D}_{key}"]
            a = np.array(cell["a"])
            blk = [cell["phys"].index(q) for q in d["block"]]
            vals.append(float(a[blk].mean()))
            NUM["repair"][f"{lab} D{D} {name}"] = dict(block=float(a[blk].mean()), usable=len(DL.usable(a)),
                                                       ring_mean=float(a.mean()))
        ax.bar(x + (k - 1) * 0.27, vals, 0.25, color=c, label=name)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["D = 12", "D = 24"])
    ax.set_title(f"kingston n = 40, dead block (38, 49-55, 59), {lab}", loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color=GRID)
axes[0].set_ylabel("mean site signal of the block")
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig("figures/self_check_repair.png", dpi=150)

json.dump(NUM, open("readable_depth_self_check_numbers.json", "w"), indent=1)
print("figures/self_check_kingston_n40_sites.png figures/self_check_width.png figures/self_check_m3b_sites.png "
      "figures/self_check_repair.png readable_depth_self_check_numbers.json")
