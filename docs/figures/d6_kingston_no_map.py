"""README headline figure for device test D6 (ibm_kingston, 2026-10-28): kept signal per n for the three arms, from
the D6 workbook values (lam_score.json); D5 part A's map-aware Qiskit point shown separately as a special condition."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, OUT = sys.argv[1], sys.argv[2]
sc = {c["name"]: c["device"] for c in json.load(open(f"{R}/job/lam_score.json"))["cells"]}
ARMS = [("fold_rule", "rule (formula)", "#2a78d6", "o"), ("fold_zipcyc", "zipper, same qubits", "#eb6834", "s"),
        ("qiskit_2_blind_template", "Qiskit level 2, no map", "#1baf7a", "^")]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, ax = plt.subplots(figsize=(9.2, 4.6))
ax.axhline(0, color=INK2, lw=0.8)
ax.grid(axis="y", color=GRID, lw=0.6)
for k, (key, lab, col, mk) in enumerate(ARMS):
    ns = [n for n in range(13, 25) if f"lam_n{n}_L16_m0-{key}" in sc]
    xs = [n + (k - 1) * 0.12 for n in ns]
    ys = [sc[f"lam_n{n}_L16_m0-{key}"]["lam"] for n in ns]
    es = [sc[f"lam_n{n}_L16_m0-{key}"]["se"] for n in ns]
    ax.errorbar(xs, ys, yerr=es, fmt=mk, ms=7, color=col, ecolor=col, elinewidth=1.2, capsize=0,
                mec=SURF, mew=1.2, label=lab, zorder=3)
    if key == "fold_rule":
        ax.plot(xs, ys, color=col, lw=2, zorder=2)
    dy = {"fold_rule": 0, "fold_zipcyc": -9, "qiskit_2_blind_template": 9}[key]
    ax.annotate(lab, (xs[-1], ys[-1]), textcoords="offset points", xytext=(12, dy), va="center", fontsize=9,
                color=INK)
ax.errorbar([20.36], [0.765], yerr=[0.099], fmt="^", ms=8, mfc=SURF, mec="#1baf7a", mew=1.6, ecolor="#1baf7a",
            elinewidth=1.0, capsize=0, zorder=3, label="Qiskit level 2 with the day's map (special condition, D5A)")
ax.annotate("Qiskit with the day's map\n(special condition, separate job: D5A)", (20.36, 0.80), xytext=(18.6, 1.07),
            fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
ax.set_xticks(range(13, 25)); ax.set_xlim(12.5, 26.6); ax.set_ylim(-0.35, 1.2)
ax.set_xlabel("ring size n (qubits), L = 16 layers")
ax.set_ylabel("kept signal λ on the device")
ax.set_title("ibm_kingston, 2026-10-28 — no arm reads the calibration map (D6)", loc="left", fontsize=11.5,
             color=INK, pad=22)
ax.text(0, 1.035, "Usable signal (λ > 2 SE): rule 12/12 · zipper 6/6 · Qiskit without map 4/12.   "
        "Rule vs Qiskit: 11 WIN, 1 TIE, 0 LOSS · vs zipper: 3 WIN, 3 TIE, 0 LOSS.",
        transform=ax.transAxes, fontsize=8.8, color=INK2)
ax.legend(frameon=False, fontsize=8.3, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2)
fig.text(0.01, -0.13, "One random classifier per n, 64 inputs × 250 shots per point; bars ±1 SE of the slope. "
         "Zipper exists only at even n. Records: TOQB results/device-d6-2026-10-28.", fontsize=7.8, color=INK2)
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print("written", OUT)
