"""Exploratory, after the fact (Addendum 488): every compiled pair run on a real IBM device so far, pooled and split by
circuit family. Not pre-registered. Reads only files already in the repository; no device, no simulation of noise.

Sources (four runs):
  HW1    ibm_kingston, 2026-10-12: PSF / QK3 (data/2026-10-12/hw1/)
  HW2    ibm_marrakesh, 2026-10-13: part R PSF / QK3 and part S S1 / S0 (data/2026-10-13/hw2/)
  TIME1  ibm_fez, 2026-10-14: T1 S1 / S0, T2 C49 / C48, T3 C49 / QK3 (data/2026-10-14/time1_results/)
  FRAC1  ibm_fez, 2026-10-14: C51 (rzz) / C49 (data/2026-10-14/frac1_results/)

A pair's ratio is always the PSF-Zero construction under test over its alternative, as each run defined it. Pairs a run
did not score (identical circuits) are absent; the GHZ control is left out. HW1's score file has no durations, so its
duration ratios are recomputed here as soon as possible on FakeKingston's Target (an approximation: the day's real
Target was not stored).

Families: "formula" = the transverse-field Ising chains (tests named tfim*), which PSF-Zero builds by formula;
"other" = QAOA and Barabasi-Albert circuits, where placement is not given by a formula. Pairs whose new side uses the
device's rzz (FRAC1) are also reported apart.

    python benchmarks/hw_pairs_by_family.py <out dir>   (refuses an existing out dir)"""
import json
import math
import os
import re
import sys

from qiskit import qpy
from qiskit_ibm_runtime.fake_provider import FakeKingston
from scipy.stats import spearmanr

OUT = sys.argv[1]
if os.path.exists(OUT):
    raise SystemExit(f"STOP: {OUT} exists (a run never overwrites another)")
D = "data"


def asap_us(circ, target):
    """Duration in microseconds, every gate as soon as possible, measurements and barriers left out."""
    finish = {}
    for ins in circ.data:
        name = ins.operation.name
        if name in ("barrier", "measure"):
            continue
        qs = tuple(circ.find_bit(q).index for q in ins.qubits)
        try:
            props = target[name][qs]
            d = (props.duration or 0.0) if props is not None else 0.0
        except KeyError:
            d = 0.0
        start = max(finish.get(q, 0.0) for q in qs)
        for q in qs:
            finish[q] = start + d
    return max(finish.values()) * 1e6


def hw1_pairs():
    base = os.path.join(D, "2026-10-12", "hw1")
    man = json.load(open(os.path.join(base, "hw1_manifest.json")))
    circs = qpy.load(open(os.path.join(base, "hw1_circuits.qpy"), "rb"))
    first = {}
    for i, pub in enumerate(man["pubs"]):
        first.setdefault(pub["entry"], circs[i])
    t = FakeKingston().target
    dur = {(e["test"], e["arm"]): asap_us(first[k], t) for k, e in enumerate(man["entries"])
           if e.get("part") == "HW1" and k in first}
    out = []
    for p in json.load(open(os.path.join(base, "hw1_score.json")))["pairs"]:
        out.append(dict(run="HW1 ibm_kingston", test=p["test"], pair="PSF/QK3",
                        dur_ratio=dur[(p["test"], "PSF")] / dur[(p["test"], "QK3")], f_ratio=p["f_ratio"],
                        f_alt=p["f_qk3"], rzz=False))
    return out


def hw2_pairs():
    s = json.load(open(os.path.join(D, "2026-10-13", "hw2", "hw2_score.json")))
    out = []
    for part, label in (("part_r", "PSF/QK3"), ("part_s", "S1/S0")):
        for p in s[part].values():
            if not p.get("same"):
                out.append(dict(run="HW2 ibm_marrakesh", test=p["test"], pair=label, dur_ratio=p["dur"],
                                f_ratio=p["f"], f_alt=p["f0"], rzz=False))
    return out


def md_rows(path):
    for line in open(path, encoding="utf-8"):
        yield [c.strip() for c in line.split("|")]


def time1_pairs():
    out = []
    for c in md_rows(os.path.join(D, "2026-10-14", "time1_results", "time1_run", "time1_score.md")):
        if len(c) > 9 and c[1] in ("T1", "T2", "T3") and "/" in c[3]:
            out.append(dict(run="TIME1 ibm_fez", test=c[2], pair=c[3], dur_ratio=float(c[4]), f_ratio=float(c[7]),
                            f_alt=float(c[9]), rzz=False))
    return out


def frac1_pairs():
    out = []
    for c in md_rows(os.path.join(D, "2026-10-14", "frac1_results", "frac1_run", "frac1_score.md")):
        if len(c) == 8 and c[1] in ("ISING", "R") and re.match(r"^[0-9.]+$", c[3]):
            out.append(dict(run="FRAC1 ibm_fez", test=c[2], pair="C51/C49", dur_ratio=float(c[3]),
                            f_ratio=float(c[4]), f_alt=float(c[6]), rzz=True))
    return out


def family(test):
    return "formula" if test.startswith("tfim") else "other"


def summary(rows, label):
    decided = [r for r in rows if r["dur_ratio"] <= 1 / 1.1 or r["dur_ratio"] >= 1.1]
    shorter_won = [r for r in decided if (r["dur_ratio"] < 1) == (r["f_ratio"] > 1)]
    rho = spearmanr([-math.log(r["dur_ratio"]) for r in rows], [math.log(r["f_ratio"]) for r in rows])
    geo = math.exp(sum(math.log(r["f_ratio"]) for r in rows) / len(rows))
    return dict(label=label, pairs=len(rows), new_side_won=sum(r["f_ratio"] > 1 for r in rows),
                f_ratio_geomean=geo, f_ratio_min=min(r["f_ratio"] for r in rows),
                f_ratio_max=max(r["f_ratio"] for r in rows), decided_by_duration=len(decided),
                shorter_won=len(shorter_won), spearman_inv_duration_f=rho.correlation, spearman_p=rho.pvalue)


if __name__ == "__main__":
    rows = [r for r in hw1_pairs() + hw2_pairs() + time1_pairs() + frac1_pairs() if not r["test"].startswith("ghz")]
    for r in rows:
        r["family"] = family(r["test"])
    groups = [
        ("formula, all", [r for r in rows if r["family"] == "formula"]),
        ("formula, compiler only (no rzz)", [r for r in rows if r["family"] == "formula" and not r["rzz"]]),
        ("formula, rzz (FRAC1)", [r for r in rows if r["family"] == "formula" and r["rzz"]]),
        ("other, all", [r for r in rows if r["family"] == "other"]),
        ("other, compiler only (no rzz)", [r for r in rows if r["family"] == "other" and not r["rzz"]]),
        ("other, rzz (FRAC1)", [r for r in rows if r["family"] == "other" and r["rzz"]]),
        ("formula, against Qiskit level 3 only", [r for r in rows if r["family"] == "formula" and "QK3" in r["pair"]]),
        ("other, against Qiskit level 3 only", [r for r in rows if r["family"] == "other" and "QK3" in r["pair"]]),
    ]
    res = [summary(g, label) for label, g in groups]
    os.makedirs(OUT)
    json.dump(dict(pairs=rows, summary=res), open(os.path.join(OUT, "hw_pairs_by_family.json"), "w"), indent=1)
    lines = ["# Real-device pairs by family (exploratory, after the fact; Addendum 488)", "",
             "| run | test | pair | duration ratio | F ratio | F of the alternative | family | rzz |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['run']} | {r['test']} | {r['pair']} | {r['dur_ratio']:.3f} | {r['f_ratio']:.3f} | "
                     f"{r['f_alt']:.3f} | {r['family']} | {'yes' if r['rzz'] else ''} |")
    lines += ["", "| group | pairs | PSF-Zero side won | F ratio geometric mean (range) | shorter won, of pairs >= 10% apart | "
              "Spearman 1/duration vs F (p) |", "|---|---|---|---|---|---|"]
    for s in res:
        lines.append(f"| {s['label']} | {s['pairs']} | {s['new_side_won']} | {s['f_ratio_geomean']:.3f} "
                     f"({s['f_ratio_min']:.3f}-{s['f_ratio_max']:.3f}) | {s['shorter_won']} of {s['decided_by_duration']} | "
                     f"{s['spearman_inv_duration_f']:+.2f} ({s['spearman_p']:.3f}) |")
    open(os.path.join(OUT, "hw_pairs_by_family.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[-(len(res) + 2):]))
