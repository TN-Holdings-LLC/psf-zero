"""score_v10_eval_r1.py -- revision 1 (before any run, 2026-09-30) of score_v10_eval.py: the evaluation is cut to the
five HELD-OUT tasks only, so that it fits in about one hour (the five tuned tasks were scored this afternoon under
v9: 14/15). H2 is rescaled to 15 task-runs per arm; G3 on fill27 is not run. Reads only files:
OUT/<arm>/run<r>/<task>/result.json (arm = v9 or v10) and OUT/retime_fill27g9.csv.

    python score_v10_eval_r1.py OUT            -> OUT/score_v10.md, OUT/score_v10.json

Criteria (as in vllm-v10-eval-preregistration-2026-09-30.md):
  H1  held-out generalisation, v10 arm, 5 held-out tasks x 3 runs:
      go >= 12/15 solved; stop <= 7/15; otherwise ambiguous
  H2  v10 against v9, the 5 held-out tasks x 3 runs (15 task-runs each):
      better if v10 solved >= v9 solved + 2; worse if v10 solved <= v9 solved - 2; otherwise no clear difference
  H3  G2 on the held-out tasks, v10 arm: device two-qubit count (PSF) of the best circuit against the StatePreparation
      baseline (PSF) on its solved held-out task-runs: go if <= on >= 80%; stop if worse on > 50% or none solved
  H4  G3 on fill27g9 (held-out), correct circuits of both arms, re-timing pass: same thresholds as G3 of the
      go/no-go tests (go: median PSF <= 1 s, median L3/PSF >= 10, PSF 2q <= L3 on >= 80%; stop: median ratio < 3, or
      PSF > L3 on > 50%, or median PSF > 1 s, or no correct circuit)
  Decisions:
    default harness: ADOPT v10 if H2 = better and H1 != stop; otherwise KEEP v9
    line: CONTINUE unless BOTH arms solve <= 7/15 held-out task-runs (then CUT: the approach does not generalise)
"""
import csv
import json
import os
import statistics
import sys

TUNED = []  # revision 1: held-out tasks only
HELD = ["w4", "dicke42", "ghz3i", "singlet3", "fill27g9"]
RUNS = [1, 2, 3]
FID = 0.9999


def load(out, arm):
    cells = {}
    for t in TUNED + HELD:
        for r in RUNS:
            f = os.path.join(out, arm, f"run{r}", t, "result.json")
            cells[(t, r)] = json.load(open(f)) if os.path.exists(f) else None
    return cells


def solved(cells, tasks):
    return sum(1 for (t, r), v in cells.items() if t in tasks and v and v["solved"])


def g3(out, task):
    f = os.path.join(out, f"retime_{task}.csv")
    if not os.path.exists(f):
        return "stop", {"reason": f"no retime_{task}.csv"}
    rows = [r for r in csv.DictReader(open(f)) if float(r["fidelity_compiled"]) >= FID]
    if not rows:
        return "stop", {"reason": f"no correct {task} circuit"}
    psf = [float(r["psf_s_median"]) for r in rows]
    ratio = [float(r["q3_s_median"]) / float(r["psf_s_median"]) for r in rows]
    le = sum(1 for r in rows if int(r["psf_2q"]) <= int(r["q3_2q"]))
    gt = len(rows) - le
    m_psf, m_ratio = statistics.median(psf), statistics.median(ratio)
    facts = dict(circuits=len(rows), median_psf_s=m_psf, median_ratio=m_ratio, min_ratio=min(ratio), max_ratio=max(ratio),
                 psf_le_l3=le, psf_gt_l3=gt, psf_2q=[int(r["psf_2q"]) for r in rows], l3_2q=[int(r["q3_2q"]) for r in rows])
    if m_psf <= 1.0 and m_ratio >= 10 and le >= 0.8 * len(rows):
        return "go", facts
    if m_ratio < 3 or gt > 0.5 * len(rows) or m_psf > 1.0:
        return "stop", facts
    return "ambiguous", facts


def main():
    out = sys.argv[1]
    arms = {a: load(out, a) for a in ("v9", "v10")}
    res, lines = {"arms": {}}, ["## Solved task-runs (a missing result counts as not solved)", "",
                                "| arm | tuned (not run) | held-out /15 | total /15 | " + " | ".join(TUNED + HELD) + " | missing |",
                                "|---|---|---|---|" + "---|" * (len(TUNED) + len(HELD)) + "---|"]
    for a, cells in arms.items():
        per = {t: sum(1 for r in RUNS if cells[(t, r)] and cells[(t, r)]["solved"]) for t in TUNED + HELD}
        miss = sum(1 for v in cells.values() if not v)
        st, sh = solved(cells, TUNED), solved(cells, HELD)
        res["arms"][a] = dict(tuned=st, held_out=sh, total=st + sh, per_task=per, missing=miss,
                              llm_s=round(sum(v["llm_s_total"] for v in cells.values() if v), 1))
        lines.append(f"| {a} | {st} | {sh} | {st + sh} | " + " | ".join(f"{per[t]}/3" for t in TUNED + HELD) + f" | {miss} |")
    h_v10 = res["arms"]["v10"]["held_out"]
    H1 = "go" if h_v10 >= 12 else ("stop" if h_v10 <= 7 else "ambiguous")
    d = res["arms"]["v10"]["total"] - res["arms"]["v9"]["total"]
    H2 = "better" if d >= 2 else ("worse" if d <= -2 else "no clear difference")
    s_held = [v for (t, r), v in arms["v10"].items() if t in HELD and v and v["solved"]]
    if not s_held:
        H3, le3, gt3 = "stop", 0, 0
    else:
        le3 = sum(1 for v in s_held if v["routed_2q"] <= v["baseline_psf_2q"])
        gt3 = len(s_held) - le3
        H3 = "go" if le3 >= 0.8 * len(s_held) else ("stop" if gt3 > 0.5 * len(s_held) else "ambiguous")
    H4, f4 = g3(out, "fill27g9")
    adopt = "ADOPT v10" if (H2 == "better" and H1 != "stop") else "KEEP v9"
    line = "CUT" if (res["arms"]["v10"]["held_out"] <= 7 and res["arms"]["v9"]["held_out"] <= 7) else "CONTINUE"
    res.update(H1=dict(verdict=H1, v10_held_out=h_v10), H2=dict(verdict=H2, v10_minus_v9=d),
               H3=dict(verdict=H3, solved=len(s_held), le=le3, gt=gt3), H4=dict(verdict=H4, **f4),
               default_harness=adopt, line=line)
    lines += ["", f"- H1 (held-out, v10): **{H1}** ({h_v10}/15)",
              f"- H2 (v10 - v9, 15 held-out task-runs each): **{H2}** ({d:+d})",
              f"- H3 (G2 on held-out, v10): **{H3}** (solved {len(s_held)}; <= baseline {le3}, worse {gt3})",
              f"- H4 (G3 on fill27g9): **{H4}** ({json.dumps(f4)})",
              "",
              f"## Default harness: **{adopt}**", f"## Line: **{line}**", ""]
    open(os.path.join(out, "score_v10.md"), "w").write("\n".join(lines) + "\n")
    json.dump(res, open(os.path.join(out, "score_v10.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
