"""score_vllm_invest.py -- pre-registered scorer (workplace design, 2026-09-30) for the vLLM x PSF-Zero
go/no-go test. Reads only files: OUT/<model tag>/run<r>/<task>/result.json and OUT/retime_fill27.csv.

    python score_vllm_invest.py OUT            -> OUT/score.md, OUT/score.json

Criteria (as in vllm-invest-preregistration-2026-09-30.md):
  G1  best model (most solved task-runs out of 5 tasks x 3 runs):
      go   >= 12/15 solved, and w3 >= 2/3 and qft3 >= 2/3
      stop <= 7/15 solved
  G2  best model, over its solved task-runs: device two-qubit count (PSF) of the model's best circuit
      versus the StatePreparation baseline compiled by PSF the same way
      go   model <= baseline on >= 80% of solved task-runs
      stop model >  baseline on >  50% of solved task-runs (or no solved task-run)
  G3  retime pass (sequential, median of the repetitions), over the distinct fill27 circuits from all
      models that are correct after compiling (fidelity >= 0.9999)
      go   median PSF time <= 1 s, and median per-circuit ratio L3/PSF >= 10, and PSF two-qubit count
           <= L3 on >= 80% of circuits
      stop median ratio < 3, or PSF > L3 two-qubit count on > 50% of circuits, or median PSF time > 1 s,
           or no correct fill27 circuit at all
  Decision: INVEST only if G1 = go, G2 != stop and G3 = go. Anything else: CUT (including ambiguous).
"""
import csv
import json
import os
import statistics
import sys

TASKS = ["ghz5", "w3", "bell3", "qft3", "fill27"]
RUNS = [1, 2, 3]
FID = 0.9999


def load(out):
    models = {}
    for tag in sorted(os.listdir(out)):
        p = os.path.join(out, tag)
        if not os.path.isdir(p) or not any(d.startswith("run") for d in os.listdir(p)):
            continue
        cells = {}
        for r in RUNS:
            for t in TASKS:
                f = os.path.join(p, f"run{r}", t, "result.json")
                cells[(t, r)] = json.load(open(f)) if os.path.exists(f) else None  # missing = failed run
        models[tag] = cells
    return models


def g1(cells):
    solved = sum(1 for v in cells.values() if v and v["solved"])
    per = {t: sum(1 for r in RUNS if cells[(t, r)] and cells[(t, r)]["solved"]) for t in TASKS}
    if solved >= 12 and per["w3"] >= 2 and per["qft3"] >= 2:
        v = "go"
    elif solved <= 7:
        v = "stop"
    else:
        v = "ambiguous"
    return v, solved, per


def g2(cells):
    s = [v for v in cells.values() if v and v["solved"]]
    if not s:
        return "stop", 0, 0, 0
    le = sum(1 for v in s if v["routed_2q"] <= v["baseline_psf_2q"])
    gt = len(s) - le
    v = "go" if le >= 0.8 * len(s) else ("stop" if gt > 0.5 * len(s) else "ambiguous")
    return v, len(s), le, gt


def g3(out):
    f = os.path.join(out, "retime_fill27.csv")
    if not os.path.exists(f):
        return "stop", {"reason": "no retime_fill27.csv"}
    rows = [r for r in csv.DictReader(open(f)) if float(r["fidelity_compiled"]) >= FID
            and not r["where"].startswith("mock")]
    if not rows:
        return "stop", {"reason": "no correct fill27 circuit from any model"}
    psf = [float(r["psf_s_median"]) for r in rows]
    ratio = [float(r["q3_s_median"]) / float(r["psf_s_median"]) for r in rows]
    le = sum(1 for r in rows if int(r["psf_2q"]) <= int(r["q3_2q"]))
    gt = len(rows) - le
    m_psf, m_ratio = statistics.median(psf), statistics.median(ratio)
    facts = dict(circuits=len(rows), median_psf_s=m_psf, median_ratio=m_ratio, min_ratio=min(ratio),
                 max_ratio=max(ratio), psf_le_l3=le, psf_gt_l3=gt,
                 psf_2q=[int(r["psf_2q"]) for r in rows], l3_2q=[int(r["q3_2q"]) for r in rows])
    if m_psf <= 1.0 and m_ratio >= 10 and le >= 0.8 * len(rows):
        v = "go"
    elif m_ratio < 3 or gt > 0.5 * len(rows) or m_psf > 1.0:
        v = "stop"
    else:
        v = "ambiguous"
    return v, facts


def main():
    out = sys.argv[1]
    models = load(out)
    lines, res = [], {"models": {}}
    lines.append("## Per model (solved task-runs; a missing result counts as not solved)\n")
    lines.append("| model | solved /15 | " + " | ".join(TASKS) + " | errors | HTTP 400 | missing | model s | PSF compile s |")
    lines.append("|---|---|" + "---|" * len(TASKS) + "---|---|---|---|---|")
    scored = {}
    for tag, cells in models.items():
        v1, solved, per = g1(cells)
        present = [v for v in cells.values() if v]
        err = sum(v["errors"] for v in present)
        h400 = sum(v["http400"] for v in present)
        miss = sum(1 for v in cells.values() if not v)
        llm = sum(v["llm_s_total"] for v in present)
        comp = sum(v.get("compile_s", 0) for v in present)
        scored[tag] = (solved, v1, per)
        res["models"][tag] = dict(solved=solved, per_task=per, g1=v1, errors=err, http400=h400, missing=miss,
                                  llm_s=llm, best_compile_s_sum=comp)
        lines.append(f"| {tag} | {solved} | " + " | ".join(f"{per[t]}/3" for t in TASKS) +
                     f" | {err} | {h400} | {miss} | {llm:.0f} | {comp:.2f} |")
    if not scored:
        print("no model results found")
        return
    best = max(scored, key=lambda k: (scored[k][0], scored[k][2]["w3"] + scored[k][2]["qft3"]))
    v1, solved, per = scored[best][1], scored[best][0], scored[best][2]
    v2, n_s, le2, gt2 = g2(models[best])
    v3, facts3 = g3(out)
    decision = "INVEST" if (v1 == "go" and v2 != "stop" and v3 == "go") else "CUT"
    res.update(best_model=best, G1=dict(verdict=v1, solved=solved, per_task=per),
               G2=dict(verdict=v2, solved_task_runs=n_s, model_le_baseline=le2, model_gt_baseline=gt2),
               G3=dict(verdict=v3, **facts3), decision=decision)
    lines += ["", f"## Best model: {best}", "",
              f"- G1: **{v1}** ({solved}/15 solved; w3 {per['w3']}/3, qft3 {per['qft3']}/3)",
              f"- G2: **{v2}** (solved task-runs {n_s}; model <= StatePreparation baseline on {le2}, worse on {gt2})",
              f"- G3: **{v3}** ({json.dumps(facts3)})", "", f"## Decision: **{decision}**", ""]
    lines += ["## G2 detail (best model)", "", "| task | run | model 2q (PSF) | baseline 2q (PSF) | baseline 2q (L3) |",
              "|---|---|---|---|---|"]
    for (t, r), v in sorted(models[best].items()):
        if v and v["solved"]:
            lines.append(f"| {t} | {r} | {v['routed_2q']} | {v['baseline_psf_2q']} | {v['baseline_q3_2q']} |")
    open(os.path.join(out, "score.md"), "w").write("\n".join(lines) + "\n")
    json.dump(res, open(os.path.join(out, "score.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
