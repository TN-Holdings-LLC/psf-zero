"""steiner6_score.py (run from the psf-zero repository root) -- STEINER6 run against Qiskit level 3 (seed 0, same process) and STEINER5's PE arm (file), by category.
    python benchmarks/steiner6_score.py <s6.jsonl> [<s6 other.jsonl> ...]
Two numbers:
  display:   two-qubit count ratio (geometric mean over all eligible tests) and ESP ratio on runnable tests where both
             outputs are clean
  essential: runnable tests (best ESP >= 0.01) with an output on a failed element counted as ESP 0: how many each
             arm loses to 0, and the ESP ratio where both > 0; wins/losses by 10%."""
import json, math, sys
surv = {r["test"]: r for r in json.load(open("data/2026-10-11/steiner6/survey.json"))}
S5 = {}
for ln in open("data/2026-10-10/steiner5/steiner5.jsonl"):
    r = json.loads(ln)
    if r.get("arm") == "PE" and "q2" in r:
        S5[(r["device"], r["test"])] = r
def cat(t):
    s = surv.get(t.split("[", 1)[1][:-1])
    if s is None: return "?"
    if s["diag"]: return "diagonal"
    if s["anti_frac"] >= 0.2: return "dense (chem etc.)"
    return "mixed"
for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r.get("eligible")]
    print(f"## {path}: {len(rows)} eligible rows; exact True {sum(1 for r in rows if r.get('exact') is True)}, "
          f"NOT exact {sum(1 for r in rows if r.get('exact') not in (True, None))}, on failed {sum(r['on_failed'] for r in rows)}")
    for dev in sorted({r["device"] for r in rows}):
        print(f"\n### {dev}\n")
        print("| category | tests | q2 S6/QK3 | q2 S6/S5PE | runnable | QK3 ESP 0 | S6 ESP 0 | ESP S6/QK3 (both>0) | S6 better / worse by 10% |")
        print("|---|---|---|---|---|---|---|---|---|")
        for c in ("diagonal", "mixed", "dense (chem etc.)", "all"):
            sel = [r for r in rows if r["device"] == dev and (c == "all" or cat(r["test"]) == c)]
            if not sel: continue
            g = lambda a, b: math.exp(sum(math.log((a(r) + 1) / (b(r) + 1)) for r in sel) / len(sel))
            q_qk = g(lambda r: r["q2"], lambda r: r["qk3_q2"])
            s5 = [r for r in sel if (dev, r["test"]) in S5]
            q_s5 = math.exp(sum(math.log((r["q2"] + 1) / (S5[(dev, r["test"])]["q2"] + 1)) for r in s5) / len(s5)) if s5 else float("nan")
            run_, z_q, z_s, lr, bw = 0, 0, 0, [], [0, 0]
            for r in sel:
                es = None if r["esp"] is None or r["on_failed"] else 10 ** r["esp"]
                eq = None if r["qk3_esp"] is None or r["qk3_failed"] else 10 ** r["qk3_esp"]
                es, eq = es or 0.0, eq or 0.0
                if max(es, eq) < 0.01: continue
                run_ += 1; z_q += eq == 0; z_s += es == 0
                if es > 0 and eq > 0:
                    lr.append(math.log10(es / eq)); bw[0] += es / eq > 1.1; bw[1] += es / eq < 1 / 1.1
                elif es > 0: bw[0] += 1
                elif eq > 0: bw[1] += 1
            er = 10 ** (sum(lr) / len(lr)) if lr else float("nan")
            print(f"| {c} | {len(sel)} | {q_qk:.3f} | {q_s5:.3f} | {run_} | {z_q} | {z_s} | {er:.3f} ({len(lr)}) | {bw[0]} / {bw[1]} |")
