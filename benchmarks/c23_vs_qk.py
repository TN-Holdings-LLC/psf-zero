"""c23_vs_qk.py -- exploratory, after the fact (Addendum 396, 2026-10-07): candidate c23's default call against
Qiskit level 2 on the Benchpress tests of BP-MOCK and BP-MOCK2, from committed output only (no compile is run).

CANCEL (Addendum 395) has no Qiskit arm. Its C23 rows are put next to the QK rows of BP-MOCK (Addendum 390) and
BP-MOCK2 (Addendum 392), which compiled the same inputs (built the same way, the same Benchpress commit) on the same
workplace PC. A test counts where every arm of its BP-MOCK or BP-MOCK2 run finished and C23 finished. Two-qubit
counts are compared as (count + 1) / (count + 1), the ratio those tests used (a circuit may have none). Times come
from different runs and are indicative only.

    python benchmarks/c23_vs_qk.py --out data/2026-10-07/c23_vs_qk
"""
import argparse
import json
import math
import os
import statistics
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SOURCES = (("BP-MOCK", "data/2026-10-07/bp_mock/bp_mock.json", ("QK", "REL", "C22")),
           ("BP-MOCK2", "data/2026-10-07/bp_mock2/bp_mock2.json", ("QK", "REL", "REL2", "C22")))
CANCEL = "data/2026-10-07/cancel/cancel.json"


def load(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        r = json.load(f)
    by = {}
    for x in r["rows"]:
        by.setdefault(x["test"], {})[x["arm"]] = x
    return r["meta"], by


def ok(x):
    return x is not None and "error" not in x


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def ratio(a, b):
    return (a["q2"] + 1) / (b["q2"] + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cmeta, ca = load(CANCEL)
    rows, lines = [], ["# C23 against Qiskit level 2 (exploratory, Addendum 396)", ""]
    lines.append(f"CANCEL git_head {cmeta['git_head']}; two-qubit count ratio (x + 1) / (y + 1); geometric means.")
    lines.append("")
    lines.append("| sample | tests | C23 / QK | C22 / QK | REL / QK | C22 count as in its own run |")
    lines.append("|---|---|---|---|---|---|")
    for name, path, arms in SOURCES:
        meta, by = load(path)
        done = [t for t in by if all(ok(by[t].get(a)) for a in arms)]
        missing = [t for t in done if t not in ca]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} tests are not in CANCEL's output")
        use = [t for t in done if ok(ca[t].get("C23")) and ok(ca[t].get("C22"))]
        same = sum(ca[t]["C22"]["q2"] == by[t]["C22"]["q2"] for t in use)
        for t in use:
            rows.append(dict(sample=name, test=t, stratum=by[t]["QK"]["stratum"], qk=by[t]["QK"]["q2"],
                             rel=by[t]["REL"]["q2"], c22=ca[t]["C22"]["q2"], c23=ca[t]["C23"]["q2"],
                             t_qk=by[t]["QK"]["t"], t_c23=ca[t]["C23"]["t"],
                             cancel_kept=ca[t]["C23"].get("cancel", {}).get("cancelled_kept", 0)))
        rs = [x for x in rows if x["sample"] == name]
        lines.append(f"| {name} (git_head {meta['git_head']}) | {len(use)} of {len(by)} | "
                     f"{gmean([(x['c23'] + 1) / (x['qk'] + 1) for x in rs]):.3f} | "
                     f"{gmean([(x['c22'] + 1) / (x['qk'] + 1) for x in rs]):.3f} | "
                     f"{gmean([(x['rel'] + 1) / (x['qk'] + 1) for x in rs]):.3f} | {same} of {len(use)} |")
    r23 = [(x["c23"] + 1) / (x["qk"] + 1) for x in rows]
    lines.append(f"| both | {len(rows)} | {gmean(r23):.3f} | "
                 f"{gmean([(x['c22'] + 1) / (x['qk'] + 1) for x in rows]):.3f} | "
                 f"{gmean([(x['rel'] + 1) / (x['qk'] + 1) for x in rows]):.3f} | |")
    lines += ["", f"C23 against QK on the {len(rows)} tests: fewer two-qubit gates on {sum(r < 1 for r in r23)}, "
                  f"as many on {sum(r == 1 for r in r23)}, more on {sum(r > 1 for r in r23)}; more by over 10% on "
                  f"{sum(r > 1.1 for r in r23)}, by over 25% on {sum(r > 1.25 for r in r23)}.", ""]
    lines += ["| stratum | tests | C23 / QK | REL / QK |", "|---|---|---|---|"]
    st = defaultdict(list)
    for x in rows:
        st[x["stratum"]].append(x)
    for s in sorted(st):
        v = st[s]
        lines.append(f"| {s} | {len(v)} | {gmean([(x['c23'] + 1) / (x['qk'] + 1) for x in v]):.3f} | "
                     f"{gmean([(x['rel'] + 1) / (x['qk'] + 1) for x in v]):.3f} |")
    order = sorted(rows, key=lambda x: (x["c23"] + 1) / (x["qk"] + 1))
    for title, part in (("Largest C23 / QK", order[::-1][:10]), ("Smallest C23 / QK", order[:10])):
        lines += ["", f"{title}:", "", "| test | QK | C23 | ratio |", "|---|---|---|---|"]
        lines += [f"| {x['test']} | {x['qk']} | {x['c23']} | {(x['c23'] + 1) / (x['qk'] + 1):.2f} |" for x in part]
    tr = [x["t_c23"] / max(x["t_qk"], 1e-3) for x in rows]
    lines += ["", f"Compile time C23 / QK (different runs, indicative): median {statistics.median(tr):.2f}, "
                  f"geometric mean {gmean(tr):.2f}, range {min(tr):.2f}-{max(tr):.2f}."]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "c23_vs_qk.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(dict(cancel_git_head=cmeta["git_head"], rows=rows), f, indent=1)
    with open(os.path.join(args.out, "c23_vs_qk.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
