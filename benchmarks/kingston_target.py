"""kingston_target.py -- KT (2026-10-10; exploratory, nothing predicted): does the live Target of ibm_kingston, as
QiskitRuntimeService returned it on 2026-09-28 (pickled by benchmarks/real_target_cliff.py, Addendum 245), keep
couplers, gates or measurements whose reported error is >= 0.5 -- as the snapshots in qiskit-ibm-runtime do
(ESP-FT, Addendum 430)? Reads the pickle only: no service call, no job.

    python <this file> PICKLE OUTDIR
"""
from __future__ import annotations

import json
import os
import pickle
import statistics
import sys

FAILED = 0.5


def describe(t):
    nq = t.num_qubits
    ops = {}
    for name in t.operation_names:
        rows = []
        for qargs, ip in t[name].items():
            if qargs is None:
                continue
            rows.append(dict(q=list(qargs), error=None if ip is None else ip.error,
                             duration=None if ip is None else ip.duration))
        ops[name] = rows
    covered = sorted({q for rows in ops.values() for r in rows for q in r["q"]})
    return nq, ops, covered


def summary(name, rows):
    errs = [r["error"] for r in rows if r["error"] is not None]
    bad = sorted(([r["q"], r["error"]] for r in rows if r["error"] is not None and r["error"] >= FAILED),
                 key=lambda x: x[0])
    return dict(op=name, entries=len(rows), with_error=len(errs), error_none=len(rows) - len(errs),
                failed=len(bad), error_1=sum(1 for e in errs if e >= 1.0),
                median=statistics.median(errs) if errs else None, max=max(errs) if errs else None,
                over_0_1=sum(1 for e in errs if e >= 0.1), failed_list=bad)


def undirected(rows):
    return {tuple(sorted(r["q"])) for r in rows}


def main():
    pkl, out = sys.argv[1], sys.argv[2]
    if os.path.exists(out):
        sys.exit(f"STOP: {out} exists")
    with open(pkl, "rb") as f:
        t = pickle.load(f)
    nq, ops, covered = describe(t)
    two = [n for n, rows in ops.items() if rows and len(rows[0]["q"]) == 2]
    one = [n for n, rows in ops.items() if rows and len(rows[0]["q"]) == 1]
    sums = {n: summary(n, ops[n]) for n in ops if ops[n]}
    edges = set().union(*(undirected(ops[n]) for n in two)) if two else set()

    from qiskit_ibm_runtime.fake_provider import FakeKingston
    ft = FakeKingston().target
    _, fops, _ = describe(ft)
    ftwo = [n for n, rows in fops.items() if rows and len(rows[0]["q"]) == 2]
    fedges = set().union(*(undirected(fops[n]) for n in ftwo)) if ftwo else set()

    L = ["# KT: the live Target of ibm_kingston (2026-09-28) -- failed elements (exploratory, nothing predicted)", "",
         f"Pickle written by benchmarks/real_target_cliff.py on 2026-09-28 (calibration 2026-09-28 19:55+09:00, as "
         f"that run printed). {nq} qubits; operations {sorted(ops)}. Failed = reported error >= {FAILED}.", "",
         "| operation | entries | error None | median error | entries >= 0.1 | failed (>= 0.5) | error 1 |",
         "|---|---|---|---|---|---|---|"]
    for n in sorted(sums, key=lambda n: (-len(ops[n][0]["q"]), n)):
        s = sums[n]
        med = "-" if s["median"] is None else f"{s['median']:.2e}"
        L.append(f"| {n} | {s['entries']} | {s['error_none']} | {med} | {s['over_0_1']} | {s['failed']} | "
                 f"{s['error_1']} |")
    L += ["", f"Couplers (undirected, any two-qubit operation): live {len(edges)}; FakeKingston (snapshot 2026-04) "
          f"{len(fedges)}; in the snapshot but not in the live Target: {sorted(fedges - edges)}; in the live Target "
          f"but not in the snapshot: {sorted(edges - fedges)}.",
          f"Qubits with no operation at all in the live Target: {sorted(set(range(nq)) - set(covered))}.", ""]
    for n in sorted(sums):
        if sums[n]["failed_list"]:
            L.append(f"Failed {n}: " + ", ".join(f"{q} ({e:.3g})" for q, e in sums[n]["failed_list"]))
    fail2 = sum(sums[n]["failed"] for n in two)
    failm = sums.get("measure", {}).get("failed", 0)
    fail1 = sum(sums[n]["failed"] for n in one if n != "measure")
    L += ["", "## Answer", ""]
    if fail2 or failm or fail1:
        L.append(f"Yes: the live Target keeps {fail2} two-qubit entries, {fail1} one-qubit gate entries and {failm} "
                 "measurements with error >= 0.5. A compiler that reads only the coupling map can place gates on "
                 "them; ESP prices such an output at (about) 0.")
    else:
        L.append("No: every entry in the live Target has error < 0.5. Elements the service flags as faulty were "
                 "removed (see the couplers and qubits missing above); what is left can be used, and the "
                 "failed-element results of ESP-FT apply to the snapshots, not to this live Target.")
    txt = "\n".join(L) + "\n"
    os.makedirs(out)
    with open(os.path.join(out, "kingston_target.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    with open(os.path.join(out, "kingston_target.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(dict(qubits=nq, summaries=sums, live_edges=sorted(edges), snapshot_edges=sorted(fedges),
                       uncovered=sorted(set(range(nq)) - set(covered))), fh, indent=1)
    print(txt)


if __name__ == "__main__":
    main()
