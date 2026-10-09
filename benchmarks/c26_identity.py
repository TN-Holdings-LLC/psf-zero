"""c26_identity.py -- C26-ID (2026-10-09): candidate 2026-10-09.c26 (changelog item 53) against release 2026-10-07.1,
function by function. Item 53 changes how the recommended call's estimates and exactness checks build matrices, not
what they compute; these functions are deterministic, so their results are compared directly (not whole compiled
outputs, which differ from process to process: Addenda 410-412).

Cases: N random circuits (qiskit.circuit.random.random_circuit, 2-10 qubits, seeds 20261009 + k), each transpiled for
FakeTorino twice (level 1, seed k; level 2, seed k + 1), and a wrong copy of the first (an X gate appended on a
touched qubit). Both modules are loaded in this one process. For each case, each version computes:
  excitation_cost, hybrid_cost, pauli_cost, kraus_cost, readout_cost   (out, target)
  _ops_of                                                              (out): matrices and qubits
  _implements                                                          (qc, out), (qc, out2), (qc, wrong)
  _same_action                                                         (out, out2), (out, out)
Values are compared exactly (== on floats; equal matrices elementwise; equal booleans). The two versions alternate
which goes first; their times are summed per function.

With --bp, the recommended call is also timed on three development tests, the two versions alternating, REPS each
(reported only: whole outputs are not compared).

    python benchmarks/c26_identity.py --out DIR [--n 300] [--bp <benchpress clone>]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
PATCH = os.path.join(REPO, "patches", "psf_compile_c26_2026-10-09")
sys.path[:0] = [HERE, REPO]
COSTS = ("excitation_cost", "hybrid_cost", "pauli_cost", "kraus_cost", "readout_cost")
HEAVY = ("test_hamlib_hamiltonians_transpile[ham_enc_gray_dvalues_8-8-8]", "test_feynman_transpile[grover_5.qasm]",
         "test_hamlib_hamiltonians_transpile[ham_ham_JW-10]")
REPS = 2


def same_ops(a, b):
    if a is None or b is None:
        return a is None and b is None
    return len(a) == len(b) and all(qa == qb and np.array_equal(ma, mb) for (ma, qa), (mb, qb) in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--bp")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    os.makedirs(a.out, exist_ok=True)
    import core_fix_c2_eval as H
    from qiskit import transpile
    from qiskit.circuit.random import random_circuit
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    H.load_module(os.path.join(HERE, "psf_smart_layout.py"), "psf_smart_layout")
    mods = {"REL": H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile_rel_c26"),
            "C26": H.load_module(os.path.join(PATCH, "psf_compile.py"), "psf_compile_c26")}
    backend = FakeTorino()
    target = backend.target
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout
    dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    times = {f: {"REL": 0.0, "C26": 0.0} for f in COSTS + ("_ops_of", "_implements", "_same_action")}
    mism, compared = [], 0
    t_start = time.perf_counter()
    for k in range(a.n):
        n = 2 + k % 9
        qc = random_circuit(n, 2 + (k * 7) % 12, max_operands=2, measure=False, seed=20261009 + k)
        out = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=k)
        out2 = transpile(qc, backend=backend, optimization_level=2, seed_transpiler=k + 1)
        wrong = out.copy()
        touched = sorted({out.find_bit(b).index for i in out.data for b in i.qubits})
        wrong.x(touched[0] if touched else 0)
        order = ("REL", "C26") if k % 2 == 0 else ("C26", "REL")
        calls = [(f, (out, target)) for f in COSTS] + [("_ops_of", (out,))]
        calls += [("_implements", (qc, c)) for c in (out, out2, wrong)] + [("_same_action", (out, c)) for c in (out2, out)]
        for f, args in calls:
            res = {}
            for v in order:
                t0 = time.perf_counter()
                res[v] = getattr(mods[v], f)(*args)
                times[f][v] += time.perf_counter() - t0
            ok = same_ops(res["REL"], res["C26"]) if f == "_ops_of" else res["REL"] == res["C26"]
            compared += 1
            if not ok:
                mism.append(dict(case=k, function=f, rel=repr(res["REL"])[:80], c26=repr(res["C26"])[:80]))
        if (k + 1) % 50 == 0:
            print(f"[{k + 1}/{a.n} {time.perf_counter() - t_start:5.0f} s] {compared} values compared, "
                  f"{len(mism)} differ", flush=True)
    heavy = []
    if a.bp:
        import bp_mock as B
        import c25_identity as C
        index = {t[1]: t for t in C.population(a.bp)}
        for tid in HEAVY:
            stratum, _, kind, arg = index[tid]
            qc, bk = B.build(a.bp, kind, arg)
            basis = [g for g in bk.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
            kw = dict(coupling_map=bk.coupling_map, basis_gates=basis, entangling_basis="cx", layout_search=True,
                      seed_transpiler=0, target=bk.target, **C.RECOMMENDED)
            row = dict(test=tid, REL=[], C26=[], q2={"REL": [], "C26": []})
            for r in range(REPS):
                for v in (("REL", "C26") if r % 2 == 0 else ("C26", "REL")):
                    t0 = time.perf_counter()
                    with contextlib.redirect_stdout(io.StringIO()):
                        o = mods[v].compile_for_hardware(qc, **kw)
                    row[v].append(round(time.perf_counter() - t0, 3))
                    row["q2"][v].append(int(o.count_ops().get(bk.two_q_gate_type, 0)))
            heavy.append(row)
            print(f"recommended call, {tid[:60]}: REL {row['REL']} s, C26 {row['C26']} s", flush=True)
    ok = (not mism and not dirty and mods["REL"].VERSION == "2026-10-07.1" and mods["C26"].VERSION == "2026-10-09.c26"
          and compared == a.n * 11)
    L = ["# C26-ID", "", f"git head {head.strip()}, uncommitted tracked changes: {'none' if not dirty else 'YES'}; "
         f"{a.n} cases, {compared} values compared (expected {a.n * 11}); versions {mods['REL'].VERSION}, "
         f"{mods['C26'].VERSION}; Python {sys.version.split()[0]}", "",
         f"- values that differ: {len(mism)}", "", "| function | REL (s) | C26 (s) | C26 / REL |", "|---|---|---|---|"]
    for f, t in times.items():
        L.append(f"| {f} | {t['REL']:.2f} | {t['C26']:.2f} | {t['C26'] / t['REL']:.3f} |" if t["REL"] else f"| {f} | 0 | 0 | |")
    if heavy:
        L += ["", "Recommended call, whole compile (reported only):", ""]
        L += [f"- {h['test']}: REL {h['REL']} s, C26 {h['C26']} s; two-qubit gates REL {h['q2']['REL']}, C26 "
              f"{h['q2']['C26']}" for h in heavy]
    L += [f"- differs: case {m['case']} {m['function']}: REL {m['rel']} / C26 {m['c26']}" for m in mism[:30]]
    L += ["", f"VERDICT {'IDENTICAL' if ok else 'NOT IDENTICAL (see above)'}"]
    txt = "\n".join(L) + "\n"
    open(os.path.join(a.out, "c26_identity.md"), "w", encoding="utf-8", newline="\n").write(txt)
    json.dump(dict(times=times, mismatches=mism, heavy=heavy, compared=compared, head=head.strip(), dirty=dirty),
              open(os.path.join(a.out, "c26_identity.json"), "w", encoding="utf-8"), indent=1)
    print(txt)


if __name__ == "__main__":
    main()
