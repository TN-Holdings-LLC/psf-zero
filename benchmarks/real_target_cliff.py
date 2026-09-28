"""real_target_cliff.py -- pre-registered (home, 2026-09-28, Addendum 245): does the
layout cliff appear on a real IBM device's Target (heavy-hex Heron r2, 156 qubits),
and which compiler meets a 1-second deadline? No job is submitted and no QPU time
is used: the device's Target (coupling map, native gates, calibration) is read once.

Stage 0 (parent): read the Target of DEVICE through QiskitRuntimeService (read-only),
pickle it for the children, and compute the maximum matching M of its coupling map.
Logical widths are n = 2M - spare, spare in SPARES, so spare 0 fills every qubit that
can be paired.

Circuits: n/2 disjoint pair24 blocks (loop_endurance.add_pair24), random angles
(seed = 1000 * spare + input). Arms, each compile in its own spawned child, timed
inside the child around the compile call only, killed at CAP_S:
  Q3  qiskit.transpile(qc, target=target, optimization_level=3, seed_transpiler=0)
  P   psf_compile.compile_for_hardware(qc, coupling_map=target's, basis_gates=native,
      entangling_basis="cx", layout_search=True, on_unsupported="raise", seed_transpiler=0)
Quality: routed two-qubit count; when no two-qubit gate joins different pairs, an exact
per-pair check by phase-aligned Frobenius distance (not infidelity; Addenda 216-221).

    python -u benchmarks/real_target_cliff.py 2>&1 | tee real_target_cliff.txt
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import multiprocessing as mp
import os
import pickle
import platform
import statistics as st
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks"), os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEVICE = "ibm_kingston"
SPARES = (0, 2, 8, 16)
INPUTS = 3
CAP_S = 180.0
DEADLINES = (0.1, 1.0, 10.0)
NATIVE = ("cz", "ecr", "cx", "rz", "sx", "x", "id", "rzz")
TARGET_PKL = "real_target_2026-09-28.pkl"
OUT_CSV = "real_target_cliff_2026-09-28.csv"
OUT_JSON = "real_target_cliff_2026-09-28.json"


def aligned(a, b):
    t = np.trace(a.conj().T @ b)
    ph = t / abs(t) if abs(t) > 0 else 1.0
    return float(np.linalg.norm(b - ph * a))


def pair_check(qc_logical, qc_out, n_logical):
    """Per-pair phase-aligned operator distance. Returns (applicable, worst)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    lay = qc_out.layout
    if lay is None:
        return False, None
    phys = list(lay.final_index_layout(filter_ancillas=True))
    pairs = [(i, i + 1) for i in range(0, n_logical - 1, 2)]
    owner = {}
    for k, (a, b) in enumerate(pairs):
        owner[phys[a]] = k
        owner[phys[b]] = k
    per_pair = {k: QuantumCircuit(2) for k in range(len(pairs))}
    for inst in qc_out.data:
        if inst.operation.name in ("barrier", "measure", "delay"):
            continue
        qs = [qc_out.find_bit(q).index for q in inst.qubits]
        ks = {owner.get(q) for q in qs}
        if None in ks:
            if len(qs) == 1:
                continue
            return False, None
        if len(ks) != 1:
            return False, None
        k = ks.pop()
        a, _ = pairs[k]
        per_pair[k].append(inst.operation, [0 if q == phys[a] else 1 for q in qs])
    worst = 0.0
    for k, (a, b) in enumerate(pairs):
        ref = QuantumCircuit(2)
        for inst in qc_logical.data:
            qs = [qc_logical.find_bit(q).index for q in inst.qubits]
            if set(qs) <= {a, b}:
                ref.append(inst.operation, [0 if q == a else 1 for q in qs])
        worst = max(worst, aligned(Operator(ref).data, Operator(per_pair[k]).data))
    return True, worst


def build(n, seed):
    from qiskit import QuantumCircuit
    import loop_endurance as le
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    th = rng.uniform(-np.pi, np.pi, (n // 2, 24))
    for k in range(n // 2):
        le.add_pair24(qc, 2 * k, 2 * k + 1, th[k])
    return qc


def _worker(arm, n, seed, q):
    from qiskit import transpile
    with open(TARGET_PKL, "rb") as f:
        target = pickle.load(f)
    qc = build(n, seed)
    native = [g for g in target.operation_names if g in NATIVE]
    cmap = target.build_coupling_map()
    if arm == "P":
        import psf_compile as pc
    t0 = time.perf_counter()
    if arm == "Q3":
        out = transpile(qc, target=target, optimization_level=3, seed_transpiler=0)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            out = pc.compile_for_hardware(qc, coupling_map=cmap, basis_gates=native, entangling_basis="cx",
                                          layout_search=True, on_unsupported="raise", seed_transpiler=0)
    elapsed = time.perf_counter() - t0
    twoq = sum(1 for i in out.data if len(i.qubits) == 2 and i.operation.name != "barrier")
    applicable, worst = pair_check(qc, out, n)
    q.put(dict(elapsed_s=elapsed, twoq=twoq, pair_check_applicable=applicable, pair_worst=worst))


def run_one(arm, n, seed):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(arm, n, seed, q))
    p.start()
    p.join(CAP_S)
    if p.is_alive():
        p.terminate()
        p.join()
        return dict(elapsed_s=None, twoq=None, pair_check_applicable=None, pair_worst=None, status="DNF")
    if p.exitcode != 0 or q.empty():
        return dict(elapsed_s=None, twoq=None, pair_check_applicable=None, pair_worst=None,
                    status=f"ERROR(exit {p.exitcode})")
    r = q.get()
    r["status"] = "OK"
    return r


def main():
    import qiskit
    import rustworkx as rx
    import psf_compile as pc
    import loop_endurance as le
    print(f"platform {platform.platform()} | cores {os.cpu_count()} | python {platform.python_version()} "
          f"| qiskit {qiskit.__version__}")
    print("LOADED", pc.__file__, pc.VERSION, le.normalized_sha256(pc.__file__),
          "| CORE_VERSION", getattr(pc, "CORE_VERSION", None))
    print("SCRIPT", os.path.abspath(__file__), le.normalized_sha256(os.path.abspath(__file__)))
    from qiskit_ibm_runtime import QiskitRuntimeService
    with contextlib.redirect_stderr(io.StringIO()):
        svc = QiskitRuntimeService()
        backend = svc.backend(DEVICE)
    target = backend.target
    try:
        cal = str(backend.properties().last_update_date)
    except Exception:  # noqa: BLE001
        cal = None
    with open(TARGET_PKL, "wb") as f:
        pickle.dump(target, f)
    cmap = target.build_coupling_map()
    nq = cmap.size()
    g = rx.PyGraph()
    g.add_nodes_from(range(nq))
    g.add_edges_from_no_data(sorted({tuple(sorted(e)) for e in cmap.get_edges()}))
    m = len(rx.max_weight_matching(g, max_cardinality=True))
    deg = sorted(g.degree(i) for i in range(nq))
    native = [x for x in target.operation_names if x in NATIVE]
    print(f"Stage 0: device {DEVICE}, {nq} qubits, degree min/median/max {deg[0]}/{st.median(deg)}/{deg[-1]}, "
          f"max matching {m} pairs ({2 * m} qubits), native {native}, calibration {cal}")
    print("No job is submitted; the Target was read once and pickled for the children.")
    rows = []
    for spare in SPARES:
        n = 2 * m - spare
        for k in range(INPUTS):
            seed = 1000 * spare + k
            for arm in ("Q3", "P"):
                r = run_one(arm, n, seed)
                row = dict(spare=spare, logical_qubits=n, physical_qubits=nq, input=k, arm=arm, **r)
                for d in DEADLINES:
                    row[f"within_{d}s"] = r["status"] == "OK" and r["elapsed_s"] <= d
                rows.append(row)
                el = "DNF" if r["elapsed_s"] is None else f"{r['elapsed_s']:.3f}s"
                print(f"spare={spare:2d} n={n} input={k} {arm:2s} {r['status']:5s} {el:>9s} 2q={r['twoq']} "
                      f"pair_check={r['pair_check_applicable']} worst={r['pair_worst']}", flush=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- scoring (Addendum 245)
    def v(ok, bad):
        return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")

    def cell(spare, arm):
        return [r for r in rows if r["spare"] == spare and r["arm"] == arm]

    def med(spare, arm):
        xs = [r["elapsed_s"] if r["status"] == "OK" else CAP_S for r in cell(spare, arm)]
        return st.median(xs)

    print("\n=== pre-registered (Addendum 245) ===")
    ok0 = all(r["status"] == "OK" for r in rows if r["arm"] == "P")
    print(f"C0 every PSF-Zero compile finished without error: {ok0}")
    p1 = sum(r["within_1.0s"] for r in cell(0, "P"))
    print(f"R1 spare 0: PSF-Zero within 1 s in {p1}/{INPUTS} (confirmed {INPUTS}/{INPUTS}, refuted <= {INPUTS - 1}) -> "
          f"{v(p1 == INPUTS, p1 < INPUTS)}")
    r2 = med(0, "Q3") / med(0, "P")
    print(f"R2 spare 0: Qiskit L3 median / PSF-Zero median = {med(0, 'Q3'):.3f} / {med(0, 'P'):.3f} s = {r2:.1f} "
          f"(confirmed >= 10, refuted < 3; DNF counted as {CAP_S:.0f} s) -> {v(r2 >= 10, r2 < 3)}")
    r3 = med(16, "Q3") / med(16, "P")
    print(f"R3 spare 16: ratio {r3:.1f} (confirmed < 3, refuted >= 10) -> {v(r3 < 3, r3 >= 10)}")
    both = [(a, b) for a, b in zip([r for r in rows if r["arm"] == "Q3"], [r for r in rows if r["arm"] == "P"])
            if a["status"] == "OK" and b["status"] == "OK"]
    same = sum(a["twoq"] == b["twoq"] for a, b in both)
    worse = sum(b["twoq"] > a["twoq"] for a, b in both)
    print(f"R4 two-qubit count: PSF-Zero equal in {same}/{len(both)}, above Qiskit in {worse} "
          f"(confirmed: never above; refuted: above in any) -> {v(worse == 0, worse > 0)}")
    pw = [r["pair_worst"] for r in rows if r["arm"] == "P" and r["pair_check_applicable"]]
    napp = sum(1 for r in rows if r["arm"] == "P" and r["status"] == "OK")
    wmax = max(pw) if pw else None
    print(f"R5 PSF-Zero exact per pair (<= 1e-12) wherever the check applies: {len(pw)}/{napp} applicable, worst {wmax} "
          f"-> {v(bool(pw) and wmax <= 1e-12, bool(pw) and wmax > 1e-12)}")
    qw = [r["pair_worst"] for r in rows if r["arm"] == "Q3" and r["pair_check_applicable"]]
    print("Reported without prediction: Qiskit L3 per-pair worst", max(qw) if qw else None,
          f"over {len(qw)} applicable; medians per spare:",
          {s: (round(med(s, 'Q3'), 3), round(med(s, 'P'), 3)) for s in SPARES})
    json.dump(dict(device=DEVICE, qubits=nq, matching=m, calibration=cal, native=native, rows=rows),
              open(OUT_JSON, "w"), indent=1, default=str)
    print(f"\nWrote {OUT_CSV} and {OUT_JSON} ({len(rows)} rows)\nDONE")


if __name__ == "__main__":
    main()
