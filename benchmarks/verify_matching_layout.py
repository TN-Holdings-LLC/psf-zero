"""verify_matching_layout.py -- Addendum 192.

Checks the matching shortcut added to psf_smart_layout.smart_vf2_layout
(LAYOUT_VERSION 2026-09-26.m1) against the unchanged VF2 path, on
FakeNighthawk, in one process, with the two arms interleaved.

M1/M2  Dense-pair-block circuits (the cliff family), spare in {0, 2, 4, 8},
       seeds 0-4, REPS calls per arm. The arm is switched with
       psf_smart_layout.USE_MATCHING_SHORTCUT (compile_for_hardware cannot
       pass the argument). Records: total compile_for_hardware time, layout
       search time and phase (the search function is wrapped to time it),
       routed 2-qubit count, and the exact per-pair check.
M3     Chain circuits (interaction graph is a path, not a matching), 40
       logical qubits, seeds 0-2: the shortcut must not be taken and both
       arms must return the same layout and 2-qubit count.
X1     Exploratory, no prediction scored: mean and sum of the backend's
       2-qubit gate error over the distinct physical edges used, for the
       old path, the new unweighted path, the new weighted path (weights =
       round(1e6 * (1 - error))), and Qiskit's transpile(optimization_level=3)
       (which runs VF2PostLayout), at spare 0 and 8, seed 0. FakeNighthawk's
       error values are, by its own warning, not representative.

Usage (repository root):
    python -u verify_matching_layout.py 2>&1 | tee matching_layout_result.txt
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import os
import platform
import statistics as st
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_ibm_runtime.fake_provider import FakeNighthawk

import psf_compile as pc
import psf_smart_layout as psl

SPARES = (0, 2, 4, 8)
SEEDS = 5
REPS = 5
GATES_PER_PAIR = 20
CHAIN_QUBITS = 40
CHAIN_SEEDS = 3
NATIVE = ("cz", "ecr", "cx", "rz", "sx", "x", "id", "rzz")
OUT_CSV = "matching_layout_2026-09-26.csv"


def normalized_sha256(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f.read().splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def build_dense_pair_blocks_circuit(num_qubits, gates_per_pair, seed):
    """Verbatim from bench_cliff_1v1.py."""
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for (a, b) in [(i, i + 1) for i in range(0, num_qubits - 1, 2)]:
        block = QuantumCircuit(2)
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            block.append(UnitaryGate(u), [0, 1])
        qc.compose(block.decompose(), [a, b], inplace=True)
    return qc


def build_chain_circuit(num_qubits, gates_per_pair, seed):
    """Same blocks as above, but on every neighbouring pair (i, i+1), so the
    interaction graph is a path rather than a matching."""
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for a in range(num_qubits - 1):
        block = QuantumCircuit(2)
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            block.append(UnitaryGate(u), [0, 1])
        qc.compose(block.decompose(), [a, a + 1], inplace=True)
    return qc


def pair_check(qc_logical, qc_out, n_logical):
    """Exact per-pair check, verbatim from nighthawk_deadline_cliff.py.
    Returns (applicable, worst_infidelity)."""
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
        qs = [qc_out.find_bit(q).index for q in inst.qubits]
        if inst.operation.name in ("barrier", "measure", "delay"):
            continue
        ks = {owner.get(q) for q in qs}
        if None in ks:
            if len(qs) == 1:
                continue
            return False, None
        if len(ks) != 1:
            return False, None
        k = ks.pop()
        a, b = pairs[k]
        local = [0 if q == phys[a] else 1 for q in qs]
        per_pair[k].append(inst.operation, local)
    worst = 0.0
    for k, (a, b) in enumerate(pairs):
        ref = QuantumCircuit(2)
        for inst in qc_logical.data:
            qs = [qc_logical.find_bit(q).index for q in inst.qubits]
            if set(qs) <= {a, b}:
                ref.append(inst.operation, [0 if q == a else 1 for q in qs])
        u, v = Operator(per_pair[k]).data, Operator(ref).data
        infid = 1.0 - abs(np.trace(v.conj().T @ u)) / 4.0
        worst = max(worst, float(infid))
    return True, worst


def twoq_count(qc):
    return sum(1 for i in qc.data if len(i.qubits) == 2 and i.operation.name != "barrier")


class LayoutProbe:
    """Wraps psf_smart_layout.smart_vf2_layout to record its time and info."""

    def __init__(self):
        self.orig = psl.smart_vf2_layout
        self.last = None

    def __enter__(self):
        def wrapper(*a, **k):
            t0 = time.perf_counter()
            out = self.orig(*a, **k)
            self.last = (time.perf_counter() - t0, out[1])
            return out
        psl.smart_vf2_layout = wrapper
        return self

    def __exit__(self, *exc):
        psl.smart_vf2_layout = self.orig
        return False


def psf_call(qc, backend, native, shortcut, initial_layout=None):
    psl.USE_MATCHING_SHORTCUT = shortcut
    pc._CX_CORE_CACHE.clear()
    kw = dict(layout_search=True) if initial_layout is None else dict(initial_layout=initial_layout)
    with LayoutProbe() as probe, contextlib.redirect_stdout(io.StringIO()):
        t0 = time.perf_counter()
        out = pc.compile_for_hardware(qc, coupling_map=backend.coupling_map, basis_gates=native,
                                      entangling_basis="cx", on_unsupported="raise",
                                      seed_transpiler=0, **kw)
        el = time.perf_counter() - t0
    lay_t, info = probe.last if probe.last else (None, {})
    return out, el, lay_t, info


def edge_errors(backend):
    twoq = next(g for g in ("cz", "ecr", "cx") if g in backend.operation_names)
    err = {}
    for qargs, props in backend.target[twoq].items():
        if qargs is None or props is None or props.error is None:
            continue
        key = tuple(sorted(qargs))
        err[key] = min(err.get(key, props.error), props.error)
    return twoq, err


def used_edge_error(qc_out, err):
    used = set()
    for inst in qc_out.data:
        if len(inst.qubits) == 2 and inst.operation.name != "barrier":
            used.add(tuple(sorted(qc_out.find_bit(q).index for q in inst.qubits)))
    vals = [err[e] for e in used if e in err]
    return len(used), (float(np.mean(vals)) if vals else None), float(np.sum(vals))


def main():
    print(f"platform {platform.platform()} | cores {os.cpu_count()} | python {platform.python_version()} "
          f"| qiskit {qiskit.__version__}")
    print("LOADED", pc.__file__, pc.VERSION, normalized_sha256(pc.__file__))
    print("LOADED", psl.__file__, getattr(psl, "LAYOUT_VERSION", "NO LAYOUT_VERSION"),
          normalized_sha256(psl.__file__))
    print("SCRIPT", os.path.abspath(__file__), normalized_sha256(os.path.abspath(__file__)))
    if getattr(psl, "LAYOUT_VERSION", None) != "2026-09-26.m1":
        print("M0 FAILED: the prototype psf_smart_layout.py is not the one loaded. Stopping.")
        return

    backend = FakeNighthawk()
    native = [g for g in backend.operation_names if g in NATIVE]
    n_phys = backend.coupling_map.size()
    rows = []

    # Warm-up (imports, caches); not recorded.
    psf_call(build_dense_pair_blocks_circuit(n_phys, GATES_PER_PAIR, 99), backend, native, True)

    # ---- M1 / M2
    print("\n=== M1/M2: dense pair blocks, old (VF2) vs new (matching) ===")
    summary = {}
    for spare in SPARES:
        acc = {False: dict(total=[], lay=[]), True: dict(total=[], lay=[])}
        m1_ok = True
        for seed in range(SEEDS):
            qc = build_dense_pair_blocks_circuit(n_phys - spare, GATES_PER_PAIR, seed)
            twoq = {}
            for rep in range(REPS):
                order = (False, True) if rep % 2 == 0 else (True, False)
                for arm in order:
                    out, el, lay_t, info = psf_call(qc, backend, native, arm)
                    acc[arm]["total"].append(el)
                    acc[arm]["lay"].append(lay_t)
                    twoq.setdefault(arm, twoq_count(out))
                    row = dict(test="M1M2", spare=spare, seed=seed, rep=rep,
                               arm="new" if arm else "old", total_s=el, layout_s=lay_t,
                               phase=info.get("phase"), found=info.get("found"),
                               twoq=twoq_count(out), pair_applicable="", pair_worst="")
                    if rep == 0:
                        applicable, worst = pair_check(qc, out, n_phys - spare)
                        row["pair_applicable"], row["pair_worst"] = applicable, worst
                        if arm:
                            ok = (info.get("phase") == 0 and applicable and worst is not None
                                  and worst <= 1e-12)
                            m1_ok &= ok
                            print(f"spare={spare} seed={seed} new: phase={info.get('phase')} "
                                  f"pair_check={applicable} worst={worst}", flush=True)
                    rows.append(row)
            if twoq[True] != twoq[False]:
                m1_ok = False
                print(f"spare={spare} seed={seed}: 2q differs old={twoq[False]} new={twoq[True]}")
        summary[spare] = (acc, m1_ok, twoq)

    print("\n--- summary (medians) ---")
    for spare, (acc, m1_ok, twoq) in summary.items():
        ot, nt = st.median(acc[False]["total"]), st.median(acc[True]["total"])
        ol, nl = st.median(acc[False]["lay"]), st.median(acc[True]["lay"])
        m2 = (nl <= 1.0e-3 and nt <= ot - 0.010) if spare == 0 else (nt <= ot + 0.001)
        print(f"spare={spare}: total old {ot * 1000:.2f} ms -> new {nt * 1000:.2f} ms | "
              f"layout old {ol * 1000:.3f} ms -> new {nl * 1000:.3f} ms | 2q old={twoq[False]} new={twoq[True]} | "
              f"M1 {'PASS' if m1_ok else 'FAIL'} | M2 {'PASS' if m2 else 'FAIL'}")

    # ---- M3
    print("\n=== M3: chain circuits (not a matching) ===")
    m3_ok = True
    for seed in range(CHAIN_SEEDS):
        qc = build_chain_circuit(CHAIN_QUBITS, GATES_PER_PAIR, seed)
        res = {}
        for arm in (False, True):
            out, el, lay_t, info = psf_call(qc, backend, native, arm)
            init = list(out.layout.initial_index_layout(filter_ancillas=True)) if out.layout else None
            res[arm] = (info.get("phase"), init, twoq_count(out))
            rows.append(dict(test="M3", spare="", seed=seed, rep=0, arm="new" if arm else "old",
                             total_s=el, layout_s=lay_t, phase=info.get("phase"), found=info.get("found"),
                             twoq=twoq_count(out), pair_applicable="", pair_worst=""))
        same = res[True] == res[False] and res[True][0] != 0
        m3_ok &= same
        print(f"seed={seed}: phase old={res[False][0]} new={res[True][0]} "
              f"same layout={res[True][1] == res[False][1]} 2q old={res[False][2]} new={res[True][2]} "
              f"-> {'PASS' if same else 'FAIL'}", flush=True)
    print(f"M3 {'PASS' if m3_ok else 'FAIL'}")

    # ---- X1
    print("\n=== X1 (exploratory): 2-qubit gate error on the edges used ===")
    twoq_name, err = edge_errors(backend)
    weights = {e: int(round(1e6 * (1.0 - v))) for e, v in err.items()}
    print(f"native 2q gate {twoq_name}; {len(err)} edges with an error value; "
          f"chip mean {np.mean(list(err.values())):.3e}")
    for spare in (0, 8):
        qc = build_dense_pair_blocks_circuit(n_phys - spare, GATES_PER_PAIR, 0)
        n_log = n_phys - spare
        pairs = [(i, i + 1) for i in range(0, n_log - 1, 2)]
        arms = {}
        arms["PSF old (VF2)"] = psf_call(qc, backend, native, False)[0]
        arms["PSF new (matching)"] = psf_call(qc, backend, native, True)[0]
        lm = psl.matching_layout(backend.coupling_map, pairs, edge_weights=weights)
        init = pc._layout_map_to_list(lm, n_log, n_phys)
        arms["PSF new (weighted matching)"] = psf_call(qc, backend, native, True, initial_layout=init)[0]
        t0 = time.perf_counter()
        arms["Qiskit L3"] = transpile(qc, backend, optimization_level=3, seed_transpiler=0)
        q_el = time.perf_counter() - t0
        for name, out in arms.items():
            n_used, mean_e, sum_e = used_edge_error(out, err)
            rows.append(dict(test="X1", spare=spare, seed=0, rep=0, arm=name,
                             twoq=twoq_count(out), edges_used=n_used, err_mean=mean_e, err_sum=sum_e))
            extra = f" (compile {q_el:.2f}s)" if name == "Qiskit L3" else ""
            print(f"spare={spare} {name:30s} edges={n_used:3d} mean err={(f'{mean_e:.3e}' if mean_e is not None else 'n/a')} "
                  f"sum={sum_e:.3e} 2q={twoq_count(out)}{extra}", flush=True)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        fields = ["test", "spare", "seed", "rep", "arm", "total_s", "layout_s", "phase", "found", "twoq",
                  "pair_applicable", "pair_worst", "edges_used", "err_mean", "err_sum"]
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
