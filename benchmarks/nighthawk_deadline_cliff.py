"""nighthawk_deadline_cliff.py -- Addendum 178.

Does the layout cliff appear on FakeNighthawk (IBM's square-lattice
generation), and which compiler meets a 1-second deadline?

Stage 0: topology prerequisite (perfect matching). Stops if absent.
Then, for spare in {0, 2, 4, 8} and 5 seeds: Q3 (Qiskit opt level 3) and
P (PSF-Zero compile_for_hardware with layout_search=True), each compile in
a fresh child process, timed inside the child, killed at 180 s (DNF).
Quality: routed 2-qubit count and, when no SWAP connects different pairs,
an exact per-pair 4x4 check.

Linux only (fork). Usage (WSL or RunPod, repository root importable):
    python -u nighthawk_deadline_cliff.py 2>&1 | tee nighthawk_result.txt
"""
from __future__ import annotations

import contextlib
import csv
import io
import multiprocessing as mp
import platform
import statistics as st
import time

import numpy as np
import qiskit
import rustworkx as rx
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_ibm_runtime.fake_provider import FakeNighthawk

from psf_compile import compile_for_hardware

SPARES = (0, 2, 4, 8)
SEEDS = 5
GATES_PER_PAIR = 20
CAP_S = 180.0
DEADLINES = (0.1, 1.0, 10.0)
NATIVE = ("cz", "ecr", "cx", "rz", "sx", "x", "id", "rzz")
OUT_CSV = "nighthawk_deadline_cliff_2026-09-25.csv"


def build_dense_pair_blocks_circuit(num_qubits, gates_per_pair, seed):
    """Verbatim from bench_cliff_1v1.py (the generator every cliff script in
    this project uses)."""
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


def stage0(backend):
    cm = backend.coupling_map
    n = cm.size()
    edges = {tuple(sorted(e)) for e in cm.get_edges()}
    g = rx.PyGraph()
    g.add_nodes_from(range(n))
    g.add_edges_from_no_data(list(edges))
    deg = sorted(g.degree(i) for i in range(n))
    bip = rx.two_color(g)
    parts = None if bip is None else (sum(1 for v in bip.values() if v == 0), sum(1 for v in bip.values() if v == 1))
    matching = rx.max_weight_matching(g, max_cardinality=True)
    print(f"Stage 0: qubits={n}, edges={len(edges)}, degree min/median/max="
          f"{deg[0]}/{st.median(deg)}/{deg[-1]}, bipartite parts={parts}, "
          f"max matching={len(matching)} pairs (perfect needs {n // 2})", flush=True)
    return n, len(matching) * 2 == n


def pair_check(qc_logical, qc_out, n_logical):
    """Exact per-pair check. Returns (applicable, worst_infidelity)."""
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
                continue  # an idle ancilla's single-qubit op
            return False, None
        if len(ks) != 1:
            return False, None  # a 2-qubit gate connects different pairs
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


def _worker(arm, spare, seed, q):
    backend = FakeNighthawk()
    n_logical = backend.coupling_map.size() - spare
    qc = build_dense_pair_blocks_circuit(n_logical, GATES_PER_PAIR, seed)
    native = [g for g in backend.operation_names if g in NATIVE]
    t0 = time.perf_counter()
    if arm == "Q3":
        out = transpile(qc, backend, optimization_level=3, seed_transpiler=0)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            out = compile_for_hardware(qc, coupling_map=backend.coupling_map, basis_gates=native,
                                       entangling_basis="cx", layout_search=True,
                                       on_unsupported="raise", seed_transpiler=0)
    elapsed = time.perf_counter() - t0
    twoq = sum(1 for i in out.data if len(i.qubits) == 2 and i.operation.name != "barrier")
    applicable, worst = pair_check(qc, out, n_logical)
    q.put(dict(elapsed_s=elapsed, twoq=twoq, pair_check_applicable=applicable, pair_worst_infid=worst))


def run_one(arm, spare, seed):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(arm, spare, seed, q))
    p.start()
    p.join(CAP_S)
    if p.is_alive():
        p.terminate()
        p.join()
        return dict(elapsed_s=None, twoq=None, pair_check_applicable=None, pair_worst_infid=None, status="DNF")
    if p.exitcode != 0 or q.empty():
        return dict(elapsed_s=None, twoq=None, pair_check_applicable=None, pair_worst_infid=None,
                    status=f"ERROR(exit {p.exitcode})")
    r = q.get()
    r["status"] = "OK"
    return r


def main():
    print(f"platform {platform.platform()} | python {platform.python_version()} | qiskit {qiskit.__version__}")
    backend = FakeNighthawk()
    n, perfect = stage0(backend)
    if not perfect:
        print("N0 FAILED: no perfect matching -- spare = 0 cannot arise on this device. Stopping.")
        return
    rows = []
    for spare in SPARES:
        for seed in range(SEEDS):
            for arm in ("Q3", "P"):
                r = run_one(arm, spare, seed)
                row = dict(spare=spare, logical_qubits=n - spare, seed=seed, arm=arm, **r)
                for d in DEADLINES:
                    row[f"within_{d}s"] = r["status"] == "OK" and r["elapsed_s"] <= d
                rows.append(row)
                el = "DNF" if r["elapsed_s"] is None else f"{r['elapsed_s']:.3f}s"
                print(f"spare={spare} seed={seed} {arm:2s} {r['status']:5s} {el:>9s} 2q={r['twoq']} "
                      f"pair_check={r['pair_check_applicable']} worst={r['pair_worst_infid']}", flush=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
