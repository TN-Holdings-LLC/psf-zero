"""verify_block_floor.py -- Addendum 209: should the default block_gate_floor
of psf_compile.py change from 12 to 8?

Addendum 206: at floor 12 the second layer of a brick-layer ansatz is not
consolidated (48 two-qubit gates instead of 33); floor 8 fixes it with no
change on HEA, QAOA-line and cliff pair24. A default is a release decision,
so this checks a wider set of circuit families, including the kind the
floor was chosen for (wide-and-shallow random circuits), at floors 12, 10, 8
6 and 4 with psf_compile.py 2026-09-27.1.

Families (3 instances each unless stated):
  brick12   the Addendum 199 training ansatz, 12 qubits (line)
  heis12    Heisenberg-chain Trotter steps, 12 qubits, 4 steps (line)
  qaoa12    QAOA on a line, p = 3, 12 qubits (line)
  hea12     hardware-efficient ansatz, 4 layers, 12 qubits (line)
  dense12   dense pair blocks (60 gates per pair), 12 qubits (line)
  random8   qiskit random_circuit, 8 qubits, depth 12, up to 2-qubit gates (line; routed)
  qft8      QFT, 8 qubits (line; routed)
  cliff120  pair24 blocks, FakeNighthawk, 120 logical qubits (1 instance)

Per compile: two-qubit gate count, depth, compile time (median of 5), and
exactness: for the line families without routing, the state from |0> and
from a fixed random product state against the uncompiled circuit; for the
routed 8-qubit families, Operator.from_circuit (which accounts for layout
and final permutation) against Operator(input); for cliff120, the exact
per-pair check.

Usage (repository root; loop_endurance.py in benchmarks/):
    python -u benchmarks/verify_block_floor.py 2>&1 | tee verify_block_floor.txt
"""
from __future__ import annotations

import contextlib
import csv
import io
import math
import os
import platform
import statistics as st
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks"), os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler import CouplingMap

import loop_endurance as le
import psf_compile as pc

PC = time.perf_counter
FLOORS = (12, 10, 8, 6, 4)
BASIS = ["cx", "rz", "sx", "x"]
REPS = 5
OUT_CSV = "verify_block_floor_2026-09-27.csv"


# ------------------------------------------------------------ families

def brick12(rng):
    return le.e_circuit(rng.uniform(-np.pi, np.pi, le.E_NPARAMS))


def heis12(rng, steps=4):
    n = 12
    qc = QuantumCircuit(n)
    for _ in range(steps):
        for start in (0, 1):
            for a in range(start, n - 1, 2):
                t = rng.uniform(-np.pi, np.pi, 3)
                qc.rxx(t[0], a, a + 1)
                qc.ryy(t[1], a, a + 1)
                qc.rzz(t[2], a, a + 1)
        for q in range(n):
            qc.rz(rng.uniform(-np.pi, np.pi), q)
    return qc


def qaoa12(rng, p=3):
    n = 12
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(p):
        g, b = rng.uniform(-np.pi, np.pi, 2)
        for q in range(n - 1):
            qc.rzz(g, q, q + 1)
        for q in range(n):
            qc.rx(b, q)
    return qc


def hea12(rng, layers=4):
    n = 12
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            qc.ry(rng.uniform(-np.pi, np.pi), q)
            qc.rz(rng.uniform(-np.pi, np.pi), q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
    return qc


def dense12(rng, per_pair=60):
    n = 12
    qc = QuantumCircuit(n)
    for a in range(0, n - 1, 2):
        for _ in range(per_pair // 4):
            qc.rz(rng.uniform(-np.pi, np.pi), a)
            qc.ry(rng.uniform(-np.pi, np.pi), a + 1)
            qc.cx(a, a + 1)
            qc.rzz(rng.uniform(-np.pi, np.pi), a, a + 1)
    return qc


def random8(rng):
    return random_circuit(8, 12, max_operands=2, seed=int(rng.integers(1 << 30)))


def qft8(rng):
    try:
        from qiskit.circuit.library import QFTGate
        qc = QuantumCircuit(8)
        qc.append(QFTGate(8), range(8))
    except ImportError:
        from qiskit.circuit.library import QFT
        qc = QFT(8)
    return qc.decompose()


LINE_FAMILIES = (("brick12", brick12, 3, False), ("heis12", heis12, 3, False), ("qaoa12", qaoa12, 3, False),
                 ("hea12", hea12, 3, False), ("dense12", dense12, 3, False), ("random8", random8, 3, True),
                 ("qft8", qft8, 1, True))


# ------------------------------------------------------------ compile and check

def line_compile(qc, floor, routed):
    n = qc.num_qubits
    pc._CX_CORE_CACHE.clear()
    t0 = PC()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        out = pc.compile_for_hardware(qc, coupling_map=CouplingMap.from_line(n), basis_gates=BASIS,
                                      block_gate_floor=floor, entangling_basis="cx",
                                      initial_layout=list(range(n)), on_unsupported="keep", seed_transpiler=0)
    return out, PC() - t0


def exactness_line(qc, out, routed):
    """1 - fidelity against the uncompiled circuit (see module docstring)."""
    n = qc.num_qubits
    if routed:
        u, v = Operator(qc).data, Operator.from_circuit(out).data
        return 1.0 - abs(np.trace(u.conj().T @ v)) / u.shape[0]
    final = list(out.layout.final_index_layout()) if out.layout is not None else list(range(n))
    if final != list(range(n)):
        return math.inf
    rng = np.random.default_rng(99)
    prep = QuantumCircuit(n)
    for q in range(n):
        prep.ry(rng.uniform(0, np.pi), q)
        prep.rz(rng.uniform(-np.pi, np.pi), q)
    worst = 0.0
    for pre in (None, prep):
        a = qc if pre is None else pre.compose(qc)
        b = out if pre is None else pre.compose(out)
        worst = max(worst, 1.0 - abs(Statevector(a).inner(Statevector(b))) ** 2)
    return worst


def twoq(qc):
    return sum(1 for i in qc.data if len(i.qubits) == 2)


def blocks_count(qc, floor):
    """Number of unitary blocks compile() consolidates at this floor."""
    captured = []
    orig = pc.PassManager

    class Capturing(orig):
        def run(self_pm, *a, **k):
            out = super().run(*a, **k)
            captured.append(out)
            return out

    pc.PassManager = Capturing
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            pc.compile(qc, block_gate_floor=floor, entangling_basis="cx", on_unsupported="keep")
    finally:
        pc.PassManager = orig
    return sum(1 for i in captured[0].data if i.operation.name == "unitary")


# ------------------------------------------------------------ main

def main():
    print(f"platform {platform.platform()} | cores {os.cpu_count()} | python {platform.python_version()} "
          f"| qiskit {qiskit.__version__}")
    print("LOADED", pc.__file__, pc.VERSION, le.normalized_sha256(pc.__file__))
    print("SCRIPT", os.path.abspath(__file__), le.normalized_sha256(os.path.abspath(__file__)))
    if pc.VERSION != "2026-09-27.1":
        print("V0 FAILED: psf_compile.py is not 2026-09-27.1. Stopping.")
        return
    t_start = time.time()
    rows = []
    res = {}  # (family, instance, floor) -> dict

    for fam, make, count, routed in LINE_FAMILIES:
        rng = np.random.default_rng([ord(c) for c in fam])
        circuits = [make(rng) for _ in range(count)]
        line_compile(circuits[0], 12, routed)  # warm-up
        for i, qc in enumerate(circuits):
            for floor in FLOORS:
                outs = [line_compile(qc, floor, routed) for _ in range(REPS)]
                out = outs[0][0]
                r = dict(twoq=twoq(out), depth=out.depth(), time=st.median(o[1] for o in outs),
                         exact=exactness_line(qc, out, routed), blocks=blocks_count(qc, floor))
                res[(fam, i, floor)] = r
                rows.append(dict(family=fam, instance=i, floor=floor, **r))
        line = []
        for floor in FLOORS:
            rs = [res[(fam, i, floor)] for i in range(count)]
            line.append(f"floor {floor}: 2q {'/'.join(str(r['twoq']) for r in rs)}, depth "
                        f"{'/'.join(str(r['depth']) for r in rs)}, blocks {'/'.join(str(r['blocks']) for r in rs)}, "
                        f"{st.mean(r['time'] for r in rs) * 1000:.1f} ms, exact {max(r['exact'] for r in rs):.1e}")
        print(f"  {fam:8s}: " + "; ".join(line), flush=True)

    # cliff120
    backend, native = le.nighthawk()
    n = backend.coupling_map.size()
    rng = np.random.default_rng(55)
    cliff = QuantumCircuit(n)
    theta = rng.uniform(-np.pi, np.pi, (n // 2, 24))
    for k in range(n // 2):
        le.add_pair24(cliff, 2 * k, 2 * k + 1, theta[k])
    line = []
    for floor in FLOORS:
        ts, out = [], None
        for _ in range(REPS):
            pc._CX_CORE_CACHE.clear()
            t0 = PC()
            with contextlib.redirect_stdout(io.StringIO()):
                o = pc.compile_for_hardware(cliff, coupling_map=backend.coupling_map, basis_gates=native,
                                            block_gate_floor=floor, entangling_basis="cx", layout_search=True,
                                            on_unsupported="raise", seed_transpiler=0)
            ts.append(PC() - t0)
            out = out or o
        ok, w = le.pair_check(cliff, out, n)
        r = dict(twoq=twoq(out), depth=out.depth(), time=st.median(ts), exact=w if ok else math.inf,
                 blocks=blocks_count(cliff, floor))
        res[("cliff120", 0, floor)] = r
        rows.append(dict(family="cliff120", instance=0, floor=floor, **r))
        line.append(f"floor {floor}: 2q {r['twoq']}, depth {r['depth']}, blocks {r['blocks']}, "
                    f"{r['time'] * 1000:.1f} ms, exact {r['exact']:.1e}")
    print(f"  cliff120: " + "; ".join(line), flush=True)

    # ---------------- predictions (floor 8 against floor 12)
    print("\n=== predictions (floor 8 against floor 12) ===")

    def v(ok):
        return "CONFIRMED" if ok else "NOT CONFIRMED"

    keys = sorted({(f, i) for (f, i, _) in res})
    worse = [(f, i, res[(f, i, 12)]["twoq"], res[(f, i, 8)]["twoq"]) for f, i in keys
             if res[(f, i, 8)]["twoq"] > res[(f, i, 12)]["twoq"]]
    print(f"F1 no instance has more two-qubit gates at floor 8: {len(keys) - len(worse)}/{len(keys)} "
          f"{worse if worse else ''} -> {v(not worse)}")
    worst_exact = max(r["exact"] for r in res.values())
    print(f"F2 every compile at every floor exact (<= 1e-10): worst {worst_exact:.1e} -> {v(worst_exact <= 1e-10)}")
    fams = sorted({f for f, _ in keys})
    ratios = {f: st.mean(res[(f, i, 8)]["time"] for g, i in keys if g == f)
              / st.mean(res[(f, i, 12)]["time"] for g, i in keys if g == f) for f in fams}
    print(f"F3 compile time at floor 8 <= 1.5 x floor 12 in every family "
          f"({', '.join(f'{f} {x:.2f}' for f, x in ratios.items())}) -> {v(all(x <= 1.5 for x in ratios.values()))}")
    heis = [(res[("heis12", i, 12)]["twoq"], res[("heis12", i, 8)]["twoq"]) for i in range(3)]
    print(f"F4 heis12: two-qubit count identical at floors 12 and 8 in every instance ({heis}) -> "
          f"{v(all(b == a for a, b in heis))}")
    dratio = max(res[(f, i, 8)]["depth"] / max(res[(f, i, 12)]["depth"], 1) for f, i in keys)
    print(f"F5 depth at floor 8 <= 1.1 x floor 12 in every instance (worst ratio {dratio:.2f}) -> {v(dratio <= 1.1)}")

    fields = ["family", "instance", "floor", "twoq", "depth", "blocks", "time", "exact"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows); total wall time {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
