"""routing_pressure_arms.py -- pre-registered: where PSF-Zero's extra two-qubit
gates under routing come from, and whether resynthesizing AFTER routing
removes them (follow-up to the GPU whole-circuit equivalence check, Z5).

Implements exactly the design locked in
"Addendum (number TBD, workplace) -- Pre-registration: two-qubit count under
routing pressure -- compress-then-route versus route-then-resynthesize"
(2026-09-28).

Arms, per circuit:
  A  compile_for_hardware, routing_optimization_level=1 (default): compress, then route
  B  compile_for_hardware, routing_optimization_level=3
  C  route first (Qiskit transpile, optimization_level=1, no basis: SWAPs kept as
     gates), then compile_for_hardware on the routed circuit with a trivial layout
     and block_gate_floor=1, so PSF-Zero consolidates each SWAP with the blocks
     next to it
  D  Qiskit transpile, optimization_level=3 (reference)
Correctness at n = 20 by the mirror test on a CPU statevector (lightning.qubit).

    python -u routing_pressure_arms.py run   2>&1 | tee ~/routing_arms_run.txt
    python -u routing_pressure_arms.py score 2>&1 | tee ~/routing_arms_score.txt
Plumbing dry run only: --sizes 12,16 --check-n 12
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import statistics as st
import sys
import time
import warnings

REPO = os.path.expanduser("~/psf-zero")
sys.path[:0] = [REPO, os.path.join(REPO, "benchmarks")]
HOME = os.path.expanduser("~")
SIZES = [20, 24, 26]
CHECK_N = 20
GRIDS = {12: (3, 4), 16: (4, 4), 20: (4, 5), 24: (4, 6), 26: (2, 13)}
INSTANCES = 5
FAMILIES = ["F1_brick_line", "F2_brick_grid", "F3_random_pairs_grid"]
ARMS = ["A_psf_route1", "B_psf_route3", "C_route_then_psf", "D_qiskit_L3"]
BASIS = ["cx", "rz", "sx", "x"]
OUT_JSON = os.path.join(HOME, "routing_arms_2026-09-28.json")
TOL = 1e-10


def brick_pairs(n):
    return [(i, i + 1) for i in range(0, n - 1, 2)] + [(i, i + 1) for i in range(1, n - 1, 2)]


def build(family, n, seed):
    """Same families as gpu_whole_circuit_equivalence.py."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import random_unitary
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.u(*rng.uniform(-np.pi, np.pi, 3), q)
    if family in ("F1_brick_line", "F2_brick_grid"):
        for (a, b) in brick_pairs(n):
            p = rng.uniform(-np.pi, np.pi, 15)
            qc.rz(p[0], a); qc.ry(p[1], a); qc.rz(p[2], a)
            qc.rz(p[3], b); qc.ry(p[4], b); qc.rz(p[5], b)
            qc.rxx(p[6], a, b); qc.ryy(p[7], a, b); qc.rzz(p[8], a, b)
            qc.rz(p[9], a); qc.ry(p[10], a); qc.rz(p[11], a)
            qc.rz(p[12], b); qc.ry(p[13], b); qc.rz(p[14], b)
    else:
        for _ in range(3):
            perm = rng.permutation(n)
            for k in range(0, n - 1, 2):
                qc.unitary(random_unitary(4, seed=int(rng.integers(0, 2**31))), [int(perm[k]), int(perm[k + 1])])
    return qc


def final_positions(out, n):
    lay = out.layout
    if lay is None:
        return list(range(n))
    return list(lay.final_index_layout(filter_ancillas=True))


def ops_of(qc, wire_of=None):
    """Qiskit -> PennyLane op list; 2-qubit `unitary` on qargs [a, b] becomes
    QubitUnitary on wires [b, a] (Qiskit is little-endian)."""
    import numpy as np
    from qiskit.quantum_info import Operator
    wmap = (lambda q: q) if wire_of is None else (lambda q: wire_of[q])
    out = []
    names = {"cx": "CNOT", "sx": "SX", "x": "PauliX", "rz": "RZ", "ry": "RY", "rx": "RX", "u": "U3",
             "rxx": "IsingXX", "ryy": "IsingYY", "rzz": "IsingZZ"}
    for inst in qc.data:
        op = inst.operation
        q = [wmap(qc.find_bit(x).index) for x in inst.qubits]
        if op.name == "unitary":
            out.append(("QubitUnitary", q[::-1], np.asarray(Operator(op).data)))
        elif op.name in ("barrier", "global_phase"):
            continue
        elif op.name in names:
            out.append((names[op.name], q, [float(v) for v in op.params] or None))
        else:
            raise RuntimeError("unexpected gate " + op.name)
    return out


def run(sizes, check_n):
    import numpy as np
    import pennylane as qml
    import qiskit
    import psf_compile as pc
    import psf_zero_core as core
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap

    env = {"platform": platform.platform(), "python": platform.python_version(), "qiskit": qiskit.__version__,
           "pennylane": qml.__version__, "numpy": np.__version__, "psf_compile_version": pc.VERSION,
           "core_version": getattr(core, "CORE_VERSION", None), "sizes": sizes, "check_n": check_n}
    print("LOADED", pc.__file__, pc.VERSION)
    print("CORE", core.__file__, "CORE_VERSION", env["core_version"])
    print("ENV", json.dumps(env))
    if pc.VERSION != "2026-09-27.7":
        print("V0 FAILED: psf_compile.py is not 2026-09-27.7. Stopping.")
        return
    G = {"CNOT": qml.CNOT, "RZ": qml.RZ, "RY": qml.RY, "RX": qml.RX, "SX": qml.SX, "PauliX": qml.PauliX,
         "U3": qml.U3, "IsingXX": qml.IsingXX, "IsingYY": qml.IsingYY, "IsingZZ": qml.IsingZZ,
         "QubitUnitary": qml.QubitUnitary}

    def mirror(out, qc, n, pos):
        dev = qml.device("lightning.qubit", wires=out.num_qubits)
        ops = ops_of(out) + ops_of(qc.inverse(), wire_of=pos)

        @qml.qnode(dev, diff_method=None)
        def f():
            for name, w, p in ops:
                if p is None:
                    G[name](wires=w)
                elif name == "QubitUnitary":
                    G[name](p, wires=w)
                else:
                    G[name](*p, wires=w)
            return qml.state()
        a = np.asarray(f())
        return float(np.sqrt(np.sum(np.abs(a[1:]) ** 2)))

    def quiet(fn):
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            return fn()

    cells = []
    for n in sizes:
        for fam in FAMILIES:
            cmap = CouplingMap.from_line(n) if fam == "F1_brick_line" else CouplingMap.from_grid(*GRIDS[n])
            kw1 = {"initial_layout": list(range(n))} if fam == "F1_brick_line" else {}
            for k in range(INSTANCES):
                qc = build(fam, n, 20000 * n + 100 * FAMILIES.index(fam) + k)
                for arm in ARMS:
                    t0 = time.perf_counter()
                    if arm in ("A_psf_route1", "B_psf_route3"):
                        pc._CX_CORE_CACHE.clear()
                        out = quiet(lambda: pc.compile_for_hardware(
                            qc, coupling_map=cmap, basis_gates=BASIS, entangling_basis="cx", on_unsupported="keep",
                            seed_transpiler=0, block_gate_floor=8,
                            routing_optimization_level=1 if arm == "A_psf_route1" else 3, **kw1))
                        pos = final_positions(out, n)
                    elif arm == "C_route_then_psf":
                        routed = quiet(lambda: transpile(qc, coupling_map=cmap, optimization_level=1,
                                                         seed_transpiler=0, **kw1))
                        p1 = final_positions(routed, n)
                        m = routed.num_qubits
                        flat = routed.copy()
                        flat._layout = None  # compile the physical circuit as it stands
                        pc._CX_CORE_CACHE.clear()
                        # block_gate_floor=1: a routed SWAP next to a 2-gate remainder must still be merged;
                        # at the default floor such short blocks are left unconsolidated.
                        out = quiet(lambda: pc.compile_for_hardware(
                            flat, coupling_map=cmap, basis_gates=BASIS, entangling_basis="cx", on_unsupported="keep",
                            seed_transpiler=0, block_gate_floor=1, initial_layout=list(range(m))))
                        p2 = final_positions(out, m)
                        pos = [p2[p1[i]] for i in range(n)]
                    else:
                        out = quiet(lambda: transpile(qc, coupling_map=cmap, basis_gates=BASIS,
                                                      optimization_level=3, seed_transpiler=0))
                        pos = final_positions(out, n)
                    tc = time.perf_counter() - t0
                    edges = set(map(tuple, cmap.get_edges()))
                    legal = all(tuple(out.find_bit(x).index for x in i.qubits) in edges
                                for i in out.data if len(i.qubits) == 2)
                    cell = {"n": n, "family": fam, "instance": k, "arm": arm, "cx": out.count_ops().get("cx", 0),
                            "ops": sum(out.count_ops().values()), "depth": out.depth(), "compile_s": tc,
                            "legal": legal, "only_basis": set(out.count_ops()) <= set(BASIS) | {"barrier"},
                            "error": mirror(out, qc, n, pos) if n == check_n else None}
                    cells.append(cell)
                    print("cell", json.dumps(cell))
        with open(OUT_JSON, "w") as fh:
            json.dump({"env": env, "cells": cells}, fh, indent=1)
    print("wrote", OUT_JSON)


def score():
    R = json.load(open(OUT_JSON)); cells = R["cells"]; env = R["env"]

    def v(ok, bad):
        return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")

    def get(n, fam, k, arm):
        return next(c for c in cells if c["n"] == n and c["family"] == fam and c["instance"] == k and c["arm"] == arm)
    print("=" * 78); print("SCORING (thresholds exactly as pre-registered)"); print("=" * 78)
    print("ENV", json.dumps(env))
    ok0 = (env["sizes"] == SIZES and env["check_n"] == CHECK_N and len(cells) == len(SIZES) * 3 * INSTANCES * 4
           and all(c["legal"] and c["only_basis"] for c in cells))
    print(f"C0 harness: sizes {env['sizes']}, cells {len(cells)}, all outputs on coupling-map edges and in basis:"
          f" {all(c['legal'] and c['only_basis'] for c in cells)}")
    if not ok0:
        print("C0 FAILED: nothing is scored.")
        return
    chk = [c for c in cells if c["error"] is not None]
    psf_chk = [c for c in chk if c["arm"] in ("A_psf_route1", "C_route_then_psf")]
    we = max(c["error"] for c in psf_chk)
    print(f"R1 every checked output of the PSF-Zero-synthesized arms A and C is exact (n={CHECK_N}, {len(psf_chk)} outputs):"
          f" worst {we:.2e} (<= {TOL:g}) ->", v(we <= TOL, we > TOL))
    ratio = lambda a, b, fam: [get(n, fam, k, a)["cx"] / get(n, fam, k, b)["cx"] for n in SIZES for k in range(INSTANCES)]
    F3 = "F3_random_pairs_grid"
    mBD = st.mean(ratio("B_psf_route3", "D_qiskit_L3", F3))
    print(f"R2 compress then route at level 3 is not worse than Qiskit L3 on F3 CX: mean B/D = {mBD:.3f}"
          " (confirmed <= 1.02, refuted >= 1.05) ->", "CONFIRMED" if mBD <= 1.02 else ("REFUTED" if mBD >= 1.05 else "AMBIGUOUS"))
    mCA = st.mean(ratio("C_route_then_psf", "A_psf_route1", F3))
    print(f"R3 route-then-resynthesize removes CX versus the default on F3: mean C/A = {mCA:.3f}"
          " (confirmed <= 0.95, refuted >= 1.00) ->", "CONFIRMED" if mCA <= 0.95 else ("REFUTED" if mCA >= 1.0 else "AMBIGUOUS"))
    mCD = st.mean(ratio("C_route_then_psf", "D_qiskit_L3", F3))
    print(f"R4 route-then-resynthesize comes close to Qiskit L3 on F3 CX: mean C/D = {mCD:.3f}"
          " (confirmed <= 1.08, refuted > 1.12) ->", "CONFIRMED" if mCD <= 1.08 else ("REFUTED" if mCD > 1.12 else "AMBIGUOUS"))
    reg = [get(n, f, k, "C_route_then_psf")["cx"] <= get(n, f, k, "A_psf_route1")["cx"]
           for n in SIZES for f in ("F1_brick_line", "F2_brick_grid") for k in range(INSTANCES)]
    print(f"R5 no regression where routing is light (F1, F2): C <= A in CX in {sum(reg)} of {len(reg)} ->",
          v(all(reg), not all(reg)))
    for arm in ("B_psf_route3", "D_qiskit_L3"):
        xs = [c["error"] for c in chk if c["arm"] == arm]
        print(f"   reported: {arm} checked outputs above {TOL:g}: {sum(x > TOL for x in xs)} of {len(xs)}, worst {max(xs):.2e}")
    eqBD = sum(get(n, F3, k, "B_psf_route3")["cx"] == get(n, F3, k, "D_qiskit_L3")["cx"] for n in SIZES for k in range(INSTANCES))
    print(f"   reported: F3 cells where B and D have the same CX count: {eqBD} of {len(SIZES) * INSTANCES}")
    print("\nReported without prediction (workplace sandbox times; not comparable with home or pod):")
    for n in SIZES:
        for fam in FAMILIES:
            row = []
            for arm in ARMS:
                xs = [c for c in cells if c["n"] == n and c["family"] == fam and c["arm"] == arm]
                row.append(f"{arm.split('_')[0]} cx {st.mean(c['cx'] for c in xs):.1f} d {st.mean(c['depth'] for c in xs):.0f}"
                           f" t {st.median(c['compile_s'] for c in xs)*1e3:.0f}ms")
            print(f"  n={n} {fam:21s} " + " | ".join(row))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "score"])
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    ap.add_argument("--check-n", type=int, default=CHECK_N)
    a = ap.parse_args()
    run([int(s) for s in a.sizes.split(",")], a.check_n) if a.mode == "run" else score()
