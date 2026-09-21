"""verify_vqa_layout_once.py -- Addendum 111.

Adds strategy D to Addendum 109/110's variational-loop comparison:

  A  compile once : transpile the PARAMETERIZED circuit once (Qiskit L3),
                    then only bind values each step
  B  Qiskit       : bind values, then Qiskit L3, every circuit
  C  PSF-Zero     : bind values, then compile_for_hardware(layout_search=True)
  D  layout once  : ONCE find a perfect layout with smart_vf2_layout; per
                    circuit, bind values, psf_compile.compile (synthesis only),
                    place by the fixed layout, convert 1q gates to the basis.
                    No layout search and no routing per circuit.

One-time and steady-state per-iteration costs are reported in SEPARATE
columns (Addendum 110, Section 3, found the previous script adding the
one-time compile to every iteration). Per-iteration steady-state cost is
extrapolated as median per-circuit time x (2P + 1) and labelled as such.

Usage:
    python verify_vqa_layout_once.py
"""
from __future__ import annotations

import contextlib
import csv
import io
import statistics
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition

import psf_compile
from psf_smart_layout import smart_vf2_layout

BASIS = ["rz", "sx", "x", "cx"]
LAYERS = 6
K_SAMPLES = 5
SEED = 1234
GRIDS = [(4, 4, True), (6, 7, False)]  # (rows, cols, statevector check)
FAMILIES = ["same_pair", "brickwork"]


def build_ansatz(n: int, family: str):
    """One ry and one rz per qubit per layer, then CX on the family's pairs.
    Identical to Addendum 109's construction."""
    theta = ParameterVector("theta", 2 * n * LAYERS)
    qc = QuantumCircuit(n)
    pairs = set()
    k = 0
    for layer in range(LAYERS):
        for q in range(n):
            qc.ry(theta[k], q); k += 1
            qc.rz(theta[k], q); k += 1
        start = 0 if (family == "same_pair" or layer % 2 == 0) else 1
        for a in range(start, n - 1, 2):
            qc.cx(a, a + 1)
            pairs.add((a, a + 1))
    return qc, theta, sorted(pairs)


def count_2q(qc):
    return sum(1 for inst in qc.data if len(inst.qubits) == 2)


def coupling_violations(qc, cm):
    edges = [tuple(e) for e in cm.get_edges()]
    allowed = set(edges) | {e[::-1] for e in edges}
    bad = 0
    for inst in qc.data:
        if len(inst.qubits) == 2:
            i = qc.find_bit(inst.qubits[0]).index
            j = qc.find_bit(inst.qubits[1]).index
            if (i, j) not in allowed:
                bad += 1
    return bad


def in_basis(qc):
    return all(inst.operation.name in BASIS for inst in qc.data)


def observable(n):
    return SparsePauliOp.from_sparse_list([("Z", [i], 1.0 / n) for i in range(n)], num_qubits=n)


def expval_error(logical, compiled, n, perm=None):
    """|<O>_compiled - <O>_ideal|, O = (1/n) sum Z_i. `perm[i]` is where
    logical qubit i ends up: from TranspileLayout.final_index_layout() for
    A/B/C (verified in Addendum 103), or D's own fixed layout."""
    obs = observable(n)
    if perm is None:
        layout = getattr(compiled, "layout", None)
        if layout is None:
            return None
        perm = list(layout.final_index_layout(filter_ancillas=True))
    mapped = obs.apply_layout(list(perm), num_qubits=compiled.num_qubits)
    ideal = Statevector(logical).expectation_value(obs).real
    got = Statevector(compiled).expectation_value(mapped).real
    return abs(ideal - got)


def compile_psf_hardware(qc, cm):
    with contextlib.redirect_stdout(io.StringIO()):
        return psf_compile.compile_for_hardware(
            qc, coupling_map=cm, basis_gates=BASIS, routing_optimization_level=1,
            entangling_basis="cx", verify=False, seed_transpiler=SEED, layout_search=True,
        )


def make_strategy_d(logical, pairs, cm, n):
    """Returns (per-circuit function, one-time seconds, perm) or (None, t, None)
    if no perfect layout exists -- D is then reported as not applicable."""
    t0 = time.perf_counter()
    layout_map, _info = smart_vf2_layout(cm, pairs, n)
    one_q = PassManager([Optimize1qGatesDecomposition(basis=BASIS)])
    one_time = time.perf_counter() - t0
    if layout_map is None:
        return None, one_time, None
    perm = [layout_map[i] for i in range(n)]

    def run(bind):
        with contextlib.redirect_stdout(io.StringIO()):
            # entangling_basis must match strategy C's ("cx"); compile()'s own
            # default is "canonical", which would emit non-CX two-qubit gates.
            synth = psf_compile.compile(logical.assign_parameters(bind), verify=False,
                                        entangling_basis="cx")
        placed = QuantumCircuit(n)
        placed.compose(synth, qubits=perm, inplace=True)
        return one_q.run(placed)

    return run, one_time, perm


def main():
    rows_out = []
    rng = np.random.default_rng(SEED)

    for r, c, do_sv in GRIDS:
        n = r * c
        cm = CouplingMap.from_grid(r, c)
        for family in FAMILIES:
            logical, theta, pairs = build_ansatz(n, family)
            P = len(theta)
            per_iter_circuits = 2 * P + 1
            samples = [rng.uniform(-np.pi, np.pi, P) for _ in range(K_SAMPLES + 1)]  # [0] = warm-up
            binds = [dict(zip(theta, v)) for v in samples]

            print("=" * 108)
            print(f"{r}x{c} (n={n}) {family}: P={P}, {per_iter_circuits} parameter-shift circuits per iteration")
            print("=" * 108)

            t0 = time.perf_counter()
            tqc = transpile(logical, coupling_map=cm, basis_gates=BASIS, optimization_level=3,
                            seed_transpiler=SEED)
            a_once = time.perf_counter() - t0

            d_run, d_once, d_perm = make_strategy_d(logical, pairs, cm, n)

            strategies = [
                ("A_compile_once", lambda b: tqc.assign_parameters(b), a_once, None),
                ("B_qiskit_L3", lambda b: transpile(logical.assign_parameters(b), coupling_map=cm,
                                                    basis_gates=BASIS, optimization_level=3,
                                                    seed_transpiler=SEED), 0.0, None),
                ("C_psf_zero", lambda b: compile_psf_hardware(logical.assign_parameters(b), cm), 0.0, None),
                ("D_layout_once", d_run, d_once, d_perm),
            ]

            for name, fn, once, perm in strategies:
                if fn is None:
                    print(f"  {name:15s} NOT APPLICABLE -- no perfect layout found "
                          f"(search took {once:.3f} s)")
                    rows_out.append(dict(grid=f"{r}x{c}", family=family, strategy=name, n=n,
                                         params=P, circuits_per_iteration=per_iter_circuits,
                                         applicable=False, one_time_s=once))
                    continue
                fn(binds[0])  # warm-up, not timed
                times, twoq, depth, errs = [], [], [], []
                viol, basis_ok = 0, True
                for b in binds[1:]:
                    t0 = time.perf_counter()
                    out = fn(b)
                    times.append(time.perf_counter() - t0)
                    twoq.append(count_2q(out))
                    depth.append(out.depth())
                    viol += coupling_violations(out, cm)
                    basis_ok &= in_basis(out)
                    if do_sv:
                        errs.append(expval_error(logical.assign_parameters(b), out, n, perm))
                med = statistics.median(times)
                steady = med * per_iter_circuits
                if do_sv:
                    err_str = f"{max(errs):.1e}" if all(e is not None for e in errs) else "NO LAYOUT"
                else:
                    err_str = "n/a"
                print(f"  {name:15s} {med*1000:9.2f} ms/circuit | 2q {statistics.median(twoq):5.0f} | "
                      f"depth {statistics.median(depth):5.0f} | viol {viol} | basis {'ok' if basis_ok else 'NO'} | "
                      f"err {err_str}")
                print(f"  {'':15s} steady-state per iteration (EXTRAPOLATED): {steady:9.2f} s"
                      + (f"   one-time: {once:.3f} s" if once else ""))
                rows_out.append(dict(
                    grid=f"{r}x{c}", family=family, strategy=name, n=n, params=P,
                    circuits_per_iteration=per_iter_circuits, applicable=True, one_time_s=once,
                    per_circuit_median_ms=med * 1000, per_circuit_min_ms=min(times) * 1000,
                    per_circuit_max_ms=max(times) * 1000,
                    steady_per_iteration_s_extrapolated=steady,
                    median_2q=statistics.median(twoq), min_2q=min(twoq), max_2q=max(twoq),
                    median_depth=statistics.median(depth), coupling_violations=viol,
                    all_gates_in_basis=basis_ok, max_expval_err=err_str,
                ))
            print()

    fields = []
    for row in rows_out:
        for k in row:
            if k not in fields:
                fields.append(k)
    out_path = "vqa_layout_once_2026-09-21.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print(f"Wrote {out_path} ({len(rows_out)} rows)")
    print("Reminder: steady-state per-iteration figures are extrapolated; one-time costs are "
          "separate; no circuit execution time is included.")


if __name__ == "__main__":
    main()
