"""compare_cx_decomposer.py -- Addendum 115.

Tests whether configuring PSF-Zero's CX decomposer for the {rz, sx} basis
removes the extra `sx` pulses Addendum 114 found between each block's CXs.

psf_compile.py is NOT modified. For each variant the module-level
`_CX_DECOMPOSER` is replaced (it is looked up at call time at both use sites:
the Cartan-core path and the degenerate-block fallback), the decomposition
cache `_CX_CORE_CACHE` is cleared, and the original is restored at the end
even if something fails.

  current   : TwoQubitBasisDecomposer(CXGate())                      (as shipped)
  zsx       : TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")
  zsx_pulse : TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX", pulse_optimize=True)

Test 1: 300 isolated blocks (20 random SU(4) on one pair each -- above
compile()'s block floor of 12, so each is really synthesized): fidelity,
synthesis actually happened, sx count.
Test 2: the Addenda 109-114 variational ansatz with strategy D (layout once +
PSF-Zero synthesis), against Qiskit L3 re-compile (strategy B) as reference.

Usage:
    python compare_cx_decomposer.py
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
from qiskit.circuit.library import CXGate, UnitaryGate
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector, random_unitary
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition

import psf_compile
from psf_smart_layout import smart_vf2_layout

BASIS = ["rz", "sx", "x", "cx"]
ONE_Q = PassManager([Optimize1qGatesDecomposition(basis=BASIS)])
LAYERS = 6
K_SAMPLES = 5
SEED = 1234
N_BLOCKS = 300
GRIDS = [(4, 4, True), (6, 7, False)]
FAMILIES = ["same_pair", "brickwork"]

VARIANTS = {
    "current": lambda: TwoQubitBasisDecomposer(CXGate()),
    "zsx": lambda: TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX"),
    "zsx_pulse": lambda: TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX", pulse_optimize=True),
}


def set_decomposer(dec):
    psf_compile._CX_DECOMPOSER = dec
    psf_compile._CX_CORE_CACHE.clear()


def psf_synth(qc):
    with contextlib.redirect_stdout(io.StringIO()):
        out = psf_compile.compile(qc, verify=False, entangling_basis="cx")
    return ONE_Q.run(out)


def fidelity(u, v):
    d = u.shape[0]
    tr = np.trace(u.conj().T @ v)
    return float((abs(tr) ** 2 + d) / (d * (d + 1)))


def test_isolated_blocks():
    """Returns dict of results for the currently installed decomposer."""
    rng = np.random.default_rng(SEED)
    worst, raised, not_synth, sx_counts, cx_counts = 1.0, 0, 0, [], []
    for _ in range(N_BLOCKS):
        qc = QuantumCircuit(2)
        for _ in range(20):
            qc.append(UnitaryGate(random_unitary(4, seed=int(rng.integers(0, 2**31)))), [0, 1])
        try:
            out = psf_synth(qc)
        except Exception:
            raised += 1
            continue
        ops = out.count_ops()
        if "unitary" in ops or not (1 <= ops.get("cx", 0) <= 3):
            not_synth += 1
        worst = min(worst, fidelity(Operator(qc).data, Operator(out).data))
        sx_counts.append(ops.get("sx", 0) + ops.get("x", 0))
        cx_counts.append(ops.get("cx", 0))
    return dict(blocks=N_BLOCKS, raised=raised, not_synthesized=not_synth, worst_fidelity=worst,
                median_sx=statistics.median(sx_counts) if sx_counts else None,
                median_cx=statistics.median(cx_counts) if cx_counts else None)


def build_ansatz(n, family):
    """Identical to Addenda 109-114's construction."""
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


def composition(qc):
    ops = qc.count_ops()
    return dict(
        cx=ops.get("cx", 0),
        cx_depth=qc.depth(filter_function=lambda i: i.operation.num_qubits == 2),
        sx=ops.get("sx", 0) + ops.get("x", 0),
        sx_depth=qc.depth(filter_function=lambda i: i.operation.name in ("sx", "x")),
        total_depth=qc.depth(),
    )


def segments(qc):
    first = next(i for i in qc.data if i.operation.num_qubits == 2)
    pair = {qc.find_bit(q).index for q in first.qubits}
    segs, cur = [], 0
    for inst in qc.data:
        idx = {qc.find_bit(q).index for q in inst.qubits}
        if not idx <= pair:
            continue
        if inst.operation.num_qubits == 2:
            segs.append(cur); cur = 0
        elif inst.operation.name in ("sx", "x"):
            cur += 1
    segs.append(cur)
    return segs  # sx per segment


def expval_error(logical, compiled, n, perm):
    obs = SparsePauliOp.from_sparse_list([("Z", [i], 1.0 / n) for i in range(n)], num_qubits=n)
    mapped = obs.apply_layout(list(perm), num_qubits=compiled.num_qubits)
    return abs(Statevector(logical).expectation_value(obs).real
               - Statevector(compiled).expectation_value(mapped).real)


def test_ansatz(variant, rows):
    rng = np.random.default_rng(SEED)
    for r, c, do_sv in GRIDS:
        n = r * c
        cm = CouplingMap.from_grid(r, c)
        for family in FAMILIES:
            logical, theta, pairs = build_ansatz(n, family)
            binds = [dict(zip(theta, rng.uniform(-np.pi, np.pi, len(theta)))) for _ in range(K_SAMPLES + 1)]
            layout_map, _ = smart_vf2_layout(cm, pairs, n)
            perm = [layout_map[i] for i in range(n)]

            def run_d(b):
                placed = QuantumCircuit(n)
                placed.compose(psf_synth(logical.assign_parameters(b)), qubits=perm, inplace=True)
                return placed

            run_d(binds[0])  # warm-up
            comps, times, errs, segs = [], [], [], None
            for b in binds[1:]:
                t0 = time.perf_counter()
                out = run_d(b)
                times.append(time.perf_counter() - t0)
                comps.append(composition(out))
                if do_sv:
                    errs.append(expval_error(logical.assign_parameters(b), out, n, perm))
                if segs is None and family == "same_pair" and (r, c) == (4, 4):
                    segs = segments(out)
            med = {k: statistics.median(x[k] for x in comps) for k in comps[0]}
            row = dict(test="ansatz", variant=variant, grid=f"{r}x{c}", family=family, **med,
                       ms_per_circuit=statistics.median(times) * 1000,
                       max_expval_err=(f"{max(errs):.1e}" if errs else "n/a"),
                       sx_per_segment=(str(segs) if segs else ""))
            rows.append(row)
            print(f"  {r}x{c} {family:9s}: cx {med['cx']:4.0f} (depth {med['cx_depth']:.0f}) | "
                  f"sx {med['sx']:4.0f} (depth {med['sx_depth']:.0f}) | total depth {med['total_depth']:.0f} | "
                  f"{row['ms_per_circuit']:7.2f} ms | err {row['max_expval_err']}"
                  + (f" | sx per segment {segs}" if segs else ""))


def reference_b(rows):
    rng = np.random.default_rng(SEED)
    print("Reference -- strategy B (Qiskit L3 re-compile):")
    for r, c, _ in GRIDS:
        n = r * c
        cm = CouplingMap.from_grid(r, c)
        for family in FAMILIES:
            logical, theta, _ = build_ansatz(n, family)
            binds = [dict(zip(theta, rng.uniform(-np.pi, np.pi, len(theta)))) for _ in range(K_SAMPLES + 1)]
            comps = [composition(transpile(logical.assign_parameters(b), coupling_map=cm, basis_gates=BASIS,
                                           optimization_level=3, seed_transpiler=SEED))
                     for b in binds[1:]]
            med = {k: statistics.median(x[k] for x in comps) for k in comps[0]}
            rows.append(dict(test="ansatz", variant="qiskit_B", grid=f"{r}x{c}", family=family, **med))
            print(f"  {r}x{c} {family:9s}: cx {med['cx']:4.0f} (depth {med['cx_depth']:.0f}) | "
                  f"sx {med['sx']:4.0f} (depth {med['sx_depth']:.0f}) | total depth {med['total_depth']:.0f}")
    print()


def main():
    rows = []
    original = psf_compile._CX_DECOMPOSER
    try:
        reference_b(rows)
        for name, make in VARIANTS.items():
            print("=" * 100)
            print(f"Variant: {name}")
            print("=" * 100)
            try:
                dec = make()
            except Exception as exc:
                print(f"  CONSTRUCTION FAILED: {type(exc).__name__}: {exc}")
                rows.append(dict(test="construct", variant=name, error=f"{type(exc).__name__}: {exc}"))
                continue
            set_decomposer(dec)
            iso = test_isolated_blocks()
            print(f"  Test 1 (isolated blocks): raised {iso['raised']}/{iso['blocks']}, "
                  f"not synthesized {iso['not_synthesized']}, worst fidelity {iso['worst_fidelity']:.15f}, "
                  f"median sx {iso['median_sx']}, median cx {iso['median_cx']}")
            rows.append(dict(test="isolated", variant=name, **iso))
            print("  Test 2 (variational ansatz, strategy D):")
            test_ansatz(name, rows)
            print()
    finally:
        set_decomposer(original)
        print("Original _CX_DECOMPOSER restored; cache cleared.")

    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    out_path = "cx_decomposer_comparison_2026-09-21.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
