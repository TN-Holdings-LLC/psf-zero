"""analyze_depth_composition.py -- Addendum 113.

What is PSF-Zero's extra circuit depth made of, and where does it sit?

For strategies A (compile once), B (Qiskit L3 re-compile) and D (layout once
+ PSF-Zero synthesis; D's depth equals strategy C's per Addendum 112), reports
per circuit:
  - total depth (all gates, as reported in Addenda 110/112)
  - CX depth    (two-qubit gates only)
  - sx depth    (sx and x only -- the physical single-qubit pulses)
  - counts of cx, sx, x, rz
On IBM-style hardware rz is a virtual frame change, so CX depth and sx depth
are the noise-relevant quantities; total depth is shown for continuity only.

For 4x4 same_pair it also prints one pair's full gate sequence for B and D,
and the number of single-qubit gates in each gap between consecutive CXs.

Usage:
    python analyze_depth_composition.py
"""
from __future__ import annotations

import contextlib
import csv
import io
import statistics

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition

import psf_compile
from psf_smart_layout import smart_vf2_layout

BASIS = ["rz", "sx", "x", "cx"]
LAYERS = 6
K_SAMPLES = 5
SEED = 1234
GRIDS = [(4, 4), (6, 7)]
FAMILIES = ["same_pair", "brickwork"]


def build_ansatz(n, family):
    """Identical to Addenda 109-112's construction."""
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
        total_depth=qc.depth(),
        cx_depth=qc.depth(filter_function=lambda inst: inst.operation.num_qubits == 2),
        sx_depth=qc.depth(filter_function=lambda inst: inst.operation.name in ("sx", "x")),
        n_cx=ops.get("cx", 0), n_sx=ops.get("sx", 0), n_x=ops.get("x", 0), n_rz=ops.get("rz", 0),
    )


def pair_sequence(qc):
    """Gate sequence on the qubit pair of the first CX, in circuit order, and
    the single-qubit gate count in each segment delimited by that pair's CXs
    (segment 0 = before the first CX, last = after the last CX)."""
    first = next(inst for inst in qc.data if inst.operation.num_qubits == 2)
    pair = {qc.find_bit(q).index for q in first.qubits}
    seq, segments, current = [], [], 0
    for inst in qc.data:
        idx = [qc.find_bit(q).index for q in inst.qubits]
        if not set(idx) <= pair:
            continue
        if inst.operation.num_qubits == 2:
            seq.append("CX")
            segments.append(current)
            current = 0
        else:
            role = "a" if idx[0] == min(pair) else "b"
            seq.append(f"{inst.operation.name}({role})")
            current += 1
    segments.append(current)
    return seq, segments


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for r, c in GRIDS:
        n = r * c
        cm = CouplingMap.from_grid(r, c)
        for family in FAMILIES:
            logical, theta, pairs = build_ansatz(n, family)
            samples = [rng.uniform(-np.pi, np.pi, len(theta)) for _ in range(K_SAMPLES)]
            binds = [dict(zip(theta, v)) for v in samples]

            tqc = transpile(logical, coupling_map=cm, basis_gates=BASIS, optimization_level=3,
                            seed_transpiler=SEED)
            layout_map, _ = smart_vf2_layout(cm, pairs, n)
            perm = [layout_map[i] for i in range(n)]
            one_q = PassManager([Optimize1qGatesDecomposition(basis=BASIS)])

            def run_d(b):
                with contextlib.redirect_stdout(io.StringIO()):
                    synth = psf_compile.compile(logical.assign_parameters(b), verify=False,
                                                entangling_basis="cx")
                placed = QuantumCircuit(n)
                placed.compose(synth, qubits=perm, inplace=True)
                return one_q.run(placed)

            strategies = {
                "A_compile_once": lambda b: tqc.assign_parameters(b),
                "B_qiskit_L3": lambda b: transpile(logical.assign_parameters(b), coupling_map=cm,
                                                   basis_gates=BASIS, optimization_level=3,
                                                   seed_transpiler=SEED),
                "D_psf_synthesis": run_d,
            }

            print("=" * 104)
            print(f"{r}x{c} (n={n}) {family}  -- medians over {K_SAMPLES} samples")
            print(f"{'strategy':16s} {'total':>6s} {'CXdep':>6s} {'SXdep':>6s} "
                  f"{'#cx':>5s} {'#sx':>5s} {'#x':>4s} {'#rz':>5s}")
            outs = {}
            for name, fn in strategies.items():
                comps = []
                for b in binds:
                    out = fn(b)
                    comps.append(composition(out))
                    outs.setdefault(name, out)  # keep the first sample for the sequence view
                med = {k: statistics.median(x[k] for x in comps) for k in comps[0]}
                print(f"{name:16s} {med['total_depth']:6.0f} {med['cx_depth']:6.0f} {med['sx_depth']:6.0f} "
                      f"{med['n_cx']:5.0f} {med['n_sx']:5.0f} {med['n_x']:4.0f} {med['n_rz']:5.0f}")
                rows.append(dict(grid=f"{r}x{c}", family=family, strategy=name, **med))

            if (r, c) == (4, 4) and family == "same_pair":
                print("\n  One pair's full block, first sample (a/b = lower/higher physical qubit):")
                for name in ("B_qiskit_L3", "D_psf_synthesis"):
                    seq, segs = pair_sequence(outs[name])
                    print(f"  {name}: 1q gates per segment (before 1st CX, between CXs..., after last CX) = {segs}")
                    print(f"    {' '.join(seq)}")
            print()

    out_path = "depth_composition_2026-09-21.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
