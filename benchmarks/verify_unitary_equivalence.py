"""verify_unitary_equivalence.py -- Addendum 103.

Addendum 101/102's own pre-registration was explicit: "This addendum
checks that the output respects the coupling map, not that it computes
the same thing." This script checks the thing that was left unchecked --
whether `compile_for_hardware(layout_search=True)` actually computes the
same unitary as the input, at a small enough scale (n=6, following this
project's own README precedent) for exact `Operator()` comparison.

## The one thing this MUST get right

Routing can move a logical qubit to a different physical position than
where it started. Comparing `Operator(input)` against `Operator(output)`
qubit-index-for-qubit-index would be WRONG whenever that happens -- a
perfectly correct circuit would then look incorrect.

Rather than rely on a Qiskit convenience method whose exact semantics
this session could not independently confirm from memory, the
permutation is applied via an explicit, hand-verified NumPy tensor
transpose (`permute_operator` below), matching Qiskit's own documented
little-endian convention. **This was verified against two independent
known cases before being used here** -- a single-qubit swap and a
3-qubit cyclic permutation, both checked against hand-built expected
operators using `numpy.kron` directly, with zero reliance on Qiskit for
the verification itself. See the accompanying note in this project's
own addendum for the verification transcript.

Usage:
    python verify_unitary_equivalence.py
"""
from __future__ import annotations

import contextlib
import csv
import io

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator, random_unitary
from qiskit.transpiler import CouplingMap

import psf_compile

BASIS_GATES = ["rz", "sx", "x", "cx"]
GATES_PER_PAIR = 20
SEEDS = [0, 1, 2]
REPEATS = 2
N_QUBITS = 6

# Both verified numerically before writing this script: 3 disjoint pairs
# (matching the 2x3 grid's own maximum matching of 3), and the smallest
# non-degenerate chain-shaped construction this project's own
# `edges_chain_shaped` logic can produce (1 bare edge + one 4-qubit chain,
# 4 edges, zero idle qubits, max matching 3).
FAMILIES = {
    "dense_pairs": [(0, 1), (2, 3), (4, 5)],
    "chain_shaped": [(0, 1), (2, 3), (3, 4), (4, 5)],
}


def build_circuit(edges, seed):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(N_QUBITS)
    for (a, b) in edges:
        for _ in range(GATES_PER_PAIR):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            qc.append(UnitaryGate(u), [a, b])
    return qc


def permute_operator(U, perm):
    """Relabels an n-qubit operator's own qubit ordering.

    `perm[i]` = which axis (qubit position) of `U` corresponds to logical
    qubit `i`. Returns a new 2^n x 2^n operator in logical-qubit order.

    Verified before use (see module docstring) against hand-built
    `numpy.kron` examples: a single-qubit swap and a 3-qubit cyclic
    permutation, both matching exactly.
    """
    n = int(np.log2(U.shape[0]))
    T = U.reshape((2,) * n + (2,) * n)
    # Qiskit's little-endian convention: axis (n-1-q) of the reshaped
    # tensor's first half corresponds to qubit q's output index; axis
    # (2n-1-q) of the second half corresponds to qubit q's input index.
    axes = [0] * (2 * n)
    for i in range(n):
        axes[n - 1 - i] = n - 1 - perm[i]
        axes[2 * n - 1 - i] = 2 * n - 1 - perm[i]
    T2 = np.transpose(T, axes)
    return T2.reshape(2 ** n, 2 ** n)


def self_test_permute_operator():
    """Re-run at import time, not just once during development, so a
    future edit to `permute_operator` cannot silently break correctness
    without this script itself catching it."""
    I2, X = np.eye(2), np.array([[0, 1], [1, 0]])
    u_x_q0 = np.kron(I2, X)
    u_x_q1 = np.kron(X, I2)
    assert np.allclose(permute_operator(u_x_q0, [1, 0]), u_x_q1), \
        "permute_operator self-test (2-qubit swap) FAILED"
    u3 = np.kron(np.kron(I2, I2), X)  # X on qubit 0 of 3
    expected3 = np.kron(np.kron(I2, X), I2)  # X moved to qubit 1
    assert np.allclose(permute_operator(u3, [2, 0, 1]), expected3), \
        "permute_operator self-test (3-qubit cyclic) FAILED"


def final_index_layout_or_none(routed, n_logical):
    """`perm[i]` = which qubit position in `routed` holds original logical
    qubit `i`. Tries Qiskit's own `TranspileLayout.final_index_layout()`
    first (the documented, purpose-built method for exactly this); falls
    back to the identity ONLY when no layout metadata exists at all
    (e.g. `transpile()` with no coupling map), which is logged rather
    than silently assumed.
    """
    layout = getattr(routed, "layout", None)
    if layout is None:
        return list(range(n_logical)), "no-layout-metadata (identity assumed)"
    if hasattr(layout, "final_index_layout"):
        try:
            perm = layout.final_index_layout(filter_ancillas=True)
            return list(perm), "final_index_layout(filter_ancillas=True)"
        except TypeError:
            perm = layout.final_index_layout()
            return list(perm), "final_index_layout()"
    raise RuntimeError(
        "This Qiskit version's TranspileLayout has no final_index_layout() "
        "-- refusing to guess a fallback for a correctness-critical "
        "permutation. Upgrade Qiskit or extend this function explicitly."
    )


def compute_fidelity(qc_in, routed):
    if routed.num_qubits != qc_in.num_qubits:
        return None, f"SKIPPED: routed has {routed.num_qubits} qubits, " \
                     f"input has {qc_in.num_qubits} (ancillas present)"
    u_in = Operator(qc_in).data
    perm, method = final_index_layout_or_none(routed, qc_in.num_qubits)
    u_out_raw = Operator(routed).data
    u_out = permute_operator(u_out_raw, perm)
    d = u_in.shape[0]
    tr = np.trace(u_in.conj().T @ u_out)
    fid = float((np.abs(tr) ** 2 + d) / (d * (d + 1)))
    return fid, method


def run_arm(arm, qc, cm):
    if arm == "qiskit_l3":
        return transpile(qc, coupling_map=cm, basis_gates=BASIS_GATES,
                         optimization_level=3, seed_transpiler=0)
    layout_search = (arm == "psf_ls_true")
    with contextlib.redirect_stdout(io.StringIO()):
        out = psf_compile.compile_for_hardware(
            qc, coupling_map=cm, basis_gates=BASIS_GATES,
            routing_optimization_level=1, entangling_basis="cx",
            verify=False, seed_transpiler=0, layout_search=layout_search,
        )
    return out


def main():
    self_test_permute_operator()
    print("self-test: permute_operator verified against 2 known cases -- OK\n")

    cm = CouplingMap.from_grid(2, 3)
    rows = []

    print("=" * 100)
    print("Addendum 103 -- exact unitary equivalence, n=6, layout permutation "
          "read from Qiskit's own TranspileLayout")
    print("=" * 100)
    print(f"{'family':>14} {'arm':>12} {'seed':>5} {'rep':>4} {'fidelity':>16} {'perm method':>38}")

    for family, edges in FAMILIES.items():
        for seed in SEEDS:
            qc = build_circuit(edges, seed)
            for rep in range(REPEATS):
                for arm in ("qiskit_l3", "psf_ls_false", "psf_ls_true"):
                    routed = run_arm(arm, qc, cm)
                    fid, method = compute_fidelity(qc, routed)
                    fid_str = f"{fid:.12f}" if fid is not None else "N/A"
                    print(f"{family:>14} {arm:>12} {seed:>5} {rep:>4} "
                          f"{fid_str:>16} {method:>38}")
                    rows.append(dict(family=family, arm=arm, seed=seed, rep=rep,
                                     fidelity=fid, perm_method=method))

    print("=" * 100)
    comparable = [r for r in rows if r["fidelity"] is not None]
    below = [r for r in comparable if r["fidelity"] < 1 - 1e-9]
    na = [r for r in rows if r["fidelity"] is None]
    print(f"P1/P2/P3: {len(comparable) - len(below)}/{len(comparable)} comparable runs "
          f"at or above 1-1e-9; {len(below)} below threshold; {len(na)} not comparable")
    if below:
        print("BELOW THRESHOLD (investigate before trusting the corresponding arm):")
        for r in below:
            print(f"  {r}")
    print("=" * 100)

    out_path = "unitary_equivalence_2026-09-20.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
