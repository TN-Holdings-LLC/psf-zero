"""
verify_compile_for_hardware_fix.py

Validates the fix proposed for `compile_for_hardware()` in `psf_compile.py`:
add a `basis_gates` parameter and pass it through to the internal
`transpile(...)` call, and default `routing_optimization_level` to 2+ so
that call actually resynthesizes to the target basis instead of only
routing.

    def compile_for_hardware(qc, coupling_map, basis_gates=None,
                              block_gate_floor=..., routing_optimization_level=2):
        qc_compressed = compile(qc, block_gate_floor=block_gate_floor)
        return transpile(qc_compressed, coupling_map=coupling_map,
                          basis_gates=basis_gates,
                          optimization_level=routing_optimization_level)

ONE CORRECTION TO THE PROPOSAL, CHECKED DIRECTLY: `transpile()`'s actual
default `optimization_level` (when left unspecified, i.e. `None`) is **2**
in the installed Qiskit version (2.5.2) -- confirmed from
`qiskit.compiler.transpiler.transpile`'s own source: "Take optimization
level from the configuration or 2 as default." It is not 1. This matters:
if the still-missing real-hardware script simply omitted
`optimization_level` entirely, it would have gotten level-2 behavior (no
inflation) in this Qiskit version -- not the level-1 behavior the fix
proposal assumed. For the 2x/4x inflation measured elsewhere in this
project's diagnostics to actually be the culprit, that missing script would
have needed to EXPLICITLY pass `optimization_level=0` or `=1` (or run an
older Qiskit version where the default was different) -- a more specific
claim than "relied on the default," and one that still can't be confirmed
without the actual file.

WHAT THIS SCRIPT CHECKS
------------------------
With `basis_gates` now supplied, does routing_optimization_level=2+ (a)
eliminate the gate-count inflation, and (b) preserve correctness (unitary
equivalence, accounting for the routing permutation via
`Operator.from_circuit`, which reads the transpiled circuit's `layout`)?

RESULT (N=50 random SU(4) unitaries, forced onto non-adjacent qubits on a
6-qubit line so every trial requires real routing/SWAP insertion; basis
`['ecr', 'rz', 'sx', 'x']`, matching `fake_sherbrooke`):

    level=0: correctness failures=0/50, mean ECR=12.00
    level=1: correctness failures=0/50, mean ECR=6.00
    level=2: correctness failures=0/50, mean ECR=3.00
    level=3: correctness failures=0/50, mean ECR=3.00

0 correctness failures at every level (the fix does not break anything),
and level >= 2 gives the same ECR count (3) as the unrouted case in
diagnose_native_gate_inflation.py -- confirming the fix is both correct and
effective. Levels 0-1 are actually WORSE here than in the unrouted
diagnostic (4x and 2x respectively, vs 2x and 2x there), because with
`basis_gates` now supplied, the routing SWAPs themselves also have to be
decomposed into the target basis, and low optimization levels don't do
that efficiently either.

USAGE
-----
    pip install qiskit
    python verify_compile_for_hardware_fix.py [--n 50]
"""

from __future__ import annotations

import argparse
import statistics as st

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, random_unitary
from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition
from qiskit.transpiler import CouplingMap


def compile_for_hardware_fixed(
    qc: QuantumCircuit,
    coupling_map: CouplingMap,
    basis_gates: list[str] | None = None,
    routing_optimization_level: int = 2,
) -> QuantumCircuit:
    """The proposed fix: `basis_gates` is now threaded through to the
    internal transpile call, so the target basis is actually known and
    RXX/RYY/RZZ (or anything else non-native) gets resynthesized rather
    than passed straight through. `qc` is assumed already PSF-compressed
    by `compile()` upstream, as in the original `compile_for_hardware()`.
    """
    return transpile(
        qc,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        optimization_level=routing_optimization_level,
    )


def run(n: int) -> None:
    coupling_map = CouplingMap.from_line(6)
    basis_gates = ["ecr", "rz", "sx", "x"]

    for level in (0, 1, 2, 3):
        ecr_counts = []
        fails = 0
        for i in range(n):
            u = random_unitary(4, seed=2000 + i).data
            block = TwoQubitWeylDecomposition(u).circuit(simplify=True)
            qc = QuantumCircuit(6)
            qc.compose(block, [0, 3], inplace=True)  # forces routing

            out = compile_for_hardware_fixed(
                qc, coupling_map, basis_gates=basis_gates,
                routing_optimization_level=level,
            )
            ecr_counts.append(out.count_ops().get("ecr", 0))

            # Operator.from_circuit reads the transpiled circuit's `layout`
            # and accounts for the routing permutation automatically -- a
            # naive Operator(out).equiv(Operator(qc)) gives false negatives
            # here because routing legitimately reorders physical qubits.
            if not Operator.from_circuit(out).equiv(Operator(qc)):
                fails += 1

        print(
            f"routing_optimization_level={level}: N={n}, "
            f"correctness failures={fails}, "
            f"mean ECR={st.mean(ecr_counts):.2f}, stdev={st.stdev(ecr_counts):.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    run(args.n)
