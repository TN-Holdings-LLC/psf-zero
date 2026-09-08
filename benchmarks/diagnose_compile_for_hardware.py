"""
diagnose_compile_for_hardware.py

Follow-up to diagnose_native_gate_inflation.py, after actually finding
`compile_for_hardware()` in the real `psf_compile.py` (lines 381-399):

    def compile_for_hardware(qc, coupling_map, block_gate_floor=...,
                              routing_optimization_level: int = 0):
        qc_compressed = compile(qc, block_gate_floor=block_gate_floor)
        return transpile(
            qc_compressed,
            coupling_map=coupling_map,
            optimization_level=routing_optimization_level,
        )

This tests a question the previous script's result raised but didn't
answer: does THIS specific transpile call -- coupling_map given, but no
basis_gates and no backend -- actually decompose the RXX/RYY/RZZ gates that
compile() emits, the way transpile(..., backend=...) does?

RESULT: no. Confirmed below -- at every routing_optimization_level (0-3),
RXX/RYY/RZZ pass straight through completely untouched; only routing
(SWAP insertion at level 0) or 1-qubit gate consolidation (level >=1)
happens. Qiskit only decomposes to a target gate set when one is given
(via `basis_gates=` or `backend=`) -- `coupling_map` alone only triggers
layout + routing.

WHAT THIS MEANS
---------------
`compile_for_hardware()`'s own output is *not* real-hardware-submittable as
written -- it still contains RXX/RYY/RZZ, and IBM Runtime's SamplerV2
rejects non-ISA circuits. So wherever `compile_for_hardware()`'s result
actually gets submitted to a real backend, there MUST be one more,
currently unseen, transpile-to-ISA step first (this is not in any file we
have). That still-missing step -- not `compile_for_hardware()` itself -- is
where diagnose_native_gate_inflation.py's measured 2x native-gate penalty
would apply, if that step uses a low optimization_level.

Notably, `compile_for_hardware()`'s own doc comment gives the exact
reasoning that would lead someone to pick a low level there too:
"`routing_optimization_level` defaults to 0 (routing only) since compile()
already did the 2-qubit optimization that a higher optimization_level
would otherwise redo." That's a true statement about LOGICAL 2-qubit gate
count. diagnose_native_gate_inflation.py shows it is false about PHYSICAL
native-gate count once a real basis (ECR, in `fake_sherbrooke`'s case) has
to be targeted -- a higher level doesn't "redo" work there, it does work
that was never done. This is circumstantial, not a confirmed diagnosis of
the real-hardware harness -- see the README's Roadmap for the one
remaining check.

USAGE
-----
    pip install qiskit
    python diagnose_compile_for_hardware.py
"""

from __future__ import annotations

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import random_unitary
from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition
from qiskit.transpiler import CouplingMap


def main() -> None:
    # A 4-qubit line: qubits 0 and 2 are not adjacent, so placing a 2-qubit
    # block there forces routing (a SWAP) regardless of optimization level.
    coupling_map = CouplingMap.from_line(4)

    u = random_unitary(4, seed=7).data
    block = TwoQubitWeylDecomposition(u).circuit(simplify=True)

    qc = QuantumCircuit(4)
    qc.compose(block, [0, 2], inplace=True)

    print("Before transpile:", dict(qc.count_ops()))
    print()

    for level in (0, 1, 2, 3):
        # This is *exactly* compile_for_hardware()'s own transpile call
        # signature: coupling_map given, no basis_gates, no backend.
        out = transpile(qc, coupling_map=coupling_map, optimization_level=level)
        ops = dict(out.count_ops())
        still_has_weyl_gates = any(g in ops for g in ("rxx", "ryy", "rzz"))
        print(
            f"routing_optimization_level={level}: {ops}\n"
            f"  -> RXX/RYY/RZZ still present (i.e. NOT ISA-decomposed): "
            f"{still_has_weyl_gates}"
        )


if __name__ == "__main__":
    main()
