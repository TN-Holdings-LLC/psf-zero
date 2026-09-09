"""
test_improved_compiler_end_to_end.py

Closes the loop between this project's two prior experiments:

  - verify_compile_for_hardware_fix.py showed the fix eliminates native-gate
    inflation for an ISOLATED `compile_for_hardware`-shaped transpile call
    on random 2-qubit unitaries.
  - experiment_fixed_compiler_fidelity.py showed a from-scratch PSF-style
    compiler reproduces section 8's fidelity gap and that applying the fix
    recovers most of it -- but that script called `transpile(...)` directly,
    not the real `compile_for_hardware()` function shape.

This script instead drives the REAL `compile()` + `compile_for_hardware()`
code from psf_compile_patched.py (see that file's own docstring for exactly
which parts are real code vs. a verified stand-in for the unusable Rust
core) end-to-end: build a circuit -> `compile_for_hardware_buggy()` (the
original, unfixed real function) vs. `compile_for_hardware()` (the same
real function, patched) -> ONE common naive final transpile, standing in
for the still-missing real-hardware submission step -> mirror-circuit
fidelity under a fake_sherbrooke noise model.

USAGE
-----
    pip install qiskit qiskit-ibm-runtime qiskit-aer
    python test_improved_compiler_end_to_end.py
"""

from __future__ import annotations

import statistics as st

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

from psf_compile_patched import compile_for_hardware, compile_for_hardware_buggy

SHOTS = 2048
BASIS_GATES = ["ecr", "rz", "sx", "x"]
FINAL_OPT_LEVEL = 1  # the naive final step, standing in for the still-missing
                      # real submission step (see README section 8)


def random_deep_block(num_qubits: int, pair: tuple[int, int], n_gates: int, seed: int) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for _ in range(n_gates):
        qc.u(*rng.uniform(0, 2 * np.pi, 3), pair[0])
        qc.u(*rng.uniform(0, 2 * np.pi, 3), pair[1])
        qc.cx(pair[0], pair[1])
    return qc


def build_family(name: str, seed: int) -> QuantumCircuit:
    if name == "deep2q":
        num_qubits, pairs, n_gates = 2, [(0, 1)], 15
    elif name == "multi_deep2q":
        num_qubits, pairs, n_gates = 8, [(0, 1), (2, 3), (4, 5), (6, 7)], 15
    elif name == "wide":
        num_qubits, pairs, n_gates = 12, [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)], 3
    else:
        raise ValueError(name)

    qc = QuantumCircuit(num_qubits)
    for i, pair in enumerate(pairs):
        block = random_deep_block(num_qubits, pair, n_gates, seed=1000 * (i + 1) + seed)
        qc.compose(block, inplace=True)
    return qc


def mirror_fidelity(compiled: QuantumCircuit, backend, sim) -> tuple[float, int]:
    mirror = compiled.copy()
    mirror.compose(compiled.inverse(), inplace=True)
    mirror.measure_all()

    final = transpile(mirror, backend=backend, optimization_level=FINAL_OPT_LEVEL, seed_transpiler=42)
    ecr_count = final.count_ops().get("ecr", 0)

    result = sim.run(final, shots=SHOTS).result()
    counts = result.get_counts()
    p_zero = sum(v for k, v in counts.items() if set(k.replace(" ", "")) <= {"0"}) / SHOTS
    return p_zero, ecr_count


def main() -> None:
    backend = FakeSherbrooke()
    sim = AerSimulator.from_backend(backend)

    families = {"deep2q": 5, "multi_deep2q": 5, "wide": 3}

    print(f"SHOTS={SHOTS}, backend={backend.name}, final naive optimization_level={FINAL_OPT_LEVEL}\n")

    for family, n_seeds in families.items():
        old_fids, new_fids = [], []
        old_ecrs, new_ecrs = [], []

        for seed in range(n_seeds):
            qc = build_family(family, seed)

            # A LINE coupling map sized to the circuit itself, not
            # fake_sherbrooke's full 127-node map: compile_for_hardware()'s
            # own transpile call pads/allocates ancilla to match whatever
            # coupling_map it's given, and if that were the full 127-node
            # graph, the mirror circuit built below would end up measuring
            # all 127 qubits -- defeating AerSimulator's idle-qubit
            # truncation and blowing past available memory on the later
            # noisy-simulation step. All of this family's own pairs are
            # already adjacent by construction, so a line map here changes
            # nothing about what's being tested (the basis-translation bug),
            # it only avoids an unrelated width explosion.
            coupling_map = CouplingMap.from_line(qc.num_qubits)

            old_out = compile_for_hardware_buggy(qc, coupling_map)
            new_out = compile_for_hardware(qc, coupling_map, basis_gates=BASIS_GATES)

            old_fid, old_ecr = mirror_fidelity(old_out, backend, sim)
            new_fid, new_ecr = mirror_fidelity(new_out, backend, sim)

            old_fids.append(old_fid); old_ecrs.append(old_ecr)
            new_fids.append(new_fid); new_ecrs.append(new_ecr)

        print(f"=== {family} (n_seeds={n_seeds}) ===")
        print(
            f"  compile_for_hardware_buggy (OLD): P(all-zero) mean={st.mean(old_fids):.4f} "
            f"stdev={st.stdev(old_fids):.4f}  |  final ecr mean={st.mean(old_ecrs):.2f}"
        )
        print(
            f"  compile_for_hardware       (NEW): P(all-zero) mean={st.mean(new_fids):.4f} "
            f"stdev={st.stdev(new_fids):.4f}  |  final ecr mean={st.mean(new_ecrs):.2f}"
        )
        print()


if __name__ == "__main__":
    main()
