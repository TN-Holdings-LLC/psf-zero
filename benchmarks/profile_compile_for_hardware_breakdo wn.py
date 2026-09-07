# profile_compile_for_hardware_breakdown.py
#
# Investigate why compile_for_hardware() is slower than standalone Qiskit transpile()
# (remaining 1.7-2.9x slower even with verify=False).
#
# Hypothesis: compile_for_hardware() sequentially runs two separate pipelines internally:
#   (a) compile()  -- Collect2qBlocks + ConsolidateBlocks + PSF synthesis
#   (b) transpile(qc_compressed, coupling_map=..., optimization_level=...)
#         -- A secondary transpile call dedicated purely to routing.
# Since standalone Qiskit completes in a single transpile() call, it might not be that
# "PSF-side synthesis is slow," but rather that "running two separate pipelines causes overhead."
# We verify this through empirical measurement.
#
# Usage: Simply run it as-is. For each size in QUBIT_SIZES, after a warm-up,
# it separately measures and displays 4 metrics over REPS repetitions:
# (a) standalone, (b) standalone, (a)+(b) split-sum, and the actual compile_for_hardware() call.
# Multiprocessing is not used (since per-process cold-start overhead is already a solved
# issue, here we only care about the inner breakdown ratios).
import time
import math
import inspect
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary
from qiskit.transpiler import CouplingMap
from psf_compile import compile as psf_compile, compile_for_hardware

QUBIT_SIZES = [50, 100, 156, 300, 500]
GATES_PER_PAIR = 20
REPS = 5
PSF_ROUTING_OPT_LEVEL = 1

_SUPPORTS_VERIFY = "verify" in inspect.signature(compile_for_hardware).parameters


def get_grid_cmap(num_qubits):
    cols = int(math.ceil(math.sqrt(num_qubits)))
    rows = int(math.ceil(num_qubits / cols))
    return CouplingMap.from_grid(rows, cols)


def build_dense_pair_blocks_circuit(num_qubits, gates_per_pair, seed):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    pairs = [(i, i + 1) for i in range(0, num_qubits - 1, 2)]
    for (a, b) in pairs:
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            qc.append(UnitaryGate(u), [a, b])
    return qc


def main():
    print(f"[breakdown] compile_for_hardware() supports verify: {_SUPPORTS_VERIFY}")
    compile_kwargs = {"verify": False} if _SUPPORTS_VERIFY else {}

    # warm-up: both compile() and a bare transpile(coupling_map=...) call,
    # so plugin-loading cost is paid once, outside every timed section below.
    warmup_cmap = get_grid_cmap(2)
    psf_compile(QuantumCircuit(2), **compile_kwargs)
    transpile(QuantumCircuit(2), coupling_map=warmup_cmap, optimization_level=PSF_ROUTING_OPT_LEVEL)
    transpile(QuantumCircuit(2), coupling_map=warmup_cmap, optimization_level=3)
    compile_for_hardware(QuantumCircuit(2), warmup_cmap,
                          routing_optimization_level=PSF_ROUTING_OPT_LEVEL, **compile_kwargs)

    print(f"{'Qubits':>7} | {'(a) compile()':>14} | {'(b) route-only':>15} | "
          f"{'(a)+(b) split-sum':>18} | {'compile_for_hardware()':>23} | {'Qiskit opt3':>12}")

    for q in QUBIT_SIZES:
        cmap = get_grid_cmap(q)
        a_times, b_times, cfh_times, qiskit_times = [], [], [], []

        for rep in range(REPS):
            seed = rep + 1
            qc = build_dense_pair_blocks_circuit(q, GATES_PER_PAIR, seed=seed)

            # (a) compile() alone
            t0 = time.perf_counter()
            qc_compressed = psf_compile(qc, **compile_kwargs)
            a_times.append(time.perf_counter() - t0)

            # (b) routing-only transpile(), fed the ALREADY-compressed circuit
            t0 = time.perf_counter()
            transpile(qc_compressed, coupling_map=cmap, optimization_level=PSF_ROUTING_OPT_LEVEL)
            b_times.append(time.perf_counter() - t0)

            # actual compile_for_hardware() call, for comparison against (a)+(b)
            t0 = time.perf_counter()
            compile_for_hardware(qc, cmap, routing_optimization_level=PSF_ROUTING_OPT_LEVEL,
                                  **compile_kwargs)
            cfh_times.append(time.perf_counter() - t0)

            # Qiskit baseline, same circuit
            t0 = time.perf_counter()
            transpile(qc, coupling_map=cmap, optimization_level=3)
            qiskit_times.append(time.perf_counter() - t0)

        a_mean = sum(a_times) / REPS
        b_mean = sum(b_times) / REPS
        cfh_mean = sum(cfh_times) / REPS
        q_mean = sum(qiskit_times) / REPS

        print(f"{q:>7} | {a_mean:>14.4f} | {b_mean:>15.4f} | {a_mean + b_mean:>18.4f} | "
              f"{cfh_mean:>23.4f} | {q_mean:>12.4f}")

    print(
        "\n[breakdown] Interpretation: (a) is PSF's own compression+synthesis cost, "
        "(b) is the SAME routing transpile() call PSF makes internally, run standalone "
        "on the already-compressed circuit. If (a)+(b) closely matches the measured "
        "compile_for_hardware() column, the 'two separate pass-manager pipelines' "
        "hypothesis is confirmed -- most of the gap vs. Qiskit is (b)'s routing "
        "overhead plus (a)'s own cost added on top, rather than a third hidden cost. "
        "Compare (a) alone against the Qiskit column to see whether PSF's own "
        "compression step alone remains faster than Qiskit's full pipeline (as "
        "Section 4 found for compile() without routing) -- revealing whether "
        "the slowdown is genuinely concentrated in routing."
    )


if __name__ == "__main__":
    main()
