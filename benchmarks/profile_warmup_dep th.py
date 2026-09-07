# profile_warmup_depth.py
#
# Between phase3_v4.py (multiprocessing, warmed up exactly once per measurement)
# and profile_compile_for_hardware_breakdown.py (single process, averaging after
# dozens of calls), the winner between PSF-Zero and Qiskit flipped.
#
# Hypothesis: A single warm-up call is insufficient to reach a fully "warmed-up"
# steady state, and repeatedly calling the same function within the same process
# yields a gradual speedup ("deep warmup" effect). To directly verify this, we
# call the same function repeatedly for REPS times with the same circuit and settings,
# and display the time for each individual repetition (without averaging).
#
# Things to look for:
#   - Is the 1st repetition (the first production call immediately following warmup)
#     clearly slower than the 10th or 20th? -> If a decay curve exists, it provides
#     direct evidence that "a single warmup is not enough."
#   - Does this decay occur to a similar degree on the Qiskit side, or is it larger
#     on the PSF side? If a large decay is unique to the PSF side, phase3_v4.py's
#     result of "PSF is slower" is likely a measurement artifact. If it occurs
#     similarly on both sides, it is simply the general rule that "software runs
#     faster when called repeatedly," meaning phase3_v4.py's single-shot result
#     better corresponds to usage scenarios where a compilation is performed
#     truly only once.
import time
import math
import inspect
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary
from qiskit.transpiler import CouplingMap
from psf_compile import compile_for_hardware

QUBIT_SIZES_TO_TEST = [156, 500]
GATES_PER_PAIR = 20
REPS = 20  # Number of repeated calls after a single warmup
PSF_ROUTING_OPT_LEVEL = 1

_SUPPORTS_VERIFY = "verify" in inspect.signature(compile_for_hardware).parameters
_compile_kwargs = {"verify": False} if _SUPPORTS_VERIFY else {}


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
    print(f"[warmup_depth] compile_for_hardware() supports verify: {_SUPPORTS_VERIFY}")

    for q in QUBIT_SIZES_TO_TEST:
        cmap = get_grid_cmap(q)
        # Reuse the exact same circuit -- what we want to see is "change across invocation counts"
        # rather than "variation between circuits," so fixing the seed is sufficient.
        qc = build_dense_pair_blocks_circuit(q, GATES_PER_PAIR, seed=1)

        print(f"\n===== {q} qubits =====")

        # PSF-Zero: Warm up exactly once, then record time across REPS individual calls
        warmup_cmap = get_grid_cmap(2)
        compile_for_hardware(QuantumCircuit(2), warmup_cmap,
                              routing_optimization_level=PSF_ROUTING_OPT_LEVEL, **_compile_kwargs)
        psf_times = []
        for i in range(REPS):
            t0 = time.perf_counter()
            compile_for_hardware(qc, cmap, routing_optimization_level=PSF_ROUTING_OPT_LEVEL,
                                  **_compile_kwargs)
            psf_times.append(time.perf_counter() - t0)

        # Qiskit: Same approach (warm up once, then REPS repeated calls)
        transpile(QuantumCircuit(2), coupling_map=warmup_cmap, optimization_level=3)
        qiskit_times = []
        for i in range(REPS):
            t0 = time.perf_counter()
            transpile(qc, coupling_map=cmap, optimization_level=3)
            qiskit_times.append(time.perf_counter() - t0)

        print(f"{'rep':>4} | {'PSF-Zero (s)':>13} | {'Qiskit (s)':>11}")
        for i in range(REPS):
            print(f"{i + 1:>4} | {psf_times[i]:>13.5f} | {qiskit_times[i]:>11.5f}")

        first5_psf = sum(psf_times[:5]) / 5
        last5_psf = sum(psf_times[-5:]) / 5
        first5_q = sum(qiskit_times[:5]) / 5
        last5_q = sum(qiskit_times[-5:]) / 5
        print(f"\n  PSF-Zero:  first-5 avg = {first5_psf:.5f}s | last-5 avg = {last5_psf:.5f}s | "
              f"ratio (first/last) = {first5_psf / last5_psf:.2f}x")
        print(f"  Qiskit:    first-5 avg = {first5_q:.5f}s | last-5 avg = {last5_q:.5f}s | "
              f"ratio (first/last) = {first5_q / last5_q:.2f}x")


if __name__ == "__main__":
    main()
