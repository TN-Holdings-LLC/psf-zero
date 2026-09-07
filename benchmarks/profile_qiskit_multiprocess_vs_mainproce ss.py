# profile_qiskit_multiprocess_vs_mainprocess.py
#
# Hypothesis: Qiskit's transpile(optimization_level>=2) internally executes multiple trials
# of SabreLayout/SabreSwap in parallel. However, if it detects that it is running inside
# a multiprocessing.Process child process, it automatically disables or reduces this internal
# parallelization. If true, running worker_qiskit as a multiprocessing.Process in phase3_v4.py
# would have caused Qiskit to yield faster (i.e., handicapped) results than normal.
#
# To verify this, we run the exact same circuit and transpile() call under both:
#   (1) Directly in this script's own main process
#   (2) By launching a single multiprocessing.Process in the exact same shape as worker_qiskit in phase3_v4.py
# and directly compare them. All other variables (circuit, coupling_map, optimization_level,
# and presence of warm-up) are kept completely identical.
import time
import math
import multiprocessing
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary
from qiskit.transpiler import CouplingMap

QUBITS = 500
GATES_PER_PAIR = 20
REPS = 5


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


def timed_transpile(circuit, cmap):
    """phase3_v4.py's worker_qiskit, minus the result_queue plumbing."""
    warmup_cmap = get_grid_cmap(2)
    transpile(QuantumCircuit(2), coupling_map=warmup_cmap, optimization_level=3)  # warm-up
    start_time = time.perf_counter()
    transpiled_qc = transpile(circuit, coupling_map=cmap, optimization_level=3)
    elapsed = time.perf_counter() - start_time
    return elapsed, transpiled_qc.depth()


def subprocess_worker(circuit, cmap, result_queue):
    elapsed, depth = timed_transpile(circuit, cmap)
    result_queue.put((elapsed, depth))


def try_report_qiskit_parallel_state():
    """Best-effort: print whatever Qiskit exposes about its own parallel-execution
    state, so we have direct evidence rather than only inferring it from timing.
    Wrapped in try/except because the exact API varies across Qiskit versions."""
    try:
        from qiskit.utils import parallel as qp
        print(f"    qiskit.utils.parallel.CPU_COUNT = {getattr(qp, 'CPU_COUNT', '?')}")
    except Exception as e:
        print(f"    (could not introspect qiskit.utils.parallel: {e})")
    try:
        import os
        print(f"    os.environ.get('QISKIT_IN_PARALLEL') = {os.environ.get('QISKIT_IN_PARALLEL')}")
        print(f"    os.environ.get('QISKIT_PARALLEL') = {os.environ.get('QISKIT_PARALLEL')}")
        print(f"    multiprocessing.current_process().daemon = {multiprocessing.current_process().daemon}")
    except Exception as e:
        print(f"    (could not introspect environment: {e})")


def main():
    cmap = get_grid_cmap(QUBITS)
    qc = build_dense_pair_blocks_circuit(QUBITS, GATES_PER_PAIR, seed=1)

    print(f"===== {QUBITS} qubits, {REPS} reps each =====\n")

    print("[1] Main process (this script's own process), direct call:")
    try_report_qiskit_parallel_state()
    main_times = []
    for i in range(REPS):
        elapsed, depth = timed_transpile(qc, cmap)
        main_times.append(elapsed)
        print(f"    rep {i + 1}: {elapsed:.5f}s (final depth {depth})")
    main_mean = sum(main_times) / REPS
    print(f"    mean: {main_mean:.5f}s\n")

    print("[2] Inside a single multiprocessing.Process (same shape as phase3_v4.py's worker_qiskit):")
    mp_times = []
    for i in range(REPS):
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=subprocess_worker, args=(qc, cmap, result_queue))
        process.start()
        elapsed, depth = result_queue.get()
        process.join()
        mp_times.append(elapsed)
        print(f"    rep {i + 1}: {elapsed:.5f}s (final depth {depth})")
    mp_mean = sum(mp_times) / REPS
    print(f"    mean: {mp_mean:.5f}s\n")

    print(f"Ratio (main-process time / multiprocessing.Process time) = {main_mean / mp_mean:.2f}x")
    print(
        "If this ratio is well above 1 (e.g. the ~3.5x seen before), it confirms Qiskit's "
        "own internal parallel trial search is being suppressed when transpile() runs "
        "inside a multiprocessing.Process -- i.e. every phase3 benchmark's 'Qiskit' column "
        "measured a throttled-down Qiskit, not the real opt_level=3 a normal script gets. "
        "This is specific to coupling_map-constrained transpile() calls (SabreLayout/"
        "SabreSwap's parallel trials) -- section 4's phase1.py/phase2.py never pass a "
        "coupling_map to transpile(), so they don't exercise this code path and should be "
        "unaffected."
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
