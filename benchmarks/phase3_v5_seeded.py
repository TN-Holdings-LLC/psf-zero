# phase3_v5.py -- Phase 3 v5: Fixed seed_transpiler + expanded to 10 seeds
import time
import math
import inspect
import multiprocessing
import numpy as np
import psutil
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary
from qiskit.transpiler import CouplingMap
from psf_compile import compile_for_hardware

# ==========================================
# Changes from v4 to v5
# ==========================================
# While tracking down v4 on real hardware, re-measuring the exact same 500-qubit, seed=1
# circuit in an independent validation script revealed that Qiskit's final depth
# (optimization_level=3) varied between 41 vs 54, and compile time between 0.06s vs 0.20s,
# despite identical code and inputs.
# A separate direct comparison (profile_qiskit_multiprocess_vs_mainprocess.py) disproved
# the hypothesis that running inside a multiprocessing.Process suppresses Qiskit's internal
# parallel search (the ratio was 0.96x, well within error bounds). The most natural remaining
# explanation is that transpile(optimization_level=3) does not fix seed_transpiler, causing
# internal randomized searches in SabreLayout/SabreSwap to yield different results (and times)
# across calls.
#
# Fix: Two changes.
#   1. Pass `seed_transpiler=seed` to Qiskit's transpile() call to eliminate search variation
#      across executions (using the same value as the circuit seed ensures total reproducibility
#      across the entire script). Warm-up calls are disposable, so a fixed value of 0 is fine.
#   2. Expand SEEDS from 3 to 10 to match the statistical rigor of Section 4.
#
# Remaining Asymmetry: compile_for_hardware() on the PSF side currently lacks an argument
# to propagate seed_transpiler to its internal transpile() call, so PSF's routing search
# (routing_optimization_level=1) remains unfixed here (remaining asymmetry -- see header comment).
# However, level=1 is likely a more deterministic heuristic that does not use Sabre's multi-trial
# search, and results up to v4 showed no variance outside of (0 fell back), so the practical
# impact is expected to be small. If desired, consider separately adding a seed_transpiler
# argument to compile_for_hardware() to propagate it to the internal transpile() call.
#
# Check whether the anomaly of "a single measurement straying widely" still occurs by
# inspecting the mean and standard deviation across the 10 seeds.
# ==========================================
QUBIT_SIZES = [50, 100, 156, 300, 500]
GATES_PER_PAIR = 20          # Must be greater than block_gate_floor=12
SEEDS = list(range(1, 11))   # Expanded from 3 in v4 to 10, aligned with Section 4
QISKIT_ROUTING_OPT_LEVEL = 3
PSF_ROUTING_OPT_LEVEL = 1
TIMEOUT_SECONDS = 300  # 5 minutes
OUTPUT_CSV = "phase3_v5_seeded_results.csv"

_PSF_SUPPORTS_VERIFY = "verify" in inspect.signature(compile_for_hardware).parameters


def get_grid_cmap(num_qubits):
    """Generate a 2D grid coupling map closest to the specified qubit count, mimicking physical device layouts."""
    cols = int(math.ceil(math.sqrt(num_qubits)))
    rows = int(math.ceil(num_qubits / cols))
    return CouplingMap.from_grid(rows, cols)


def build_dense_pair_blocks_circuit(num_qubits, gates_per_pair, seed):
    """Build a circuit applying gates_per_pair random SU(4) unitaries consecutively 
    to each adjacent logical qubit pair. Ensures deep continuous blocks exceeding block_gate_floor
    to properly trigger PSF-Zero's synthesis path (same technique as phase1.py)."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    pairs = [(i, i + 1) for i in range(0, num_qubits - 1, 2)]
    for (a, b) in pairs:
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            qc.append(UnitaryGate(u), [a, b])
    return qc


def count_coupling_violations(qc, cmap):
    """Verify whether all two-qubit gates in the output circuit reside strictly on edges of the coupling_map.
    If violations > 0, the circuit cannot be physically executed on this hardware topology."""
    edges = set()
    for a, b in cmap.get_edges():
        edges.add((a, b))
        edges.add((b, a))
    total_2q = 0
    violations = 0
    for inst in qc.data:
        if len(inst.qubits) == 2:
            total_2q += 1
            qi = qc.find_bit(inst.qubits[0]).index
            qj = qc.find_bit(inst.qubits[1]).index
            if (qi, qj) not in edges:
                violations += 1
    return total_2q, violations


def worker_qiskit(circuit, cmap, seed, result_queue):
    """Qiskit: Performs global optimization and routing simultaneously. Fixes seed_transpiler
    to remove execution jitter caused by internal randomized search."""
    warmup_cmap = get_grid_cmap(2)
    transpile(QuantumCircuit(2), coupling_map=warmup_cmap,
              optimization_level=QISKIT_ROUTING_OPT_LEVEL, seed_transpiler=0)  # warm-up, outside the timer
    start_time = time.perf_counter()
    try:
        transpiled_qc = transpile(circuit, coupling_map=cmap,
                                   optimization_level=QISKIT_ROUTING_OPT_LEVEL,
                                   seed_transpiler=seed)
        elapsed_time = time.perf_counter() - start_time
        total_2q, violations = count_coupling_violations(transpiled_qc, cmap)
        status = "success" if violations == 0 else f"invalid ({violations} coupling violations)"
        result_queue.put({
            "status": status, "time": elapsed_time,
            "final_2q_gates": total_2q, "final_depth": transpiled_qc.depth(),
            "coupling_violations": violations,
        })
    except Exception as e:
        result_queue.put({"status": f"error: {str(e)}", "time": None,
                           "final_2q_gates": None, "final_depth": None, "coupling_violations": None})


def worker_psf(circuit, cmap, seed, result_queue):
    """PSF-Zero: Block compression + routing (compile_for_hardware).
    Note: compile_for_hardware() currently lacks an argument to propagate seed_transpiler
    to the internal transpile() call, so PSF's routing search is not fixed here either
    (remaining asymmetry -- see header comment)."""
    psf_kwargs = {"routing_optimization_level": PSF_ROUTING_OPT_LEVEL}
    if _PSF_SUPPORTS_VERIFY:
        psf_kwargs["verify"] = False

    warmup_cmap = get_grid_cmap(2)
    compile_for_hardware(QuantumCircuit(2), warmup_cmap, **psf_kwargs)  # warm-up, outside the timer
    start_time = time.perf_counter()
    try:
        transpiled_qc = compile_for_hardware(circuit, cmap, **psf_kwargs)
        elapsed_time = time.perf_counter() - start_time
        total_2q, violations = count_coupling_violations(transpiled_qc, cmap)
        status = "success" if violations == 0 else f"invalid ({violations} coupling violations)"
        result_queue.put({
            "status": status, "time": elapsed_time,
            "final_2q_gates": total_2q, "final_depth": transpiled_qc.depth(),
            "coupling_violations": violations,
        })
    except Exception as e:
        result_queue.put({"status": f"error: {str(e)}", "time": None,
                           "final_2q_gates": None, "final_depth": None, "coupling_violations": None})


def run_with_monitor(target_worker, circuit, cmap, seed):
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=target_worker,
        args=(circuit, cmap, seed, result_queue)
    )

    process.start()
    p = psutil.Process(process.pid)

    peak_memory_mb = 0.0
    start_time = time.time()

    while process.is_alive():
        elapsed = time.time() - start_time
        if elapsed > TIMEOUT_SECONDS:
            process.terminate()
            process.join()
            return {"status": "timeout (Dead Zone)", "time": TIMEOUT_SECONDS,
                    "final_2q_gates": None, "final_depth": None, "coupling_violations": None}, peak_memory_mb

        try:
            mem_info = p.memory_info()
            current_memory_mb = mem_info.rss / (1024 * 1024)
            if current_memory_mb > peak_memory_mb:
                peak_memory_mb = current_memory_mb
        except psutil.NoSuchProcess:
            break

        time.sleep(0.5)

    process.join()

    if not result_queue.empty():
        result = result_queue.get()
        return result, peak_memory_mb
    else:
        return {"status": "crash (OOM)", "time": None,
                "final_2q_gates": None, "final_depth": None, "coupling_violations": None}, peak_memory_mb


def main():
    if _PSF_SUPPORTS_VERIFY:
        print("[phase3_v5] compile_for_hardware() supports `verify` -- measuring "
              "PSF-Zero with verify=False (the validated fast path).")
    else:
        print("[phase3_v5] WARNING: this installed compile_for_hardware() does NOT "
              "expose a `verify` parameter yet. Falling back to its current default "
              "behavior (presumably verify=True inside compile()). Label any PSF-Zero "
              "timings from this run as verify=True (default), not the confirmed "
              "verify=False numbers from Section 4.")
    print(f"[phase3_v5] Qiskit optimization_level={QISKIT_ROUTING_OPT_LEVEL} "
          f"(seed_transpiler=<circuit seed>, fixed for reproducibility), "
          f"PSF routing_optimization_level={PSF_ROUTING_OPT_LEVEL} (seed_transpiler "
          f"NOT fixed -- compile_for_hardware() doesn't expose it yet), "
          f"GATES_PER_PAIR={GATES_PER_PAIR}, SEEDS={SEEDS} "
          f"(block_gate_floor is 12 -- watch the '[Debug] ... executed for N/M blocks' "
          f"lines below; N should be > 0.)")

    results = []
    for q in QUBIT_SIZES:
        print(f"\n==========================================")
        print(f"🔥 Real Topology Validation v5: {q} Qubits (dense pair blocks, seeded) 🔥")
        print(f"==========================================")

        cmap = get_grid_cmap(q)

        for seed in SEEDS:
            print(f"\n[Seed {seed}] Generating circuit...")
            qc = build_dense_pair_blocks_circuit(q, GATES_PER_PAIR, seed=seed)

            print(f" -> [Qiskit] Running routing & search (Timeout: {TIMEOUT_SECONDS}s)...")
            q_result, q_mem = run_with_monitor(worker_qiskit, qc, cmap, seed)
            results.append({
                "Compiler": "Qiskit", "Qubits": q, "GatesPerPair": GATES_PER_PAIR, "Seed": seed,
                "Status": q_result["status"], "Compile_Time_s": q_result["time"],
                "Peak_Memory_MB": q_mem, "Final_2Q_Gates": q_result["final_2q_gates"],
                "Final_Depth": q_result["final_depth"],
                "Coupling_Violations": q_result["coupling_violations"],
            })
            print(f"    Result: {q_result['status']} | Time: {q_result['time']}s | Mem: {q_mem:.1f}MB | "
                  f"2Q Gates: {q_result['final_2q_gates']} | Depth: {q_result['final_depth']}")

            print(f" -> [PSF-Zero] Running geometric projection compilation + routing...")
            p_result, p_mem = run_with_monitor(worker_psf, qc, cmap, seed)
            results.append({
                "Compiler": "PSF-Zero", "Qubits": q, "GatesPerPair": GATES_PER_PAIR, "Seed": seed,
                "Status": p_result["status"], "Compile_Time_s": p_result["time"],
                "Peak_Memory_MB": p_mem, "Final_2Q_Gates": p_result["final_2q_gates"],
                "Final_Depth": p_result["final_depth"],
                "Coupling_Violations": p_result["coupling_violations"],
            })
            print(f"    Result: {p_result['status']} | Time: {p_result['time']}s | Mem: {p_mem:.1f}MB | "
                  f"2Q Gates: {p_result['final_2q_gates']} | Depth: {p_result['final_depth']}")

            df = pd.DataFrame(results)
            df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
