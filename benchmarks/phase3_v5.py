# phase3_v5.py -- Phase 3 v5: Fix seed_transpiler + Expand to 10 seeds
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
# Changes from v4 -> v5
# ==========================================
# In the process of running v4 on the real device, we remeasured the exact same 
# 500-qubit, seed=1 circuit with an independent validation script. We found that 
# the Qiskit side (optimization_level=3) resulted in a final depth of 41 vs 54 
# and a compile time of 0.06s vs 0.20s. Even though the code and input were 
# identical, the results varied with each execution.
# The hypothesis that Qiskit suppresses its internal parallel search when run 
# inside a multiprocessing.Process was refuted by a separate direct comparison 
# (profile_qiskit_multiprocess_vs_mainprocess.py) which showed a ratio of 0.96x, 
# well within the margin of error. The most natural remaining explanation is that 
# transpile(optimization_level=3) does not fix the seed_transpiler, causing 
# the internal SabreLayout/SabreSwap random search to produce different results 
# (and consequently different execution times) on every call.
#
# Fix: 2 points.
#   1. Pass `seed_transpiler=seed` to the Qiskit transpile() call so the search 
#      does not change across executions (by using the same value as the circuit 
#      seed, the entire script becomes completely reproducible). The warm-up call 
#      is a throwaway, so a fixed value of 0 is fine.
#   2. Expand SEEDS from 3 to 10 to match the statistical rigor of section 4.
#
# Remaining asymmetry: PSF's compile_for_hardware() currently lacks an argument 
# to propagate seed_transpiler to the internal transpile() call, so the routing 
# search on the PSF side (routing_optimization_level=1) is still unfixed. 
# However, level=1 does not use Sabre's multi-trial search and is highly likely 
# to be a more deterministic heuristic. Since no variance (other than 0 fell back) 
# was observed in the results up to v4, we consider the practical harm to be minimal. 
# If this is a concern, consider a separate update to add the seed_transpiler 
# argument to compile_for_hardware() and propagate it to the internal transpile().
#
# Please review the mean and standard deviation of the 10 seeds to judge whether 
# the phenomenon from the last run ("a single measurement deviated significantly") 
# still occurs.
# ==========================================
QUBIT_SIZES = [50, 100, 156, 300, 500]
GATES_PER_PAIR = 20          # Must be larger than block_gate_floor=12
SEEDS = list(range(1, 11))   # Expanded from 3 in v4 to 10, matching section 4
QISKIT_ROUTING_OPT_LEVEL = 3
PSF_ROUTING_OPT_LEVEL = 1
TIMEOUT_SECONDS = 300  # 5 minutes
OUTPUT_CSV = "phase3_v5_seeded_results.csv"

_PSF_SUPPORTS_VERIFY = "verify" in inspect.signature(compile_for_hardware).parameters


def get_grid_cmap(num_qubits):
    """Generate a 2D grid closest to the specified number of qubits (mimics physical hardware layout)."""
    cols = int(math.ceil(math.sqrt(num_qubits)))
    rows = int(math.ceil(num_qubits / cols))
    return CouplingMap.from_grid(rows, cols)


def build_dense_pair_blocks_circuit(num_qubits, gates_per_pair, seed):
    """Create a circuit that sequentially applies gates_per_pair random SU(4) 
    unitaries to each adjacent logical qubit pair. By reliably creating continuous 
    blocks that exceed block_gate_floor, we trigger PSF-Zero's actual synthesis 
    path (same approach as phase1.py)."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    pairs = [(i, i + 1) for i in range(0, num_qubits - 1, 2)]
    for (a, b) in pairs:
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            qc.append(UnitaryGate(u), [a, b])
    return qc


def count_coupling_violations(qc, cmap):
    """Verify if all 2-qubit gates in the output circuit actually lie on the edges 
    of the coupling_map. If this is >0, the circuit cannot be physically executed 
    on this hardware topology."""
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
    """Qiskit: Perform global optimization and routing simultaneously. Fix seed_transpiler 
    to eliminate variance across executions caused by internal random search."""
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
    Note: compile_for_hardware() currently lacks an argument to propagate 
    seed_transpiler to the internal transpile(), so the routing search on the PSF side 
    is still unfixed (remaining asymmetry -- see header comment)."""
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
              "verify=False numbers from section 4.")
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
        print(f"🔥 Real-Device Topology Validation v5: {q} Qubits (dense pair blocks, seeded) 🔥")
        print(f"==========================================")

        cmap = get_grid_cmap(q)

        for seed in SEEDS:
            print(f"\n[Seed {seed}] Generating circuit...")
            qc = build_dense_pair_blocks_circuit(q, GATES_PER_PAIR, seed=seed)

            print(f" -> [Qiskit] Executing routing & search (Timeout: {TIMEOUT_SECONDS}s)...")
            q_result, q_mem = run_with_monitor(worker_qiskit, qc, cmap, seed)
            results.append({
                "Compiler": "Qiskit", "Qubits": q, "GatesPerPair": GATES_PER_PAIR, "Seed": seed,
                "Status": q_result["status"], "Compile_Time_s": q_result["time"],
                "Peak_Memory_MB": q_mem, "Final_2Q_Gates": q_result["final_2q_gates"],
                "Final_Depth": q_result["final_depth"],
                "Coupling_Violations": q_result["coupling_violations"],
            })
            print(f"    Result: {q_result['status']} | Time: {q_result['time']}s | Mem: {q_mem:.1f}MB | "
                  f"2Q gates: {q_result['final_2q_gates']} | Depth: {q_result['final_depth']}")

            print(f" -> [PSF-Zero] Executing geometric projection compile + routing...")
            p_result, p_mem = run_with_monitor(worker_psf, qc, cmap, seed)
            results.append({
                "Compiler": "PSF-Zero", "Qubits": q, "GatesPerPair": GATES_PER_PAIR, "Seed": seed,
                "Status": p_result["status"], "Compile_Time_s": p_result["time"],
                "Peak_Memory_MB": p_mem, "Final_2Q_Gates": p_result["final_2q_gates"],
                "Final_Depth": p_result["final_depth"],
                "Coupling_Violations": p_result["coupling_violations"],
            })
            print(f"    Result: {p_result['status']} | Time: {p_result['time']}s | Mem: {p_mem:.1f}MB | "
                  f"2Q gates: {p_result['final_2q_gates']} | Depth: {p_result['final_depth']}")

            df = pd.DataFrame(results)
            df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
