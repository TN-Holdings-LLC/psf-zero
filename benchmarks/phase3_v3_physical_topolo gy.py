# phase3_v3.py -- Phase 3 v3: Added "warm-up" and "verify=False" to v2
import time
import math
import inspect
import multiprocessing
import psutil
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit.transpiler import CouplingMap
from psf_compile import compile_for_hardware

# ==========================================
# Changes from v2 to v3
# ==========================================
# The core design of v2 (de28e470-test1.py) --- actual routing using
# compile_for_hardware(), physical validity checks via count_coupling_violations(),
# and recording gate counts/depths --- is fully retained. Only the following two items were added:
#
#   4. v2 lacked a warm-up call. As confirmed in phase1.py/phase2.py,
#      the first transpile()/compile_for_hardware() call inside a
#      multiprocessing.Process ('spawn' on Windows = full restart every time)
#      incurs a fixed per-process cost of roughly 1.3 to 2.4 seconds (already measured
#      as 2.41s vs 0.07s = a 32x difference on the identical 156-qubit circuit).
#      At scales around 50 to 100 qubits, this fixed cost can completely mask actual
#      performance differences. Therefore, both engines warm up an empty circuit once
#      outside the timed block before recording start_time.
#
#   5. worker_psf was not passing verify. To utilize verify=False (the fast path
#      skipping self-checks) added in compile_optional_verify.patch, the caller must
#      explicitly specify it. Because v2 omitted this, it was constantly measuring
#      the default of verify=True (if present) = the slower path including self-verification.
#
#      However, we could not verify beforehand whether your installed compile_for_hardware()
#      was already propagating verify down to compile() (in 010a5cbb-psf_compile.py,
#      compile_for_hardware() hardcoded compile(qc, ..., verify=True)). Thus, we use
#      inspect.signature() to automatically detect support at runtime:
#        - If supported, pass verify=False to measure the fast path.
#        - If not supported, output a clear warning message once at startup
#          (preventing silent measurement with verify=True and result confusion)
#          and continue measuring with the current behavior.
#      When reviewing results, always check this startup message to confirm
#      which path was actually measured.
# ==========================================
QUBIT_SIZES = [50, 100, 156, 300, 500]
DEPTH = 100
SEEDS = [1, 2, 3]
TIMEOUT_SECONDS = 300  # 5 minutes
OUTPUT_CSV = "phase3_v3_physical_topology_results.csv"

_PSF_SUPPORTS_VERIFY = "verify" in inspect.signature(compile_for_hardware).parameters


def get_grid_cmap(num_qubits):
    """Generate a 2D grid coupling map closest to the specified qubit count (mimicking a physical backend layout)."""
    cols = int(math.ceil(math.sqrt(num_qubits)))
    rows = int(math.ceil(num_qubits / cols))
    return CouplingMap.from_grid(rows, cols)


def count_coupling_violations(qc, cmap):
    """Verify whether all two-qubit gates in the output circuit lie strictly on edges of the coupling_map.
    If this is >0, the circuit cannot be physically executed on this hardware topology."""
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


def worker_qiskit(circuit, cmap, result_queue):
    """Qiskit: Perform global optimization and routing simultaneously."""
    warmup_cmap = get_grid_cmap(2)
    transpile(QuantumCircuit(2), coupling_map=warmup_cmap, optimization_level=3)  # warm-up, outside the timer
    start_time = time.perf_counter()
    try:
        transpiled_qc = transpile(circuit, coupling_map=cmap, optimization_level=3)
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


def worker_psf(circuit, cmap, result_queue):
    """PSF-Zero: Block compression + routing (compile_for_hardware)"""
    psf_kwargs = {"routing_optimization_level": 1}
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


def run_with_monitor(target_worker, circuit, cmap):
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=target_worker,
        args=(circuit, cmap, result_queue)
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
        print("[phase3_v3] compile_for_hardware() supports `verify` -- measuring "
              "PSF-Zero with verify=False (the validated fast path).")
    else:
        print("[phase3_v3] WARNING: this installed compile_for_hardware() does NOT "
              "expose a `verify` parameter yet. Falling back to its current default "
              "behavior (presumably verify=True inside compile()). The PSF-Zero "
              "timings in this run therefore include the per-block self-check cost "
              "-- they are NOT the confirmed 2.4x-5.2x-faster numbers from Section 4, "
              "and should be labeled as verify=True (default) results, not conflated "
              "with the verify=False numbers. See compile_optional_verify.patch to add "
              "verify support to compile_for_hardware() itself.")

    results = []
    for q in QUBIT_SIZES:
        print(f"\n==========================================")
        print(f"🔥 Real Topology Validation v3: {q} Qubits / Depth {DEPTH} 🔥")
        print(f"==========================================")

        cmap = get_grid_cmap(q)

        for seed in SEEDS:
            print(f"\n[Seed {seed}] Generating circuit...")
            qc = random_circuit(num_qubits=q, depth=DEPTH, measure=False, seed=seed)

            print(f" -> [Qiskit] Running routing & search (Timeout: {TIMEOUT_SECONDS}s)...")
            q_result, q_mem = run_with_monitor(worker_qiskit, qc, cmap)
            results.append({
                "Compiler": "Qiskit", "Qubits": q, "Depth": DEPTH, "Seed": seed,
                "Status": q_result["status"], "Compile_Time_s": q_result["time"],
                "Peak_Memory_MB": q_mem, "Final_2Q_Gates": q_result["final_2q_gates"],
                "Final_Depth": q_result["final_depth"],
                "Coupling_Violations": q_result["coupling_violations"],
            })
            print(f"    Result: {q_result['status']} | Time: {q_result['time']}s | Mem: {q_mem:.1f}MB | "
                  f"2Q Gates: {q_result['final_2q_gates']} | Depth: {q_result['final_depth']}")

            print(f" -> [PSF-Zero] Running geometric projection compilation + routing...")
            p_result, p_mem = run_with_monitor(worker_psf, qc, cmap)
            results.append({
                "Compiler": "PSF-Zero", "Qubits": q, "Depth": DEPTH, "Seed": seed,
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
