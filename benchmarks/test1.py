# phase1.py -- Complete version (reflects basis_gates fix for Qiskit worker)
import time
import multiprocessing
import numpy as np
import psutil
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary
from psf_compile import compile as psf_compile  # Call PSF-Zero

# ==========================================
# Changes from v1 -> v2 (2026-09-03, empirical fix based on actual measurements)
# ==========================================
# The original phase1.py used qiskit.circuit.random.random_circuit(),
# which by default mixes in a large number of >=3-qubit gates like ccx/cswap/c3sx.
# Upon actual verification (15/50/100/156 qubits, depth=100,
# across all 10 seeds), the 2-qubit blocks found by Collect2qBlocks were at most
# 3 to 4 gates long.
#
# Fix: Replaced with build_dense_pair_blocks_circuit(), which creates a circuit
# that sequentially applies GATES_PER_PAIR random SU(4) unitaries to each
# adjacent logical qubit pair.
QUBIT_SIZES = [15, 50, 100, 156]
GATES_PER_PAIR = 20  
SEEDS = list(range(1, 11))  
TIMEOUT_SECONDS = 3600      
OUTPUT_CSV = "phase1_v2_benchmark_results.csv"


def build_dense_pair_blocks_circuit(num_qubits, gates_per_pair, seed=2):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    pairs = [(i, i + 1) for i in range(0, num_qubits - 1, 2)]
    for (a, b) in pairs:
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            qc.append(UnitaryGate(u), [a, b])
    return qc

def worker_qiskit(circuit, backend_mock, result_queue):
    """Execute Qiskit compilation and return the result to the queue (explicitly specifying basis_gates to measure accurate conversion load)"""
    start_time = time.perf_counter()
    try:
        transpiled_qc = transpile(circuit, basis_gates=["rz", "sx", "x", "cx"], optimization_level=0)
        elapsed_time = time.perf_counter() - start_time
        result_queue.put({"status": "success", "time": elapsed_time})
    except Exception as e:
        result_queue.put({"status": f"error: {str(e)}", "time": None})

def worker_psf(circuit, backend_mock, result_queue):
    """Execute PSF-Zero compilation and return the result to the queue"""
    start_time = time.perf_counter()
    try:
        transpiled_qc = psf_compile(circuit)
        elapsed_time = time.perf_counter() - start_time
        result_queue.put({"status": "success", "time": elapsed_time})
    except Exception as e:
        result_queue.put({"status": f"error: {str(e)}", "time": None})

def run_with_monitor(target_worker, circuit, backend_mock):
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=target_worker, 
        args=(circuit, backend_mock, result_queue)
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
            return "timeout", TIMEOUT_SECONDS, peak_memory_mb
        
        try:
            mem_info = p.memory_info()
            current_memory_mb = mem_info.rss / (1024 * 1024)
            if current_memory_mb > peak_memory_mb:
                peak_memory_mb = current_memory_mb
        except psutil.NoSuchProcess:
            break
            
        time.sleep(0.1)
        
    process.join()
    
    if not result_queue.empty():
        result = result_queue.get()
        return result["status"], result["time"], peak_memory_mb
    else:
        return "crash", None, peak_memory_mb

def main():
    results = []
    backend_mock = None 
    for q in QUBIT_SIZES:
        print(f"\n--- Starting evaluation: {q} Qubits ---")
        for seed in SEEDS:
            qc = build_dense_pair_blocks_circuit(q, GATES_PER_PAIR, seed=seed)

            print(f"[Qiskit] Qubits: {q}, Seed: {seed} running...")
            q_status, q_time, q_mem = run_with_monitor(worker_qiskit, qc, backend_mock)
            results.append({
                "Compiler": "Qiskit", "Qubits": q, "GatesPerPair": GATES_PER_PAIR, "Seed": seed,
                "Status": q_status, "Compile_Time_s": q_time, "Peak_Memory_MB": q_mem
            })

            print(f"[PSF-Zero] Qubits: {q}, Seed: {seed} running...")
            p_status, p_time, p_mem = run_with_monitor(worker_psf, qc, backend_mock)
            results.append({
                "Compiler": "PSF-Zero", "Qubits": q, "GatesPerPair": GATES_PER_PAIR, "Seed": seed,
                "Status": p_status, "Compile_Time_s": p_time, "Peak_Memory_MB": p_mem
            })
            
            df = pd.DataFrame(results)
            df.to_csv(OUTPUT_CSV, index=False)
            
            print(f"  -> Qiskit: {q_status} ({q_time}s, {q_mem:.1f}MB)")
            print(f"  -> PSF-Zero: {p_status} ({p_time}s, {p_mem:.1f}MB)")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
