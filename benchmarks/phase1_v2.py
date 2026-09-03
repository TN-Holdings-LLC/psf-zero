import time
import multiprocessing
import psutil
import pandas as pd
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from psf_compile import compile as psf_compile  # Calling PSF-Zero

# ==========================================
# Experimental Parameter Setup
# ==========================================
QUBIT_SIZES = [15, 50, 100, 156]
DEPTH = 100
SEEDS = list(range(1, 11))  # 10 random circuit patterns per qubit size
TIMEOUT_SECONDS = 3600      # 1-hour timeout
OUTPUT_CSV = "phase1_benchmark_results.csv"

def worker_qiskit(circuit, backend_mock, result_queue):
    """Executes Qiskit compilation and returns the result to the queue."""
    start_time = time.perf_counter()
    try:
        transpiled_qc = transpile(circuit, backend=backend_mock, optimization_level=3)
        elapsed_time = time.perf_counter() - start_time
        result_queue.put({"status": "success", "time": elapsed_time})
    except Exception as e:
        result_queue.put({"status": f"error: {str(e)}", "time": None})

def worker_psf(circuit, backend_mock, result_queue):
    """Executes PSF-Zero compilation and returns the result to the queue."""
    start_time = time.perf_counter()
    try:
        # Executes geometric projection compilation calling the Rust core
        transpiled_qc = psf_compile(circuit)
        elapsed_time = time.perf_counter() - start_time
        result_queue.put({"status": "success", "time": elapsed_time})
    except Exception as e:
        result_queue.put({"status": f"error: {str(e)}", "time": None})

def run_with_monitor(target_worker, circuit, backend_mock):
    """Runs the compilation in a separate process and monitors Peak Memory and Timeout."""
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
        print(f"\n--- Starting Evaluation: {q} Qubits ---")
        for seed in SEEDS:
            qc = random_circuit(num_qubits=q, depth=DEPTH, measure=False, seed=seed)
            
            # 1. Qiskit Measurement
            print(f"[Qiskit] Executing Qubits: {q}, Seed: {seed} ...")
            q_status, q_time, q_mem = run_with_monitor(worker_qiskit, qc, backend_mock)
            results.append({
                "Compiler": "Qiskit", "Qubits": q, "Depth": DEPTH, "Seed": seed,
                "Status": q_status, "Compile_Time_s": q_time, "Peak_Memory_MB": q_mem
            })
            
            # 2. PSF-Zero Measurement
            print(f"[PSF-Zero] Executing Qubits: {q}, Seed: {seed} ...")
            p_status, p_time, p_mem = run_with_monitor(worker_psf, qc, backend_mock)
            results.append({
                "Compiler": "PSF-Zero", "Qubits": q, "Depth": DEPTH, "Seed": seed,
                "Status": p_status, "Compile_Time_s": p_time, "Peak_Memory_MB": p_mem
            })
            
            # Save to CSV
            df = pd.DataFrame(results)
            df.to_csv(OUTPUT_CSV, index=False)
            
            print(f"  -> Qiskit: {q_status} ({q_time}s, {q_mem:.1f}MB)")
            print(f"  -> PSF-Zero: {p_status} ({p_time}s, {p_mem:.1f}MB)")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
