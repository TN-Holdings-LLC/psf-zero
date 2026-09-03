import time
import multiprocessing
import psutil
import pandas as pd
from qiskit import transpile
from qiskit.circuit.random import random_circuit
from psf_compile import compile as psf_compile

# ==========================================
# Phase 2: "Dead Zone (1000 qubits)" Scale Explosion Test
# ==========================================
QUBIT_SIZES = [156, 300, 500, 1000]
DEPTH = 500  # Significantly increased depth to induce complete combinatorial explosion
SEEDS = [1, 2, 3]  # Since it's heavy, 3 trials each are sufficient
TIMEOUT_SECONDS = 300  # 5-minute timeout (freeze detection for existing compilers)
OUTPUT_CSV = "phase2_deadzone_results.csv"

def worker_qiskit(circuit, backend_mock, result_queue):
    start_time = time.perf_counter()
    try:
        transpiled_qc = transpile(circuit, backend=backend_mock, optimization_level=3)
        elapsed_time = time.perf_counter() - start_time
        result_queue.put({"status": "success", "time": elapsed_time})
    except Exception as e:
        result_queue.put({"status": f"error: {str(e)}", "time": None})

def worker_psf(circuit, backend_mock, result_queue):
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
        # Timeout (Dead Zone detection)
        if elapsed > TIMEOUT_SECONDS:
            process.terminate()
            process.join()
            return "timeout (Dead Zone)", TIMEOUT_SECONDS, peak_memory_mb
        
        try:
            mem_info = p.memory_info()
            current_memory_mb = mem_info.rss / (1024 * 1024)
            if current_memory_mb > peak_memory_mb:
                peak_memory_mb = current_memory_mb
        except psutil.NoSuchProcess:
            break
            
        time.sleep(0.5) # Set slightly longer to reduce monitoring load
        
    process.join()
    
    if not result_queue.empty():
        result = result_queue.get()
        return result["status"], result["time"], peak_memory_mb
    else:
        return "crash (OOM)", None, peak_memory_mb

def main():
    results = []
    backend_mock = None 

    for q in QUBIT_SIZES:
        print(f"\n==========================================")
        print(f"🔥 Starting Dead Zone Verification: {q} Qubits / Depth {DEPTH} 🔥")
        print(f"==========================================")
        for seed in SEEDS:
            print(f"\n[Seed {seed}] Generating circuit...")
            qc = random_circuit(num_qubits=q, depth=DEPTH, measure=False, seed=seed)
            
            # 1. Qiskit Measurement (Confirmation of combinatorial explosion)
            print(f" -> [Qiskit] Executing search compilation (Timeout: {TIMEOUT_SECONDS}s)...")
            q_status, q_time, q_mem = run_with_monitor(worker_qiskit, qc, backend_mock)
            results.append({
                "Compiler": "Qiskit", "Qubits": q, "Depth": DEPTH, "Seed": seed,
                "Status": q_status, "Compile_Time_s": q_time, "Peak_Memory_MB": q_mem
            })
            print(f"    Result: {q_status} | Time: {q_time}s | Mem: {q_mem:.1f}MB")
            
            # 2. PSF-Zero Measurement (Proof of constant-time projection)
            print(f" -> [PSF-Zero] Executing geometric projection compilation...")
            p_status, p_time, p_mem = run_with_monitor(worker_psf, qc, backend_mock)
            results.append({
                "Compiler": "PSF-Zero", "Qubits": q, "Depth": DEPTH, "Seed": seed,
                "Status": p_status, "Compile_Time_s": p_time, "Peak_Memory_MB": p_mem
            })
            print(f"    Result: {p_status} | Time: {p_time}s | Mem: {p_mem:.1f}MB")
            
            df = pd.DataFrame(results)
            df.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
