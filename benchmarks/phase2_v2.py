# phase2.py -- Dead Zone Validation Script (Complete version with corrected Qiskit measurement)
from __future__ import annotations
import time
import multiprocessing
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary

import psf_compile as pcp

GATES_PER_PAIR = 20
QUBIT_SIZES = [156, 300, 500, 1000]


def build_dense_pair_blocks_circuit(num_qubits: int, gates_per_pair: int, seed: int = 0) -> QuantumCircuit:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    pairs = [(i, i + 1) for i in range(0, num_qubits - 1, 2)]
    for (a, b) in pairs:
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            qc.append(UnitaryGate(u), [a, b])
    return qc


def worker_qiskit(circuit, backend_mock, result_queue):
    start_time = time.perf_counter()
    try:
        # Avoid invalid path via backend=None, specify basis_gates to properly measure transpile workload[cite: 4]
        transpiled_qc = transpile(circuit, basis_gates=["rz", "sx", "x", "cx"], optimization_level=3)
        elapsed_time = time.perf_counter() - start_time
        result_queue.put({"status": "success", "time": elapsed_time})
    except Exception as e:
        result_queue.put({"status": "error", "message": str(e)})


def worker_psf(circuit, coupling_map, result_queue):
    start_time = time.perf_counter()
    try:
        transpiled_qc = pcp.compile(circuit)
        elapsed_time = time.perf_counter() - start_time
        result_queue.put({"status": "success", "time": elapsed_time})
    except Exception as e:
        result_queue.put({"status": "error", "message": str(e)})


def main() -> None:
    print("==========================================")
    print("🔥 Dead Zone Validation Started (Patched Complete Version) 🔥")
    print("==========================================")

    for q in QUBIT_SIZES:
        print(f"\n[Scale] {q} Qubits / {GATES_PER_PAIR} gates-per-pair")
        qc = build_dense_pair_blocks_circuit(q, GATES_PER_PAIR, seed=1)

        # Run Qiskit
        q_queue = multiprocessing.Queue()
        p_qiskit = multiprocessing.Process(target=worker_qiskit, args=(qc, None, q_queue))
        p_qiskit.start()
        q_res = q_queue.get()
        p_qiskit.join()

        if q_res["status"] == "success":
            print(f"  -> [Qiskit] Success | Time: {q_res['time']:.4f}s")
        else:
            print(f"  -> [Qiskit] Error: {q_res.get('message')}")

        # Run PSF-Zero
        p_queue = multiprocessing.Queue()
        p_psf = multiprocessing.Process(target=worker_psf, args=(qc, None, p_queue))
        p_psf.start()
        p_res = p_queue.get()
        p_psf.join()

        if p_res["status"] == "success":
            print(f"  -> [PSF-Zero] Success | Time: {p_res['time']:.4f}s")
        else:
            print(f"  -> [PSF-Zero] Error: {p_res.get('message')}")


if __name__ == "__main__":
    main()
