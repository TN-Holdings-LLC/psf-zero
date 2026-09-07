"""
test_cumulative_compile_scale.py -- Scalable cumulative compile time benchmark
supporting 3000, 10000, 50000+ iterations with dynamic progress reporting.
"""
import sys
import io
import time
import contextlib
import argparse
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary, Operator
import psf_compile

N_QUBITS = 15
GATES_PER_PAIR = 20
BASIS_GATES = ["rz", "sx", "x", "cx"]


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
    parser = argparse.ArgumentParser(description="Scalable PSF-Zero Compile Benchmark")
    parser.add_argument("--iters", type=int, default=10000, help="Number of iterations (e.g., 3000, 10000, 50000)")
    args = parser.parse_args()

    n_iter = args.iters
    print(f"Circuit family: {N_QUBITS} qubits, {GATES_PER_PAIR} gates/pair")
    print(f"Target N_ITER = {n_iter}\n")

    print("One-time correctness check (6-qubit version):")
    small_qc = build_dense_pair_blocks_circuit(6, GATES_PER_PAIR, seed=999)
    small_qiskit = transpile(small_qc, basis_gates=BASIS_GATES, optimization_level=3)
    with contextlib.redirect_stdout(io.StringIO()):
        small_psf_true = psf_compile.compile(small_qc, verify=True)
        small_psf_false = psf_compile.compile(small_qc, verify=False)
    U0 = Operator(small_qc).data
    d = U0.shape[0]
    for label, out in [("qiskit", small_qiskit), ("psf_true", small_psf_true), ("psf_false", small_psf_false)]:
        U1 = Operator(out).data
        fid = abs(np.trace(U0.conj().T @ U1)) / d
        print(f"  {label:10s} fidelity={fid:.12f}")
    print()

    # Warm-up
    warm_qc = build_dense_pair_blocks_circuit(N_QUBITS, GATES_PER_PAIR, seed=0)
    _ = transpile(warm_qc, basis_gates=BASIS_GATES, optimization_level=3)
    with contextlib.redirect_stdout(io.StringIO()):
        _ = psf_compile.compile(warm_qc, verify=True)
        _ = psf_compile.compile(warm_qc, verify=False)

    qiskit_times = np.empty(n_iter)
    psf_true_times = np.empty(n_iter)
    psf_false_times = np.empty(n_iter)

    # 動的な進捗表示の間隔（50000回などの高回数にも対応）
    report_step = 2500 if n_iter >= 20000 else (1000 if n_iter >= 10000 else 500)

    t_loop_start = time.perf_counter()
    for i in range(n_iter):
        qc = build_dense_pair_blocks_circuit(N_QUBITS, GATES_PER_PAIR, seed=1000 + i)

        t0 = time.perf_counter()
        _ = transpile(qc, basis_gates=BASIS_GATES, optimization_level=3)
        qiskit_times[i] = time.perf_counter() - t0

        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            _ = psf_compile.compile(qc, verify=True)
            psf_true_times[i] = time.perf_counter() - t0

            t0 = time.perf_counter()
            _ = psf_compile.compile(qc, verify=False)
            psf_false_times[i] = time.perf_counter() - t0

        if (i + 1) % report_step == 0 or (i + 1) == n_iter:
            elapsed = time.perf_counter() - t_loop_start
            print(f"  ...{i+1}/{n_iter} done ({elapsed:.1f}s elapsed)")

    def summarize(name, arr):
        total = arr.sum()
        print(f"{name:22s} total={total:8.3f}s  mean={arr.mean()*1000:6.3f}ms  "
              f"median={np.median(arr)*1000:6.3f}ms  stdev={arr.std()*1000:6.3f}ms")
        return total

    t_q = summarize("Qiskit L3", qiskit_times)
    t_pt = summarize("PSF-Zero verify=True", psf_true_times)
    t_pf = summarize("PSF-Zero verify=False", psf_false_times)

    print(f"\nCumulative speedup over {n_iter} iterations:")
    print(f"  Qiskit / PSF(verify=True)  = {t_q/t_pt:.2f}x")
    print(f"  Qiskit / PSF(verify=False) = {t_q/t_pf:.2f}x")
    print(f"  Time saved vs Qiskit, verify=True : {t_q - t_pt:8.2f}s over {n_iter} iterations")
    print(f"  Time saved vs Qiskit, verify=False: {t_q - t_pf:8.2f}s over {n_iter} iterations")

    filename = f"cumulative_compile_times_{n_iter}.npz"
    np.savez(filename, qiskit=qiskit_times, psf_true=psf_true_times, psf_false=psf_false_times)
    print(f"\nSaved raw per-iteration timings to {filename}")


if __name__ == "__main__":
    main()
