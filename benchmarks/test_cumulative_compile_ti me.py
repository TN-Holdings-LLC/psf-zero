"""
test_cumulative_compile_time.py -- quantify cumulative wall-clock savings
from PSF-Zero's compile-time speed advantage over many repeated
compile calls, on the SAME machine, in a tight loop (no real QPU time
involved -- this only tests the classical compile step).

This does NOT test whether faster compile translates into a real-hardware
fidelity advantage via "less calibration drift" -- that would require an
actual iterative real-hardware session. What this DOES answer honestly:
exactly how much wall-clock time PSF-Zero saves, cumulatively, over N
repeated compiles of the same circuit family used elsewhere in this
project's benchmarks (15-qubit, dense-pair-blocks, matching section 4/7's
own circuit generator) -- so any claim about "N iterations fit in the same
session" can be backed by a real, local, reproducible number instead of a
guess.
"""
import sys, io, time, contextlib
sys.path.insert(0, '.')
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary, Operator
import psf_compile

N_QUBITS = 15
GATES_PER_PAIR = 20
N_ITER = 3000
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
    print(f"Circuit family: {N_QUBITS} qubits, {GATES_PER_PAIR} gates/pair "
          f"(matches this project's section 4/7 circuit generator)")
    print(f"N_ITER = {N_ITER}\n")

    # One-time correctness confirmation on a SMALL version of the same
    # circuit family (Operator() on the full 15-qubit circuit would need a
    # 32768x32768 / 8GB matrix per call -- infeasible to do 3000x, and
    # unnecessary: psf_compile's correctness at this exact gate/block
    # structure was already exhaustively verified earlier in this project
    # against real gates, degenerate points, and 100s of random SU(4)
    # samples. This is just a final sanity check at small scale before the
    # timing-only loop below.
    print("One-time correctness check (6-qubit version of the same circuit family):")
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

    # Warm-up (outside the timed loop) -- section 4 of this project's own
    # README found that skipping this makes the FIRST measurement in a
    # fresh process misleadingly slow for both engines.
    warm_qc = build_dense_pair_blocks_circuit(N_QUBITS, GATES_PER_PAIR, seed=0)
    _ = transpile(warm_qc, basis_gates=BASIS_GATES, optimization_level=3)
    with contextlib.redirect_stdout(io.StringIO()):
        _ = psf_compile.compile(warm_qc, verify=True)
        _ = psf_compile.compile(warm_qc, verify=False)

    qiskit_times = np.empty(N_ITER)
    psf_true_times = np.empty(N_ITER)
    psf_false_times = np.empty(N_ITER)

    t_loop_start = time.perf_counter()
    for i in range(N_ITER):
        qc = build_dense_pair_blocks_circuit(N_QUBITS, GATES_PER_PAIR, seed=1000 + i)

        t0 = time.perf_counter()
        qc_qiskit = transpile(qc, basis_gates=BASIS_GATES, optimization_level=3)
        qiskit_times[i] = time.perf_counter() - t0

        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            qc_psf_true = psf_compile.compile(qc, verify=True)
            psf_true_times[i] = time.perf_counter() - t0

            t0 = time.perf_counter()
            qc_psf_false = psf_compile.compile(qc, verify=False)
            psf_false_times[i] = time.perf_counter() - t0

        if (i + 1) % 500 == 0:
            elapsed = time.perf_counter() - t_loop_start
            print(f"  ...{i+1}/{N_ITER} done ({elapsed:.1f}s elapsed)")

    def summarize(name, arr):
        total = arr.sum()
        print(f"{name:22s} total={total:8.3f}s  mean={arr.mean()*1000:6.3f}ms  "
              f"median={np.median(arr)*1000:6.3f}ms  stdev={arr.std()*1000:6.3f}ms")
        return total

    t_q = summarize("Qiskit L3", qiskit_times)
    t_pt = summarize("PSF-Zero verify=True", psf_true_times)
    t_pf = summarize("PSF-Zero verify=False", psf_false_times)

    print(f"\nCumulative speedup over {N_ITER} iterations:")
    print(f"  Qiskit / PSF(verify=True)  = {t_q/t_pt:.2f}x")
    print(f"  Qiskit / PSF(verify=False) = {t_q/t_pf:.2f}x")
    print(f"  Time saved vs Qiskit, verify=True : {t_q - t_pt:8.2f}s over {N_ITER} iterations")
    print(f"  Time saved vs Qiskit, verify=False: {t_q - t_pf:8.2f}s over {N_ITER} iterations")

    np.savez("cumulative_compile_times.npz",
             qiskit=qiskit_times, psf_true=psf_true_times, psf_false=psf_false_times)
    print("\nSaved raw per-iteration timings to cumulative_compile_times.npz")


if __name__ == "__main__":
    main()
