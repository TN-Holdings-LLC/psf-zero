import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import FullPeepholeOptimise

from psf_compile import compile as psf_compile

FULL = os.environ.get("PSF_MAX_LOAD_FULL") == "1"

# ==========================================================================
# Changes from v1 -> v2 (2026-09-03, corrections based on empirical measurement)
# ==========================================================================
# The original generate_scalable_dense_circuit() was the quintessential example of 
# a "broad and shallow" circuit explicitly named in psf_compile.py's own comments 
# (BUG#4). Upon actual verification (160 qubits, depth=5): The 2-qubit blocks 
# found by Collect2qBlocks averaged 3.0 gates, maxing out at only 7 gates, 
# never crossing the block_gate_floor=12 threshold. In other words, psf_compile() 
# returning "0/0 blocks" was the intended, correct behavior, not a bug 
# (block_gate_floor was introduced precisely to prevent touching these 
# "unrewarding" blocks).
#
# Looking at the actual execution results (10-160 qubits, 0/0 blocks across all scales), 
# PSF Depth was identical to or 1 greater than TKET Depth (20 vs 19) — meaning 
# PSF-Zero was merely returning the original circuit almost as-is, while TKET 
# was actually searching and optimizing with FullPeepholeOptimise. The "500-800x faster" 
# claim (0.002s vs 17.043s) was solely because PSF-Zero was doing no work. 
# It cannot be used as evidence for claims like the "1000-Qubit Scalability Checkmate" 
# in the README.
#
# Fix: Replaced with a circuit that applies GATES_PER_PAIR random SU(4) unitaries 
# consecutively to each adjacent logical qubit pair (build_dense_pair_blocks_circuit, 
# the same one used in the other tests tonight). With this structure, Collect2qBlocks 
# can detect each pair as a single large block, reliably exceeding block_gate_floor.
# ==========================================================================

GATES_PER_PAIR = 20  # Depth to reliably exceed block_gate_floor(=12)


def build_dense_pair_blocks_circuit(num_qubits, gates_per_pair, seed=0):
    """A circuit applying consecutive random SU(4) unitaries to each adjacent logical qubit pair. 
    It has a structure that PSF-Zero can actually compress (deep 2-qubit interactions on the same pair).

    Note: If raw UnitaryGates are appended directly, TKET's qiskit_to_tk() converts them 
    into Unitary2qBox, and FullPeepholeOptimise() crashes with 
    "Can only build replacement circuits for basic gates: Unitary2qBox" 
    (This is the same known limitation already reported in psf_compile.py's BUG#1 comment). 
    Therefore, we decompose the block for each pair into a standard gate sequence (like u/cx) 
    before incorporating it into the main circuit. The internal interaction remains the same, 
    but the format can be handled without issue by both TKET and 
    PSF-Zero (Collect2qBlocks/ConsolidateBlocks)."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    pairs = [(i, i + 1) for i in range(0, num_qubits - 1, 2)]
    for (a, b) in pairs:
        block = QuantumCircuit(2)
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            block.append(UnitaryGate(u), [0, 1])
        qc.compose(block.decompose(), [a, b], inplace=True)
    return qc

def run_tket_compile(qiskit_circ: QuantumCircuit) -> QuantumCircuit:
    tket_circ = qiskit_to_tk(qiskit_circ)
    FullPeepholeOptimise().apply(tket_circ)
    return tk_to_qiskit(tket_circ)

def test_native_showdown_scale_sweep():
    qubits_list = [10, 20, 40, 80, 160] + ([320, 640] if FULL else [])
    
    tket_times = []
    psf_times = []
    tket_depths = []
    psf_depths = []

    print("\n🚀 [NATIVE SHOWDOWN] Starting pure TKET vs PSF-Zero core comparison...")

    for n in qubits_list:
        qc_orig = build_dense_pair_blocks_circuit(n, GATES_PER_PAIR, seed=42)

        # 1. TKET Native
        t0 = time.perf_counter()
        qc_tket = run_tket_compile(qc_orig)
        t_tket = time.perf_counter() - t0
        tket_times.append(t_tket)
        tket_depths.append(qc_tket.depth())

        # 2. PSF-Zero Native
        t0 = time.perf_counter()
        qc_psf = psf_compile(qc_orig)
        t_psf = time.perf_counter() - t0
        psf_times.append(t_psf)
        psf_depths.append(qc_psf.depth())

        print(f"  [{n}q] TKET Time: {t_tket:.3f}s (Depth: {qc_tket.depth()}) | PSF Time: {t_psf:.3f}s (Depth: {qc_psf.depth()})")

    print("\n" + "="*65)
    print("🏆 [NATIVE SHOWDOWN RESULTS]")
    print("="*65)
    for n, tt, tp, dt, dp in zip(qubits_list, tket_times, psf_times, tket_depths, psf_depths):
        print(f"  Qubits: {n:4d} | TKET Time: {tt:8.3f}s (Depth: {dt:4d}) | PSF Time: {tp:8.3f}s (Depth: {dp:4d})")
    print("="*65)

    # Plot graphs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Compile time comparison
    ax1.plot(qubits_list, tket_times, marker="o", color="red", label="TKET Native")
    ax1.plot(qubits_list, psf_times, marker="s", color="blue", label="PSF-Zero Native")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Qubit count")
    ax1.set_ylabel("Compilation Time (seconds, log scale)")
    ax1.set_title("Native Speed Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Depth comparison
    ax2.plot(qubits_list, tket_depths, marker="o", color="red", label="TKET Native")
    ax2.plot(qubits_list, psf_depths, marker="s", color="blue", label="PSF-Zero Native")
    ax2.set_xscale("log")
    ax2.set_xlabel("Qubit count")
    ax2.set_ylabel("Circuit Depth")
    ax2.set_title("Native Depth Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("native_showdown_checkmate.png", dpi=300)
    print("🎨 Native showdown graph successfully saved to 'native_showdown_checkmate.png'.")
