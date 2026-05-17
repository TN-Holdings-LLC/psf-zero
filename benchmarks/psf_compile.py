"""
psf_compile.py — FINAL PRODUCTION CORE
======================================

PSF-Zero Packet Router + Geometric Compiler

Key properties:
- Zero global unitary construction (no memory explosion)
- DAG packetization (2-qubit blocks)
- Numerically stable SU(4) purification
- Parallel-ready architecture
"""

from __future__ import annotations
import numpy as np
from typing import List

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.library import UnitaryGate

# Core Engine (Rust-backed)
from psf_synthesis import SU4GeodesicPSFSynthesizer, GeodesicPSFHyper


# ================================================================
# 🔧 Core Compiler
# ================================================================
def compile(circuit: QuantumCircuit, **kwargs) -> QuantumCircuit:
    """
    PSF Packet Compiler (Production Version)

    Input:
        Arbitrary quantum circuit (N qubits)

    Output:
        Packet-optimized circuit using SU(4) geometric synthesis
    """

    # --- Hyperparameters ---
    hyper = GeodesicPSFHyper(
        m=kwargs.get("m", 5),
        iters=kwargs.get("iters", 120),
        tol=kwargs.get("tol", 1e-9),
    )
    synth = SU4GeodesicPSFSynthesizer(hyper)

    # --- STEP 1: Packetization ---
    pm = PassManager([
        Collect2qBlocks(),
        ConsolidateBlocks(force_consolidate=True),
    ])
    packetized = pm.run(circuit)

    dag = circuit_to_dag(packetized)

    # --- STEP 2: Extract Target Nodes ---
    target_nodes = [
        node for node in dag.op_nodes()
        if isinstance(node.op, UnitaryGate) and node.op.num_qubits == 2
    ]

    # --- STEP 3: Process Packets (parallel-ready loop) ---
    optimized_blocks = []

    for node in target_nodes:
        U = node.op.to_matrix()

        # ✅ 안전한 유니타리 복원 (SVD)
        U_u, _, U_vh = np.linalg.svd(U)
        U_pure = U_u @ U_vh

        # ✅ SU(4) 정규화
        det = np.linalg.det(U_pure)
        if abs(det) < 1e-12:
            U_su4 = np.eye(4)
        else:
            U_su4 = U_pure / (det ** 0.25)

        # ✅ PSF 합성
        optimized = synth.synthesize(U_su4)
        optimized_blocks.append((node, optimized))

    # --- STEP 4: DAG Rewrite ---
    for node, block in optimized_blocks:
        dag.substitute_node_with_dag(node, circuit_to_dag(block))

    # --- STEP 5: Reassemble ---
    return dag_to_circuit(dag)


# ================================================================
# ⚡ Optional Hybrid Pipeline
# ================================================================
def compile_hybrid(circuit: QuantumCircuit, backend_pass=None, **kwargs):
    """
    PSF → (optional) external optimizer (TKET/Qiskit/etc)

    This is the real "production winner" pipeline.
    """
    psf_circ = compile(circuit, **kwargs)

    if backend_pass is not None:
        optimized = psf_circ.copy()
        backend_pass.apply(optimized)
        return optimized

    return psf_circ


# ================================================================
# 🧪 Debug / Visualization
# ================================================================
def analyze(circuit: QuantumCircuit, optimized: QuantumCircuit):
    print("=== STATS ===")
    print(f"Original depth: {circuit.depth()}")
    print(f"Optimized depth: {optimized.depth()}")

    def count_2q(qc):
        return sum(
            qc.count_ops().get(op, 0)
            for op in ["cx", "cz", "rzz", "rxx", "ryy"]
        )

    print(f"Original 2Q: {count_2q(circuit)}")
    print(f"Optimized 2Q: {count_2q(optimized)}")


# ================================================================
# 🧪 Example Use
# ================================================================
if __name__ == "__main__":
    from qiskit.circuit.random import random_circuit

    qc = random_circuit(4, 10, seed=1)

    print("=== ORIGINAL ===")
    print(qc)

    opt = compile(qc)

    print("\n=== PSF OPTIMIZED ===")
    print(opt)

    analyze(qc, opt)
