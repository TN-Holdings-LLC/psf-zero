"""
psf_compile.py — FINAL PRODUCTION CORE (ULTIMATE EDITION)
=========================================================

PSF-Zero Packet Router + Geometric Compiler
- Qiskit Deep Memory Registration (Monkey Patch)
- Vectorized Math (Batch SVD & SU(4) Purification)
- Concurrent Rust Execution (Zero-Friction Parallelism)
"""

from __future__ import annotations
import numpy as np
import concurrent.futures
from typing import List, Tuple

# Qiskit Imports
from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.library import UnitaryGate
from qiskit.dagcircuit import DAGNode
from qiskit.transpiler.passes.synthesis import plugin

# Core Engine (Rust-backed)
import psf_synthesis
from psf_synthesis import SU4GeodesicPSFSynthesizer, GeodesicPSFHyper


# ======================================================================
# [Monkey Patch] Forcible Registration into Qiskit Deep Memory
# ======================================================================
_original_init = plugin.UnitarySynthesisPluginManager.__init__

def _hacked_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    class FakeExtension:
        def __init__(self):
            self.name = 'psf_zero'
            self.plugin = psf_synthesis.SU4GeodesicPSFUnitarySynthesis
            self.obj = psf_synthesis.SU4GeodesicPSFUnitarySynthesis()
            
    fake_ext = FakeExtension()
    self.ext_plugins.extensions.append(fake_ext)
    
    if hasattr(self.ext_plugins, '_extensions_by_name'):
        if self.ext_plugins._extensions_by_name is None:
            self.ext_plugins._extensions_by_name = {}
        self.ext_plugins._extensions_by_name['psf_zero'] = fake_ext

# Apply the patch globally to intercept Qiskit's default synthesis
plugin.UnitarySynthesisPluginManager.__init__ = _hacked_init


# ================================================================
# Core Compiler Engine
# ================================================================
def compile(circuit: QuantumCircuit, max_workers: int = None, **kwargs) -> QuantumCircuit:
    """
    Optimized PSF Packet Compiler (Vectorized & Concurrent)
    
    Args:
        circuit (QuantumCircuit): The input quantum circuit to be optimized.
        max_workers (int, optional): Maximum number of threads for parallel execution.
        **kwargs: Hyperparameters for the Geodesic PSF Synthesizer.
        
    Returns:
        QuantumCircuit: The geometrically optimized quantum circuit.
    """
    # --- Hyperparameters ---
    hyper = GeodesicPSFHyper(
        m=kwargs.get("m", 5),
        iters=kwargs.get("iters", 120),
        tol=kwargs.get("tol", 1e-9),
    )
    synth = SU4GeodesicPSFSynthesizer(hyper)

    # --- STEP 1: Packetization ---
    # Convert circuit to DAG and consolidate into isolated 2-qubit packets
    pm = PassManager([
        Collect2qBlocks(),
        ConsolidateBlocks(force_consolidate=True),
    ])
    packetized = pm.run(circuit)
    dag = circuit_to_dag(packetized)

    # --- STEP 2: Extract Target Nodes ---
    # Filter nodes that are strictly 2-qubit unitary blocks
    target_nodes = [
        node for node in dag.op_nodes()
        if isinstance(node.op, UnitaryGate) and node.op.num_qubits == 2
    ]
    
    if not target_nodes:
        return circuit

    # ================================================================
    # STEP 3: Vectorized Math (GPU-Ready Batch Processing)
    # ================================================================
    # Stack N packets into a single 3D tensor of shape (N, 4, 4)
    U_stack = np.array([node.op.to_matrix() for node in target_nodes])

    # Batch SVD computation (simultaneous evaluation for N matrices)
    U_u, _, U_vh = np.linalg.svd(U_stack)
    U_pure_stack = U_u @ U_vh

    # Batch determinant calculation and SU(4) normalization
    dets = np.linalg.det(U_pure_stack)
    
    # Safe handling for near-zero determinants via vectorized masking
    valid_mask = np.abs(dets) > 1e-12
    phase_corrections = np.ones_like(dets, dtype=np.complex128)
    phase_corrections[valid_mask] = dets[valid_mask] ** -0.25
    
    # Broadcast phase corrections to perfectly project all N matrices into SU(4)
    U_su4_stack = U_pure_stack * phase_corrections[:, np.newaxis, np.newaxis]

    # ================================================================
    # STEP 4: Concurrent Execution (Zero-Friction Rust Integration)
    # ================================================================
    optimized_blocks: List[Tuple[DAGNode, QuantumCircuit]] = [None] * len(target_nodes)

    def _synthesize_task(idx: int) -> Tuple[int, QuantumCircuit]:
        # Invoke the Rust core engine directly for geometric projection
        opt_circ = synth.synthesize(U_su4_stack[idx])
        return idx, opt_circ

    # Dispatch all SU(4) packets to the Rust core simultaneously via multi-threading
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_synthesize_task, i) for i in range(len(target_nodes))]
        
        for future in concurrent.futures.as_completed(futures):
            idx, opt_circ = future.result()
            optimized_blocks[idx] = (target_nodes[idx], opt_circ)

    # --- STEP 5: DAG Rewrite & Reassemble ---
    # Substitute the optimized circuits back into the DAG
    for node, block in optimized_blocks:
        dag.substitute_node_with_dag(node, circuit_to_dag(block))

    return dag_to_circuit(dag)


# ================================================================
# Hybrid Pipeline (Optional)
# ================================================================
def compile_hybrid(circuit: QuantumCircuit, backend_pass=None, **kwargs):
    """
    Hybrid Execution Pipeline:
    PSF-Zero (Y-Axis pure geometric projection) -> External Optimizer (X-Axis mapping/routing)
    """
    psf_circ = compile(circuit, **kwargs)

    if backend_pass is not None:
        optimized = psf_circ.copy()
        backend_pass.apply(optimized)
        return optimized

    return psf_circ


# ================================================================
# Debug & Visualization
# ================================================================
def analyze(circuit: QuantumCircuit, optimized: QuantumCircuit):
    """Prints benchmark statistics comparing the original and optimized circuits."""
    print("\n=== SYSTEM ARCHITECTURE STATS ===")
    print(f"Original depth:  {circuit.depth()}")
    print(f"Optimized depth: {optimized.depth()}")

    def count_2q(qc):
        return sum(
            qc.count_ops().get(op, 0)
            for op in ["cx", "cz", "rzz", "rxx", "ryy"]
        )

    print(f"Original 2Q gates:  {count_2q(circuit)}")
    print(f"Optimized 2Q gates: {count_2q(optimized)}")
    print("=================================\n")


# ================================================================
# Execution Entry Point
# ================================================================
if __name__ == "__main__":
    import time
    from qiskit.circuit.random import random_circuit

    print("[SYSTEM] Booting PSF-Zero Geometric Compiler...")
    
    # Generate a large random circuit for stress testing
    qc = random_circuit(10, 100, seed=42)
    
    print("[SYSTEM] Circuit Generated. Initiating Projection...")
    
    start_time = time.time()
    
    # Execute PSF-Zero compilation utilizing all available CPU cores
    opt = compile(qc, max_workers=None)
    
    end_time = time.time()

    analyze(qc, opt)
    print(f"[SUCCESS] Projection Execution Completed in {end_time - start_time:.4f} seconds.")
