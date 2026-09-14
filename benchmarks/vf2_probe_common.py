#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers imported by the six scripts below. Keep this file alongside them.

    verify_vf2_call_limit_tuple.py
    verify_vf2_steps_to_first_match.py
    verify_vf2_ordering_structure.py
    verify_vf2_cross_implementation.py
    verify_vf2_topologies.py
    verify_qiskit_source_2_5_2.py

No measurement lives here. Only the circuit generator, the grid helper,
environment collection, and CSV output.
The circuit generator and grid function are copied **verbatim** from
`phase3_v5_spare_qubits.py`.
"""
from __future__ import annotations

import glob
import math
import os
import platform
import time

import numpy as np

BASIS_GATES = ["rz", "sx", "x", "cx"]
GATES_PER_PAIR = 20


# ------------------------------------------------------------ fixtures
def get_grid_cmap(num_qubits):
    """Copied verbatim from phase3_v5_spare_qubits.py."""
    from qiskit.transpiler import CouplingMap
    cols = int(math.ceil(math.sqrt(num_qubits)))
    rows = int(math.ceil(num_qubits / cols))
    return CouplingMap.from_grid(rows, cols)


def build_dense_pair_blocks_circuit(num_qubits, gates_per_pair=GATES_PER_PAIR, seed=0):
    """Copied verbatim from phase3_v5_spare_qubits.py."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for (a, b) in [(i, i + 1) for i in range(0, num_qubits - 1, 2)]:
        block = QuantumCircuit(2)
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            block.append(UnitaryGate(u), [0, 1])
        qc.compose(block.decompose(), [a, b], inplace=True)
    return qc


def interaction_pairs(num_qubits):
    """The logical pairs the circuit requires. Used to check for a perfect matching."""
    return [(i, i + 1) for i in range(0, num_qubits - 1, 2)]


def has_perfect_matching(cmap, num_logical_pairs):
    """Whether the coupling map has enough disjoint edges (a maximum matching)
    to cover the required number of logical pairs.

    Separates "no solution exists so none is found" from "one exists but is not
    found". Returns None if networkx is unavailable.
    """
    try:
        import networkx as nx
    except ImportError:
        return None
    g = nx.Graph()
    g.add_nodes_from(range(cmap.size()))
    g.add_edges_from([tuple(e) for e in cmap.get_edges()])
    m = nx.max_weight_matching(g, maxcardinality=True)
    return len(m) >= num_logical_pairs


# ------------------------------------------------------------ timing
def timed(fn, reps=3):
    """One warm-up call outside the timer. Returns (min, median, last return value)."""
    out = fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return min(ts), float(np.median(ts)), out


# ------------------------------------------------------------ environment
def environment():
    """Environment fields to carry as CSV columns. **Not just printed.**"""
    env = {
        "Platform": platform.platform(),
        "Python": platform.python_version(),
        "CPU": platform.processor(),
        "CPU_count": os.cpu_count(),
        "StartMethod": "spawn",
    }
    for name in ("qiskit", "rustworkx", "networkx", "numpy"):
        try:
            env[name] = __import__(name).__version__
        except Exception:
            env[name] = "absent"
    # Record which core build was loaded. The 2026-09-13 anomaly could not be
    # explained because this column did not exist.
    try:
        import psf_zero_core
        f = getattr(psf_zero_core, "__file__", None)
        exts = []
        if f:
            for pat in ("*.pyd", "*.so", "*.dll"):
                for g in glob.glob(os.path.join(os.path.dirname(f), pat)):
                    exts.append(f"{os.path.basename(g)}:{os.path.getsize(g)}")
        env["core_ext"] = ";".join(sorted(exts)) or "none(pure-python?)"
    except Exception:
        env["core_ext"] = "absent"
    return env


def write_csv(rows, out, env):
    import pandas as pd
    df = pd.DataFrame([{**r, **env} for r in rows])
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows)")
    return df


def default_out(stem):
    """Never a fixed filename. A fixed name overwrote earlier evidence once already."""
    return f"{stem}_{time.strftime('%Y-%m-%d')}.csv"


def banner(title, predictions):
    print("=" * 78)
    print(title)
    print("=" * 78)
    print("Environment:", environment())
    print("\nPre-registered predictions (written **before** running, not moved afterward):")
    for p in predictions:
        print("  - " + p)
    print()
