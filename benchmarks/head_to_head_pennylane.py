"""head_to_head_pennylane.py -- Addendum 131.

The first comparison of the IBM route (pennylane-qiskit) against the
PSF-Zero route on Paper 1's own failure-region instances (6x7 dense_pairs,
spare 0 and spare 2).

Split into two separate checks (Addendum 131, Section 3a-2), after finding
that neither route can execute a 42-qubit circuit exactly (a 42-qubit
statevector needs tens of terabytes; pennylane-qiskit itself refuses above
29 wires):

  Compilation time, 42 qubits (P1, P2, P4):
    Route A (IBM):      Qiskit's own transpile() directly, with the same
                         coupling map, basis gates, optimization_level and
                         seed_transpiler that pennylane_qiskit's device
                         passes to transpile() internally (confirmed from its
                         source, get_transpile_args/compile_circuits) -- no
                         PennyLane device involved on this side, so what is
                         measured is exactly Qiskit's own transpile cost.
    Route B (PSF-Zero):  psf_layout + r0_psf_zero_transform applied directly
                         to a PennyLane tape, not executed.

  Correctness, n=6 (P3), executed for real on both routes' full pipelines:
    Route A: qml.device("qiskit.aer", backend=AerSimulator(statevector))
    Route B: qml.device("default.qubit") + psf_for_device

Run in psf_h2h_env (Python 3.12, pennylane-qiskit 0.45.0 -> Qiskit 2.3.0).
A wall-clock timeout guards the 42-qubit compilation calls (predicted to run
long on spare-0 seeds, Addendum 131 P1); cross-platform (no SIGALRM, which
does not exist on Windows -- found and fixed before any run, Section 3b of
the pre-registration).

Usage:
    python head_to_head_pennylane.py
"""
from __future__ import annotations

import csv
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

import numpy as np
import pennylane as qml
from pennylane.tape import QuantumTape
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import random_unitary
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

from psf_pennylane import grid_edges, psf_for_device, psf_layout
from r0_psf_zero_transform import r0_psf_zero_transform

ROWS, COLS = 6, 7
N = ROWS * COLS
SEEDS = (0, 1, 2)
TIMEOUT_S = 60
BASIS = ["id", "rz", "sx", "x", "cx"]
N_SMALL = 6  # for the n=6 correctness check; well under Aer's 29-wire limit

_POOL = ThreadPoolExecutor(max_workers=1)


def timed_call(fn, timeout_s=TIMEOUT_S):
    """Runs fn() with a wall-clock timeout, cross-platform (no SIGALRM,
    which does not exist on Windows). Returns (result_or_None, elapsed_s,
    timed_out). A timed-out call's thread keeps running in the background
    until it finishes on its own; its result is discarded."""
    t0 = time.perf_counter()
    future = _POOL.submit(fn)
    try:
        result = future.result(timeout=timeout_s)
        return result, time.perf_counter() - t0, False
    except FutureTimeout:
        return None, time.perf_counter() - t0, True


def dense_pairs_unitaries(n_active, seed):
    """Paper 1's dense_pairs family: one random SU(4) per disjoint pair,
    shared between the PennyLane and Qiskit constructions below so both
    represent the IDENTICAL logical circuit."""
    rng = np.random.default_rng(seed)
    return [random_unitary(4, seed=int(rng.integers(0, 2**31))).data
           for _ in range(n_active // 2)]


def pennylane_tape(n_active, mats):
    ops = [qml.QubitUnitary(U, wires=[2 * i, 2 * i + 1]) for i, U in enumerate(mats)]
    meas = [qml.expval(qml.PauliZ(0) @ qml.PauliZ(n_active - 1))]
    return QuantumTape(ops, meas)


def qiskit_circuit(n_active, mats):
    qc = QuantumCircuit(n_active)
    for i, U in enumerate(mats):
        qc.append(UnitaryGate(U), [2 * i, 2 * i + 1])
    return qc


# ---------------- 42-qubit compilation-only timing ----------------
def compile_route_a(n_active, seed):
    cm = CouplingMap.from_grid(ROWS, COLS)
    qc = qiskit_circuit(n_active, dense_pairs_unitaries(n_active, seed))
    return transpile(qc, coupling_map=cm, basis_gates=BASIS,
                     optimization_level=3, seed_transpiler=seed)


def compile_route_b(n_active, seed):
    tape = pennylane_tape(n_active, dense_pairs_unitaries(n_active, seed))
    (laid_out,), _ = psf_layout(tape, device_edges=grid_edges(ROWS, COLS), device_size=N)
    (final,), _ = r0_psf_zero_transform(laid_out)
    return final


# ---------------- n=6 correctness, executed for real ----------------
def correctness_check(seed):
    mats = dense_pairs_unitaries(N_SMALL, seed)

    def body():
        for i, U in enumerate(mats):
            qml.QubitUnitary(U, wires=[2 * i, 2 * i + 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(N_SMALL - 1))

    dev_a = qml.device("qiskit.aer", wires=N_SMALL,
                       backend=AerSimulator(method="statevector", basis_gates=BASIS),
                       shots=None)
    ref = qml.QNode(body, dev_a)

    dev_b = qml.device("default.qubit", wires=N_SMALL)
    plain = qml.QNode(body, dev_b)
    new = psf_for_device(plain, device_edges=grid_edges(2, 3), device_size=N_SMALL)

    return float(ref()), float(new())


def main():
    rows = []
    print(f"{ROWS}x{COLS} dense_pairs head-to-head -- compilation time (42 qubits), "
          f"Route A timeout {TIMEOUT_S}s\n")

    for spare, n_active in ((0, N), (2, N - 2)):
        for seed in SEEDS:
            label = f"spare {spare} seed {seed}"

            out_a, t_a, to_a = timed_call(lambda: compile_route_a(n_active, seed))
            cx_a = sum(1 for inst in out_a.data if len(inst.qubits) == 2) if out_a else None
            print(f"  {label:16s} Route A compile (IBM)     : "
                  + (f"TIMED OUT after {t_a:.1f}s" if to_a else f"{t_a:8.4f}s  cx={cx_a}"))

            out_b, t_b, to_b = timed_call(lambda: compile_route_b(n_active, seed))
            cx_b = sum(1 for op in out_b.operations if len(op.wires) == 2) if out_b else None
            print(f"  {label:16s} Route B compile (PSF-Zero): "
                  + (f"TIMED OUT after {t_b:.1f}s" if to_b else f"{t_b:8.4f}s  cx={cx_b}"))
            print()

            rows.append(dict(check="compile", spare=spare, seed=seed, n_active=n_active,
                             route_a_s=t_a, route_a_timed_out=to_a, route_a_cx=cx_a,
                             route_b_s=t_b, route_b_timed_out=to_b, route_b_cx=cx_b))

    print(f"n={N_SMALL} correctness -- both routes executed for real:\n")
    for seed in SEEDS:
        val_a, val_b = correctness_check(seed)
        diff = abs(val_a - val_b)
        print(f"  seed {seed}: Route A={val_a:.8f}  Route B={val_b:.8f}  |diff|={diff:.2e}")
        rows.append(dict(check="correctness", spare=None, seed=seed, n_active=N_SMALL,
                         route_a_value=val_a, route_b_value=val_b, agreement=diff))

    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open("head_to_head_2026-09-22.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("\nWrote head_to_head_2026-09-22.csv")


if __name__ == "__main__":
    main()
