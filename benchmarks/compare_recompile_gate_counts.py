"""compare_recompile_gate_counts.py -- Addendum 123.

Is PSF-Zero's per-circuit simulation speed (Addendum 122) specific to
PSF-Zero, or does any re-compile of bound circuits give the same?

No training loop. The same 50 parameter vectors per ansatz are compiled five
ways, and each result is counted and simulated under Addendum 121's noise
model. The problem definition is imported from train_heisenberg_torch.py so
it is identical by construction.

Usage (same folder as train_heisenberg_torch.py):
    python compare_recompile_gate_counts.py
"""
from __future__ import annotations

import contextlib
import csv
import io
import statistics
import time

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator

import psf_compile
from psf_smart_layout import smart_vf2_layout
from train_heisenberg_torch import (BASIS, COLS, COMPILE_SEED, N, ONE_Q, ROWS, build_ansatz,
                                    heisenberg, noise_model)

N_SAMPLES = 50
SIM_REPEATS = 3
SAMPLE_SEED = 7


def counts(qc):
    ops = qc.count_ops()
    return dict(cx=ops.get("cx", 0), sx=ops.get("sx", 0) + ops.get("x", 0), rz=ops.get("rz", 0),
                total=sum(ops.values()), depth=qc.depth())


def main():
    cm = CouplingMap.from_grid(ROWS, COLS)
    H, _ = heisenberg()
    sim = AerSimulator(method="density_matrix", noise_model=noise_model())
    rng = np.random.default_rng(SAMPLE_SEED)
    ansatze = {"red": build_ansatz(2, "redundant"), "opt": build_ansatz(2, "optimal")}
    values = {k: [rng.uniform(-np.pi, np.pi, len(a[1])) for _ in range(N_SAMPLES)] for k, a in ansatze.items()}

    # One-time preparations (not part of per-circuit compile time).
    once = {}
    for k, (logical, theta, pairs) in ansatze.items():
        tqc = transpile(logical, coupling_map=cm, basis_gates=BASIS, optimization_level=3,
                        seed_transpiler=COMPILE_SEED)
        once[("A", k)] = tqc
    layout_map, _ = smart_vf2_layout(cm, ansatze["red"][2], N)
    d_perm = [layout_map[i] for i in range(N)]

    def compile_one(method, k, v):
        logical, theta, _ = ansatze[k]
        bind = dict(zip(theta, v))
        if method == "A":
            out = once[("A", k)].assign_parameters(bind)
            return out, list(out.layout.final_index_layout(filter_ancillas=True))
        if method == "B":
            out = transpile(logical.assign_parameters(bind), coupling_map=cm, basis_gates=BASIS,
                            optimization_level=3, seed_transpiler=COMPILE_SEED)
            return out, list(out.layout.final_index_layout(filter_ancillas=True))
        if method == "D":
            with contextlib.redirect_stdout(io.StringIO()):
                synth = psf_compile.compile(logical.assign_parameters(bind), verify=False,
                                            entangling_basis="cx")
            placed = QuantumCircuit(N)
            placed.compose(ONE_Q.run(synth), qubits=d_perm, inplace=True)
            return placed, d_perm
        raise ValueError(method)

    runs = [("A_red", "A", "red"), ("B_red", "B", "red"), ("D_red", "D", "red"),
            ("A_opt", "A", "opt"), ("B_opt", "B", "opt")]
    rows = []
    print(f"{ROWS}x{COLS} Heisenberg, deep ansatze, {N_SAMPLES} parameter vectors each\n")
    print(f"{'method':7s} {'cx':>4s} {'sx':>5s} {'rz':>5s} {'total':>6s} {'depth':>6s} "
          f"{'compile ms':>11s} {'sim ms/circuit':>15s} {'max P4 err':>11s}")
    for name, method, k in runs:
        logical, theta, _ = ansatze[k]
        compile_one(method, k, values[k][0])  # warm-up
        comp_t, circs, cnts, errs = [], [], [], []
        for v in values[k]:
            t0 = time.perf_counter()
            out, perm = compile_one(method, k, v)
            comp_t.append(time.perf_counter() - t0)
            cnts.append(counts(out))
            ideal = Statevector(logical.assign_parameters(dict(zip(theta, v)))).expectation_value(H).real
            got = Statevector(out).expectation_value(H.apply_layout(perm, num_qubits=N)).real
            errs.append(abs(ideal - got))
            c = out.copy()
            c.save_density_matrix()
            circs.append(c)
        sim.run(circs).result()  # warm-up batch
        sim_t = []
        for _ in range(SIM_REPEATS):
            t0 = time.perf_counter()
            sim.run(circs).result()
            sim_t.append((time.perf_counter() - t0) / N_SAMPLES)
        med = {key: statistics.median(c[key] for c in cnts) for key in cnts[0]}
        row = dict(method=name, ansatz=k, **med,
                   sx_min=min(c["sx"] for c in cnts), sx_max=max(c["sx"] for c in cnts),
                   compile_ms_median=statistics.median(comp_t) * 1000,
                   sim_ms_per_circuit=statistics.median(sim_t) * 1000,
                   sim_ms_per_circuit_min=min(sim_t) * 1000, sim_ms_per_circuit_max=max(sim_t) * 1000,
                   max_p4_err=max(errs))
        rows.append(row)
        print(f"{name:7s} {med['cx']:4.0f} {med['sx']:5.0f} {med['rz']:5.0f} {med['total']:6.0f} "
              f"{med['depth']:6.0f} {row['compile_ms_median']:11.2f} {row['sim_ms_per_circuit']:15.2f} "
              f"{row['max_p4_err']:11.1e}", flush=True)

    out_path = "recompile_gate_counts_2026-09-21.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
