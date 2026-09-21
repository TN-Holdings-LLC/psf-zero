"""train_noisy_vqe_torch.py -- Addendum 119.

PyTorch-driven variational training under a fixed depolarizing noise model,
comparing how the circuits are compiled for the (simulated) device:

  ideal_L6 / ideal_L3 : noiseless, no compilation (logical statevector)
  A_L6                : Qiskit L3 compile ONCE (parameterized), then bind
  D_L6                : PSF-Zero layout ONCE + re-synthesis per circuit
  A_L3                : compile once, shallow ansatz (the "just use fewer
                        layers" control)

The PyTorch bridge is one torch.autograd.Function: forward returns <H>,
backward returns the exact parameter-shift gradient (+-pi/2 shifts; each
parameter appears in exactly one ry or rz). Every circuit it evaluates is
compiled by the strategy under test, so one training iteration costs 2P + 1
circuit evaluations.

Noisy expectation values are exact (Aer density matrix, no shots).

Usage:
    python train_noisy_vqe_torch.py
"""
from __future__ import annotations

import contextlib
import csv
import io
import time

import numpy as np
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

import psf_compile
from psf_smart_layout import smart_vf2_layout

# ---------------- fixed settings (pre-registered in Addendum 119) ----------------
ROWS, COLS = 2, 4
N = ROWS * COLS
BASIS = ["rz", "sx", "x", "cx"]
P_1Q, P_2Q = 1e-3, 1e-2
ITERATIONS = 40
LEARNING_RATE = 0.1
INIT_SEEDS = (0, 1, 2)
HAMILTONIAN_SEED = 2026
COMPILE_SEED = 1234
ONE_Q = PassManager([Optimize1qGatesDecomposition(basis=BASIS)])
TWO_Q_PAULIS = [a + b for a in "IXYZ" for b in "IXYZ"][1:]  # 15 non-identity


# ---------------- problem ----------------
def build_ansatz(n: int, layers: int):
    """Addenda 109-118's `same_pair` family."""
    theta = ParameterVector("theta", 2 * n * layers)
    qc = QuantumCircuit(n)
    k = 0
    for _ in range(layers):
        for q in range(n):
            qc.ry(theta[k], q); k += 1
            qc.rz(theta[k], q); k += 1
        for a in range(0, n - 1, 2):
            qc.cx(a, a + 1)
    pairs = [(a, a + 1) for a in range(0, n - 1, 2)]
    return qc, theta, pairs


def build_hamiltonian(n: int, seed: int):
    """One random two-qubit Hermitian term per pair. Returns (H, exact ground
    energy); the pairs are disjoint, so the ground energy is the sum of the
    per-pair 4x4 minima."""
    rng = np.random.default_rng(seed)
    terms, exact = [], 0.0
    for a in range(0, n - 1, 2):
        coeffs = rng.normal(size=len(TWO_Q_PAULIS))
        coeffs /= np.linalg.norm(coeffs)
        local = SparsePauliOp.from_sparse_list(
            [(p, [0, 1], c) for p, c in zip(TWO_Q_PAULIS, coeffs)], num_qubits=2)
        exact += float(np.linalg.eigvalsh(local.to_matrix()).min())
        terms += [(p, [a, a + 1], c) for p, c in zip(TWO_Q_PAULIS, coeffs)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n).simplify(), exact


def noise_model():
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(P_1Q, 1), ["sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(P_2Q, 2), ["cx"])
    return nm


# ---------------- executors: one per strategy ----------------
class Executor:
    """Evaluates <H> for a batch of parameter vectors under one strategy."""

    def __init__(self, strategy, logical, theta, pairs, H, cm):
        self.strategy, self.logical, self.theta, self.H = strategy, logical, theta, H
        self.compile_s = 0.0
        self.sim_s = 0.0
        self.sim = None
        if strategy == "ideal":
            return
        self.sim = AerSimulator(method="density_matrix", noise_model=noise_model())
        t0 = time.perf_counter()
        if strategy == "A":
            self.tqc = transpile(logical, coupling_map=cm, basis_gates=BASIS,
                                 optimization_level=3, seed_transpiler=COMPILE_SEED)
            self.perm = list(self.tqc.layout.final_index_layout(filter_ancillas=True))
        elif strategy == "D":
            layout_map, _ = smart_vf2_layout(cm, pairs, N)
            if layout_map is None:
                raise RuntimeError("D not applicable: no perfect layout found")
            self.perm = [layout_map[i] for i in range(N)]
        else:
            raise ValueError(strategy)
        self.one_time_s = time.perf_counter() - t0
        self.H_mapped = H.apply_layout(self.perm, num_qubits=N)

    def _bind(self, values):
        return dict(zip(self.theta, values))

    def compiled(self, values):
        if self.strategy == "A":
            return self.tqc.assign_parameters(self._bind(values))
        with contextlib.redirect_stdout(io.StringIO()):
            synth = psf_compile.compile(self.logical.assign_parameters(self._bind(values)),
                                        verify=False, entangling_basis="cx")
        placed = QuantumCircuit(N)
        placed.compose(ONE_Q.run(synth), qubits=self.perm, inplace=True)
        return placed

    def energies(self, batch):
        if self.strategy == "ideal":
            return np.array([Statevector(self.logical.assign_parameters(self._bind(v)))
                             .expectation_value(self.H).real for v in batch])
        t0 = time.perf_counter()
        circs = []
        for v in batch:
            c = self.compiled(v).copy()
            c.save_density_matrix()
            circs.append(c)
        self.compile_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        result = self.sim.run(circs).result()
        out = np.array([result.data(i)["density_matrix"].expectation_value(self.H_mapped).real
                        for i in range(len(circs))])
        self.sim_s += time.perf_counter() - t0
        return out

    def noiseless_compiled_energy(self, values):
        """P0: the compiled circuit, simulated WITHOUT noise, must reproduce
        the logical energy."""
        return Statevector(self.compiled(values)).expectation_value(self.H_mapped).real


# ---------------- the PyTorch bridge ----------------
class QuantumExpectation(torch.autograd.Function):
    """<H>(params) with an exact parameter-shift backward pass."""

    @staticmethod
    def forward(ctx, params, executor):
        values = params.detach().cpu().numpy().astype(float)
        ctx.executor = executor
        ctx.save_for_backward(params)
        return params.new_tensor(executor.energies([values])[0])

    @staticmethod
    def backward(ctx, grad_output):
        (params,) = ctx.saved_tensors
        v = params.detach().cpu().numpy().astype(float)
        shifted = []
        for k in range(len(v)):
            for s in (np.pi / 2, -np.pi / 2):
                w = v.copy()
                w[k] += s
                shifted.append(w)
        e = ctx.executor.energies(shifted)
        grad = 0.5 * (e[0::2] - e[1::2])
        return grad_output * params.new_tensor(grad), None


def train(executor, init):
    params = torch.tensor(init, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([params], lr=LEARNING_RATE)
    history = []
    for _ in range(ITERATIONS):
        opt.zero_grad()
        loss = QuantumExpectation.apply(params, executor)
        loss.backward()
        opt.step()
        history.append(loss.item())
    return params.detach().numpy().astype(float), history


# ---------------- experiment ----------------
def main():
    cm = CouplingMap.from_grid(ROWS, COLS)
    H, exact = build_hamiltonian(N, HAMILTONIAN_SEED)
    ansatze = {L: build_ansatz(N, L) for L in (6, 3)}
    print(f"{ROWS}x{COLS} grid, {N} qubits, noise: sx/x {P_1Q}, cx {P_2Q}; "
          f"exact ground energy {exact:.6f}\n")

    runs = [("ideal_L6", "ideal", 6), ("ideal_L3", "ideal", 3),
            ("A_L6", "A", 6), ("D_L6", "D", 6), ("A_L3", "A", 3)]
    summary, traj = [], []
    for name, strategy, L in runs:
        logical, theta, pairs = ansatze[L]
        for seed in INIT_SEEDS:
            ex = Executor(strategy, logical, theta, pairs, H, cm)
            ideal_ex = Executor("ideal", logical, theta, pairs, H, cm)
            init = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(theta))

            p0_err = ""
            if strategy != "ideal":
                p0_err = abs(ex.noiseless_compiled_energy(init) - ideal_ex.energies([init])[0])

            t0 = time.perf_counter()
            final, history = train(ex, init)
            wall = time.perf_counter() - t0

            final_run = ex.energies([final])[0]           # what the (noisy) device reports
            final_ideal = ideal_ex.energies([final])[0]   # quality of what was learned
            row = dict(run=name, layers=L, seed=seed, params=len(theta), exact=exact,
                       final_energy_own_execution=final_run, gap_own_execution=final_run - exact,
                       final_energy_noiseless=final_ideal, gap_noiseless=final_ideal - exact,
                       p0_noiseless_compiled_err=p0_err, wall_s=wall,
                       compile_s=ex.compile_s, sim_s=ex.sim_s)
            summary.append(row)
            traj += [dict(run=name, seed=seed, iteration=i, loss=l) for i, l in enumerate(history)]
            p0 = f" | P0 err {p0_err:.1e}" if p0_err != "" else ""
            print(f"{name:9s} seed {seed}: gap (own execution) {row['gap_own_execution']:8.4f} | "
                  f"gap (noiseless) {row['gap_noiseless']:8.4f} | {wall:6.1f} s "
                  f"(compile {ex.compile_s:5.1f} s, sim {ex.sim_s:5.1f} s){p0}")
        print()

    with open("noisy_vqe_torch_summary_2026-09-21.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    with open("noisy_vqe_torch_trajectories_2026-09-21.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(traj[0].keys()))
        w.writeheader(); w.writerows(traj)
    print("Wrote noisy_vqe_torch_summary_2026-09-21.csv and noisy_vqe_torch_trajectories_2026-09-21.csv")


if __name__ == "__main__":
    main()
