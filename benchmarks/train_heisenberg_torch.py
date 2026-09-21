"""train_heisenberg_torch.py -- Addendum 121.

PyTorch-driven training under noise on a task that needs entangling depth:
the antiferromagnetic Heisenberg model on a 2x3 grid. Compares a redundant
block ansatz compiled once (A) or re-synthesized by PSF-Zero (D) against two
compile-once controls: an optimally written block ansatz (3 CX per block from
the start) and a shallower one.

Same PyTorch bridge, noise model, optimizer and simulator as Addendum 119
(train_noisy_vqe_torch.py). Usage:
    python train_heisenberg_torch.py
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

# ---------------- fixed settings (pre-registered in Addendum 121) ----------------
ROWS, COLS = 2, 3
N = ROWS * COLS
BASIS = ["rz", "sx", "x", "cx"]
P_1Q, P_2Q = 1e-3, 1e-2
ITERATIONS = 40
LEARNING_RATE = 0.1
INIT_SEEDS = (0, 1, 2)
COMPILE_SEED = 1234
REDUNDANT_K = 6
PAIRINGS = [
    [(0, 1), (3, 4)],            # horizontal-a
    [(0, 3), (1, 4), (2, 5)],    # vertical
    [(1, 2), (4, 5)],            # horizontal-b
]
ONE_Q = PassManager([Optimize1qGatesDecomposition(basis=BASIS)])


# ---------------- problem ----------------
def grid_bonds():
    bonds = []
    for r in range(ROWS):
        for c in range(COLS):
            q = r * COLS + c
            if c + 1 < COLS:
                bonds.append((q, q + 1))
            if r + 1 < ROWS:
                bonds.append((q, q + COLS))
    return bonds


def heisenberg():
    terms = [(p, [a, b], 1.0) for (a, b) in grid_bonds() for p in ("XX", "YY", "ZZ")]
    H = SparsePauliOp.from_sparse_list(terms, num_qubits=N).simplify()
    exact = float(np.linalg.eigvalsh(H.to_matrix()).min())
    return H, exact


def build_ansatz(cycles: int, form: str):
    """Blocks on alternating pairings. Returns (circuit, parameters, pairs)."""
    blocks = [pair for _ in range(cycles) for pairing in PAIRINGS for pair in pairing]
    per_block = 4 * REDUNDANT_K if form == "redundant" else 15
    theta = ParameterVector("theta", per_block * len(blocks))
    qc = QuantumCircuit(N)
    k = 0

    def nxt():
        nonlocal k
        p = theta[k]
        k += 1
        return p

    def local(q):  # rz ry rz: a general single-qubit rotation, 3 parameters
        qc.rz(nxt(), q); qc.ry(nxt(), q); qc.rz(nxt(), q)

    for a, b in blocks:
        if form == "redundant":
            for _ in range(REDUNDANT_K):
                qc.ry(nxt(), a); qc.rz(nxt(), a)
                qc.ry(nxt(), b); qc.rz(nxt(), b)
                qc.cx(a, b)
        elif form == "optimal":
            # Standard 3-CX universal two-qubit circuit, 15 parameters.
            local(a); local(b)
            qc.cx(b, a)
            qc.rz(nxt(), a); qc.ry(nxt(), b)
            qc.cx(a, b)
            qc.ry(nxt(), b)
            qc.cx(b, a)
            local(a); local(b)
        else:
            raise ValueError(form)
    assert k == len(theta)
    return qc, theta, sorted(set(blocks))


def noise_model():
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(P_1Q, 1), ["sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(P_2Q, 2), ["cx"])
    return nm


# ---------------- executors (unchanged from Addendum 119) ----------------
class Executor:
    def __init__(self, strategy, logical, theta, pairs, H, cm):
        self.strategy, self.logical, self.theta, self.H = strategy, logical, theta, H
        self.compile_s = 0.0
        self.sim_s = 0.0
        if strategy == "ideal":
            return
        self.sim = AerSimulator(method="density_matrix", noise_model=noise_model())
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
        return Statevector(self.compiled(values)).expectation_value(self.H_mapped).real

    def device_counts(self, values):
        ops = self.compiled(values).count_ops()
        return ops.get("cx", 0), ops.get("sx", 0) + ops.get("x", 0)


# ---------------- the PyTorch bridge (unchanged from Addendum 119) ----------------
class QuantumExpectation(torch.autograd.Function):
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
        return grad_output * params.new_tensor(0.5 * (e[0::2] - e[1::2])), None


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
    H, exact = heisenberg()
    ansatze = {
        ("deep", "redundant"): build_ansatz(2, "redundant"),
        ("shallow", "redundant"): build_ansatz(1, "redundant"),
        ("deep", "optimal"): build_ansatz(2, "optimal"),
        ("shallow", "optimal"): build_ansatz(1, "optimal"),
    }
    print(f"{ROWS}x{COLS} Heisenberg, {N} qubits, {len(grid_bonds())} bonds; exact ground energy "
          f"{exact:.6f}; noise sx/x {P_1Q}, cx {P_2Q}\n")

    runs = [
        ("ideal_deep_red", "ideal", "deep", "redundant"),
        ("ideal_shallow_red", "ideal", "shallow", "redundant"),
        ("ideal_deep_opt", "ideal", "deep", "optimal"),
        ("A_deep_red", "A", "deep", "redundant"),
        ("D_deep_red", "D", "deep", "redundant"),
        ("A_deep_opt", "A", "deep", "optimal"),
        ("A_shallow_opt", "A", "shallow", "optimal"),
    ]
    summary, traj = [], []
    for name, strategy, depth, form in runs:
        logical, theta, pairs = ansatze[(depth, form)]
        for seed in INIT_SEEDS:
            ex = Executor(strategy, logical, theta, pairs, H, cm)
            ideal_ex = Executor("ideal", logical, theta, pairs, H, cm)
            init = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(theta))
            p0_err, cx, sx = "", "", ""
            if strategy != "ideal":
                p0_err = abs(ex.noiseless_compiled_energy(init) - ideal_ex.energies([init])[0])
                cx, sx = ex.device_counts(init)

            t0 = time.perf_counter()
            final, history = train(ex, init)
            wall = time.perf_counter() - t0
            final_run = ex.energies([final])[0]
            final_ideal = ideal_ex.energies([final])[0]
            row = dict(run=name, depth=depth, form=form, seed=seed, params=len(theta),
                       device_cx=cx, device_sx=sx, exact=exact,
                       final_energy_own_execution=final_run, gap_own_execution=final_run - exact,
                       final_energy_noiseless=final_ideal, gap_noiseless=final_ideal - exact,
                       p0_noiseless_compiled_err=p0_err, wall_s=wall,
                       compile_s=ex.compile_s, sim_s=ex.sim_s)
            summary.append(row)
            traj += [dict(run=name, seed=seed, iteration=i, loss=l) for i, l in enumerate(history)]
            extra = f" | cx {cx} sx {sx} | P0 err {p0_err:.1e}" if p0_err != "" else ""
            print(f"{name:17s} seed {seed}: gap (own) {row['gap_own_execution']:8.4f} | "
                  f"gap (noiseless) {row['gap_noiseless']:8.4f} | {wall:6.1f} s{extra}", flush=True)
        print()

    with open("heisenberg_torch_summary_2026-09-21.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    with open("heisenberg_torch_trajectories_2026-09-21.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(traj[0].keys()))
        w.writeheader(); w.writerows(traj)
    print("Wrote heisenberg_torch_summary_2026-09-21.csv and heisenberg_torch_trajectories_2026-09-21.csv")


if __name__ == "__main__":
    main()
