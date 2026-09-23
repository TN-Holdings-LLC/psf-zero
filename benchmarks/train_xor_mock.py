"""train_xor_mock.py -- Addendum 147.

Trains a small variational circuit to solve XOR, in two conditions (ideal,
noisy), 5 seeds each, applying the pre-registered decision rule for
whether to proceed to real IBM hardware on 2026-09-28.

Ideal: default.qubit, exact autograd gradients.
Noisy: Qiskit's AerSimulator (density-matrix, CPU), parameter-shift
gradients (AerSimulator is not autograd-differentiable through PennyLane).

Also times 10 iterations on lightning.qubit vs lightning.gpu (P4,
informational only, not part of the decision rule).

Usage (inside psf_zero_wsl_env_312, which has both PennyLane/lightning.gpu
and Qiskit 2.5.2 installed):
    python train_xor_mock.py
"""
from __future__ import annotations

import csv
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

N = 4
LAYERS = 3
ITERATIONS = 200
SEEDS = (0, 1, 2, 3, 4)
LR = 0.1
XOR_INPUTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
XOR_LABELS = [-1, 1, 1, -1]
P_1Q, P_2Q = 1e-3, 1e-2
BASIS = ["rz", "sx", "x", "cx"]


def n_params():
    return 2 * N * LAYERS


def circuit_ops(params, bits):
    k = 0
    qml.RX(np.pi * bits[0], wires=0)
    qml.RX(np.pi * bits[1], wires=1)
    for _ in range(LAYERS):
        for q in range(N):
            qml.RY(params[k], wires=q); k += 1
            qml.RZ(params[k], wires=q); k += 1
        for a in range(N - 1):
            qml.CNOT(wires=[a, a + 1])


# ---------------- ideal condition: default.qubit, autograd ----------------
dev_ideal = qml.device("default.qubit", wires=N)


@qml.qnode(dev_ideal, interface="autograd", diff_method="backprop")
def ideal_qnode(params, bits):
    circuit_ops(params, bits)
    return qml.expval(qml.PauliZ(0))


def ideal_loss(params):
    preds = pnp.stack([ideal_qnode(params, bits) for bits in XOR_INPUTS])
    labels = pnp.array(XOR_LABELS, dtype=float)
    return pnp.mean((preds - labels) ** 2)


def ideal_predict(params):
    return [1 if ideal_qnode(params, bits) > 0 else -1 for bits in XOR_INPUTS]


# ---------------- noisy condition: AerSimulator, parameter-shift ----------------
def noise_model():
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(P_1Q, 1), ["sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(P_2Q, 2), ["cx"])
    return nm


def build_qc(params, bits):
    """The same circuit, built directly in Qiskit for AerSimulator."""
    qc = QuantumCircuit(N)
    if bits[0]:
        qc.rx(np.pi, 0)
    if bits[1]:
        qc.rx(np.pi, 1)
    k = 0
    for _ in range(LAYERS):
        for q in range(N):
            qc.ry(float(params[k]), q); k += 1
            qc.rz(float(params[k]), q); k += 1
        for a in range(N - 1):
            qc.cx(a, a + 1)
    return qc


_sim = AerSimulator(method="density_matrix", device="CPU", noise_model=noise_model(),
                    basis_gates=BASIS)
_obs0 = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=N)


def noisy_expval(params, bits):
    qc = build_qc(params, bits)
    qc.save_density_matrix()
    res = _sim.run(qc).result()
    return res.data(0)["density_matrix"].expectation_value(_obs0).real


def noisy_loss(params):
    preds = np.array([noisy_expval(params, bits) for bits in XOR_INPUTS])
    labels = np.array(XOR_LABELS, dtype=float)
    return float(np.mean((preds - labels) ** 2))


def noisy_grad(params, preds_at_params=None):
    """Parameter-shift gradient of noisy_loss. `preds_at_params` (the four
    predictions AT the current params) is accepted as an optional argument
    so the caller can reuse a value it already computed (e.g. when also
    reporting the loss this same iteration) instead of recomputing it once
    per parameter -- avoids len(params) redundant simulator calls per
    gradient step, found before this was ever run, not as an optimization
    made necessary by observed slowness."""
    labels = np.array(XOR_LABELS, dtype=float)
    if preds_at_params is None:
        preds_at_params = np.array([noisy_expval(params, b) for b in XOR_INPUTS])
    loss_grad_wrt_pred = 2.0 * (preds_at_params - labels) / len(labels)
    g = np.zeros_like(params)
    for i in range(len(params)):
        shift = np.zeros_like(params)
        shift[i] = np.pi / 2
        preds_p = np.array([noisy_expval(params + shift, bits) for bits in XOR_INPUTS])
        preds_m = np.array([noisy_expval(params - shift, bits) for bits in XOR_INPUTS])
        dpred_di = 0.5 * (preds_p - preds_m)
        g[i] = float(np.dot(loss_grad_wrt_pred, dpred_di))
    return g


def noisy_predict(params):
    return [1 if noisy_expval(params, bits) > 0 else -1 for bits in XOR_INPUTS]


# ---------------- training loops ----------------
def train_ideal(seed):
    params = pnp.array(np.random.default_rng(seed).uniform(-np.pi, np.pi, n_params()),
                       requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=LR)
    history = []
    for _ in range(ITERATIONS):
        params, loss = opt.step_and_cost(ideal_loss, params)
        history.append(float(loss))
    return params, history


def train_noisy(seed):
    params = np.random.default_rng(seed).uniform(-np.pi, np.pi, n_params())
    m1 = np.zeros_like(params); v1 = np.zeros_like(params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    history = []
    for t in range(1, ITERATIONS + 1):
        preds_now = np.array([noisy_expval(params, b) for b in XOR_INPUTS])
        loss = float(np.mean((preds_now - np.array(XOR_LABELS, dtype=float)) ** 2))
        history.append(loss)
        g = noisy_grad(params, preds_at_params=preds_now)
        m1 = beta1 * m1 + (1 - beta1) * g
        v1 = beta2 * v1 + (1 - beta2) * g ** 2
        mhat = m1 / (1 - beta1 ** t)
        vhat = v1 / (1 - beta2 ** t)
        params = params - LR * mhat / (np.sqrt(vhat) + eps)
    return params, history


def main():
    rows_summary, rows_traj = [], []

    print(f"=== Ideal condition (default.qubit, autograd), {len(SEEDS)} seeds, {ITERATIONS} iterations ===")
    t0 = time.perf_counter()
    for seed in SEEDS:
        params, history = train_ideal(seed)
        preds = ideal_predict(params)
        correct = sum(int(p == l) for p, l in zip(preds, XOR_LABELS))
        print(f"  seed {seed}: final loss {history[-1]:.4f}  accuracy {correct}/4  preds {preds}")
        rows_summary.append(dict(condition="ideal", seed=seed, final_loss=history[-1],
                                 accuracy=correct, preds=str(preds)))
        rows_traj += [dict(condition="ideal", seed=seed, iteration=i, loss=l) for i, l in enumerate(history)]
    ideal_wall_s = time.perf_counter() - t0
    print(f"  wall time for all {len(SEEDS)} seeds: {ideal_wall_s:.1f} s "
          f"(C3 check: {ITERATIONS} iters/seed under 600s? {'PASS' if ideal_wall_s/len(SEEDS) < 600 else 'FAIL'})\n")

    print(f"=== Noisy condition (AerSimulator density_matrix CPU), {len(SEEDS)} seeds, {ITERATIONS} iterations ===")
    t0 = time.perf_counter()
    for seed in SEEDS:
        params, history = train_noisy(seed)
        preds = noisy_predict(params)
        correct = sum(int(p == l) for p, l in zip(preds, XOR_LABELS))
        last50 = history[-50:]
        improving = last50[-1] < last50[0]
        print(f"  seed {seed}: final loss {history[-1]:.4f}  accuracy {correct}/4  preds {preds}  "
              f"last-50 improving={improving} ({last50[0]:.4f}->{last50[-1]:.4f})")
        rows_summary.append(dict(condition="noisy", seed=seed, final_loss=history[-1],
                                 accuracy=correct, preds=str(preds)))
        rows_traj += [dict(condition="noisy", seed=seed, iteration=i, loss=l) for i, l in enumerate(history)]
    noisy_wall_s = time.perf_counter() - t0
    print(f"  wall time for all {len(SEEDS)} seeds: {noisy_wall_s:.1f} s\n")

    print("=== P4 (informational): lightning.qubit vs lightning.gpu, 10 iterations, n=4 ===")
    params0 = pnp.array(np.random.default_rng(0).uniform(-np.pi, np.pi, n_params()), requires_grad=True)
    for device_name in ("lightning.qubit", "lightning.gpu"):
        dev = qml.device(device_name, wires=N)

        @qml.qnode(dev, interface="autograd", diff_method="adjoint")
        def qnode(params, bits):
            circuit_ops(params, bits)
            return qml.expval(qml.PauliZ(0))

        def loss_fn(p):
            preds = pnp.stack([qnode(p, bits) for bits in XOR_INPUTS])
            labels = pnp.array(XOR_LABELS, dtype=float)
            return pnp.mean((preds - labels) ** 2)

        opt = qml.AdamOptimizer(stepsize=LR)
        p = params0.copy()
        opt.step_and_cost(loss_fn, p)  # warm-up
        t0 = time.perf_counter()
        for _ in range(10):
            p, _ = opt.step_and_cost(loss_fn, p)
        dt = (time.perf_counter() - t0) / 10 * 1000
        print(f"  {device_name:16s}: {dt:8.2f} ms/iteration")

    with open("xor_mock_summary_2026-09-23.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_summary[0].keys()))
        w.writeheader(); w.writerows(rows_summary)
    with open("xor_mock_trajectories_2026-09-23.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_traj[0].keys()))
        w.writeheader(); w.writerows(rows_traj)
    print("\nWrote xor_mock_summary_2026-09-23.csv and xor_mock_trajectories_2026-09-23.csv")


if __name__ == "__main__":
    main()
