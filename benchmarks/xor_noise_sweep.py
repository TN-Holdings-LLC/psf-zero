"""xor_noise_sweep.py -- Addendum 149.

Sweeps CX depolarizing error to find where XOR training (Addendum 147/148's
own noisy condition) stops converging -- a safety margin check before
sending anything to real IBM hardware on 2026-09-28.

Reuses Addendum 147/148's own circuit, task and training procedure exactly
(density_matrix AerSimulator, CPU, parameter-shift gradients, 200
iterations, 5 seeds); only the noise level is swept.

Usage (inside psf_zero_wsl_env_312):
    python xor_noise_sweep.py
"""
from __future__ import annotations

import csv

import numpy as np
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
P2_LEVELS = (0.01, 0.02, 0.05, 0.1, 0.2)
BASIS = ["rz", "sx", "x", "cx"]


def n_params():
    return 2 * N * LAYERS


def noise_model(p2):
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p2 / 10, 1), ["sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return nm


def build_qc(params, bits):
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


_obs0 = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=N)


def expval(sim, params, bits):
    qc = build_qc(params, bits)
    qc.save_density_matrix()
    res = sim.run(qc).result()
    return res.data(0)["density_matrix"].expectation_value(_obs0).real


def grad(sim, params, preds_now):
    labels = np.array(XOR_LABELS, dtype=float)
    loss_grad_wrt_pred = 2.0 * (preds_now - labels) / len(labels)
    g = np.zeros_like(params)
    for i in range(len(params)):
        shift = np.zeros_like(params)
        shift[i] = np.pi / 2
        preds_p = np.array([expval(sim, params + shift, b) for b in XOR_INPUTS])
        preds_m = np.array([expval(sim, params - shift, b) for b in XOR_INPUTS])
        dpred_di = 0.5 * (preds_p - preds_m)
        g[i] = float(np.dot(loss_grad_wrt_pred, dpred_di))
    return g


def train(sim, seed):
    params = np.random.default_rng(seed).uniform(-np.pi, np.pi, n_params())
    m1 = np.zeros_like(params); v1 = np.zeros_like(params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    history = []
    for t in range(1, ITERATIONS + 1):
        preds_now = np.array([expval(sim, params, b) for b in XOR_INPUTS])
        loss = float(np.mean((preds_now - np.array(XOR_LABELS, dtype=float)) ** 2))
        history.append(loss)
        g = grad(sim, params, preds_now)
        m1 = beta1 * m1 + (1 - beta1) * g
        v1 = beta2 * v1 + (1 - beta2) * g ** 2
        mhat = m1 / (1 - beta1 ** t)
        vhat = v1 / (1 - beta2 ** t)
        params = params - LR * mhat / (np.sqrt(vhat) + eps)
    return params, history


def converged_by(history, absolute_threshold=0.1):
    """Returns the first iteration at which loss drops below a fixed,
    ABSOLUTE threshold -- not a threshold relative to the final value.

    A first version compared each point against `final * tol_factor`
    (i.e. "close to wherever training ended up"). Found, before running
    anything, to misfire exactly on the case this whole check exists to
    detect: a training run that never improves (loss oscillating near a
    BAD value throughout) has every early point already "close to" its
    own bad final value, so the old check reported convergence at
    iteration 0 for a run that never actually solved anything. A fixed,
    absolute threshold (well below this problem's own random-guess loss
    of ~2, and comfortably above Addendum 148's own converged values of
    0.0009-0.0016) does not have this failure mode: a run stuck near
    loss ~2 never crosses 0.1, and correctly returns None.
    """
    for i, v in enumerate(history):
        if v <= absolute_threshold:
            return i
    return None


def main():
    rows_summary, rows_traj = [], []
    print(f"CX depolarizing error sweep: {P2_LEVELS} (1q error = p2/10 throughout)\n")

    for p2 in P2_LEVELS:
        sim = AerSimulator(method="density_matrix", device="CPU",
                           noise_model=noise_model(p2), basis_gates=BASIS)
        print(f"=== p2={p2} ===")
        accs = []
        for seed in SEEDS:
            params, history = train(sim, seed)
            preds = [1 if expval(sim, params, b) > 0 else -1 for b in XOR_INPUTS]
            correct = sum(int(p == l) for p, l in zip(preds, XOR_LABELS))
            accs.append(correct)
            conv_iter = converged_by(history)
            print(f"  seed {seed}: final loss {history[-1]:.4f}  accuracy {correct}/4  "
                  f"converged by iter {conv_iter}")
            rows_summary.append(dict(p2=p2, seed=seed, final_loss=history[-1],
                                     accuracy=correct, converged_by_iter=conv_iter))
            rows_traj += [dict(p2=p2, seed=seed, iteration=i, loss=l) for i, l in enumerate(history)]
        print(f"  mean accuracy: {sum(accs)/len(accs):.2f}/4  "
              f"(seeds at 4/4: {sum(1 for a in accs if a == 4)}/{len(SEEDS)})\n")

    with open("xor_noise_sweep_summary_2026-09-23.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_summary[0].keys()))
        w.writeheader(); w.writerows(rows_summary)
    with open("xor_noise_sweep_trajectories_2026-09-23.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_traj[0].keys()))
        w.writeheader(); w.writerows(rows_traj)
    print("Wrote xor_noise_sweep_summary_2026-09-23.csv and xor_noise_sweep_trajectories_2026-09-23.csv")


if __name__ == "__main__":
    main()
