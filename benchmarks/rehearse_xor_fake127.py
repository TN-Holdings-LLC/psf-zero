"""rehearse_xor_fake127.py -- Addendum 163.

2026-09-28 rehearsal without credentials: the trained XOR classifier
(Addendum 148, seed 0) through the verified routing / logical-measurement /
SamplerV2 path (Addenda 155-162), on FakeBrisbane (127-qubit heavy-hex
snapshot), plus a diagnostic of a suspected qubit-order bug in the
prototype's tape_to_qiskit (P5).

Usage (WSL, psf_zero_wsl_env_fresh):
    python rehearse_xor_fake127.py 2>&1 | tee rehearse_result.txt
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

from psf_ibm_real_submit import make_sampler_submit_fn
from psf_pennylane_gpu_ibm_prototype import is_isa_compliant, logical_measurement, route_for_backend
from psf_pennylane_gpu_prototype import collect_and_consolidate, tape_to_qiskit

N = 4
LAYERS = 3
ITERATIONS = 200
SEED = 0
LR = 0.1
XOR_INPUTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
XOR_LABELS = [-1, 1, 1, -1]
SHOTS = 4000
SIM_SEED = 42


def circuit_ops(params, bits):
    qml.RX(np.pi * bits[0], wires=0)
    qml.RX(np.pi * bits[1], wires=1)
    k = 0
    for _ in range(LAYERS):
        for q in range(N):
            qml.RY(params[k], wires=q); k += 1
            qml.RZ(params[k], wires=q); k += 1
        for a in range(N - 1):
            qml.CNOT(wires=[a, a + 1])


dev = qml.device("default.qubit", wires=N)


@qml.qnode(dev, interface="autograd", diff_method="backprop")
def ideal_qnode(params, bits):
    circuit_ops(params, bits)
    return qml.expval(qml.PauliZ(0))


def retrain_seed0():
    """Addendum 148's seed-0 ideal training, deterministic (Addendum 151
    reproduced its exact final loss)."""
    params = pnp.array(np.random.default_rng(SEED).uniform(-np.pi, np.pi, 2 * N * LAYERS), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=LR)

    def loss(p):
        preds = pnp.stack([ideal_qnode(p, b) for b in XOR_INPUTS])
        return pnp.mean((preds - pnp.array(XOR_LABELS, dtype=float)) ** 2)

    for _ in range(ITERATIONS):
        params, final_loss = opt.step_and_cost(loss, params)
    return params, float(final_loss)


def build_qiskit(params, bits):
    """Gate-by-gate, Qiskit qubit i = PennyLane wire i, named gates only --
    no matrix is handed across the library boundary, so no bit-order
    convention is involved (Addendum 163, P5)."""
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


def z0_from_counts(counts):
    total = sum(counts.values())
    return sum(c * (1 if key[-1] == "0" else -1) for key, c in counts.items()) / total


def diagnostic_p5():
    """Does tape_to_qiskit preserve a 2-wire QubitUnitary's meaning?"""
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    tape = qml.tape.QuantumTape([qml.QubitUnitary(cnot, wires=[0, 1])], measurements=[], shots=None)
    qc, _ = tape_to_qiskit(tape)
    qiskit_op = Operator(qc).data
    # PennyLane's own matrix for the tape, written in Qiskit's bit order
    # (wire 1 as the most significant index).
    pennylane_in_qiskit_order = qml.matrix(tape, wire_order=[1, 0])
    same = np.allclose(qiskit_op, pennylane_in_qiskit_order)
    print(f"P5 diagnostic: tape_to_qiskit preserves 2-wire QubitUnitary meaning? {same}")
    return same


def main():
    print("Retraining seed 0 ...")
    params, final_loss = retrain_seed0()
    print(f"  final loss {final_loss:.6f}")

    backend = FakeBrisbane()
    noisy_submit = make_sampler_submit_fn(backend, seed_simulator=SIM_SEED)
    ideal_submit = make_sampler_submit_fn(AerSimulator(), seed_simulator=SIM_SEED)
    obs = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=N)

    print(f"\nBackend: {backend.name}, {backend.num_qubits} qubits\n")
    print(f"{'input':>6} {'label':>5} {'PL <Z0>':>9} {'QK <Z0>':>9} {'|diff|':>8} {'blocks':>6} "
          f"{'routed2q':>8} {'bits':>4} {'ideal<Z0>':>9} {'noisy<Z0>':>9} {'ideal ok':>8} {'noisy ok':>8}")
    for bits, label in zip(XOR_INPUTS, XOR_LABELS):
        pl = float(ideal_qnode(params, bits))
        qc = build_qiskit(params, bits)
        qk = float(Statevector(qc).expectation_value(obs).real)
        n_blocks = sum(1 for i in collect_and_consolidate(qc).data if i.operation.name == "unitary")
        qc_routed = route_for_backend(qc, backend)
        ok, reason = is_isa_compliant(qc_routed, backend)
        if not ok:
            raise RuntimeError(reason)
        measured = logical_measurement(qc_routed)
        routed2q = sum(1 for i in qc_routed.data if len(i.qubits) == 2)
        c_ideal = ideal_submit(measured, SHOTS)
        c_noisy = noisy_submit(measured, SHOTS)
        widths = {len(k) for k in c_noisy}
        zi, zn = z0_from_counts(c_ideal), z0_from_counts(c_noisy)
        print(f"{str(bits[0]) + str(bits[1]):>6} {label:>5} {pl:>9.5f} {qk:>9.5f} {abs(pl - qk):>8.1e} {n_blocks:>6} "
              f"{routed2q:>8} {str(widths):>4} {zi:>9.4f} {zn:>9.4f} "
              f"{str(np.sign(zi) == label):>8} {str(np.sign(zn) == label):>8}", flush=True)

    print()
    diagnostic_p5()


if __name__ == "__main__":
    main()
