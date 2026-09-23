"""compile_for_hardware.py -- practice run for the 2026-09-28 submission.

Re-trains seed 0 of Addendum 148's ideal condition (deterministic, same
seed -- reproduces the exact parameters that run reached, since no
trajectory file stores the parameters themselves, only the loss), then
compiles the resulting 4-qubit XOR circuit with PSF-Zero for a heavy-hex
target (the topology real IBM devices use), and verifies the compiled
circuit reproduces the logical circuit's own expectation values exactly.

No IBM Quantum connection is made here -- this only prepares the circuit.
Requires psf_compile.py and psf_smart_layout.py in the same directory.

Usage (inside psf_zero_wsl_env_312):
    python compile_for_hardware.py
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit import qasm2

import psf_compile
from psf_smart_layout import smart_vf2_layout
from psf_pennylane import EdgeListCouplingMap  # reused, not reinvented

N = 4
LAYERS = 3
ITERATIONS = 200
SEED = 0
LR = 0.1
XOR_INPUTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
XOR_LABELS = [-1, 1, 1, -1]
BASIS = ["rz", "sx", "x", "cx"]


def n_params():
    return 2 * N * LAYERS


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
    """Reproduces Addendum 148's own seed-0 training exactly (same seed,
    same optimizer, same iteration count -- deterministic)."""
    params = pnp.array(np.random.default_rng(SEED).uniform(-np.pi, np.pi, n_params()),
                       requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=LR)

    def loss(p):
        preds = pnp.stack([ideal_qnode(p, bits) for bits in XOR_INPUTS])
        labels = pnp.array(XOR_LABELS, dtype=float)
        return pnp.mean((preds - labels) ** 2)

    for _ in range(ITERATIONS):
        params, final_loss = opt.step_and_cost(loss, params)
    return params, final_loss


def build_qiskit_circuit(params, bits):
    """The trained circuit, for one fixed XOR input, as a Qiskit
    QuantumCircuit -- what would actually be sent to hardware, one input
    at a time (a real submission needs one circuit per input to test)."""
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


def heavy_hex_edges(d):
    """Same construction as Addendum 104/106/135's own heavy-hex instances."""
    import rustworkx as rx
    graph = rx.generators.heavy_hex_graph(d)
    edges = list(graph.edge_list())
    return sorted(set(edges) | {(b, a) for a, b in edges})


def main():
    print("Re-training seed 0 (Addendum 148's ideal condition, deterministic)...")
    params, final_loss = retrain_seed0()
    preds = [1 if ideal_qnode(params, bits) > 0 else -1 for bits in XOR_INPUTS]
    correct = sum(int(p == l) for p, l in zip(preds, XOR_LABELS))
    print(f"  final loss {final_loss:.6f}  accuracy {correct}/4  preds {preds}")
    assert correct == 4, "seed 0 did not reproduce Addendum 148's own 4/4 result -- stopping."
    print("  Matches Addendum 148's own recorded seed-0 result (4/4).\n")

    hh_edges = heavy_hex_edges(3)  # smallest heavy-hex big enough for n=4
    hh_n = max(max(e) for e in hh_edges) + 1
    print(f"Target topology: heavy-hex d=3, {hh_n} physical qubits available "
          f"(only {N} needed for this circuit)\n")

    print("Compiling each of the 4 XOR-input circuits with PSF-Zero for this topology...\n")
    for bits in XOR_INPUTS:
        qc = build_qiskit_circuit(params, bits)
        # This circuit has no two-qubit gate saturation concern at n=4 on a
        # much larger device -- included for correctness, not because a
        # cliff is expected here (Addenda 88-136's own cliff needs a
        # saturated, large instance; this is neither).
        pairs = [(0, 1), (1, 2), (2, 3)]
        cm = EdgeListCouplingMap(hh_edges, size=hh_n)
        layout_map, info = smart_vf2_layout(cm, pairs, N)
        if layout_map is None:
            raise RuntimeError(f"no layout found for input {bits} (info={info})")
        perm = [layout_map[i] for i in range(N)]

        with contextlib.redirect_stdout(io.StringIO()):
            compiled = psf_compile.compile(qc, verify=False, entangling_basis="cx")

        placed = QuantumCircuit(hh_n)
        placed.compose(compiled, qubits=perm, inplace=True)

        # Correctness: the placed circuit's own logical-qubit-0 expectation
        # value (via the same layout-tracking method as Addendum 103/122)
        # must match the untransformed circuit's.
        obs = SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits=N)
        ideal_val = Statevector(qc).expectation_value(obs).real
        mapped = obs.apply_layout(perm, num_qubits=hh_n)
        placed_val = Statevector(placed).expectation_value(mapped).real
        cx_count = sum(1 for inst in placed.data if len(inst.qubits) == 2)

        print(f"  input {bits}: 2q gates={cx_count}  ideal_expval={ideal_val:.6f}  "
              f"placed_expval={placed_val:.6f}  diff={abs(ideal_val - placed_val):.2e}")
        assert abs(ideal_val - placed_val) < 1e-9, "compiled circuit does not match logical circuit!"

        # QuantumCircuit.qasm() was removed in Qiskit 1.0; qiskit.qasm2.dumps()
        # is the current replacement (confirmed from Qiskit's own current
        # documentation, not assumed from the old, removed API).
        out_path = f"xor_hw_ready_input_{bits[0]}{bits[1]}.qasm"
        with open(out_path, "w") as f:
            f.write(qasm2.dumps(placed))

    print("\nAll 4 input circuits compiled, verified exact, and ready.")
    print("No IBM connection was made. Submission itself is deferred to 2026-09-28.")


if __name__ == "__main__":
    main()
