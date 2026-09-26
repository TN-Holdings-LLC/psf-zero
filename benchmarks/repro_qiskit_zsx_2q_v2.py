"""repro_qiskit_zsx_2q_v2.py -- does ordinary transpile() hit the ZSX decomposer failure?

repro_qiskit_zsx_2q.py showed TwoQubitBasisDecomposer(CXGate(),
euler_basis="ZSX") returning a 3-CX circuit with 1 - F_avg = 7.0e-2 for
exp(i(0.6 XX + 0.3 YY + c ZZ)) at c = 3e-8 .. 3e-7, while the default Euler
basis was exact. transpile() of the same gate written as RXX/RYY/RZZ was
fine at every level. The earlier window scan, which fed transpile() a
UnitaryGate of the core dressed with single-qubit layers, found failures at
levels 1-3. This script checks that difference with fixed, explicit inputs
(no random seeds):

  input forms   (1) UnitaryGate(core)              (2) UnitaryGate(dressed core)
                (3) the dressed core as gates (u + rxx/ryy/rzz + u)
  targets       basis_gates=[cx, rz, sx, x], levels 0-3
                basis_gates=[cz, rz, sx, x], levels 1 and 3
                GenericBackendV2(2 qubits, cz/rz/sx/x, with error data), levels 1 and 3

Cells: 1 - average gate fidelity against the input (number of 2q gates).
Usage:
    python -u repro_qiskit_zsx_2q_v2.py 2>&1 | tee repro_qiskit_zsx_2q_v2.txt
"""
import platform

import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator, average_gate_fidelity

A, B = 0.6, 0.3
CS = [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 1e-5]
CONTRACT = 1e-8
# Fixed single-qubit dressing: u(theta, phi, lam) on q0 and q1, before and after.
PRE = [(0.7, 0.2, -1.1), (1.9, -0.4, 0.3)]
POST = [(2.3, 1.0, 0.5), (0.4, -2.2, 1.7)]


def core_gates(qc, c):
    qc.rxx(-2 * A, 0, 1)
    qc.ryy(-2 * B, 0, 1)
    qc.rzz(-2 * c, 0, 1)


def dressed_gates(c):
    qc = QuantumCircuit(2)
    for q, (t, p, l) in enumerate(PRE):
        qc.u(t, p, l, q)
    core_gates(qc, c)
    for q, (t, p, l) in enumerate(POST):
        qc.u(t, p, l, q)
    return qc


def as_unitary_gate(qc):
    out = QuantumCircuit(2)
    out.append(UnitaryGate(Operator(qc).data), [0, 1])
    return out


def main():
    print(f"qiskit {qiskit.__version__} | python {platform.python_version()} | {platform.platform()}")
    backend = GenericBackendV2(num_qubits=2, basis_gates=["cz", "rz", "sx", "x"], seed=0)
    targets = {}
    for lvl in range(4):
        targets[f"[cx,rz,sx,x] L{lvl}"] = dict(basis_gates=["cx", "rz", "sx", "x"], optimization_level=lvl)
    for lvl in (1, 3):
        targets[f"[cz,rz,sx,x] L{lvl}"] = dict(basis_gates=["cz", "rz", "sx", "x"], optimization_level=lvl)
    for lvl in (1, 3):
        targets[f"GenericBackendV2 L{lvl}"] = dict(backend=backend, optimization_level=lvl)

    bad = []
    for form in ("UnitaryGate(core)", "UnitaryGate(dressed)", "dressed as gates"):
        print(f"\n--- input: {form} ---")
        print(f"{'c':>8s} | " + " | ".join(f"{t:>22s}" for t in targets))
        for c in CS:
            core = QuantumCircuit(2)
            core_gates(core, c)
            qc = {"UnitaryGate(core)": as_unitary_gate(core),
                  "UnitaryGate(dressed)": as_unitary_gate(dressed_gates(c)),
                  "dressed as gates": dressed_gates(c)}[form]
            u = Operator(qc)
            cells = []
            for tname, kw in targets.items():
                out = transpile(qc, seed_transpiler=0, **kw)
                v = Operator.from_circuit(out) if out.layout is not None else Operator(out)
                infid = 1.0 - average_gate_fidelity(v, u)
                n2 = sum(1 for i in out.data if len(i.qubits) == 2)
                cells.append(f"{infid:.2e} ({n2})")
                if infid > CONTRACT:
                    bad.append((form, tname, c, infid, n2))
            print(f"{c:8.1e} | " + " | ".join(f"{x:>22s}" for x in cells))
    print(f"\nabove {CONTRACT:g}: {len(bad)} cells")
    for form, tname, c, infid, n2 in bad:
        print(f"  {form:22s} {tname:22s} c={c:.1e}  1-F_avg={infid:.3e}  2q={n2}")


if __name__ == "__main__":
    main()
