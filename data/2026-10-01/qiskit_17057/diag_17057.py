"""diag_17057.py -- Qiskit issue #17057: is the wrong ZSX/CX synthesis near the two-CX boundary a Python->Rust
translation slip, or does the pre-Rust Python algorithm fail in the same way?

Runs in the installed Qiskit (2.5.2). It takes the KAK pieces from the installed (Rust) decomposer,
`decomp3_supercontrolled`, and feeds them to:
  RUST  the installed TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX") end to end;
  PY045 a verbatim port of _get_sx_vz_3cx_efficient_euler from Qiskit 0.45.0
        (qiskit/quantum_info/synthesis/two_qubit_decompose.py, lines 1270-1410), global phase omitted
        because the average gate fidelity does not depend on it.
It also prints the branch variables (x12 and the angles the branches test) so the failing branch can be named.

    python diag_17057.py
"""
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, RXGate
from qiskit.quantum_info import Operator, average_gate_fidelity
from qiskit.synthesis import OneQubitEulerDecomposer, TwoQubitBasisDecomposer
from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition


def target(c):
    core = QuantumCircuit(2)
    core.rxx(-1.2, 0, 1)
    core.ryy(-0.6, 0, 1)
    core.rzz(-2 * c, 0, 1)
    return Operator(core)


def euler_tables(decomposition):
    """The euler_q0 / euler_q1 tables exactly as 0.45 builds them."""
    n = len(decomposition)
    q0 = np.empty((n // 2, 3))
    q1 = np.empty((n // 2, 3))
    zxz, xzx = OneQubitEulerDecomposer("ZXZ"), OneQubitEulerDecomposer("XZX")
    for i, d in enumerate(decomposition[0::2]):
        q0[i, [1, 2, 0]] = zxz.angles_and_phase(d)[:3]
    for i, d in enumerate(decomposition[1::2]):
        q1[i, [1, 2, 0]] = xzx.angles_and_phase(d)[:3]
    return q0, q1


def py045_3cx(decomposition, pulse_optimize=None):
    """Port of Qiskit 0.45.0 TwoQubitBasisDecomposer._get_sx_vz_3cx_efficient_euler (phase lines omitted)."""
    dec1q = OneQubitEulerDecomposer("ZSX")
    euler_q0, euler_q1 = euler_tables(decomposition)
    atol = 1e-10
    qc = QuantumCircuit(2)
    x12 = euler_q0[1][2] + euler_q0[2][0]
    x12_isNonZero = not math.isclose(x12, 0, abs_tol=atol)
    x12_isOddMult = None
    x12_isPiMult = math.isclose(math.sin(x12), 0, abs_tol=atol)
    if x12_isPiMult:
        x12_isOddMult = math.isclose(math.cos(x12), -1, abs_tol=atol)
    x02_add = x12 - euler_q0[1][0]
    x12_isHalfPi = math.isclose(x12, math.pi / 2, abs_tol=atol)

    circ = QuantumCircuit(1)
    circ.rz(euler_q0[0][0], 0)
    circ.rx(euler_q0[0][1], 0)
    if x12_isNonZero and x12_isPiMult:
        circ.rz(euler_q0[0][2] - x02_add, 0)
    else:
        circ.rz(euler_q0[0][2] + euler_q0[1][0], 0)
    circ.h(0)
    qc.compose(dec1q(Operator(circ).data), [0], inplace=True)

    circ = QuantumCircuit(1)
    circ.rx(euler_q1[0][0], 0)
    circ.rz(euler_q1[0][1], 0)
    circ.rx(euler_q1[0][2] + euler_q1[1][0], 0)
    circ.h(0)
    qc.compose(dec1q(Operator(circ).data), [1], inplace=True)

    qc.cx(1, 0)
    branch = []
    if x12_isPiMult:
        if x12_isNonZero and x12_isOddMult:
            qc.rz(-euler_q0[1][1], 0)
            branch.append("pi-mult odd: rz(-theta1)")
        else:
            qc.rz(euler_q0[1][1], 0)
            branch.append("pi-mult even/zero: rz(theta1)")
    if x12_isHalfPi:
        qc.sx(0)
        branch.append("half-pi: sx")
    elif x12_isNonZero and not x12_isPiMult:
        if pulse_optimize is None:
            qc.compose(dec1q(Operator(RXGate(x12)).data), [0], inplace=True)
            branch.append("other: rx(x12) only")
        else:
            raise RuntimeError("possible non-pulse-optimal decomposition encountered")
    if math.isclose(euler_q1[1][1], math.pi / 2, abs_tol=atol):
        qc.sx(1)
    else:
        qc.compose(dec1q(Operator(RXGate(euler_q1[1][1])).data), [1], inplace=True)
        branch.append("q1 middle 1 not pi/2")
    qc.rz(euler_q1[1][2] + euler_q1[2][0], 1)
    qc.cx(1, 0)
    qc.rz(euler_q0[2][1], 0)
    if math.isclose(euler_q1[2][1], math.pi / 2, abs_tol=atol):
        qc.sx(1)
    else:
        qc.compose(dec1q(Operator(RXGate(euler_q1[2][1])).data), [1], inplace=True)
        branch.append("q1 middle 2 not pi/2")
    qc.cx(1, 0)

    circ = QuantumCircuit(1)
    circ.h(0)
    circ.rz(euler_q0[2][2] + euler_q0[3][0], 0)
    circ.rx(euler_q0[3][1], 0)
    circ.rz(euler_q0[3][2], 0)
    qc.compose(dec1q(Operator(circ).data), [0], inplace=True)
    circ = QuantumCircuit(1)
    circ.h(0)
    circ.rx(euler_q1[2][2] + euler_q1[3][0], 0)
    circ.rz(euler_q1[3][1], 0)
    circ.rx(euler_q1[3][2], 0)
    qc.compose(dec1q(Operator(circ).data), [1], inplace=True)
    info = dict(x12=x12, sin_x12=math.sin(x12), theta1=euler_q0[1][1], theta2=euler_q0[2][1],
                q0_1=list(euler_q0[1]), q0_2=list(euler_q0[2]), branch=branch)
    return qc, info


def infid(circ, u):
    return 1 - average_gate_fidelity(Operator(circ), u)


def main():
    import qiskit
    print("qiskit", qiskit.__version__)
    zsx = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")
    plain = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX", pulse_optimize=False)
    print(f"{'c':>9} {'RUST cx':>7} {'RUST infid':>11} {'no-pulse':>10} {'PY045 infid':>11} {'x12':>12} "
          f"{'sin(x12)':>10} {'theta1':>9} {'theta2':>10}  branch")
    for c in (1e-9, 1e-8, 2e-8, 3e-8, 5e-8, 1e-7, 2e-7, 3e-7, 5e-7, 1e-6, 1e-5, 1e-3, 0.1):
        u = target(c)
        out = zsx(u.data)
        ncx = sum(1 for ins in out.data if ins.operation.name == "cx")
        dec = zsx.decomp3_supercontrolled(TwoQubitWeylDecomposition(u.data)._inner_decomposition)
        py, info = py045_3cx(dec)
        print(f"{c:9.0e} {ncx:7d} {infid(out, u):11.3e} {infid(plain(u.data), u):10.1e} {infid(py, u):11.3e} "
              f"{info['x12']:12.8f} {info['sin_x12']:10.2e} {info['theta1']:9.5f} {info['theta2']:10.3e}  "
              f"{'; '.join(info['branch'])}")
    u = target(1e-7)
    dec = zsx.decomp3_supercontrolled(TwoQubitWeylDecomposition(u.data)._inner_decomposition)
    _, info = py045_3cx(dec)
    print("c=1e-7 euler_q0[1] (lam, theta, phi):", info["q0_1"])
    print("c=1e-7 euler_q0[2] (lam, theta, phi):", info["q0_2"])


if __name__ == "__main__":
    main()
