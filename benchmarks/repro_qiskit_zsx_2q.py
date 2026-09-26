"""repro_qiskit_zsx_2q.py -- minimal, PSF-Zero-free reproduction.

Input: the canonical two-qubit gate exp(i(a XX + b YY + c ZZ)) with
a = 0.6, b = 0.3 and a small c, written with Qiskit's own RXX/RYY/RZZ.
Checked: TwoQubitBasisDecomposer(CXGate()) with euler_basis "ZSX" and with
the default, and transpile(basis_gates=["cx", "rz", "sx", "x"]) at
optimization levels 0-3.

Reported per method: 1 - average gate fidelity against the input. Qiskit's
two-qubit synthesis may approximate (drop a tiny interaction) when the
result stays within its requested fidelity, 1 - 1e-9 by default, so a value
below ~1e-9 is expected behaviour, and anything far above it is not.

Usage:
    python -u repro_qiskit_zsx_2q.py 2>&1 | tee repro_qiskit_zsx_2q.txt
"""
import platform

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Operator, average_gate_fidelity
from qiskit.synthesis import TwoQubitBasisDecomposer

A, B = 0.6, 0.3
CS = [3e-9, 1e-8, 3e-8, 1e-7, 2e-7, 3e-7, 5e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
CONTRACT = 1e-8  # generous margin above the default requested fidelity 1 - 1e-9


def canonical(a, b, c):
    qc = QuantumCircuit(2)
    qc.rxx(-2 * a, 0, 1)
    qc.ryy(-2 * b, 0, 1)
    qc.rzz(-2 * c, 0, 1)
    return qc


def main():
    print(f"qiskit {qiskit.__version__} | python {platform.python_version()} | {platform.platform()}")
    dec_zsx = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")
    dec_def = TwoQubitBasisDecomposer(CXGate())
    methods = {
        "TwoQubitBasisDecomposer(CX, 'ZSX')": lambda qc: dec_zsx(Operator(qc).data),
        "TwoQubitBasisDecomposer(CX)": lambda qc: dec_def(Operator(qc).data),
    }
    for lvl in range(4):
        methods[f"transpile L{lvl} [cx,rz,sx,x]"] = (
            lambda qc, lvl=lvl: transpile(qc, basis_gates=["cx", "rz", "sx", "x"],
                                          optimization_level=lvl, seed_transpiler=0))
    print(f"\ninput: exp(i({A} XX + {B} YY + c ZZ)); cells: 1 - average gate fidelity (number of CX)")
    print(f"{'c':>8s} | " + " | ".join(f"{m:>34s}" for m in methods))
    bad = []
    for c in CS:
        qc = canonical(A, B, c)
        u = Operator(qc)
        cells = []
        for name, fn in methods.items():
            out = fn(qc)
            infid = 1.0 - average_gate_fidelity(Operator(out), u)
            ncx = out.count_ops().get("cx", 0)
            cells.append(f"{infid:26.3e} ({ncx} CX)")
            if infid > CONTRACT:
                bad.append((name, c, infid, ncx))
        print(f"{c:8.1e} | " + " | ".join(f"{x:>34s}" for x in cells))
    print(f"\nabove {CONTRACT:g}: {len(bad)} cells")
    for name, c, infid, ncx in bad:
        print(f"  {name:36s} c={c:.1e}  1-F_avg={infid:.3e}  CX={ncx}")
    if bad:
        name, c, _, _ = max(bad, key=lambda x: x[2])
        print(f"\nminimal case: canonical({A}, {B}, {c:g}) through {name}")


if __name__ == "__main__":
    main()
