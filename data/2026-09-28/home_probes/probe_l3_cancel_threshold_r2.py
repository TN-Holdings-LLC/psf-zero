"""probe_l3_cancel_threshold_r2.py -- exploratory (home, 2026-09-28): at what merged angle does
Qiskit's CommutativeCancellation drop a merged Z rotation as the identity?

probe_l3_cancel.py showed that in both real-target pairs of Addendum 245 the pass merged
rz(x) . cz . rz(pi/2 - ...) on one qubit into a single rotation of about 1e-4
(-1.0772e-4 and +7.1466e-5) and then removed it; the pair's operator error equals that
angle. This script finds the angle below which the pass removes the merged rotation, on
2-qubit toy circuits (no Target, no transpile), for several shapes:

  across_cz  rz(a) on q0, cz(q0, q1), rz(theta - a) on q0   (the shape seen in the probe)
  adjacent   rz(a), rz(theta - a) on q0, no cz
  single     rz(theta) alone on q0 (nothing to merge)

with a = -pi/2 and a = 0.3, both signs of theta, and approximation_degree None / 1.0 / 0.99
when the installed CommutativeCancellation accepts it (revision 2: Qiskit 2.5.2 does not,
so those variants are skipped instead of raising). Revision 2 also checks merged angles
near 2*pi and 4*pi (theta + 2*pi, theta + 4*pi) for the across_cz shape.
For each setting it bisects the boundary between "removed" and "kept" on |theta| in
[1e-9, 1e-1] and reports it together with the operator distance at the boundary.

    python -u benchmarks/probe_l3_cancel_threshold_r2.py 2>&1 | tee probe_l3_cancel_threshold_r2.txt
"""
from __future__ import annotations

import inspect
import math
import os
import sys

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CommutativeCancellation

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks"), os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import loop_endurance as le  # noqa: E402


def aligned(u, v):
    """Phase-aligned Frobenius distance min_phi ||u - e^{i phi} v||."""
    t = np.trace(v.conj().T @ u)
    ph = t / abs(t) if abs(t) > 0 else 1.0
    return float(np.linalg.norm(u - ph * v))


APPROX_OK = "approximation_degree" in inspect.signature(CommutativeCancellation.__init__).parameters


def build(shape, a, theta):
    qc = QuantumCircuit(2)
    if shape == "across_cz":
        qc.rz(a, 0)
        qc.cz(0, 1)
        qc.rz(theta - a, 0)
    elif shape == "adjacent":
        qc.rz(a, 0)
        qc.rz(theta - a, 0)
    else:
        qc.rz(theta, 0)
    return qc


def run(shape, a, theta, approx):
    qc = build(shape, a, theta)
    kw = {} if approx is None else {"approximation_degree": approx}
    out = PassManager([CommutativeCancellation(**kw)]).run(qc)
    n_rz = sum(1 for inst in out.data if inst.operation.name == "rz")
    return n_rz == 0, aligned(Operator(qc).data, Operator(out).data), out


def boundary(shape, a, sign, approx, offset=0.0):
    """Largest |theta| that is removed, by bisection in log space (None if none/all).
    The merged angle is offset + sign * |theta|."""
    lo, hi = 1e-9, 1e-1
    rem_lo, _, _ = run(shape, a, offset + sign * lo, approx)
    rem_hi, _, _ = run(shape, a, offset + sign * hi, approx)
    if not rem_lo:
        return "never removed (even 1e-9)"
    if rem_hi:
        return "removed up to 1e-1"
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if run(shape, a, offset + sign * mid, approx)[0]:
            lo = mid
        else:
            hi = mid
    _, d_lo, _ = run(shape, a, offset + sign * lo, approx)
    _, d_hi, _ = run(shape, a, offset + sign * hi, approx)
    return f"removed below |theta| = {lo:.6e} (distance there {d_lo:.3e}); kept from {hi:.6e} (distance {d_hi:.3e})"


def main():
    print("SCRIPT", os.path.abspath(__file__), le.normalized_sha256(os.path.abspath(__file__)))
    print("qiskit", qiskit.__version__, "| approximation_degree accepted:", APPROX_OK)
    print("reference: 4*pi*1e-5 =", f"{4 * math.pi * 1e-5:.6e}", "| 2*pi*1e-5 =", f"{2 * math.pi * 1e-5:.6e}")
    # the two angles of the real-target cases, checked directly
    for theta in (-1.0772320510343825e-4, 7.146579489658578e-05):
        for a in (-math.pi / 2, 0.3):
            rem, d, out = run("across_cz", a, theta, None)
            print(f"case theta {theta:+.6e} a {a:+.6f}: removed {rem}, distance {d:.3e}, gates {[i.operation.name for i in out.data]}")
    print()
    for shape in ("across_cz", "adjacent", "single"):
        for a in (-math.pi / 2, 0.3):
            if shape == "single" and a != -math.pi / 2:
                continue
            for sign in (+1, -1):
                for approx in ((None, 1.0, 0.99) if APPROX_OK else (None,)):
                    print(f"{shape:9s} a {a:+.6f} sign {sign:+d} approx {approx}: {boundary(shape, a, sign, approx)}", flush=True)
    print()
    for offset, label in ((2 * math.pi, "2*pi"), (4 * math.pi, "4*pi")):
        for sign in (+1, -1):
            print(f"across_cz a -1.570796 offset {label} sign {sign:+d}: "
                  f"{boundary('across_cz', -math.pi / 2, sign, None, offset)}", flush=True)
    print("\nDONE")


if __name__ == "__main__":
    main()
