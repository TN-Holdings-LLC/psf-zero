"""ka_frobenius_recheck.py -- re-checks the meaning test of the compound
round-trip chain (workplace roundtrip-chain, home Addendum 175/176) with the
phase-aligned Frobenius distance instead of the average gate infidelity.

Why: infidelity is quadratic in the operator error, so the chain's meaning
tolerance of 1e-12 in infidelity admits operator errors of order 1e-6. The
chain's tapes, converters and 100 steps are rebuilt exactly as in
roundtrip_compound_chain.py; at every step the Qiskit circuit is compared with
the ORIGINAL tape's PennyLane matrix by both measures.

    python -u ka_frobenius_recheck.py <repository root> <pre-fix converter file>

Exploratory (not pre-registered). Deterministic on a given machine.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys

import numpy as np
import pennylane as qml
from qiskit.quantum_info import Operator, random_unitary

REPO, OLD_PATH = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(REPO, "benchmarks"))
NEW_PATH = os.path.join(REPO, "benchmarks", "psf_pennylane_gpu_prototype.py")
STEPS = 100
ONE_Q = ("Hadamard", "PauliX", "PauliY", "PauliZ", "RX", "RY", "RZ")
CNOT_MAT = qml.matrix(qml.CNOT(wires=[0, 1]), wire_order=[0, 1])


def norm_sha(path):
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f.read().strip().splitlines()]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def infidelity(u, v):
    d = u.shape[0]
    tr = np.trace(u.conj().T @ v)
    return max(0.0, 1.0 - float((abs(tr) ** 2 + d) / (d * (d + 1))))


def frobenius(u, v):
    z = np.vdot(v, u)
    z = z / abs(z) if abs(z) > 0 else 1.0
    return float(np.linalg.norm(u - z * v))


def random_tape(seed, wires, n_ops=20):
    rng = np.random.default_rng(seed)
    ops = []
    for k in range(n_ops):
        if k % 2 == 0:
            a, b = rng.choice(len(wires), size=2, replace=False)
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            ops.append(qml.QubitUnitary(u, wires=[wires[a], wires[b]]))
        else:
            name = ONE_Q[int(rng.integers(0, len(ONE_Q)))]
            w = wires[int(rng.integers(0, len(wires)))]
            if name in ("RX", "RY", "RZ"):
                ops.append(getattr(qml, name)(float(rng.uniform(-np.pi, np.pi)), wires=w))
            else:
                ops.append(getattr(qml, name)(wires=w))
    return qml.tape.QuantumTape(ops, measurements=[], shots=None)


def xor_tapes():
    import rehearse_xor_fake127 as R
    res = R.retrain_seed0()
    params = res[0] if isinstance(res, tuple) else res
    tapes = []
    for bits in R.XOR_INPUTS:
        with qml.queuing.AnnotatedQueue() as q:
            R.circuit_ops(params, bits)
        raw = qml.tape.QuantumScript.from_queue(q)
        ops = [qml.QubitUnitary(CNOT_MAT, wires=list(op.wires)) if op.name == "CNOT" else op
               for op in raw.operations]
        tapes.append(qml.tape.QuantumTape(ops, measurements=[], shots=None))
    return tapes


def chain(conv, tape0):
    tape, fro, inf = tape0, [], []
    for _ in range(STEPS):
        qc, wo = conv.tape_to_qiskit(tape)
        u = qml.matrix(tape0, wire_order=list(wo))
        v = Operator(qc).reverse_qargs().data
        fro.append(frobenius(u, v)); inf.append(infidelity(u, v))
        tape = conv.qiskit_to_tape(qc, wo)
    return fro, inf


def main():
    print("NEW", os.path.relpath(NEW_PATH, REPO), norm_sha(NEW_PATH))
    print("OLD", os.path.basename(OLD_PATH), norm_sha(OLD_PATH))
    new, old = load("conv_new", NEW_PATH), load("conv_old", OLD_PATH)
    fams = [("F1_int_wires", random_tape(s, [0, 1, 2, 3])) for s in range(10)]
    fams += [("F2_mixed_labels", random_tape(100 + s, ["q3", "a", 7, "b"])) for s in range(5)]
    fams += [("F3_xor", t) for t in xor_tapes()]
    summary = {}
    for fam, t in fams:
        for lab, conv in (("NEW", new), ("OLD", old)):
            fro, inf = chain(conv, t)
            s = summary.setdefault((fam, lab), {"tapes": 0, "fro_max": 0.0, "fro_min": np.inf,
                                                 "inf_max": 0.0, "distinct_per_chain": 0})
            s["tapes"] += 1
            s["fro_max"] = max(s["fro_max"], max(fro)); s["fro_min"] = min(s["fro_min"], min(fro))
            s["inf_max"] = max(s["inf_max"], max(inf))
            s["distinct_per_chain"] = max(s["distinct_per_chain"], len(set(fro)))
    print("\nfamily            conv  tapes  Frobenius min..max        infidelity max   distinct Frobenius values per chain")
    for (fam, lab), s in summary.items():
        print(f"{fam:17s} {lab:4s}  {s['tapes']:5d}  {s['fro_min']:.2e} .. {s['fro_max']:.2e}   "
              f"{s['inf_max']:.2e}         {s['distinct_per_chain']}")
    nmax = max(s["fro_max"] for (f, l), s in summary.items() if l == "NEW")
    omin = min(s["fro_min"] for (f, l), s in summary.items() if l == "OLD")
    print(f"\nNEW: largest phase-aligned Frobenius distance over all tapes and steps = {nmax:.2e}")
    print(f"OLD: smallest phase-aligned Frobenius distance over all tapes and steps = {omin:.2e}")


if __name__ == "__main__":
    main()
