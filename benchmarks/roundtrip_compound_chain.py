"""roundtrip_compound_chain.py -- compound (chained) round-trip test (workplace, number TBD).

Implements exactly the design locked in
"Addendum (number TBD, workplace) -- Pre-registration: compound round-trip
chain of the PennyLane <-> Qiskit conversion" (2026-09-25).

Each step feeds the previous step's output back in:
    tape_k --tape_to_qiskit--> qc_k --qiskit_to_tape--> tape_{k+1}
for 100 steps, with two converters loaded side by side:
  NEW  /root/psf-zero/benchmarks/psf_pennylane_gpu_prototype.py (bit-order fix)
  OLD  ~/pennylane_gpu_mock_test/psf_pennylane_gpu_prototype.py  (pre-fix copy;
       positive control)
At every step two things are checked against the ORIGINAL tape:
  round trip : is tape_{k+1} exactly the original (fingerprint), and its matrix?
  meaning    : does qc_k mean what the original tape means (PennyLane matrix vs
               Qiskit operator in PennyLane's qubit order)?
Families: F1 10 random tapes on wires 0-3; F2 5 random tapes on mixed wire
labels; F3 the four 2026-09-28 XOR tapes with each CNOT written as a 2-qubit
QubitUnitary. Both converters run on every tape.
No timing, no GPU, no backend. Deterministic: rerunning gives the same file.

Run (from ~/pennylane_gpu_mock_test):
    python -u roundtrip_compound_chain.py 2>&1 | tee ~/roundtrip_chain_run.txt
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
import sys

REPO_BENCH = "/root/psf-zero/benchmarks"
sys.path.insert(0, REPO_BENCH)

import numpy as np
import pennylane as qml
from qiskit.quantum_info import Operator, random_unitary

STEPS = 100
NEW_PATH = os.path.join(REPO_BENCH, "psf_pennylane_gpu_prototype.py")
OLD_PATH = os.path.expanduser("~/pennylane_gpu_mock_test/psf_pennylane_gpu_prototype.py")
OUT = os.path.expanduser("~/roundtrip_chain_2026-09-25.csv")
MEANING_TOL = 1e-12
OLD_DETECT = 1e-3


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def infidelity(u, v):
    d = u.shape[0]
    tr = np.trace(u.conj().T @ v)
    return max(0.0, 1.0 - float((abs(tr) ** 2 + d) / (d * (d + 1))))


def fingerprint(tape):
    items = []
    for op in tape.operations:
        ps = []
        for p in op.parameters:
            a = np.asarray(p)
            ps.append(float(a).hex() if a.ndim == 0 and np.isrealobj(a) else a.astype(complex).tobytes().hex())
        items.append((op.name, tuple(op.wires.tolist()), tuple(ps)))
    return hashlib.sha256(repr(items).encode()).hexdigest()[:16]


ONE_Q = ("Hadamard", "PauliX", "PauliY", "PauliZ", "RX", "RY", "RZ")


def random_tape(seed, wires, n_ops=20):
    rng = np.random.default_rng(seed)
    ops = []
    for k in range(n_ops):
        if k % 2 == 0:
            a, b = rng.choice(len(wires), size=2, replace=False)  # ordered pair, often reversed
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


CNOT_MAT = qml.matrix(qml.CNOT(wires=[0, 1]), wire_order=[0, 1])


def xor_tapes():
    """The 2026-09-28 XOR tapes with every CNOT written as a 2-qubit
    QubitUnitary (the converter has no CNOT mapping and raises on it; the
    2026-09-28 path itself builds its circuit gate by gate and never calls the
    converter). CNOT is not symmetric in its two qubits, so a bit-order error
    in the converter changes these tapes' meaning."""
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


def chain(conv, tape0, family, name, label):
    w0 = list(tape0.wires)
    u0 = qml.matrix(tape0, wire_order=w0)
    fp0 = fingerprint(tape0)
    rows = []
    tape = tape0
    for k in range(1, STEPS + 1):
        qc, wo = conv.tape_to_qiskit(tape)
        meaning = infidelity(qml.matrix(tape0, wire_order=list(wo)), Operator(qc).reverse_qargs().data)
        tape = conv.qiskit_to_tape(qc, wo)
        rows.append(dict(converter=label, family=family, tape=name, step=k, fp0=fp0,
                         fp_equal=int(fingerprint(tape) == fp0),
                         rt_infid=infidelity(u0, qml.matrix(tape, wire_order=w0)),
                         meaning_infid=meaning, n_ops=len(tape.operations)))
    return rows


def verdict(confirmed, refuted):
    return "REFUTED" if refuted else ("CONFIRMED" if confirmed else "AMBIGUOUS")


def main():
    new, old = load("conv_new", NEW_PATH), load("conv_old", OLD_PATH)
    families = [("F1_int_wires", "seed%d" % s, random_tape(s, [0, 1, 2, 3])) for s in range(10)]
    families += [("F2_mixed_labels", "seed%d" % s, random_tape(100 + s, ["q3", "a", 7, "b"])) for s in range(5)]
    try:
        xt = xor_tapes()
    except Exception as exc:  # noqa: BLE001 -- reported, not swallowed
        raise SystemExit("C0 FAILED: could not build the XOR tapes: " + type(exc).__name__ + ": " + str(exc))
    families += [("F3_xor", "input%d" % i, t) for i, t in enumerate(xt)]

    rows = []
    for fam, name, t in families:
        try:
            rows += chain(new, t, fam, name, "NEW")
        except Exception as exc:  # noqa: BLE001
            raise SystemExit("C0 FAILED: NEW converter could not convert " + fam + "/" + name + ": "
                             + type(exc).__name__ + ": " + str(exc))
        rows += chain(old, t, fam, name, "OLD")
        print("  done", fam, name, "fp0", fingerprint(t))
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nWrote", len(rows), "rows to", OUT)

    N = [r for r in rows if r["converter"] == "NEW"]
    O = [r for r in rows if r["converter"] == "OLD"]
    print("\n" + "=" * 78)
    print("SCORING (thresholds exactly as pre-registered)")
    print("=" * 78)
    a1 = all(r["fp_equal"] for r in N)
    print("A1 NEW round trip exact at every step: steps not exact =", sum(1 - r["fp_equal"] for r in N),
          " max rt_infid =", format(max(r["rt_infid"] for r in N), ".2e"), "->", verdict(a1, not a1))
    mx = max(r["meaning_infid"] for r in N)
    print("A2 NEW meaning at every step: max meaning_infid =", format(mx, ".2e"), "->",
          verdict(mx <= MEANING_TOL, mx > MEANING_TOL))
    a3 = all(r["fp_equal"] for r in O)
    print("A3 OLD round trip exact at every step: steps not exact =", sum(1 - r["fp_equal"] for r in O),
          "->", verdict(a3, not a3))
    first = [r for r in O if r["step"] == 1 and r["family"] != "F3_xor"]
    det = [r["meaning_infid"] > OLD_DETECT for r in first]
    print("A4 OLD meaning check catches the pre-fix bit order at step 1:", sum(det), "of", len(det),
          "tapes  (min meaning_infid", format(min(r["meaning_infid"] for r in first), ".3f"), ") ->",
          verdict(all(det), not all(det)))
    x_new = max(r["meaning_infid"] for r in N if r["family"] == "F3_xor")
    x_old = [r["meaning_infid"] for r in O if r["family"] == "F3_xor" and r["step"] == 1]
    a5_ok = x_new <= MEANING_TOL and all(v > OLD_DETECT for v in x_old)
    print("A5 XOR tapes: NEW max meaning_infid =", format(x_new, ".2e"),
          "; OLD step-1 meaning_infid > 1e-3 in", sum(v > OLD_DETECT for v in x_old), "of", len(x_old),
          "(min", format(min(x_old), ".3f"), ") ->", verdict(a5_ok, not a5_ok))

    print("\nPer family, NEW: steps exact / max meaning_infid | OLD: steps exact / min step-1 meaning_infid")
    for fam in ("F1_int_wires", "F2_mixed_labels", "F3_xor"):
        n = [r for r in N if r["family"] == fam]
        o = [r for r in O if r["family"] == fam]
        line = "  {:<16} {}/{} {:.2e}".format(fam, sum(r["fp_equal"] for r in n), len(n), max(r["meaning_infid"] for r in n))
        if o:
            line += " | {}/{} {:.3f}".format(sum(r["fp_equal"] for r in o), len(o),
                                             min(r["meaning_infid"] for r in o if r["step"] == 1))
        print(line)


if __name__ == "__main__":
    main()
