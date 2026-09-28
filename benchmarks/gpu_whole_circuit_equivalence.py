"""gpu_whole_circuit_equivalence.py -- pre-registered: whole-circuit equivalence
of compiled circuits, checked on a GPU statevector (mirror test).

Implements exactly the design locked in
"Addendum (number TBD, workplace) -- Pre-registration: whole-circuit
equivalence of compiled circuits on a GPU statevector" (2026-09-28).

For each circuit C (n logical qubits) and each compiled output C' (m >= n
physical qubits, final layout pi), the GPU simulates C' followed by the
inverse of the ORIGINAL circuit C applied on wires pi(i). If C' implements C
under pi, the result is |0...0> up to a global phase. The error is the norm
of all other amplitudes, sqrt(sum_{k != 0} |a_k|^2), summed directly (no
1 - |a_0| cancellation, no infidelity).

    python -u gpu_whole_circuit_equivalence.py run   2>&1 | tee ~/eq_gpu_run.txt
    python -u gpu_whole_circuit_equivalence.py score 2>&1 | tee ~/eq_gpu_score.txt
Options for a plumbing dry run only: --device lightning.qubit --sizes 12,16
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import sys
import time
import warnings

REPO = os.path.expanduser("~/psf-zero")
sys.path[:0] = [REPO, os.path.join(REPO, "benchmarks")]
HOME = os.path.expanduser("~")
SIZES = [20, 24, 26]
GRIDS = {12: (3, 4), 16: (4, 4), 20: (4, 5), 24: (4, 6), 26: (2, 13), 10: (2, 5)}
INPUTS = 3
BASIS = ["cx", "rz", "sx", "x"]
OUT_JSON = os.path.join(HOME, "eq_gpu_2026-09-28.json")
TOL = 1e-10


def brick_pairs(n):
    return [(i, i + 1) for i in range(0, n - 1, 2)] + [(i, i + 1) for i in range(1, n - 1, 2)]


def build(family, n, seed):
    """Random product input layer, then the family's body."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import random_unitary
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.u(*rng.uniform(-np.pi, np.pi, 3), q)
    if family in ("F1_brick_line", "F2_brick_grid"):
        for (a, b) in brick_pairs(n):
            p = rng.uniform(-np.pi, np.pi, 15)
            qc.rz(p[0], a); qc.ry(p[1], a); qc.rz(p[2], a)
            qc.rz(p[3], b); qc.ry(p[4], b); qc.rz(p[5], b)
            qc.rxx(p[6], a, b); qc.ryy(p[7], a, b); qc.rzz(p[8], a, b)
            qc.rz(p[9], a); qc.ry(p[10], a); qc.rz(p[11], a)
            qc.rz(p[12], b); qc.ry(p[13], b); qc.rz(p[14], b)
    elif family == "F3_random_pairs_grid":
        for _ in range(3):
            perm = rng.permutation(n)
            for k in range(0, n - 1, 2):
                a, b = int(perm[k]), int(perm[k + 1])
                qc.unitary(random_unitary(4, seed=int(rng.integers(0, 2**31))), [a, b])
    else:
        raise ValueError(family)
    return qc


def ops_of(qc, wire_of=None, reverse_unitary=True):
    """Qiskit circuit -> list of (name, wires, params) for PennyLane. Qiskit
    qubit q goes to wire wire_of[q] (identity by default). A Qiskit 2-qubit
    `unitary` on qargs [a, b] is little-endian (a is the low bit), so in
    PennyLane it is QubitUnitary(M, wires=[b, a]); reverse_unitary=False
    reproduces the old bit-order bug and is used only as a positive control."""
    import numpy as np
    from qiskit.quantum_info import Operator
    wmap = (lambda q: q) if wire_of is None else (lambda q: wire_of[q])
    out = []
    for inst in qc.data:
        op = inst.operation
        name = op.name
        q = [wmap(qc.find_bit(x).index) for x in inst.qubits]
        p = [float(v) for v in op.params] if name != "unitary" else None
        if name == "cx":
            out.append(("CNOT", q, None))
        elif name in ("rz", "ry", "rx"):
            out.append((name.upper(), q, p))
        elif name == "sx":
            out.append(("SX", q, None))
        elif name == "x":
            out.append(("PauliX", q, None))
        elif name == "u":
            out.append(("U3", q, p))
        elif name in ("rxx", "ryy", "rzz"):
            out.append(({"rxx": "IsingXX", "ryy": "IsingYY", "rzz": "IsingZZ"}[name], q, p))
        elif name == "unitary":
            m = np.asarray(Operator(op).data)
            out.append(("QubitUnitary", q[::-1] if reverse_unitary else q, m))
        elif name in ("barrier", "global_phase"):
            continue
        else:
            raise RuntimeError("C0: unexpected gate " + name)
    return out


def final_positions(out, n):
    lay = out.layout
    if lay is None:
        return list(range(n))
    return list(lay.final_index_layout(filter_ancillas=True))


def run(device, sizes):
    import numpy as np
    import pennylane as qml
    import qiskit
    import psf_compile as pc
    import psf_zero_core as core
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    from qiskit.transpiler import CouplingMap

    env = {"platform": platform.platform(), "python": platform.python_version(), "qiskit": qiskit.__version__,
           "pennylane": qml.__version__, "numpy": np.__version__, "device": device,
           "psf_compile_version": pc.VERSION, "core_version": getattr(core, "CORE_VERSION", None)}
    print("LOADED", pc.__file__, pc.VERSION)
    print("CORE", core.__file__, "CORE_VERSION", env["core_version"])
    if pc.VERSION != "2026-09-27.7":
        print("V0 FAILED: psf_compile.py is not 2026-09-27.7. Stopping.")
        return
    try:
        import subprocess
        env["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                     "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception:
        env["gpu"] = None
    print("ENV", json.dumps(env))

    G = {"CNOT": qml.CNOT, "RZ": qml.RZ, "RY": qml.RY, "RX": qml.RX, "SX": qml.SX, "PauliX": qml.PauliX,
         "U3": qml.U3, "IsingXX": qml.IsingXX, "IsingYY": qml.IsingYY, "IsingZZ": qml.IsingZZ,
         "QubitUnitary": qml.QubitUnitary}

    def apply(ops):
        for name, w, p in ops:
            if p is None:
                G[name](wires=w)
            elif name == "QubitUnitary":
                G[name](p, wires=w)
            else:
                G[name](*p, wires=w)

    def state(m, ops):
        dev = qml.device(device, wires=m)

        @qml.qnode(dev, diff_method=None)
        def f():
            apply(ops)
            return qml.state()
        return np.asarray(f())

    res = {"env": env, "c0": {}, "cells": [], "control": []}

    # ---- C0: converter check against Qiskit's Statevector (n = 10), with the bit-order bug as positive control
    for fam in ("F2_brick_grid", "F3_random_pairs_grid"):
        qc = build(fam, 10, 7)
        ref = np.asarray(Statevector(qc).data).reshape([2] * 10).transpose(list(range(9, -1, -1))).reshape(-1)
        good = np.max(np.abs(state(10, ops_of(qc)) - ref))
        bad = np.max(np.abs(state(10, ops_of(qc, reverse_unitary=False)) - ref))
        res["c0"][fam] = {"max_diff": float(good), "bug_control_max_diff": float(bad)}
        print(f"C0 {fam}: converter vs Qiskit Statevector {good:.2e}; bit-order-bug control {bad:.2e}")

    def mirror_error(out, qc, n, wrong=False):
        m = out.num_qubits
        pos = final_positions(out, n)
        if wrong:
            pos = pos.copy(); pos[0], pos[1] = pos[1], pos[0]
        ops = ops_of(out) + ops_of(qc.inverse(), wire_of=pos)
        a = state(m, ops)
        return float(np.sqrt(np.sum(np.abs(a[1:]) ** 2))), pos != list(range(n))

    for n in sizes:
        for fam in ("F1_brick_line", "F2_brick_grid", "F3_random_pairs_grid"):
            cmap = CouplingMap.from_line(n) if fam == "F1_brick_line" else CouplingMap.from_grid(*GRIDS[n])
            for k in range(INPUTS):
                qc = build(fam, n, 10000 * n + 100 * ["F1_brick_line", "F2_brick_grid", "F3_random_pairs_grid"].index(fam) + k)
                for arm in ("PSF", "QISKIT_L3"):
                    t0 = time.perf_counter()
                    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                        warnings.simplefilter("ignore")
                        if arm == "PSF":
                            pc._CX_CORE_CACHE.clear()
                            kw = {"initial_layout": list(range(n))} if fam == "F1_brick_line" else {}
                            out = pc.compile_for_hardware(qc, coupling_map=cmap, basis_gates=BASIS,
                                                          entangling_basis="cx", on_unsupported="keep",
                                                          seed_transpiler=0, block_gate_floor=8, **kw)
                        else:
                            out = transpile(qc, coupling_map=cmap, basis_gates=BASIS, optimization_level=3,
                                            seed_transpiler=0)
                    tc = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    err, moved = mirror_error(out, qc, n)
                    ts = time.perf_counter() - t0
                    cell = {"n": n, "family": fam, "input": k, "arm": arm, "error": err, "moved": moved,
                            "cx": out.count_ops().get("cx", 0), "ops": sum(out.count_ops().values()),
                            "physical_qubits": out.num_qubits, "compile_s": tc, "check_s": ts}
                    res["cells"].append(cell)
                    print("cell", json.dumps(cell))
                    if arm == "PSF" and k == 0:
                        werr, _ = mirror_error(out, qc, n, wrong=True)
                        res["control"].append({"n": n, "family": fam, "error_wrong_layout": werr})
                        print("control", json.dumps(res["control"][-1]))
        with open(OUT_JSON, "w") as fh:
            json.dump(res, fh, indent=1)
    print("wrote", OUT_JSON)


def score():
    R = json.load(open(OUT_JSON))

    def v(ok, bad):
        return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")
    print("=" * 78); print("SCORING (thresholds exactly as pre-registered)"); print("=" * 78)
    print("ENV", json.dumps(R["env"]))
    c0 = R["c0"]; cells = R["cells"]
    ns = sorted({c["n"] for c in cells})
    moved_any = any(c["moved"] for c in cells if c["arm"] == "PSF")
    c0_ok = (all(x["max_diff"] <= 1e-12 for x in c0.values())
             and c0["F3_random_pairs_grid"]["bug_control_max_diff"] >= 0.01)  # only F3 contains 2-qubit unitaries
    print(f"C0 converter: {json.dumps(c0)} | sizes {ns} | a PSF output with a non-identity final layout exists: {moved_any}")
    if not (c0_ok and ns == SIZES and moved_any and len(cells) == len(SIZES) * 3 * INPUTS * 2):
        print("C0 FAILED: nothing is scored.")
        return
    psf = [c for c in cells if c["arm"] == "PSF"]; qk = [c for c in cells if c["arm"] == "QISKIT_L3"]
    wp, wq = max(c["error"] for c in psf), max(c["error"] for c in qk)
    print(f"Z1 every PSF-Zero output equivalent to its circuit (error <= {TOL:g}): worst {wp:.2e} over {len(psf)} ->",
          v(wp <= TOL, wp > TOL))
    print(f"Z2 every Qiskit L3 output equivalent (reference, same gauge): worst {wq:.2e} over {len(qk)} ->",
          v(wq <= TOL, wq > TOL))
    wc = min(c["error_wrong_layout"] for c in R["control"])
    print(f"Z3 wrong final layout detected (error >= 0.1) in all {len(R['control'])} controls: min {wc:.3f} ->",
          v(wc >= 0.1, wc < 0.1))
    f3 = [c for c in psf if c["family"] == "F3_random_pairs_grid"]
    mv = sum(c["moved"] for c in f3)
    print(f"Z4 routing exercised: PSF-Zero F3 outputs with a non-identity final layout: {mv} of {len(f3)} (>= 5) ->",
          v(mv >= 5, mv < 5))
    ratios = [p["cx"] / q["cx"] for p, q in zip(psf, qk) if p["family"] == "F3_random_pairs_grid"]
    mr = sum(ratios) / len(ratios)
    print(f"Z5 under routing (F3) PSF-Zero does not beat Qiskit L3 on CX: mean CX ratio PSF/L3 = {mr:.3f}"
          " (confirmed >= 1.00, refuted <= 0.95) ->", "CONFIRMED" if mr >= 1.0 else ("REFUTED" if mr <= 0.95 else "AMBIGUOUS"))
    print("\nReported without prediction:")
    for n in ns:
        for fam in ("F1_brick_line", "F2_brick_grid", "F3_random_pairs_grid"):
            a = [c for c in psf if c["n"] == n and c["family"] == fam]; b = [c for c in qk if c["n"] == n and c["family"] == fam]
            print(f"  n={n} {fam:21s} PSF worst {max(c['error'] for c in a):.1e} cx {sorted({c['cx'] for c in a})} moved {sum(c['moved'] for c in a)}/3"
                  f" | L3 worst {max(c['error'] for c in b):.1e} cx {sorted({c['cx'] for c in b})} moved {sum(c['moved'] for c in b)}/3")
    for c in R["control"]:
        print("  control", c)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "score"])
    ap.add_argument("--device", default="lightning.gpu")
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    a = ap.parse_args()
    run(a.device, [int(s) for s in a.sizes.split(",")]) if a.mode == "run" else score()
