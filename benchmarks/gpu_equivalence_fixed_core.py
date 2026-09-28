"""gpu_equivalence_fixed_core.py -- pre-registered: the whole-circuit
equivalence check of 2026-09-28 (gpu_whole_circuit_equivalence.py) repeated
with the fixed Rust core (CORE_VERSION 2026-09-28.1), at the current
REFINE_THRESHOLD (1e-13, arm PSF_T13) and at the candidate 1e-14 (arm
PSF_T14, set in memory; psf_compile.py is not edited).

Implements exactly the design locked in
"Addendum (number TBD, workplace) -- Pre-registration: whole-circuit
equivalence with the fixed core, at REFINE_THRESHOLD 1e-13 and 1e-14"
(2026-09-28).

Same circuits (families, sizes, seeds), same compile call and the same GPU
mirror test as the first run; the Qiskit L3 arm is not repeated. The PSF-Zero
CX counts of the first run (pre-fix core) are embedded as BASE_CX.

    python -u gpu_equivalence_fixed_core.py run   2>&1 | tee ~/eq2_run.txt
    python -u gpu_equivalence_fixed_core.py score 2>&1 | tee ~/eq2_score.txt
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
OUT_JSON = os.path.join(HOME, "eq2_fixed_core_2026-09-28.json")
ARMS = {"PSF_T13": 1e-13, "PSF_T14": 1e-14}
# PSF-Zero CX counts of the first run (pre-fix core), key "n/family/input"
BASE_CX = {"20/F1/0": 57, "20/F1/1": 57, "20/F1/2": 57, "20/F2/0": 57, "20/F2/1": 57, "20/F2/2": 57,
           "20/F3/0": 126, "20/F3/1": 126, "20/F3/2": 120, "24/F1/0": 69, "24/F1/1": 69, "24/F1/2": 69,
           "24/F2/0": 69, "24/F2/1": 69, "24/F2/2": 69, "24/F3/0": 153, "24/F3/1": 156, "24/F3/2": 159,
           "26/F1/0": 75, "26/F1/1": 75, "26/F1/2": 75, "26/F2/0": 75, "26/F2/1": 75, "26/F2/2": 75,
           "26/F3/0": 234, "26/F3/1": 240, "26/F3/2": 189}
BASE_WORST = 8.863075028251012e-14


def set_threshold(pc, t):
    pc.REFINE_THRESHOLD = t
    pc._refine_batch.__defaults__ = (t, 3)
    pc._refine_decomposition.__defaults__ = (t, 3)
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
    from qiskit.quantum_info import Statevector
    from qiskit.transpiler import CouplingMap

    env = {"platform": platform.platform(), "python": platform.python_version(), "qiskit": qiskit.__version__,
           "pennylane": qml.__version__, "numpy": np.__version__, "device": device,
           "psf_compile_version": pc.VERSION, "core_version": getattr(core, "CORE_VERSION", None)}
    print("LOADED", pc.__file__, pc.VERSION)
    print("CORE", core.__file__, "CORE_VERSION", env["core_version"])
    if pc.VERSION != "2026-09-27.7" or env["core_version"] != "2026-09-28.1":
        print("V0 FAILED: need psf_compile.py 2026-09-27.7 and CORE_VERSION 2026-09-28.1. Stopping.")
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
                for arm, thr in ARMS.items():
                    set_threshold(pc, thr)
                    t0 = time.perf_counter()
                    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                        warnings.simplefilter("ignore")
                        pc._CX_CORE_CACHE.clear()
                        kw = {"initial_layout": list(range(n))} if fam == "F1_brick_line" else {}
                        out = pc.compile_for_hardware(qc, coupling_map=cmap, basis_gates=BASIS,
                                                      entangling_basis="cx", on_unsupported="keep",
                                                      seed_transpiler=0, block_gate_floor=8, **kw)
                    tc = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    err, moved = mirror_error(out, qc, n)
                    ts = time.perf_counter() - t0
                    cell = {"n": n, "family": fam, "input": k, "arm": arm, "threshold": thr, "error": err,
                            "moved": moved, "cx": out.count_ops().get("cx", 0), "ops": sum(out.count_ops().values()),
                            "physical_qubits": out.num_qubits, "compile_s": tc, "check_s": ts}
                    res["cells"].append(cell)
                    print("cell", json.dumps(cell))
                    if arm == "PSF_T13" and k == 0:
                        werr, _ = mirror_error(out, qc, n, wrong=True)
                        res["control"].append({"n": n, "family": fam, "error_wrong_layout": werr})
                        print("control", json.dumps(res["control"][-1]))
                set_threshold(pc, 1e-13)
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
    moved_any = any(c["moved"] for c in cells)
    c0_ok = (all(x["max_diff"] <= 1e-12 for x in c0.values())
             and c0["F3_random_pairs_grid"]["bug_control_max_diff"] >= 0.01
             and R["env"]["core_version"] == "2026-09-28.1" and R["env"]["device"] == "lightning.gpu")
    print(f"C0 converter: {json.dumps(c0)} | core {R['env']['core_version']} | device {R['env']['device']} | sizes {ns}"
          f" | an output with a non-identity final layout exists: {moved_any}")
    if not (c0_ok and ns == SIZES and moved_any and len(cells) == len(SIZES) * 3 * INPUTS * len(ARMS)):
        print("C0 FAILED: nothing is scored.")
        return
    w = {a: max(c["error"] for c in cells if c["arm"] == a) for a in ARMS}
    wa = max(w.values())
    print(f"E1 every output of both arms equivalent (error <= {TOL:g}): worst T13 {w['PSF_T13']:.2e}, "
          f"T14 {w['PSF_T14']:.2e} over {len(cells)} ->", v(wa <= TOL, wa > TOL))
    diff = {a: [f"{c['n']}/{c['family'][:2]}/{c['input']}: {c['cx']} vs {BASE_CX[str(c['n']) + '/' + c['family'][:2] + '/' + str(c['input'])]}"
                for c in cells if c["arm"] == a
                and c["cx"] != BASE_CX[f"{c['n']}/{c['family'][:2]}/{c['input']}"]] for a in ARMS}
    nd = sum(len(x) for x in diff.values())
    print(f"E2 CX count equal to the pre-fix run in all 27 cells, both arms: differing cells {nd} {json.dumps(diff)} ->",
          v(nd == 0, nd > 0))
    wc = min(c["error_wrong_layout"] for c in R["control"])
    print(f"E3 wrong final layout detected (error >= 0.1) in all {len(R['control'])} controls: min {wc:.3f} ->",
          v(wc >= 0.1, wc < 0.1))
    print(f"E4 worst error of each arm <= 1e-12 (pre-fix run: {BASE_WORST:.2e}): T13 {w['PSF_T13']:.2e}, "
          f"T14 {w['PSF_T14']:.2e} ->", v(wa <= 1e-12, wa > TOL))
    print("\nReported without prediction:")
    for n in ns:
        for fam in ("F1_brick_line", "F2_brick_grid", "F3_random_pairs_grid"):
            parts = []
            for a in ARMS:
                x = [c for c in cells if c["arm"] == a and c["n"] == n and c["family"] == fam]
                parts.append(f"{a} worst {max(c['error'] for c in x):.1e} cx {[c['cx'] for c in x]} "
                             f"moved {sum(c['moved'] for c in x)}/3")
            print(f"  n={n} {fam:21s} " + " | ".join(parts))
    for a in ARMS:
        x = sorted(c["compile_s"] for c in cells if c["arm"] == a)
        print(f"  {a}: median compile {x[len(x) // 2] * 1e3:.1f} ms (RunPod pod)")
    for c in R["control"]:
        print("  control", c)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "score"])
    ap.add_argument("--device", default="lightning.gpu")
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    a = ap.parse_args()
    run(a.device, [int(s) for s in a.sizes.split(",")]) if a.mode == "run" else score()
