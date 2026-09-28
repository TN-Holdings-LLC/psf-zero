"""x4_gpu_bind_vs_recompile.py -- pre-registered: per-evaluation cost of the two
training-loop routes with a GPU statevector simulator (home Addendum 204, X4).

Implements exactly the design locked in
"Addendum (number TBD, workplace) -- Pre-registration: transpile-once-and-bind
versus recompile-every-step with a GPU statevector simulator" (2026-09-28).

Route A (PSF-Zero): bind numeric angles into the brick-layer circuit, compile
    with psf_compile.compile_for_hardware (line, basis cx/rz/sx/x, trivial
    layout, block_gate_floor 8), convert, simulate.
Route B (Qiskit):   transpile the parameterized circuit ONCE (optimization
    level 3, same line/basis/layout); per evaluation assign the parameters,
    convert, simulate.
Both simulate on the same PennyLane device and return the energy of the
nearest-neighbour ZZ chain. Evaluations alternate A, B with identical angles.

    python -u x4_gpu_bind_vs_recompile.py run   2>&1 | tee ~/x4_gpu_run.txt
    python -u x4_gpu_bind_vs_recompile.py score 2>&1 | tee ~/x4_gpu_score.txt
Options for a plumbing dry run only: --device lightning.qubit --sizes 12,16
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import io
import json
import os
import platform
import statistics as st
import sys
import time
import warnings

REPO = os.path.expanduser("~/psf-zero")
sys.path[:0] = [REPO, os.path.join(REPO, "benchmarks")]
HOME = os.path.expanduser("~")
SIZES = [12, 16, 20, 24, 26, 28]
WARMUP, EVALS = 2, 20
BASIS = ["cx", "rz", "sx", "x"]
OUT_CSV = os.path.join(HOME, "x4_gpu_2026-09-28.csv")
OUT_JSON = os.path.join(HOME, "x4_gpu_2026-09-28.json")
FIELDS = ["n", "route", "eval", "energy", "total_s", "build_s", "compile_s", "bind_s", "convert_s",
          "simulate_s", "cx", "ops", "trivial_layout"]


def blocks(n):
    return [(i, i + 1) for i in range(0, n - 1, 2)] + [(i, i + 1) for i in range(1, n - 1, 2)]


def add_block(qc, a, b, p):
    """Same gate sequence as loop_endurance.add_block (Addenda 199-222)."""
    qc.rz(p[0], a); qc.ry(p[1], a); qc.rz(p[2], a)
    qc.rz(p[3], b); qc.ry(p[4], b); qc.rz(p[5], b)
    qc.rxx(p[6], a, b); qc.ryy(p[7], a, b); qc.rzz(p[8], a, b)
    qc.rz(p[9], a); qc.ry(p[10], a); qc.rz(p[11], a)
    qc.rz(p[12], b); qc.ry(p[13], b); qc.rz(p[14], b)


def brick(n, params):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    for k, (a, b) in enumerate(blocks(n)):
        add_block(qc, a, b, params[15 * k:15 * (k + 1)])
    return qc


def to_ops(qc):
    """Qiskit circuit in {cx, rz, sx, x} -> list of (name, wires, param). Qiskit
    qubit i -> PennyLane wire i for both routes; the ZZ chain is symmetric
    under reversal, so the energy does not depend on the bit-order convention."""
    ops = []
    for inst in qc.data:
        name = inst.operation.name
        q = tuple(qc.find_bit(x).index for x in inst.qubits)
        if name == "cx":
            ops.append(("CNOT", q, None))
        elif name == "rz":
            ops.append(("RZ", q, float(inst.operation.params[0])))
        elif name == "sx":
            ops.append(("SX", q, None))
        elif name == "x":
            ops.append(("PauliX", q, None))
        elif name in ("barrier", "global_phase"):
            continue
        else:
            raise RuntimeError("C0: unexpected gate in compiled circuit: " + name)
    return ops


def layout_is_trivial(qc, n):
    lay = qc.layout
    if lay is None:
        return True
    return list(lay.final_index_layout(filter_ancillas=True)) == list(range(n))


def run(device, sizes):
    import numpy as np
    import pennylane as qml
    import qiskit
    import psf_compile as pc
    import psf_zero_core as core
    from qiskit import transpile
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit.transpiler import CouplingMap

    env = {"platform": platform.platform(), "python": platform.python_version(), "qiskit": qiskit.__version__,
           "pennylane": qml.__version__, "numpy": np.__version__, "device": device,
           "psf_compile": pc.__file__, "psf_compile_version": pc.VERSION,
           "core_version": getattr(core, "CORE_VERSION", None)}
    print("LOADED", pc.__file__, pc.VERSION)
    print("CORE", core.__file__, "CORE_VERSION", env["core_version"])
    print("ENV", json.dumps(env))
    if pc.VERSION != "2026-09-27.7":
        print("V0 FAILED: psf_compile.py is not 2026-09-27.7. Stopping.")
        return
    try:
        import subprocess
        env["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                     "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception:
        env["gpu"] = None
    print("GPU", env["gpu"])

    rows, summary = [], {"env": env, "sizes": {}}
    for n in sizes:
        cmap = CouplingMap.from_line(n)
        nb = len(blocks(n))
        npar = 15 * nb
        rng = np.random.default_rng(1000 + n)
        base = rng.uniform(-np.pi, np.pi, npar)
        angles = [base + rng.normal(0.0, 0.5, npar) for _ in range(WARMUP + EVALS)]

        theta = ParameterVector("t", npar)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tb = transpile(brick(n, theta), coupling_map=cmap, basis_gates=BASIS,
                           initial_layout=list(range(n)), optimization_level=3, seed_transpiler=0)
        t_once = time.perf_counter() - t0
        b_trivial = layout_is_trivial(tb, n)

        dev = qml.device(device, wires=n)
        H = qml.Hamiltonian([1.0] * (n - 1), [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(n - 1)])
        gates = {"CNOT": qml.CNOT, "RZ": qml.RZ, "SX": qml.SX, "PauliX": qml.PauliX}

        @qml.qnode(dev, diff_method=None)
        def energy(ops):
            for name, w, p in ops:
                if p is None:
                    gates[name](wires=list(w))
                else:
                    gates[name](p, wires=list(w))
            return qml.expval(H)

        def route_a(x):
            s = {}
            t = time.perf_counter(); qc = brick(n, x); s["build"] = time.perf_counter() - t
            t = time.perf_counter()
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("ignore")
                pc._CX_CORE_CACHE.clear()
                out = pc.compile_for_hardware(qc, coupling_map=cmap, basis_gates=BASIS, entangling_basis="cx",
                                              initial_layout=list(range(n)), on_unsupported="keep",
                                              seed_transpiler=0, block_gate_floor=8)
            s["compile"] = time.perf_counter() - t
            t = time.perf_counter(); ops = to_ops(out); s["convert"] = time.perf_counter() - t
            t = time.perf_counter(); e = float(energy(ops)); s["simulate"] = time.perf_counter() - t
            return e, s, out

        def route_b(x):
            s = {}
            t = time.perf_counter(); bound = tb.assign_parameters(dict(zip(theta, x))); s["bind"] = time.perf_counter() - t
            t = time.perf_counter(); ops = to_ops(bound); s["convert"] = time.perf_counter() - t
            t = time.perf_counter(); e = float(energy(ops)); s["simulate"] = time.perf_counter() - t
            return e, s, bound

        gc.collect(); gc.freeze()
        c0 = None
        for i, x in enumerate(angles):
            gc.disable()
            ea, sa, outa = route_a(x)
            eb, sb, outb = route_b(x)
            gc.enable()
            if i == 0 and n == sizes[0]:
                ref = Statevector(brick(n, x))
                zz = SparsePauliOp.from_sparse_list([("ZZ", [j, j + 1], 1.0) for j in range(n - 1)], num_qubits=n)
                c0 = abs(float(np.real(ref.expectation_value(zz))) - ea)
                print(f"C0 n={n}: |E_A - E_ref(Qiskit Statevector, uncompiled)| = {c0:.2e}")
            if i < WARMUP:
                continue
            ca, cb = outa.count_ops(), outb.count_ops()
            for route, e, s, co, out in (("A", ea, sa, ca, outa), ("B", eb, sb, cb, outb)):
                rows.append({"n": n, "route": route, "eval": i - WARMUP, "energy": e,
                             "total_s": sum(s.values()), **{k + "_s": v for k, v in s.items()},
                             "cx": co.get("cx", 0), "ops": sum(co.values()),
                             "trivial_layout": layout_is_trivial(out, n) if route == "A" else b_trivial})
            gc.collect()
        gc.unfreeze()
        R = [r for r in rows if r["n"] == n]
        ta = st.median(r["total_s"] for r in R if r["route"] == "A")
        tbm = st.median(r["total_s"] for r in R if r["route"] == "B")
        simA = st.median(r["simulate_s"] for r in R if r["route"] == "A")
        de = max(abs(a["energy"] - b["energy"]) for a, b in zip(R[0::2], R[1::2]))
        summary["sizes"][str(n)] = {
            "transpile_once_s": t_once, "median_A_s": ta, "median_B_s": tbm, "ratio_A_over_B": ta / tbm,
            "median_A_compile_s": st.median(r["compile_s"] for r in R if r["route"] == "A"),
            "median_A_simulate_s": simA, "median_B_simulate_s": st.median(r["simulate_s"] for r in R if r["route"] == "B"),
            "median_B_bind_s": st.median(r["bind_s"] for r in R if r["route"] == "B"),
            "cx_A": sorted({r["cx"] for r in R if r["route"] == "A"}), "cx_B": sorted({r["cx"] for r in R if r["route"] == "B"}),
            "ops_A": sorted({r["ops"] for r in R if r["route"] == "A"}), "ops_B": sorted({r["ops"] for r in R if r["route"] == "B"}),
            "max_energy_diff": de, "all_trivial_layout": all(r["trivial_layout"] for r in R)}
        if c0 is not None:
            summary["c0"] = c0
        print(f"n={n}: {json.dumps(summary['sizes'][str(n)])}")
        with open(OUT_CSV, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, restval=""); w.writeheader(); w.writerows(rows)
        with open(OUT_JSON, "w") as fh:
            json.dump(summary, fh, indent=1)
    print("wrote", OUT_CSV, OUT_JSON)


def score():
    S = json.load(open(OUT_JSON)); Z = S["sizes"]; ns = sorted(int(k) for k in Z)

    def v(ok, bad):
        return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")
    print("=" * 78); print("SCORING (thresholds exactly as pre-registered)"); print("=" * 78)
    print("ENV", json.dumps(S["env"]))
    c0 = S.get("c0", float("inf"))
    triv = all(Z[str(n)]["all_trivial_layout"] for n in ns)
    print(f"C0 harness: converter check {c0:.2e} (<= 1e-10), trivial layouts everywhere {triv}, sizes {ns}")
    if not (c0 <= 1e-10 and triv and ns == SIZES):
        print("C0 FAILED: nothing is scored.")
        return
    de = max(Z[str(n)]["max_energy_diff"] for n in ns)
    print(f"Y1 routes agree: max |E_A - E_B| = {de:.2e} (<= 1e-10) ->", v(de <= 1e-10, de > 1e-10))
    r = {n: Z[str(n)]["ratio_A_over_B"] for n in ns}
    print("   median time per evaluation, A/B:", {n: round(r[n], 3) for n in ns})
    print(f"Y2 n=12: A/B = {r[12]:.3f} (confirmed < 2.0: the transpile-once route is NOT twice as fast; refuted >= 2.0) ->",
          v(r[12] < 2.0, r[12] >= 2.0))
    print(f"Y3 n=28: A/B = {r[28]:.3f} (confirmed <= 0.6, refuted >= 0.8) ->",
          "CONFIRMED" if r[28] <= 0.6 else ("REFUTED" if r[28] >= 0.8 else "AMBIGUOUS"))
    cross = next((n for n in ns if r[n] <= 1.0), None)
    print(f"Y4 smallest n with A/B <= 1: {cross} (confirmed if 12 or 16) ->",
          v(cross in (12, 16), cross not in (12, 16)))
    g = {n: max(Z[str(n)]["cx_A"]) / min(Z[str(n)]["cx_B"]) for n in ns}
    print("Y5 two-qubit gates A/B <= 0.7 at every n:", {n: round(g[n], 3) for n in ns}, "->",
          v(all(x <= 0.7 for x in g.values()), any(x > 0.7 for x in g.values())))
    fr = Z["28"]["median_A_simulate_s"] / Z["28"]["median_A_s"]
    print(f"Y6 n=28: simulation share of route A = {fr:.3f} (>= 0.9) ->", v(fr >= 0.9, fr < 0.9))
    o = {n: min(Z[str(n)]["ops_B"]) / max(Z[str(n)]["ops_A"]) for n in ns}
    print("Y7 total gates B/A >= 2.0 at every n:", {n: round(o[n], 2) for n in ns}, "->",
          v(all(x >= 2.0 for x in o.values()), any(x < 2.0 for x in o.values())))
    print("\nReported without prediction (RunPod pod times):")
    for n in ns:
        z = Z[str(n)]
        print(f"  n={n}: A {z['median_A_s']*1e3:.1f} ms (compile {z['median_A_compile_s']*1e3:.1f}, sim {z['median_A_simulate_s']*1e3:.1f})"
              f" | B {z['median_B_s']*1e3:.1f} ms (bind {z['median_B_bind_s']*1e3:.2f}, sim {z['median_B_simulate_s']*1e3:.1f})"
              f" | transpile once {z['transpile_once_s']*1e3:.0f} ms | cx {z['cx_A']} vs {z['cx_B']} | ops {z['ops_A']} vs {z['ops_B']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "score"])
    ap.add_argument("--device", default="lightning.gpu")
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    a = ap.parse_args()
    run(a.device, [int(s) for s in a.sizes.split(",")]) if a.mode == "run" else score()
