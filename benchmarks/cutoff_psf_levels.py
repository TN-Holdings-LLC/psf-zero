"""cutoff_psf_levels.py -- pre-registered (workplace, 2026-09-29): does
PSF-Zero's compile_for_hardware meet Qiskit's CommutativeCancellation angle
cutoff (Addendum 247 B) at its default routing level 1, and at level 3?

Qiskit 2.5.2 (crates/transpiler/src/passes/commutation_cancellation.rs)
treats a merged Z rotation as the identity when angle / (4 pi) is within
_CUTOFF_PRECISION = 1e-5 of an integer, i.e. |angle mod 4 pi| < 1.2566e-4, and
drops it. The preset pass managers run CommutativeCancellation at optimization
levels 2 and 3 only (init and optimization stages), not at level 1.

Part M (minimal circuits, 2 qubits, line coupling map, basis cz/rz/sx/x):
  rz(a) q0 . cz(0, 1) . rz(delta - a) q0 with a = -pi/2 and the merged angle
  delta in OFFSETS; arms Q3 (transpile level 3), P1 (compile_for_hardware,
  routing level 1) and P3 (routing level 3). Error: phase-aligned Frobenius
  distance of the 4x4 operators (final layout applied).
Part R (realistic, FakeNighthawk, 120 qubits): 60 disjoint pair24 blocks
  (loop_endurance.add_pair24), random angles, N_R circuits (seed = 5000 + i);
  arms Q3 (transpile(target=..., optimization_level=3, seed_transpiler=0)),
  P1 and P3 (compile_for_hardware with the backend's coupling map and native
  basis, entangling_basis="cx", layout_search=True, seed_transpiler=0).
  Error: per-pair phase-aligned distance (as in real_target_cliff.pair_check).

    python -u cutoff_psf_levels.py run   2>&1 | tee cutoff_run.txt
    python -u cutoff_psf_levels.py score 2>&1 | tee cutoff_score.txt
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import platform
import sys
import time
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks"), os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CUTOFF = 4 * np.pi * 1e-5
OFFSETS = [5e-5, -5e-5, 1.0e-4, -1.0e-4, 1.2e-4, -1.2e-4, 1.3e-4, -1.3e-4, 2e-4, -2e-4, 1e-3, -1e-3]
N_R = 50
OUT_JSON = "cutoff_psf_levels_2026-09-29.json"
ARMS = ("Q3", "P1", "P3")


def aligned(a, b):
    t = np.trace(a.conj().T @ b)
    ph = t / abs(t) if abs(t) > 0 else 1.0
    return float(np.linalg.norm(b - ph * a))


def quiet(fn):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return fn()


def run():
    import qiskit
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Operator
    from qiskit.transpiler import CouplingMap
    import psf_compile as pc
    import psf_zero_core as core
    import loop_endurance as le

    env = {"platform": platform.platform(), "python": platform.python_version(), "qiskit": qiskit.__version__,
           "numpy": np.__version__, "psf_compile": pc.VERSION, "core_version": getattr(core, "CORE_VERSION", None),
           "refine_threshold": pc.REFINE_THRESHOLD, "offsets": OFFSETS, "n_r": N_R}
    print("ENV", json.dumps(env))
    print("LOADED", os.path.basename(pc.__file__), pc.VERSION, le.normalized_sha256(pc.__file__))
    print("SCRIPT", os.path.basename(__file__), le.normalized_sha256(os.path.abspath(__file__)))
    if pc.VERSION != "2026-09-28.1" or env["core_version"] != "2026-09-28.1":
        print("V0 FAILED: need psf_compile.py 2026-09-28.1 and CORE_VERSION 2026-09-28.1. Stopping.")
        return
    res = {"env": env, "M": [], "R": []}

    # ---- Part M
    basis = ["cz", "rz", "sx", "x"]
    cmap2 = CouplingMap.from_line(2)
    a = -np.pi / 2
    for delta in OFFSETS:
        qc = QuantumCircuit(2)
        qc.rz(a, 0); qc.cz(0, 1); qc.rz(delta - a, 0)
        ref = Operator(qc).data
        for arm in ARMS:
            if arm == "Q3":
                out = quiet(lambda: transpile(qc, coupling_map=cmap2, basis_gates=basis, optimization_level=3,
                                              seed_transpiler=0))
            else:
                pc._CX_CORE_CACHE.clear()
                out = quiet(lambda: pc.compile_for_hardware(
                    qc, coupling_map=cmap2, basis_gates=basis, entangling_basis="cx", seed_transpiler=0,
                    routing_optimization_level=1 if arm == "P1" else 3))
            err = aligned(ref, Operator.from_circuit(out).data)
            rec = {"delta": delta, "arm": arm, "error": err, "ops": dict(out.count_ops())}
            res["M"].append(rec)
            print("M", json.dumps(rec), flush=True)

    # ---- Part R
    backend, native = le.nighthawk()
    n = backend.coupling_map.size()
    for i in range(N_R):
        seed = 5000 + i
        rng = np.random.default_rng(seed)
        th = rng.uniform(-np.pi, np.pi, (n // 2, 24))
        qc = QuantumCircuit(n)
        for k in range(n // 2):
            le.add_pair24(qc, 2 * k, 2 * k + 1, th[k])
        for arm in ARMS:
            t0 = time.perf_counter()
            if arm == "Q3":
                out = quiet(lambda: transpile(qc, target=backend.target, optimization_level=3, seed_transpiler=0))
            else:
                pc._CX_CORE_CACHE.clear()
                out = quiet(lambda: pc.compile_for_hardware(
                    qc, coupling_map=backend.coupling_map, basis_gates=native, entangling_basis="cx",
                    layout_search=True, on_unsupported="raise", seed_transpiler=0,
                    routing_optimization_level=1 if arm == "P1" else 3))
            el = time.perf_counter() - t0
            per = pair_errors(qc, out, n)
            rec = {"seed": seed, "arm": arm, "applicable": per is not None, "compile_s": el,
                   "twoq": sum(v for k, v in out.count_ops().items() if k in ("cz", "cx", "ecr")),
                   "worst": max(per) if per else None,
                   "inexact": [[k, e] for k, e in enumerate(per or []) if e > 1e-12]}
            res["R"].append(rec)
            print("R", json.dumps(rec), flush=True)
        with open(OUT_JSON, "w") as fh:
            json.dump(res, fh, indent=1)
    print("wrote", OUT_JSON)


def pair_errors(qc_logical, qc_out, n):
    """Per-pair phase-aligned distances, in pair order (the construction of
    real_target_cliff.pair_check); None if a two-qubit gate joins pairs."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    lay = qc_out.layout
    if lay is None:
        return None
    phys = list(lay.final_index_layout(filter_ancillas=True))
    pairs = [(i, i + 1) for i in range(0, n - 1, 2)]
    owner, local = {}, {}
    for k, (a, b) in enumerate(pairs):
        owner[phys[a]] = k; local[phys[a]] = 0
        owner[phys[b]] = k; local[phys[b]] = 1
    per = {k: QuantumCircuit(2) for k in range(len(pairs))}
    for inst in qc_out.data:
        if inst.operation.name in ("barrier", "measure", "delay"):
            continue
        qs = [qc_out.find_bit(q).index for q in inst.qubits]
        ks = {owner.get(q) for q in qs}
        if None in ks:
            if len(qs) == 1:
                continue
            return None
        if len(ks) != 1:
            return None
        k = ks.pop()
        per[k].append(inst.operation, [local[q] for q in qs])
    ref = {}
    for k, (a, b) in enumerate(pairs):
        ref[k] = QuantumCircuit(2)
    for inst in qc_logical.data:
        qs = [qc_logical.find_bit(q).index for q in inst.qubits]
        k = qs[0] // 2
        ref[k].append(inst.operation, [q - 2 * k for q in qs])
    return [aligned(Operator(ref[k]).data, Operator(per[k]).data) for k in range(len(pairs))]


def score():
    R = json.load(open(OUT_JSON))
    env = R["env"]

    def v(ok, bad):
        return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")
    print("=" * 78); print("SCORING (thresholds exactly as pre-registered)"); print("=" * 78)
    print("ENV", json.dumps(env))
    M, Rr = R["M"], R["R"]
    napp = {a: sum(r["applicable"] for r in Rr if r["arm"] == a) for a in ARMS}
    ok0 = (env["psf_compile"] == "2026-09-28.1" and env["core_version"] == "2026-09-28.1"
           and len(M) == len(OFFSETS) * 3 and len(Rr) == N_R * 3 and min(napp.values()) >= 0.9 * N_R)
    print(f"C0: versions {env['psf_compile']}/{env['core_version']}; M cells {len(M)}; R cells {len(Rr)}; "
          f"per-pair check applicable per arm {napp} (>= 90%) -> {'passed' if ok0 else 'FAILED'}")
    Rr = [r for r in Rr if r["applicable"]]
    if not ok0:
        print("C0 FAILED: nothing is scored.")
        return

    def m(arm):
        return {r["delta"]: r["error"] for r in M if r["arm"] == arm}
    for arm, kid in (("Q3", "K1"), ("P3", "K3")):
        e = m(arm)
        inside = [d for d in OFFSETS if abs(d) < CUTOFF]
        outside = [d for d in OFFSETS if abs(d) >= CUTOFF]
        rem = all(abs(e[d] - abs(d)) <= 0.01 * abs(d) for d in inside)
        keep = all(e[d] <= 1e-12 for d in outside)
        bad = any(e[d] <= 1e-12 for d in inside) or any(e[d] > 1e-10 for d in outside)
        print(f"{kid} {arm} minimal: removed inside the cutoff (error = |delta| within 1%): {rem}; exact outside: "
              f"{keep} -> {v(rem and keep, bad)}")
        print("   " + ", ".join(f"{d:+.1e}: {e[d]:.3e}" for d in OFFSETS))
    e = m("P1")
    w = max(e.values())
    print(f"K2 P1 minimal exact at every offset (<= 1e-12): worst {w:.2e} -> {v(w <= 1e-12, w > 1e-10)}")
    p1 = [r for r in Rr if r["arm"] == "P1"]
    wp1 = max(r["worst"] for r in p1)
    np1 = sum(len(r["inexact"]) for r in p1)
    print(f"K4 P1 realistic: pairs > 1e-12: {np1} of {len(p1) * 60}; worst {wp1:.2e} (confirmed 0 and worst <= 1e-12,"
          f" refuted any > 1e-10) -> {v(np1 == 0, wp1 > 1e-10)}")
    q3 = [r for r in Rr if r["arm"] == "Q3"]
    wq3 = max(r["worst"] for r in q3)
    nq3 = sum(len(r["inexact"]) for r in q3)
    print(f"K5 Q3 realistic: every pair error <= 5e-4 (cutoff-sized): worst {wq3:.3e}; inexact pairs {nq3} of "
          f"{len(q3) * 60} -> {v(wq3 <= 5e-4, wq3 > 1e-3)}")
    p3 = [r for r in Rr if r["arm"] == "P3"]
    np3 = sum(len(r["inexact"]) for r in p3)
    wp3 = max(r["worst"] for r in p3)
    print(f"K6 P3 realistic: at least one pair above 1e-12: {np3} of {len(p3) * 60} (worst {wp3:.3e}) -> "
          f"{'CONFIRMED' if np3 >= 1 else 'AMBIGUOUS'}")
    print("\nReported without prediction:")
    for arm in ARMS:
        rs = [r for r in Rr if r["arm"] == arm]
        ts = sorted(r["compile_s"] for r in rs)
        tq = sorted({r["twoq"] for r in rs})
        errs = sorted(e for r in rs for _, e in r["inexact"])
        print(f"   {arm}: median compile {ts[len(ts) // 2]:.3f} s (sandbox); two-qubit counts {tq[:5]}"
              f"{'...' if len(tq) > 5 else ''}; inexact pair errors {[f'{x:.3e}' for x in errs][:20]}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    run() if mode == "run" else score()
