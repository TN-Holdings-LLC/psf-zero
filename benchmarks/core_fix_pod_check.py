"""core_fix_pod_check.py -- pre-registered pod check of the Rust core fix
(changelog item 11, CORE_VERSION 2026-09-28.1) against the core of
repository commit 100e768, with psf_compile.py 2026-09-27.7 unchanged.

Implements exactly the design locked in
"Addendum (number TBD, workplace) -- Pre-registration: the Rust core fix
(CORE_VERSION 2026-09-28.1) on the RunPod pod" (2026-09-28).

Run each measurement in the venv holding that core, base first:
    source ~/psf_zero_runpod_env/bin/activate
    python -u ~/core_fix_pod_check.py measure --label base  2>&1 | tee ~/core_fix_base.txt
    source ~/psf_zero_fixed_env/bin/activate
    python -u ~/core_fix_pod_check.py measure --label fixed 2>&1 | tee ~/core_fix_fixed.txt
    python -u ~/core_fix_pod_check.py score 2>&1 | tee ~/core_fix_score.txt

Outputs (home directory): core_fix_{base,fixed}.json, core_fix_blocks_base.npz.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import json
import os
import platform
import re
import statistics as st
import subprocess
import sys
import time
import warnings
from collections import Counter

REPO = os.path.expanduser("~/psf-zero")
BENCH = os.path.join(REPO, "benchmarks")
HOME = os.path.expanduser("~")
sys.path[:0] = [REPO, BENCH]

E_LAPS = 3000
CLIFF_REPS = 6
LAP_2138_POS = 8          # capture position of pair (7, 8) in the E circuit (Addendum section 2)
FELL = re.compile(r"(\d+) block\(s\) fell back to CX-basis synthesis")
TESTS = ["test_exact_fallback.py", "test_closed_form_core.py", "test_batched_polish.py",
         "test_guard_v4.py", "test_matching_layout.py"]


def out(name):
    return os.path.join(HOME, name)


def e_setup(le, np):
    rng_e = np.random.default_rng(7)
    tt = rng_e.uniform(-np.pi, np.pi, le.E_NPARAMS)
    tt[8::15] = 0.0
    rng_t = np.random.default_rng(202)
    thetas = [tt + rng_t.normal(0.0, 0.5, le.E_NPARAMS) for _ in range(E_LAPS)]
    return tt, thetas


def compile_e(pc, le, qc, cmap):
    with warnings.catch_warnings(record=True) as ws, contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("always")
        pc._CX_CORE_CACHE.clear()
        t0 = time.perf_counter()
        o = pc.compile_for_hardware(qc, coupling_map=cmap, basis_gates=["cx", "rz", "sx", "x"],
                                    entangling_basis="cx", initial_layout=list(range(le.E_QUBITS)),
                                    on_unsupported="keep", seed_transpiler=0, block_gate_floor=8)
        el = time.perf_counter() - t0
    fb = sum(int(m.group(1)) for w in ws for m in [FELL.search(str(w.message))] if m)
    return o, el, fb


def measure(label):
    import numpy as np
    import psf_compile as pc
    import psf_zero_core as core
    import loop_endurance as le
    import qiskit
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from qiskit.synthesis import TwoQubitWeylDecomposition
    from qiskit.quantum_info import Operator
    from qiskit.transpiler import CouplingMap

    res = {"label": label, "platform": platform.platform(), "python": platform.python_version(),
           "qiskit": qiskit.__version__, "numpy": np.__version__,
           "psf_compile_file": pc.__file__, "psf_compile_version": pc.VERSION,
           "core_file": core.__file__, "core_version": getattr(core, "CORE_VERSION", None)}
    print("LOADED", pc.__file__, pc.VERSION)
    print("CORE", core.__file__, "CORE_VERSION", res["core_version"])
    print("ENV", res["platform"], "| python", res["python"], "| qiskit", res["qiskit"], "| numpy", res["numpy"])
    want = {"base": None, "fixed": "2026-09-28.1"}[label]
    if pc.VERSION != "2026-09-27.7" or res["core_version"] != want or not pc.__file__.startswith(REPO):
        print("V0 FAILED: wrong psf_compile.py or wrong core for label", label, "- stopping.")
        return
    print("V0 passed")

    cmap = CouplingMap.from_line(le.E_QUBITS)
    tt, thetas = e_setup(le, np)
    target = Statevector(le.e_circuit(tt))

    # ---- E, pass 1: timing, fallbacks, GUARD_STATS, loss error, CX (no wrapper)
    compile_e(pc, le, le.e_circuit(tt), cmap)
    gc.collect(); gc.freeze()
    for k in pc.GUARD_STATS:
        pc.GUARD_STATS[k] = 0 if isinstance(pc.GUARD_STATS[k], int) else 0.0
    times, fbs, cxs, worst = [], 0, Counter(), 0.0
    for lap, th in enumerate(thetas):
        qc = le.e_circuit(th)
        gc.disable(); o, el, fb = compile_e(pc, le, qc, cmap); gc.enable()
        if lap % 50 == 0:
            gc.collect()
        times.append(el); fbs += fb; cxs[o.count_ops().get("cx", 0)] += 1
        err = abs((1 - abs(target.inner(Statevector(o))) ** 2) - (1 - abs(target.inner(Statevector(qc))) ** 2))
        worst = max(worst, err)
    res["e"] = {"laps": E_LAPS, "fallbacks": fbs, "worst_loss_error": worst,
                "cx_counts": {str(k): v for k, v in cxs.items()},
                "median_ms": st.median(times) * 1e3, "p90_ms": float(np.percentile(times, 90)) * 1e3,
                "guard_stats": {k: (float(v) if isinstance(v, float) else int(v)) for k, v in pc.GUARD_STATS.items()}}
    print("E pass 1:", json.dumps(res["e"]))

    # ---- E, pass 2: capture every block handed to the core and the error type
    cap_u, cap_tag = [], []
    orig_checked, orig_plain = pc._CORE_CHECKED, pc.geometric_decompose

    def wrap(f):
        def g(u_r, u_i):
            u = np.array(u_r) + 1j * np.array(u_i)
            try:
                r = f(u_r, u_i); cap_u.append(u); cap_tag.append("ok"); return r
            except Exception as e:
                cap_u.append(u); cap_tag.append(type(e).__name__); raise
        return g
    if orig_checked is not None:
        pc._CORE_CHECKED = wrap(orig_checked)
    pc.geometric_decompose = wrap(orig_plain)
    for th in thetas:
        compile_e(pc, le, le.e_circuit(th), cmap)
    pc._CORE_CHECKED, pc.geometric_decompose = orig_checked, orig_plain
    res["capture"] = {"blocks": len(cap_tag), "tags": dict(Counter(cap_tag))}
    print("E pass 2 (capture):", json.dumps(res["capture"]))
    if label == "base":
        np.savez(out("core_fix_blocks_base.npz"), U=np.array(cap_u), tags=np.array(cap_tag))

    # ---- raw core error on the base capture (same inputs for both cores)
    bpath = out("core_fix_blocks_base.npz")
    if not os.path.exists(bpath):
        print("base capture missing: run --label base first. Stopping.")
        return
    d = np.load(bpath); U = d["U"]
    per = len(U) // E_LAPS

    def raw(u):
        try:
            c, k1, k2, ph = core.geometric_decompose(u.real.tolist(), u.imag.tolist())
        except Exception as e:
            return type(e).__name__
        v = pc._reconstruct(c, k1, k2, ph)
        z = np.vdot(v, u); z = z / abs(z)
        return float(np.linalg.norm(u - z * v))
    lap = [raw(U[2137 * per + p]) for p in range(per)]
    res["lap2138"] = {"blocks_per_lap": per, "raw": lap}
    # position check: capture position LAP_2138_POS must be the E circuit's pair (7, 8)
    th = thetas[2137]; k78 = le.E_BLOCKS.index((7, 8))
    q = QuantumCircuit(2); p = th[15 * k78:15 * k78 + 15]
    q.rxx(p[6], 0, 1); q.ryy(p[7], 0, 1); q.rzz(p[8], 0, 1)
    wa = TwoQubitWeylDecomposition(Operator(q).data); wb = TwoQubitWeylDecomposition(U[2137 * per + LAP_2138_POS])
    res["lap2138"]["pos_is_pair_7_8"] = bool(np.allclose([wa.a, wa.b, wa.c], [wb.a, wb.b, wb.c], atol=1e-6))
    alld = [raw(u) for u in U]
    nums = [x for x in alld if isinstance(x, float)]
    res["raw_all"] = {"failed": len(alld) - len(nums), "max": max(nums),
                      "over_1e_13": sum(x > 1e-13 for x in nums)}
    print("lap 2138:", json.dumps(res["lap2138"]))
    print("raw, all captured blocks:", json.dumps(res["raw_all"]))

    # ---- cliff (FakeNighthawk, pair24, spare 0)
    backend, native = le.nighthawk(); n = backend.coupling_map.size()
    rng = np.random.default_rng(101); rows = []
    for rep in range(CLIFF_REPS):
        th = rng.uniform(-np.pi, np.pi, (n // 2, 24))
        qc = QuantumCircuit(n)
        for k in range(n // 2):
            le.add_pair24(qc, 2 * k, 2 * k + 1, th[k])
        with warnings.catch_warnings(record=True) as ws, contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("always"); pc._CX_CORE_CACHE.clear()
            t0 = time.perf_counter()
            o = pc.compile_for_hardware(qc, coupling_map=backend.coupling_map, basis_gates=native,
                                        entangling_basis="cx", layout_search=True, on_unsupported="keep",
                                        seed_transpiler=0, block_gate_floor=8)
            el = time.perf_counter() - t0
        fb = sum(int(m.group(1)) for w in ws for m in [FELL.search(str(w.message))] if m)
        ok, w = le.pair_check(qc, o, n)
        ops = o.count_ops()
        h = hashlib.sha256(repr([(i.operation.name, tuple(o.find_bit(x).index for x in i.qubits),
                                  tuple(float(v) for v in i.operation.params)) for i in o.data]).encode()).hexdigest()
        rows.append({"rep": rep, "twoq": int(sum(v for k2, v in ops.items() if k2 in ("cz", "cx", "ecr"))),
                     "pair_ok": bool(ok), "worst": w, "fallbacks": fb, "ms": el * 1e3, "hash": h})
        print("cliff", json.dumps(rows[-1]))
    res["cliff"] = rows

    # ---- unit tests
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
                       cwd=BENCH, capture_output=True, text=True)
    text = r.stdout + r.stderr
    m = re.search(r"(\d+) passed", text); f = re.search(r"(\d+) failed", text)
    res["tests"] = {"passed": int(m.group(1)) if m else 0, "failed": int(f.group(1)) if f else 0,
                    "su2_warning_lines": text.count("SU2ExtractionSingular"), "returncode": r.returncode}
    print("tests:", json.dumps(res["tests"]))

    with open(out(f"core_fix_{label}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", out(f"core_fix_{label}.json"))


def score():
    B = json.load(open(out("core_fix_base.json"))); F = json.load(open(out("core_fix_fixed.json")))

    def v(ok, bad):
        return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")
    print("=" * 78); print("SCORING (thresholds exactly as pre-registered)"); print("=" * 78)
    print("V0 harness: base core", B["core_version"], "| fixed core", F["core_version"],
          "| psf_compile", B["psf_compile_version"], F["psf_compile_version"])
    bt, ft = B["capture"]["tags"], F["capture"]["tags"]
    bs = bt.get("PsfSU2SingularError", 0); fs = ft.get("PsfSU2SingularError", 0)
    f_other = sum(n for k, n in ft.items() if k not in ("ok", "PsfSU2SingularError"))
    print(f"C1 base SU2 failures in [4000, 4600]: {bs} ->", v(4000 <= bs <= 4600, not 4000 <= bs <= 4600))
    print(f"C2 fixed SU2 failures = 0 and other core errors <= 2: SU2 {fs}, other {f_other} ->",
          v(fs == 0 and f_other <= 2, fs > 0 or f_other > 2))
    ok3 = all(x["e"]["worst_loss_error"] <= 1e-13 and list(x["e"]["cx_counts"]) == ["33"] for x in (B, F))
    print("C3 both: worst loss error <= 1e-13 and CX 33 in every compile:",
          f"{B['e']['worst_loss_error']:.2e} {B['e']['cx_counts']} | {F['e']['worst_loss_error']:.2e} {F['e']['cx_counts']} ->",
          v(ok3, not ok3))
    g = F["e"]["guard_stats"]; ok4 = g["exact_rebuilt"] == 0 and g["psf_rerouted"] == 0 and g["best_effort"] == 0
    print("C4 fixed: exact_rebuilt = psf_rerouted = best_effort = 0:",
          g["exact_rebuilt"], g["psf_rerouted"], g["best_effort"], "| base:",
          B["e"]["guard_stats"]["exact_rebuilt"], B["e"]["guard_stats"]["psf_rerouted"],
          B["e"]["guard_stats"]["best_effort"], "->", v(ok4, not ok4))
    lb, lf = B["lap2138"]["raw"][LAP_2138_POS], F["lap2138"]["raw"][LAP_2138_POS]
    pos = B["lap2138"]["pos_is_pair_7_8"]
    ok5 = pos and isinstance(lb, float) and lb >= 1e-7 and isinstance(lf, float) and lf <= 1e-12
    print(f"C5 lap 2138, position {LAP_2138_POS} (pair (7,8): {pos}): base {lb} >= 1e-7 and fixed {lf} <= 1e-12 ->",
          v(ok5, not ok5))
    cb, cf = B["cliff"], F["cliff"]
    ok6 = all(a["twoq"] == b["twoq"] == 180 and a["pair_ok"] and b["pair_ok"] and a["worst"] <= 1e-13
              and b["worst"] <= 1e-13 and a["fallbacks"] == b["fallbacks"] == 0 for a, b in zip(cb, cf))
    print("C6 cliff: both 180 two-qubit gates, exact, 0 fallbacks in all reps ->", v(ok6, not ok6))
    ndiff = sum(a["hash"] != b["hash"] for a, b in zip(cb, cf))
    print(f"C7 cliff outputs bit-different in all {len(cb)} reps: {ndiff} ->", v(ndiff == len(cb), ndiff < len(cb)))
    tb, tf = B["tests"], F["tests"]
    ok8 = tb["passed"] == tf["passed"] == 75 and tb["failed"] == tf["failed"] == 0 \
        and tb["su2_warning_lines"] >= 1 and tf["su2_warning_lines"] == 0
    print("C8 tests: 75 passed in both; SU2 warning in base only:", tb, tf, "->", v(ok8, not ok8))
    r = F["e"]["median_ms"] / B["e"]["median_ms"]
    print(f"C9 (timing, RunPod pod) E median fixed/base = {F['e']['median_ms']:.2f}/{B['e']['median_ms']:.2f} = {r:.3f} ->",
          "CONFIRMED" if r <= 0.95 else ("REFUTED" if r >= 1.0 else "AMBIGUOUS"))
    print("\nReported without prediction: base GUARD_STATS", B["e"]["guard_stats"])
    print("raw error over all captured blocks: base", B["raw_all"], "| fixed", F["raw_all"])
    print("cliff median ms: base", st.median(x["ms"] for x in cb), "| fixed", st.median(x["ms"] for x in cf))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["measure", "score"])
    ap.add_argument("--label", choices=["base", "fixed"])
    a = ap.parse_args()
    measure(a.label) if a.mode == "measure" else score()
