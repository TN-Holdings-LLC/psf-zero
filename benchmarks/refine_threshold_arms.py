"""refine_threshold_arms.py -- pre-registered (workplace, 2026-09-28): does
lowering psf_compile.REFINE_THRESHOLD remove the compounding drift that the
fixed Rust core (CORE_VERSION 2026-09-28.1) doubled, and at what compile-time
cost?

Arms (the threshold is set in memory; psf_compile.py itself is not changed):
  T13  REFINE_THRESHOLD = 1e-13 (current default; control)
  T14  REFINE_THRESHOLD = 1e-14 (= _REFINE_TARGET)
  T0   REFINE_THRESHOLD = 0     (polish every block)

Modes (run in this order; each writes its own JSON next to the working dir):
  drift  --arm A --laps N   part C of long_loop_100k_v3.py (seed 13, same
                            compile call) for N laps; distance to lap 0 every
                            CHECK laps and at N.
  timing                    fresh cliff circuits (part F of v3, seed 101) and
                            fresh training circuits (part E, seeds 7 / 202),
                            the three arms interleaved in chunks of CHUNK
                            compiles; per-compile times, exactness checks,
                            fallbacks, GUARD_STATS per arm; then an untimed
                            pass counting how many blocks each arm polishes.
  score                     scoring exactly as pre-registered.

    python -u refine_threshold_arms.py drift --arm T13 --laps 1000
    python -u refine_threshold_arms.py drift --arm T14 --laps 20000
    python -u refine_threshold_arms.py drift --arm T0  --laps 20000
    python -u refine_threshold_arms.py timing
    python -u refine_threshold_arms.py score
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import re
import statistics as st
import sys
import time
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks"), os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

import loop_endurance as le
import psf_compile as pc
import psf_zero_core as core

ARMS = {"T13": 1e-13, "T14": 1e-14, "T0": 0.0}
CHECK = 1000
F_TIMED, E_TIMED, CHUNK = 2000, 6000, 100
F_CHECK_EVERY, E_CHECK_EVERY = 100, 10
COUNT_F, COUNT_E = 100, 300
V3_LAP1000_T13 = 8.312e-11          # long_loop_100k_v3.py, part C, lap 1,000
FELL = re.compile(r"(\d+) block\(s\) fell back to CX-basis synthesis")
OUT = {"drift": "refine_arms_drift_{arm}_2026-09-28.json", "timing": "refine_arms_timing_2026-09-28.json"}


def set_threshold(t):
    pc.REFINE_THRESHOLD = t
    pc._refine_batch.__defaults__ = (t, 3)
    pc._refine_decomposition.__defaults__ = (t, 3)


def header():
    env = {"platform": platform.platform(), "python": platform.python_version(), "qiskit": qiskit.__version__,
           "numpy": np.__version__, "psf_compile": pc.VERSION, "core_version": getattr(core, "CORE_VERSION", None),
           "cpu": platform.processor(), "cores": os.cpu_count()}
    print("ENV", json.dumps(env))
    print("LOADED", os.path.basename(pc.__file__), pc.VERSION, le.normalized_sha256(pc.__file__))
    print("SCRIPT", os.path.basename(__file__), le.normalized_sha256(os.path.abspath(__file__)))
    print("CORE_VERSION", env["core_version"])
    return env


def v0(env):
    if env["psf_compile"] != "2026-09-27.7" or env["core_version"] != "2026-09-28.1":
        print("V0 FAILED: need psf_compile.py 2026-09-27.7 and CORE_VERSION 2026-09-28.1. Stopping.")
        return False
    return True


def quiet(fn):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            out = fn()
            el = time.perf_counter() - t0
    fb = 0
    for w in ws:
        m = FELL.search(str(w.message))
        if m:
            fb = int(m.group(1))
    return out, el, fb


def cliff(qc, backend, native):
    pc._CX_CORE_CACHE.clear()
    return pc.compile_for_hardware(qc, coupling_map=backend.coupling_map, basis_gates=native,
                                   entangling_basis="cx", layout_search=True, on_unsupported="keep",
                                   seed_transpiler=0, block_gate_floor=8)


def line(qc, cmap):
    pc._CX_CORE_CACHE.clear()
    return pc.compile_for_hardware(qc, coupling_map=cmap, basis_gates=["cx", "rz", "sx", "x"],
                                   entangling_basis="cx", initial_layout=list(range(le.E_QUBITS)),
                                   on_unsupported="keep", seed_transpiler=0, block_gate_floor=8)


def reset_guard():
    for k in list(pc.GUARD_STATS):
        pc.GUARD_STATS[k] = 0 if isinstance(pc.GUARD_STATS[k], int) else 0.0


def drift(arm, laps):
    env = header()
    if not v0(env):
        return
    set_threshold(ARMS[arm])
    print(f"ARM {arm} REFINE_THRESHOLD {pc.REFINE_THRESHOLD} laps {laps}")
    backend, native = le.nighthawk()
    n = backend.coupling_map.size()
    rng = np.random.default_rng(13)
    cur = QuantumCircuit(n)
    th = rng.uniform(-np.pi, np.pi, (n // 2, 24))
    for k in range(n // 2):
        le.add_pair24(cur, 2 * k, 2 * k + 1, th[k])
    ref = le.pair_matrices(cur, n)
    checks, fbs, t0 = {}, 0, time.time()
    for lap in range(1, laps + 1):
        out, _, fb = quiet(lambda: cliff(cur, backend, native))
        fbs += fb
        cur = le.back_to_logical(out, n)
        if lap % CHECK == 0 or lap == laps or lap == laps // 2:
            mats = le.pair_matrices(cur, n)
            d = {k: le.aligned(ref[k], mats[k]) for k in ref}
            w = max(d, key=d.get)
            checks[lap] = {"max": d[w], "pair": w}
            if lap % (CHECK * 5) == 0 or lap == laps:
                print(f"  lap {lap}: max distance {d[w]:.4e} (pair {w}); fallbacks {fbs}; {time.time() - t0:.0f} s",
                      flush=True)
    res = {"env": env, "arm": arm, "threshold": ARMS[arm], "laps": laps, "checks": checks, "fallbacks": fbs,
           "guard": dict(pc.GUARD_STATS), "wall_s": time.time() - t0}
    path = OUT["drift"].format(arm=arm)
    json.dump(res, open(path, "w"), indent=1)
    print("wrote", path)


def timing():
    env = header()
    if not v0(env):
        return
    backend, native = le.nighthawk()
    n = backend.coupling_map.size()
    cmap = CouplingMap.from_line(le.E_QUBITS)
    rng_f = np.random.default_rng(101)
    theta_f = rng_f.uniform(-np.pi, np.pi, (n // 2, 24))
    rng_e = np.random.default_rng(7)
    target_theta = rng_e.uniform(-np.pi, np.pi, le.E_NPARAMS)
    target_theta[8::15] = 0.0
    target = Statevector(le.e_circuit(target_theta))
    rng_t = np.random.default_rng(202)
    # the same circuits for every arm: pre-draw all inputs
    f_thetas, th = [], theta_f
    for _ in range(F_TIMED):
        th = th + rng_f.normal(0.0, 0.02, th.shape)
        f_thetas.append(th)
    e_thetas = [target_theta + rng_t.normal(0.0, 0.5, le.E_NPARAMS) for _ in range(E_TIMED)]

    def f_circ(i):
        qc = QuantumCircuit(n)
        for k in range(n // 2):
            le.add_pair24(qc, 2 * k, 2 * k + 1, f_thetas[i][k])
        return qc

    for t in ARMS.values():          # warm-up, one compile of each kind per arm
        set_threshold(t)
        quiet(lambda: cliff(f_circ(0), backend, native))
        quiet(lambda: line(le.e_circuit(e_thetas[0]), cmap))
    R = {a: {"f_s": [], "e_s": [], "f_worst": 0.0, "e_worst": 0.0, "f_fb": 0, "e_fb": 0, "errors": 0,
             "guard": {}} for a in ARMS}
    guards = {a: {} for a in ARMS}
    t_start = time.time()
    for part, total in (("f", F_TIMED), ("e", E_TIMED)):
        for start in range(0, total, CHUNK):
            for a, t in ARMS.items():
                set_threshold(t)
                saved = dict(pc.GUARD_STATS)
                reset_guard()
                for k, v in guards[a].items():
                    pc.GUARD_STATS[k] = v
                for i in range(start, min(total, start + CHUNK)):
                    try:
                        if part == "f":
                            qc = f_circ(i)
                            out, el, fb = quiet(lambda: cliff(qc, backend, native))
                            R[a]["f_s"].append(el); R[a]["f_fb"] += fb
                            if i % F_CHECK_EVERY == 0:
                                ok, worst = le.pair_check(qc, out, n)
                                R[a]["f_worst"] = max(R[a]["f_worst"], worst if ok else float("inf"))
                        else:
                            qc = le.e_circuit(e_thetas[i])
                            out, el, fb = quiet(lambda: line(qc, cmap))
                            R[a]["e_s"].append(el); R[a]["e_fb"] += fb
                            if i % E_CHECK_EVERY == 0:
                                d = abs((1 - abs(target.inner(Statevector(out))) ** 2)
                                        - (1 - abs(target.inner(Statevector(qc))) ** 2))
                                R[a]["e_worst"] = max(R[a]["e_worst"], d)
                    except Exception as exc:  # noqa: BLE001 -- recorded
                        R[a]["errors"] += 1
                        print(f"  {a} {part} {i}: {type(exc).__name__}: {exc}", flush=True)
                guards[a] = dict(pc.GUARD_STATS)
                reset_guard()
                for k, v in saved.items():
                    pc.GUARD_STATS[k] = v
        print(f"  part {part} done, {time.time() - t_start:.0f} s", flush=True)
    for a in ARMS:
        R[a]["guard"] = guards[a]
    # untimed: how many blocks each arm polishes
    orig = pc._refine_batch
    counts = {}
    for a, t in ARMS.items():
        set_threshold(t)
        c = {"blocks": 0, "polished": 0}

        def counting(us, p, threshold=t, max_iter=3):
            out = orig(us, p, threshold, max_iter)
            c["blocks"] += len(us); c["polished"] += int((out[1] > threshold).sum())
            return out
        pc._refine_batch = counting
        for i in range(COUNT_F):
            quiet(lambda: cliff(f_circ(i), backend, native))
        cf = dict(c); c["blocks"] = c["polished"] = 0
        for i in range(COUNT_E):
            quiet(lambda: line(le.e_circuit(e_thetas[i]), cmap))
        counts[a] = {"f": cf, "e": dict(c)}
        pc._refine_batch = orig
    set_threshold(1e-13)
    res = {"env": env, "arms": ARMS, "timed": {"F": F_TIMED, "E": E_TIMED, "chunk": CHUNK}, "results": R,
           "polish_counts": counts, "wall_s": time.time() - t_start}
    json.dump(res, open(OUT["timing"], "w"), indent=1)
    print("wrote", OUT["timing"])


def score():
    def verdict(ok, bad):
        return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")
    D = {a: json.load(open(OUT["drift"].format(arm=a))) for a in ARMS}
    T = json.load(open(OUT["timing"]))
    R = T["results"]
    print("=" * 78); print("SCORING (thresholds exactly as pre-registered)"); print("=" * 78)
    print("ENV", json.dumps(T["env"]))
    c13 = D["T13"]["checks"].get("1000", {}).get("max")
    envs_ok = all(d["env"]["core_version"] == "2026-09-28.1" and d["env"]["psf_compile"] == "2026-09-27.7"
                  for d in list(D.values()) + [T])
    laps_ok = D["T13"]["laps"] == 1000 and D["T14"]["laps"] == 20000 and D["T0"]["laps"] == 20000
    c0 = envs_ok and laps_ok and c13 is not None and abs(c13 - V3_LAP1000_T13) <= 0.01 * V3_LAP1000_T13
    print(f"C0: versions {envs_ok}; laps {laps_ok}; T13 lap-1000 distance {c13} vs v3 {V3_LAP1000_T13} (within 1%)"
          f" -> {'passed' if c0 else 'FAILED'}")
    if not c0:
        print("C0 FAILED: nothing is scored.")
        return
    d14 = D["T14"]["checks"]["20000"]["max"]
    d0 = D["T0"]["checks"]["20000"]["max"]
    print(f"P1 T14 drift at lap 20,000 {d14:.3e} (confirmed <= 3e-10, refuted >= 8.3e-10) -> "
          f"{verdict(d14 <= 3e-10, d14 >= 8.3e-10)}")
    print(f"P2 T0 drift at lap 20,000 {d0:.3e} (confirmed <= 1e-10, refuted >= 4e-10) -> "
          f"{verdict(d0 <= 1e-10, d0 >= 4e-10)}")
    ex = {a: (R[a]["errors"] == 0 and R[a]["f_worst"] <= 1e-12 and R[a]["e_worst"] <= 1e-13 and R[a]["e_fb"] <= 5)
          for a in ARMS}
    print("P3 exact in every arm (no exception, F per-pair <= 1e-12, E loss <= 1e-13, E fallbacks <= 5): " +
          ", ".join(f"{a} F {R[a]['f_worst']:.2e} E {R[a]['e_worst']:.2e} fb {R[a]['e_fb']} err {R[a]['errors']}"
                    for a in ARMS) + f" -> {verdict(all(ex.values()), not all(ex.values()))}")
    med = {a: (st.median(R[a]["f_s"]), st.median(R[a]["e_s"])) for a in ARMS}
    r14 = (med["T14"][0] / med["T13"][0], med["T14"][1] / med["T13"][1])
    r0 = (med["T0"][0] / med["T13"][0], med["T0"][1] / med["T13"][1])
    print(f"   compile medians (sandbox) F/E ms: " +
          ", ".join(f"{a} {med[a][0] * 1e3:.2f}/{med[a][1] * 1e3:.2f}" for a in ARMS))
    print(f"P4 T14 cost: median ratio to T13, F {r14[0]:.3f}, E {r14[1]:.3f} (confirmed both <= 1.10, refuted any > 1.25)"
          f" -> {verdict(max(r14) <= 1.10, max(r14) > 1.25)}")
    print(f"P5 T0 cost: median ratio to T13, F {r0[0]:.3f}, E {r0[1]:.3f} (confirmed both <= 1.25, refuted any > 1.50)"
          f" -> {verdict(max(r0) <= 1.25, max(r0) > 1.50)}")
    print("\nReported without prediction:")
    for a in ARMS:
        print(f"   {a}: drift checks {json.dumps(D[a]['checks'])}; drift-run fallbacks {D[a]['fallbacks']}")
        print(f"   {a}: polished blocks {json.dumps(T['polish_counts'][a])}; timing GUARD_STATS {json.dumps(R[a]['guard'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["drift", "timing", "score"])
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--laps", type=int)
    a = ap.parse_args()
    if a.mode == "drift":
        drift(a.arm, a.laps)
    elif a.mode == "timing":
        timing()
    else:
        score()
