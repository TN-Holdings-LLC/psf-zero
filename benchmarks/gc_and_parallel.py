"""gc_and_parallel.py -- Addendum 201: two remedies suggested for the loop, measured.

Part G  Garbage-collection strategies in the compounding loop (the worst
        case of Addendum 200: a generation-2 collection every ~22 laps,
        ~77 ms each, landing inside compiles). Each arm runs LAPS laps in
        its own freshly spawned process, so no arm inherits another's heap:
          default        -- collector untouched
          freeze         -- gc.collect(); gc.freeze() once after set-up
          pause          -- collector disabled during each compile and
                            back-mapping; gc.collect() at every lap boundary
          freeze+pause   -- both
        Per lap: compile time, boundary-collection time, gen-2 collections
        inside the compile. Per arm: total wall time and resident memory.
Part P  Batch compilation in parallel processes. A batch of B circuits that
        differ only in their parameters (as in SPSA or parameter-shift
        gradients) is compiled serially in this process and by a pool of W
        spawned worker processes (W = 2, 4, 8, 12), the pool created once
        and warmed up outside the timing. Two sizes: cliff scale
        (FakeNighthawk, 120 logical qubits) and small (12 qubits, line).
        Every compiled circuit is hashed (OpenQASM 2 text) so parallel
        output can be compared with serial output exactly.

Usage (repository root; loop_endurance.py must be in benchmarks/):
    python -u benchmarks/gc_and_parallel.py 2>&1 | tee gc_and_parallel.txt
"""
from __future__ import annotations

import contextlib
import csv
import gc
import hashlib
import io
import multiprocessing as mp
import os
import platform
import statistics as st
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks"), os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qiskit
from qiskit import QuantumCircuit, qasm2
from qiskit.transpiler import CouplingMap

import loop_endurance as le
import psf_compile as pc

G_LAPS = 800
G_ARMS = ("default", "freeze", "pause", "freeze+pause")
P_BATCH = 24
P_REPEATS = 5
P_WORKERS = (2, 4, 8, 12)
OUT_CSV = "gc_and_parallel_2026-09-27.csv"


def normalized_sha256(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f.read().splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def pct(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


# ------------------------------------------------------------ Part G

def g_worker(arm, laps, q):
    backend, native = le.nighthawk()
    n = backend.coupling_map.size()
    rng = np.random.default_rng(13)
    theta = rng.uniform(-np.pi, np.pi, (n // 2, 24))
    current = QuantumCircuit(n)
    for k in range(n // 2):
        le.add_pair24(current, 2 * k, 2 * k + 1, theta[k])
    le.cfh(current, backend, native)  # warm-up, outside the record

    g2 = {"n": 0}

    def cb(phase, info):
        if phase == "stop" and info["generation"] == 2:
            g2["n"] += 1

    gc.callbacks.append(cb)
    freeze = arm in ("freeze", "freeze+pause")
    pause = arm in ("pause", "freeze+pause")
    if freeze:
        gc.collect()
        gc.freeze()
    rss0 = le.rss_mb()
    compile_t, boundary_t, g2_in_compile = [], [], []
    t_start = time.perf_counter()
    for _ in range(laps):
        if pause:
            gc.disable()
        before = g2["n"]
        out, el, _ = le.cfh(current, backend, native)
        g2_in_compile.append(g2["n"] - before)
        current = le.back_to_logical(out, n)
        compile_t.append(el)
        if pause:
            gc.enable()
            b0 = time.perf_counter()
            gc.collect()
            boundary_t.append(time.perf_counter() - b0)
    wall = time.perf_counter() - t_start
    gc.callbacks.remove(cb)
    q.put(dict(arm=arm, compile_t=compile_t, boundary_t=boundary_t, g2_in_compile=g2_in_compile,
               wall=wall, rss0=rss0, rss1=le.rss_mb()))


def part_g(rows):
    print(f"\n=== G: garbage-collection strategies, compounding loop, {G_LAPS} laps per arm "
          f"(each arm in a fresh process) ===")
    ctx = mp.get_context("spawn")
    res = {}
    for arm in G_ARMS:
        q = ctx.Queue()
        p = ctx.Process(target=g_worker, args=(arm, G_LAPS, q))
        p.start()
        r = q.get()
        p.join()
        res[arm] = r
        ct, bt = r["compile_t"], r["boundary_t"]
        med = st.median(ct)
        p99 = pct(ct, 0.99)
        slow = sum(1 for t in ct if t > 2 * med)
        g2 = sum(1 for x in r["g2_in_compile"] if x > 0)
        bmsg = (f"; boundary collect median {st.median(bt) * 1000:.1f} ms, total {sum(bt):.1f} s"
                if bt else "")
        print(f"  {arm:13s}: compile median {med * 1000:.1f} ms, p99 {p99 * 1000:.1f} ms, max {max(ct) * 1000:.1f} ms; "
              f"laps > 2 x median {slow}; laps with a gen-2 collection inside the compile {g2}{bmsg}; "
              f"loop wall time {r['wall']:.1f} s; RSS {r['rss0']:.0f} -> {r['rss1']:.0f} MB", flush=True)
        for i, t in enumerate(ct):
            rows.append(dict(part="G", arm=arm, index=i + 1, compile_s=t,
                             boundary_s=bt[i] if bt else "", g2_in_compile=r["g2_in_compile"][i]))
        rows.append(dict(part="G", arm=arm, index="summary", compile_s=med, p99_s=p99, max_s=max(ct),
                         slow=slow, wall_s=r["wall"], rss_start=r["rss0"], rss_end=r["rss1"],
                         boundary_s=st.median(bt) if bt else ""))
    return res


# ------------------------------------------------------------ Part P

_W = {}


def _init_worker():
    backend, native = le.nighthawk()
    _W["backend"], _W["native"] = backend, native
    _W["line"] = CouplingMap.from_line(le.E_QUBITS)


def _circuit(kind, theta):
    if kind == "cliff":
        n = 2 * theta.shape[0]
        qc = QuantumCircuit(n)
        for k in range(theta.shape[0]):
            le.add_pair24(qc, 2 * k, 2 * k + 1, theta[k])
        return qc
    return le.e_circuit(theta)


def compile_task(kind, theta):
    """Build and compile one circuit; return (sha256 of its OpenQASM 2 text,
    compile seconds). Runs in the parent (serial) or in a worker."""
    if not _W:
        _init_worker()
    qc = _circuit(kind, theta)
    if kind == "cliff":
        out, el, _ = le.cfh(qc, _W["backend"], _W["native"])
    else:
        pc._CX_CORE_CACHE.clear()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            out = pc.compile_for_hardware(qc, coupling_map=_W["line"], basis_gates=["cx", "rz", "sx", "x"],
                                          entangling_basis="cx", initial_layout=list(range(le.E_QUBITS)),
                                          on_unsupported="keep", seed_transpiler=0)
        el = time.perf_counter() - t0
    return hashlib.sha256(qasm2.dumps(out).encode()).hexdigest(), el


def batches(kind):
    rng = np.random.default_rng(21 if kind == "cliff" else 22)
    if kind == "cliff":
        base = rng.uniform(-np.pi, np.pi, (60, 24))
    else:
        base = rng.uniform(-np.pi, np.pi, le.E_NPARAMS)
    out = []
    for _ in range(P_REPEATS):
        base = base + rng.normal(0.0, 0.02, base.shape)
        out.append([base + rng.normal(0.0, 0.1, base.shape) for _ in range(P_BATCH)])
    return out


def part_p(rows):
    print(f"\n=== P: batch compilation, batch of {P_BATCH} circuits, {P_REPEATS} batches per setting ===")
    summary = {}
    for kind in ("cliff", "small"):
        work = batches(kind)
        _init_worker()
        compile_task(kind, work[0][0])  # warm-up
        serial_t, serial_hash = [], []
        for b in work:
            t0 = time.perf_counter()
            hs = [compile_task(kind, th)[0] for th in b]
            serial_t.append(time.perf_counter() - t0)
            serial_hash.append(hs)
        s_med = st.median(serial_t)
        print(f"  [{kind}] serial: batch median {s_med:.3f} s ({s_med / P_BATCH * 1000:.1f} ms per circuit)", flush=True)
        rows.append(dict(part="P", arm=f"{kind} serial", index="summary", wall_s=s_med, workers=1, speedup=1.0))
        summary[(kind, 1)] = 1.0
        for w in P_WORKERS:
            t_pool = time.perf_counter()
            with ProcessPoolExecutor(max_workers=w, mp_context=mp.get_context("spawn"),
                                     initializer=_init_worker) as ex:
                list(ex.map(compile_task, [kind] * w, [work[0][0]] * w))  # warm every worker
                startup = time.perf_counter() - t_pool
                par_t, mismatches = [], 0
                for b, hs in zip(work, serial_hash):
                    t0 = time.perf_counter()
                    got = list(ex.map(compile_task, [kind] * len(b), b))
                    par_t.append(time.perf_counter() - t0)
                    mismatches += sum(1 for (h, _), hs_i in zip(got, hs) if h != hs_i)
            p_med = st.median(par_t)
            speed = s_med / p_med
            summary[(kind, w)] = speed
            print(f"  [{kind}] {w:2d} workers: batch median {p_med:.3f} s ({p_med / P_BATCH * 1000:.1f} ms per circuit), "
                  f"speed-up {speed:.2f}x; outputs differing from serial: {mismatches}/{P_REPEATS * P_BATCH}; "
                  f"pool start-up incl. warm-up {startup:.1f} s", flush=True)
            rows.append(dict(part="P", arm=f"{kind} {w} workers", index="summary", wall_s=p_med, workers=w,
                             speedup=speed, mismatches=mismatches, startup_s=startup))
    return summary


def main():
    print(f"platform {platform.platform()} | cores {os.cpu_count()} | python {platform.python_version()} "
          f"| qiskit {qiskit.__version__}")
    print("LOADED", pc.__file__, pc.VERSION, normalized_sha256(pc.__file__))
    print("SCRIPT", os.path.abspath(__file__), normalized_sha256(os.path.abspath(__file__)))
    if pc.VERSION != "2026-09-26.4":
        print("V0 FAILED: psf_compile.py is not 2026-09-26.4. Stopping.")
        return
    rows = []
    t0 = time.time()
    g = part_g(rows)
    p = part_p(rows)

    def tail(arm):
        ct = g[arm]["compile_t"]
        return pct(ct, 0.99), st.median(ct)

    d_p99, d_med = tail("default")
    print("\n=== scoring ===")
    f_p99, f_med = tail("freeze")
    print(f"G1 freeze: p99 {f_p99 * 1000:.1f} vs default {d_p99 * 1000:.1f} ms -> "
          f"{'lower' if f_p99 < d_p99 else 'not lower'}")
    for arm in ("pause", "freeze+pause"):
        a_p99, a_med = tail(arm)
        rss_growth = g[arm]["rss1"] - g[arm]["rss0"]
        ok = a_p99 <= 2 * a_med and rss_growth <= 50
        print(f"G2 {arm}: compile p99 <= 2 x median ({a_p99 * 1000:.1f} vs {2 * a_med * 1000:.1f} ms) and "
              f"RSS growth <= 50 MB ({rss_growth:+.0f}) -> {'PASS' if ok else 'FAIL'}")
    bp = st.median(g["pause"]["boundary_t"])
    bfp = st.median(g["freeze+pause"]["boundary_t"])
    print(f"G3 freeze+pause boundary collect <= half of pause: {bfp * 1000:.1f} vs {bp * 1000:.1f} ms -> "
          f"{'PASS' if bfp <= 0.5 * bp else 'FAIL'}")
    mism = sum(r.get("mismatches", 0) or 0 for r in rows if r.get("part") == "P")
    print(f"P1 every parallel output identical to serial: {'PASS' if mism == 0 else 'FAIL'} ({mism} differing)")
    best_cliff = max(p[("cliff", w)] for w in P_WORKERS)
    best_small = max(p[("small", w)] for w in P_WORKERS)
    print(f"P2 cliff-scale best speed-up >= 4x: {best_cliff:.2f}x -> {'PASS' if best_cliff >= 4 else 'FAIL'}")
    print(f"P3 small-circuit best speed-up below the cliff-scale best: {best_small:.2f}x vs {best_cliff:.2f}x -> "
          f"{'PASS' if best_small < best_cliff else 'FAIL'}")
    fields = ["part", "arm", "index", "compile_s", "boundary_s", "g2_in_compile", "p99_s", "max_s", "slow",
              "wall_s", "rss_start", "rss_end", "workers", "speedup", "mismatches", "startup_s"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows); total wall time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
