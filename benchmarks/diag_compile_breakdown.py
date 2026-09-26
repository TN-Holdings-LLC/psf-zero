"""diag_compile_breakdown.py -- where does PSF-Zero's ~0.08 s go on the
FakeNighthawk cliff, and where do Qiskit L3's ~13 s go?

Diagnostic only (no pass/fail criteria). Its output decides what a speed-up
prototype should target; nothing is changed in psf_compile.py or
psf_smart_layout.py. Timing is done by wrapping the real functions in place
(module attributes are swapped for timed wrappers and restored afterwards),
so the measured code path is exactly the one compile_for_hardware() runs.

Part A  PSF-Zero compile_for_hardware(), same arguments as
        nighthawk_deadline_cliff.py (entangling_basis="cx",
        layout_search=True, on_unsupported="raise", seed_transpiler=0).
        For spare in {0, 8} and seeds 0-4: REPS plain calls (no wrappers),
        then REPS instrumented calls. The plain calls give the reference
        wall time; the instrumented ones give the breakdown, and the ratio
        of the two medians is reported as instrumentation overhead.
        The CX-core LRU cache is cleared before every call, so each call
        sees the cache state a freshly generated circuit would see.
Part B  The layout feasibility check in isolation: networkx
        max_weight_matching (what psf_smart_layout uses today) vs
        rustworkx.max_weight_matching vs networkx Hopcroft-Karp, on the
        Nighthawk coupling graph. Checks the three agree on cardinality.
Part C  Qiskit transpile(optimization_level=3) per-pass times via
        callback, spare in {0, 8}, seeds 0-2, one call each, plus
        VF2Layout's stop reason (tests the "budget exhausted" hypothesis
        for the ~13 s plateau).

All runs are in one process (no spawn): the first PSF-Zero call in the
process is timed and reported separately as "first call" (imports, caches),
and excluded from the medians.

Usage (repository root, the same environment as the Nighthawk runs):
    python -u diag_compile_breakdown.py 2>&1 | tee diag_compile_breakdown.txt
Options:
    --no-qiskit     skip Part C (saves ~1.5 minutes)
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import os
import platform
import statistics as st
import sys
import time
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import qiskit
import rustworkx as rx
import networkx as nx
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeNighthawk

import psf_compile as pc
import psf_smart_layout as psl

SPARES = (0, 8)
SEEDS = 5
REPS = 5
GATES_PER_PAIR = 20
QISKIT_SEEDS = 3
MATCHING_REPS = 5
NATIVE = ("cz", "ecr", "cx", "rz", "sx", "x", "id", "rzz")
OUT_CSV = "diag_compile_breakdown_2026-09-26.csv"
OUT_PASS_CSV = "diag_compile_breakdown_passes_2026-09-26.csv"


# ---------------------------------------------------------------- helpers

def normalized_sha256(path):
    """SHA-256 after stripping trailing whitespace per line and trailing
    blank lines (the project's pre-registration hashing convention)."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f.read().splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def build_dense_pair_blocks_circuit(num_qubits, gates_per_pair, seed):
    """Verbatim from bench_cliff_1v1.py (the generator every cliff script in
    this project uses)."""
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)
    for (a, b) in [(i, i + 1) for i in range(0, num_qubits - 1, 2)]:
        block = QuantumCircuit(2)
        for _ in range(gates_per_pair):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            block.append(UnitaryGate(u), [0, 1])
        qc.compose(block.decompose(), [a, b], inplace=True)
    return qc


def twoq_count(qc):
    return sum(1 for i in qc.data if len(i.qubits) == 2 and i.operation.name != "barrier")


# ---------------------------------------------------------------- timing

class Timers:
    def __init__(self):
        self.t = defaultdict(float)
        self.n = defaultdict(int)
        self.extra = {}

    def add(self, key, dt):
        self.t[key] += dt
        self.n[key] += 1


class Instrumentation:
    """Swaps timed wrappers into psf_compile / psf_smart_layout and restores
    the originals on exit. Every wrapped name is looked up as a module (or
    class) attribute at call time by the code under test, which is what
    makes wrapping in place observe the real call path."""

    def __init__(self):
        self.T = Timers()
        self._saved = []

    def _wrap(self, owner, name, key, post=None):
        orig = getattr(owner, name)
        T = self.T

        def wrapper(*a, **k):
            t0 = time.perf_counter()
            try:
                out = orig(*a, **k)
            finally:
                T.add(key, time.perf_counter() - t0)
            if post is not None:
                post(out)
            return out

        self._saved.append((owner, name, orig))
        setattr(owner, name, wrapper)

    def __enter__(self):
        T = self.T

        # compile(): total, and its PassManager (Collect2qBlocks + Consolidate).
        self._wrap(pc, "compile", "A1 psf.compile [total]")
        orig_pm = pc.PassManager

        class TimedPassManager(orig_pm):
            def run(self_pm, *a, **k):
                t0 = time.perf_counter()
                try:
                    return super().run(*a, **k)
                finally:
                    T.add("A1a  consolidate (Collect2qBlocks+ConsolidateBlocks)", time.perf_counter() - t0)

        self._saved.append((pc, "PassManager", orig_pm))
        pc.PassManager = TimedPassManager

        # Per-block synthesis and its parts.
        self._wrap(pc.SU4GeodesicPSFSynthesizer, "synthesize", "A1b  synthesize [all blocks]")
        if pc._CORE_CHECKED is not None:
            self._wrap(pc, "_CORE_CHECKED", "A1b.1 core decompose (Rust)")
        self._wrap(pc, "geometric_decompose", "A1b.1 core decompose (Rust)")

        def after_refine(out):
            _, before, _after = out
            if before > pc.REFINE_THRESHOLD:
                T.n["refine_triggered"] += 1

        self._wrap(pc, "_refine_decomposition", "A1b.2 polish (check + Gauss-Newton)", post=after_refine)
        self._wrap(pc.SU4GeodesicPSFSynthesizer, "_build_circuit", "A1b.3 build circuit (incl. CX core)")

        orig_cache = pc._cx_core_cached

        def cx_core(a, b, c):
            hit = (a, b, c) in pc._CX_CORE_CACHE
            t0 = time.perf_counter()
            try:
                return orig_cache(a, b, c)
            finally:
                dt = time.perf_counter() - t0
                T.add("A1b.3a cx core lookup/build", dt)
                T.n["cx_cache_hit" if hit else "cx_cache_miss"] += 1

        self._saved.append((pc, "_cx_core_cached", orig_cache))
        pc._cx_core_cached = cx_core

        # Layout search.
        def after_layout(out):
            _, info = out
            T.extra["layout_phase"] = info.get("phase")
            T.extra["layout_attempts"] = len(info.get("attempts", []))
            T.extra["layout_found"] = info.get("found")

        self._wrap(psl, "smart_vf2_layout", "A2 layout search [total]", post=after_layout)
        self._wrap(psl, "_has_feasible_matching", "A2a  matching check (networkx)")
        self._wrap(psl, "_candidate_orderings", "A2b  orderings")
        self._wrap(psl, "_fallback_orderings", "A2b  orderings")
        self._wrap(psl, "_relabel", "A2c  relabel graph")
        self._wrap(psl, "_try_mapping", "A2d  vf2_mapping (rustworkx)")
        self._wrap(pc, "_layout_map_to_list", "A3 layout map -> list")

        # Routing / translation.
        self._wrap(pc, "transpile", "A4 qiskit transpile L1 [total]")
        return self

    def __exit__(self, *exc):
        for owner, name, orig in reversed(self._saved):
            setattr(owner, name, orig)
        self._saved.clear()
        return False


def make_pass_callback(store):
    def cb(**kw):
        name = kw["pass_"].name()
        store[name] = store.get(name, 0.0) + kw["time"]
        if name == "VF2Layout":
            store["_vf2_stop_reason"] = str(kw["property_set"].get("VF2Layout_stop_reason"))
    return cb


def psf_call(qc, backend, native, callback=None):
    pc._CX_CORE_CACHE.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        t0 = time.perf_counter()
        out = pc.compile_for_hardware(qc, coupling_map=backend.coupling_map, basis_gates=native,
                                      entangling_basis="cx", layout_search=True,
                                      on_unsupported="raise", seed_transpiler=0,
                                      callback=callback)
        el = time.perf_counter() - t0
    return out, el


# ---------------------------------------------------------------- parts

def part_a(backend, native, rows, pass_rows):
    n_phys = backend.coupling_map.size()
    summary = {}
    first_done = False
    for spare in SPARES:
        per_key = defaultdict(list)
        plain_totals, instr_totals = [], []
        layout_info = []
        pass_acc = defaultdict(list)
        for seed in range(SEEDS):
            qc = build_dense_pair_blocks_circuit(n_phys - spare, GATES_PER_PAIR, seed)
            if not first_done:
                _, el = psf_call(qc, backend, native)
                print(f"first PSF-Zero call in process (imports, caches): {el:.4f}s", flush=True)
                rows.append(dict(part="A", spare=spare, seed=seed, mode="first_call", rep=0,
                                 component="total", seconds=el, count=1))
                first_done = True
            ref_2q = None
            for rep in range(REPS):
                out, el = psf_call(qc, backend, native)
                plain_totals.append(el)
                rows.append(dict(part="A", spare=spare, seed=seed, mode="plain", rep=rep,
                                 component="total", seconds=el, count=1))
                ref_2q = twoq_count(out)
            for rep in range(REPS):
                passes = {}
                with Instrumentation() as ins:
                    out, el = psf_call(qc, backend, native, callback=make_pass_callback(passes))
                if twoq_count(out) != ref_2q:
                    print(f"WARNING spare={spare} seed={seed}: instrumented 2q count "
                          f"{twoq_count(out)} != plain {ref_2q}", flush=True)
                instr_totals.append(el)
                T = ins.T
                comp = dict(T.t)
                comp["A0 compile_for_hardware [total]"] = el
                comp["A1c  compile other (to_matrix, compose, loop)"] = (
                    T.t["A1 psf.compile [total]"] - T.t["A1a  consolidate (Collect2qBlocks+ConsolidateBlocks)"]
                    - T.t["A1b  synthesize [all blocks]"])
                comp["A1b.4 synthesize other (tolist, verify, checks)"] = (
                    T.t["A1b  synthesize [all blocks]"] - T.t["A1b.1 core decompose (Rust)"]
                    - T.t["A1b.2 polish (check + Gauss-Newton)"] - T.t["A1b.3 build circuit (incl. CX core)"])
                comp["A5 compile_for_hardware other (pair collection, import)"] = (
                    el - T.t["A1 psf.compile [total]"] - T.t["A2 layout search [total]"]
                    - T.t["A3 layout map -> list"] - T.t["A4 qiskit transpile L1 [total]"])
                for k, v in comp.items():
                    per_key[k].append(v)
                    rows.append(dict(part="A", spare=spare, seed=seed, mode="instrumented", rep=rep,
                                     component=k, seconds=v, count=T.n.get(k, "")))
                for k in ("refine_triggered", "cx_cache_hit", "cx_cache_miss"):
                    rows.append(dict(part="A", spare=spare, seed=seed, mode="instrumented", rep=rep,
                                     component=k, seconds="", count=T.n.get(k, 0)))
                    per_key[k].append(T.n.get(k, 0))
                layout_info.append((T.extra.get("layout_found"), T.extra.get("layout_phase"),
                                    T.extra.get("layout_attempts")))
                for name, v in passes.items():
                    if name.startswith("_"):
                        continue
                    pass_acc[name].append(v)
                    pass_rows.append(dict(part="A-L1", spare=spare, seed=seed, rep=rep, pass_name=name, seconds=v))
            print(f"spare={spare} seed={seed} plain median={st.median(plain_totals[-REPS:]):.4f}s "
                  f"instr median={st.median(instr_totals[-REPS:]):.4f}s 2q={ref_2q} "
                  f"layout(found,phase,attempts)={layout_info[-1]}", flush=True)
        summary[spare] = (per_key, plain_totals, instr_totals, layout_info, pass_acc)

    print("\n=== Part A: PSF-Zero breakdown (median over seeds x reps, instrumented) ===")
    for spare, (per_key, plain, instr, layout_info, pass_acc) in summary.items():
        tot = st.median(per_key["A0 compile_for_hardware [total]"])
        print(f"\n-- spare={spare} (logical {n_phys - spare}) --")
        print(f"plain total median {st.median(plain):.4f}s | instrumented {st.median(instr):.4f}s "
              f"| overhead x{st.median(instr) / st.median(plain):.3f}")
        print(f"layout (found, phase, attempts) seen: {sorted(set(layout_info), key=str)}")
        for k in sorted(k for k in per_key if k[0] == "A"):
            m = st.median(per_key[k])
            print(f"  {k:58s} {m * 1000:9.2f} ms  {100 * m / tot:5.1f}%")
        for k in ("refine_triggered", "cx_cache_hit", "cx_cache_miss"):
            print(f"  {k:58s} {st.median(per_key[k]):9.1f} (count per call)")
        top = sorted(((st.median(v), n) for n, v in pass_acc.items()), reverse=True)[:8]
        print("  transpile L1 top passes (median ms): "
              + ", ".join(f"{n} {m * 1000:.1f}" for m, n in top))


def part_b(backend, rows):
    cm = backend.coupling_map
    n = cm.size()
    edges = sorted({tuple(sorted(e)) for e in cm.get_edges()})
    need = n // 2

    def nx_blossom():
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(edges)
        return len(nx.max_weight_matching(g, maxcardinality=True))

    def rx_blossom():
        g = rx.PyGraph()
        g.add_nodes_from(range(n))
        g.add_edges_from_no_data(edges)
        return len(rx.max_weight_matching(g, max_cardinality=True))

    def nx_hopcroft_karp():
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from(edges)
        if not nx.is_bipartite(g):
            return None  # not applicable; the general-graph checks above still are
        top = {v for v, c in nx.bipartite.color(g).items() if c == 0}
        return len(nx.bipartite.hopcroft_karp_matching(g, top_nodes=top)) // 2

    print("\n=== Part B: feasibility check alone (Nighthawk coupling graph) ===")
    for name, fn in (("networkx max_weight_matching (current)", nx_blossom),
                     ("rustworkx max_weight_matching", rx_blossom),
                     ("networkx hopcroft_karp (bipartite only)", nx_hopcroft_karp)):
        ts, card = [], None
        for rep in range(MATCHING_REPS):
            t0 = time.perf_counter()
            card = fn()
            ts.append(time.perf_counter() - t0)
            rows.append(dict(part="B", spare="", seed="", mode=name, rep=rep,
                             component="matching", seconds=ts[-1], count=card))
        print(f"  {name:42s} median {st.median(ts) * 1000:8.2f} ms  pairs={card} (perfect needs {need})")


def part_c(backend, native, rows, pass_rows):
    n_phys = backend.coupling_map.size()
    print("\n=== Part C: Qiskit transpile(optimization_level=3) per-pass ===")
    for spare in SPARES:
        for seed in range(QISKIT_SEEDS):
            qc = build_dense_pair_blocks_circuit(n_phys - spare, GATES_PER_PAIR, seed)
            passes = {}
            t0 = time.perf_counter()
            out = transpile(qc, backend, optimization_level=3, seed_transpiler=0,
                            callback=make_pass_callback(passes))
            el = time.perf_counter() - t0
            stop = passes.pop("_vf2_stop_reason", None)
            rows.append(dict(part="C", spare=spare, seed=seed, mode="qiskit_L3", rep=0,
                             component="total", seconds=el, count=twoq_count(out)))
            for name, v in passes.items():
                pass_rows.append(dict(part="C-L3", spare=spare, seed=seed, rep=0, pass_name=name, seconds=v))
            top = sorted(((v, n) for n, v in passes.items()), reverse=True)[:6]
            print(f"spare={spare} seed={seed} total {el:.3f}s 2q={twoq_count(out)} "
                  f"VF2Layout stop_reason={stop}", flush=True)
            print("    top passes: " + ", ".join(f"{n} {v:.3f}s" for v, n in top), flush=True)


def main():
    import rustworkx
    print(f"platform {platform.platform()} | cpu {platform.processor() or platform.machine()} "
          f"| cores {os.cpu_count()} | python {platform.python_version()}")
    print(f"qiskit {qiskit.__version__} | rustworkx {rustworkx.__version__} | networkx {nx.__version__} "
          f"| RAYON_NUM_THREADS={os.environ.get('RAYON_NUM_THREADS')}")
    print("LOADED", pc.__file__, pc.VERSION, normalized_sha256(pc.__file__))
    print("LOADED", psl.__file__, normalized_sha256(psl.__file__))
    print("SCRIPT", os.path.abspath(__file__), normalized_sha256(os.path.abspath(__file__)))

    backend = FakeNighthawk()
    native = [g for g in backend.operation_names if g in NATIVE]
    rows, pass_rows = [], []
    part_a(backend, native, rows, pass_rows)
    part_b(backend, rows)
    if "--no-qiskit" not in sys.argv:
        part_c(backend, native, rows, pass_rows)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["part", "spare", "seed", "mode", "rep", "component", "seconds", "count"])
        w.writeheader()
        w.writerows(rows)
    with open(OUT_PASS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["part", "spare", "seed", "rep", "pass_name", "seconds"])
        w.writeheader()
        w.writerows(pass_rows)
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows) and {OUT_PASS_CSV} ({len(pass_rows)} rows)")


if __name__ == "__main__":
    main()
