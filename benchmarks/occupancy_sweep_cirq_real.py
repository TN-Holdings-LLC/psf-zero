"""occupancy_sweep_cirq_real.py -- does Cirq's own, real `get_placements()`
show an occupancy cliff, and if so, in which direction?

## Why this exists, and why it replaces `occupancy_sweep_cirq.py`

The first attempt at this comparison (`occupancy_sweep_cirq.py`) called
`networkx.algorithms.isomorphism.GraphMatcher.subgraph_is_monomorphic()`
directly -- an *existence* check that stops at the first match, chosen
because `cirq` itself was not installed in the environment that wrote
that script. Once `cirq` was actually available, reading Cirq's own
source (`cirq-core/cirq/devices/named_topologies.py`,
`quantumlib/Cirq` on GitHub, fetched directly rather than assumed) showed
`cirq.get_placements()` -- the actual function `RandomDevicePlacer` calls
internally -- does something meaningfully different:

    def get_placements(big_graph, small_graph, max_placements=100_000):
        matcher = nx.algorithms.isomorphism.GraphMatcher(big_graph, small_graph)
        dedupe = {}
        for big_to_small_map in matcher.subgraph_monomorphisms_iter():
            dedupe[frozenset(big_to_small_map.keys())] = big_to_small_map
            if len(dedupe) > max_placements:
                raise ValueError(...)
        ...
        return small_to_bigs

**It enumerates every distinct placement** (deduplicating only exact
rotations/reflections that reuse the same device qubits), not "the first
one it finds." This is a fundamentally different computation from
Qiskit's `VF2Layout` (which needs and returns exactly one valid layout,
bounded by `call_limit`) and from the existence-only check the first
version of this script used.

**This produces a specific, testable, and counter-intuitive prediction
worth stating before running anything at full scale**: at spare=0, the
interaction graph nearly fills the device, so there should be very few
ways to place it (little room to shift or rotate the embedding) --
enumeration should be fast. At large spare, the same interaction graph
can be placed in many more positions within the larger empty region --
enumeration could plausibly be *slower*, not faster. If this holds, Cirq
would show the *opposite* direction of cliff from Qiskit's (which gets
catastrophically slow *at* saturation, not away from it). This has not
been checked before this script runs; treat it as an open prediction, not
a known result.

## What changed from the first version

  - Uses real `cirq.GridQubit` device-graph nodes and calls the actual
    `cirq.get_placements()`, not a hand-rolled `networkx` call.
  - Records `n_placements_found` (the enumeration count itself), since
    per the mechanism above, this count -- not just the elapsed time --
    is the thing most directly explaining any timing pattern found.
  - `max_placements` is exposed and defaults lower than Cirq's own
    100,000 default (see `--max-placements`), since an unbounded
    enumeration at large spare could otherwise make this sweep take
    unpredictably long; hitting this cap is recorded as its own outcome
    (`hit_cap`), distinct from "found" or "not found," not silently
    treated as failure.

## Usage

    python occupancy_sweep_cirq_real.py --rows 6 --cols 7 \\
        --spares 0,1,2,3,4,6,8,16,24 --seeds 1 --repeats 3 \\
        --max-placements 2000

Requires `cirq` (already installed) and `networkx`/`pandas`/`numpy`.
"""
from __future__ import annotations

import argparse
import platform
import time
from datetime import date

import cirq
import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- graphs
def build_dense_pair_interaction_graph(num_qubits: int) -> nx.Graph:
    """Same interaction graph as `occupancy_sweep.py` / `cross_compiler_cliff.py`
    / the first `occupancy_sweep_cirq.py`: plain-integer nodes, edges
    (0,1), (2,3), (4,5), ... -- kept identical across all four scripts so
    results are directly comparable."""
    g = nx.Graph()
    g.add_nodes_from(range(num_qubits))
    for i in range(0, num_qubits - 1, 2):
        g.add_edge(i, i + 1)
    return g


def build_grid_device_graph(rows: int, cols: int) -> nx.Graph:
    """The device graph, built with real `cirq.GridQubit` nodes -- matching
    what Cirq's own device-graph objects actually look like (per
    `rainbow_device.metadata.nx_graph` / `Sycamore23.metadata.qubit_set`
    in Cirq's own documentation), rather than plain-integer nodes as the
    first version of this comparison used."""
    g = nx.Graph()
    qubits = [cirq.GridQubit(r, c) for r in range(rows) for c in range(cols)]
    g.add_nodes_from(qubits)
    for r in range(rows):
        for c in range(cols):
            q = cirq.GridQubit(r, c)
            if c + 1 < cols:
                g.add_edge(q, cirq.GridQubit(r, c + 1))
            if r + 1 < rows:
                g.add_edge(q, cirq.GridQubit(r + 1, c))
    return g


# ---------------------------------------------------------------- feasibility
def matching_feasibility(device_graph: nx.Graph, n_pairs: int) -> dict:
    """Independent check via `networkx`'s general max-matching routine --
    a different algorithm from `GraphMatcher`, so this is not the same
    code path being timed twice."""
    matching = nx.max_weight_matching(device_graph, maxcardinality=True)
    return dict(max_matching_size=len(matching), required=n_pairs,
               embedding_feasible=len(matching) >= n_pairs)


# ---------------------------------------------------------------- timed call
def _get_placements_worker(rows, cols, n, max_placements, result_queue):
    """Runs in a *separate process* so it can be killed on a hard
    wall-clock timeout. This replaces an earlier version of this function
    that only checked `elapsed_s` *after* `cirq.get_placements()`
    returned -- which cannot actually interrupt a blocking call. That gap
    was exposed directly: a first real run hung indefinitely (confirmed
    via Ctrl+C, whose traceback showed execution genuinely inside
    `networkx`'s VF2 search, `isomorphvf2.py` line ~1079 -- not a frozen
    terminal) on a grid as small as 4x4, spare=0. The graphs are rebuilt
    inside the worker (not passed in) since `cirq.GridQubit`-keyed
    `nx.Graph` objects are not guaranteed cheap/safe to pickle across a
    process boundary for every networkx/cirq version; rebuilding from
    plain ints (rows, cols, n) is small, fast, and avoids that entirely.
    """
    try:
        device_graph = build_grid_device_graph(rows, cols)
        interaction_graph = build_dense_pair_interaction_graph(n)
        t0 = time.perf_counter()
        placements = cirq.get_placements(device_graph, interaction_graph,
                                         max_placements=max_placements)
        elapsed = time.perf_counter() - t0
        result_queue.put(dict(elapsed_s=elapsed, n_placements_found=len(placements),
                              hit_cap=False, error="", timed_out=False))
    except ValueError as e:
        elapsed = time.perf_counter() - t0
        if "more than" in str(e) and "placements" in str(e):
            result_queue.put(dict(elapsed_s=elapsed, n_placements_found=None,
                                  hit_cap=True, error=str(e), timed_out=False))
        else:
            result_queue.put(dict(elapsed_s=elapsed, n_placements_found=None,
                                  hit_cap=False, error=f"ValueError: {e}", timed_out=False))
    except Exception as e:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        result_queue.put(dict(elapsed_s=elapsed, n_placements_found=None,
                              hit_cap=False, error=f"{type(e).__name__}: {e}",
                              timed_out=False))


def timed_get_placements(rows: int, cols: int, n: int, max_placements: int,
                         wall_clock_timeout_s: float) -> dict:
    """Times a single real call to `cirq.get_placements()` in a subprocess
    with a genuine, enforced wall-clock timeout -- see
    `_get_placements_worker`'s docstring for why this replaced a version
    that could not actually interrupt a hung call. On timeout, the
    subprocess is terminated and `timed_out=True` is recorded with
    `elapsed_s` equal to the timeout itself (a lower bound on the true
    time, not the true time)."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_get_placements_worker,
                       args=(rows, cols, n, max_placements, result_queue))
    t0 = time.perf_counter()
    proc.start()
    proc.join(timeout=wall_clock_timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5.0)
        if proc.is_alive():  # still alive after terminate -- escalate
            proc.kill()
            proc.join()
        return dict(elapsed_s=time.perf_counter() - t0, n_placements_found=None,
                   hit_cap=False, error="", timed_out=True)
    if not result_queue.empty():
        result = result_queue.get()
        result.setdefault("timed_out", False)
        return result
    # Process exited without putting a result and without hitting the
    # join timeout above -- e.g. it crashed. Recorded honestly rather
    # than silently treated as a timeout or a success.
    return dict(elapsed_s=time.perf_counter() - t0, n_placements_found=None,
               hit_cap=False, error="worker process exited with no result "
               "(crashed or was killed by the OS, e.g. out of memory)",
               timed_out=False)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--cols", type=int, default=7)
    ap.add_argument("--spares", type=str, default="0,1,2,3,4,6,8,16,24")
    ap.add_argument("--seeds", type=int, default=1,
                    help="the interaction graph shape is deterministic given "
                         "qubit count (no randomness in which pairs interact), "
                         "so seeds do not vary the graph itself here -- kept "
                         "as an argument only for output-format parity with "
                         "this project's other cliff scripts")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-placements", type=int, default=2000,
                    help="capped well below Cirq's own default (100,000) so "
                         "a large-spare condition with many valid placements "
                         "cannot make the sweep's wall-clock time unbounded; "
                         "hitting this cap is recorded as its own outcome, "
                         "not conflated with success or failure")
    ap.add_argument("--wall-clock-timeout", type=float, default=10.0,
                    help="hard per-call timeout in seconds, enforced by "
                         "killing a subprocess -- not a soft check after the "
                         "call returns. Needed because an earlier version of "
                         "this script only checked elapsed time *after* "
                         "cirq.get_placements() returned, which cannot "
                         "interrupt a call that never returns: a real run "
                         "hung indefinitely on a 4x4 grid at spare=0, "
                         "confirmed via Ctrl+C showing execution genuinely "
                         "inside networkx's VF2 search, not a frozen "
                         "terminal. The likely cause (documented, not yet "
                         "separately confirmed): this project's dense-pair-"
                         "blocks interaction graph is a disjoint union of "
                         "N/2 unconnected 2-node edges -- an extremely "
                         "symmetric, disconnected structure that is a known "
                         "pathological case for VF2-family *enumeration* "
                         "(not existence-only) search, unrelated to grid "
                         "size.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    spares = [int(x) for x in args.spares.split(",")]
    n_physical = args.rows * args.cols
    device_graph = build_grid_device_graph(args.rows, args.cols)

    print(f"Device graph: {args.rows}x{args.cols} grid, {n_physical} cirq.GridQubit nodes")
    print(f"Spare values: {spares}")
    print(f"Qubit counts: {[n_physical - s for s in spares]}")
    print(f"max_placements cap: {args.max_placements}")
    print(f"Wall-clock timeout per call: {args.wall_clock_timeout}s (hard, via subprocess kill)\n")

    rows_out = []
    hdr = (f"{'spare':>6} {'n':>4} {'rep':>4} {'elapsed_s':>10} "
           f"{'n_placements':>13} {'hit_cap':>8} {'timed_out':>9} {'feasible':>9}")
    print(hdr)
    print("-" * len(hdr))

    for spare in spares:
        n = n_physical - spare
        if n < 2 or n % 2:
            print(f"spare={spare}: n={n} invalid (need >=2, even) -- skipped")
            continue
        n_pairs = n // 2
        feas = matching_feasibility(device_graph, n_pairs)

        for seed in range(args.seeds):
            for rep in range(args.repeats):
                result = timed_get_placements(args.rows, args.cols, n,
                                              args.max_placements,
                                              args.wall_clock_timeout)
                print(f"{spare:>6} {n:>4} {rep:>4} {result['elapsed_s']:10.4f} "
                      f"{str(result['n_placements_found']):>13} "
                      f"{str(result['hit_cap']):>8} "
                      f"{str(result['timed_out']):>9} "
                      f"{str(feas['embedding_feasible']):>9}", flush=True)
                rows_out.append(dict(
                    Spare=spare, Qubits=n, Seed=seed, Rep=rep,
                    Elapsed_s=result["elapsed_s"],
                    NPlacementsFound=result["n_placements_found"],
                    HitCap=result["hit_cap"], TimedOut=result["timed_out"],
                    Error=result["error"],
                    EmbeddingFeasible=feas["embedding_feasible"],
                    MaxMatchingSize=feas["max_matching_size"],
                    RequiredPairs=feas["required"], MaxPlacementsCap=args.max_placements,
                    Rows=args.rows, Cols=args.cols,
                    Platform=platform.platform(), Python=platform.python_version(),
                    CPU=platform.processor(),
                    CirqVersion=getattr(cirq, "__version__", "unknown"),
                ))

    df = pd.DataFrame(rows_out)
    out_path = args.out or f"occupancy_sweep_cirq_real_{args.rows}x{args.cols}_{date.today().isoformat()}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} rows)")

    # ------------------------------------------------------------ summary
    print("\n" + "=" * 78)
    print("Summary -- median elapsed time (s) and median placement count by spare")
    print("=" * 78)
    ok = df[df["Error"] == ""]
    summary_time = ok.groupby("Spare")["Elapsed_s"].median()
    summary_count = ok.groupby("Spare")["NPlacementsFound"].median()
    print(f"{'spare':>6} {'median_s':>10} {'median_n_placements':>20}")
    for spare in sorted(summary_time.index):
        print(f"{spare:>6} {summary_time[spare]:10.4f} "
              f"{summary_count.get(spare, float('nan')):20.1f}")

    if len(summary_time) >= 2 and 0 in summary_time.index:
        others = [s for s in summary_time.index if s != 0]
        if others:
            s_next = min(others)
            ratio = summary_time[0] / summary_time[s_next] if summary_time[s_next] > 0 else float("inf")
            print(f"\nspare=0 / spare={s_next} elapsed-time ratio: {ratio:.3f}x")
            print("  -> Per this script's own pre-stated prediction: if this ratio is")
            print("     LESS than 1 (spare=0 FASTER than spare>0), that is consistent with")
            print("     the placement-count mechanism described in the docstring, and would")
            print("     mean Cirq's cliff (if any) runs the OPPOSITE direction from Qiskit's")
            print("     and TKET's. If this ratio is large and positive like Qiskit's, the")
            print("     placement-count theory is wrong and something else is happening.")

    timed_out_rows = df[df["TimedOut"]]
    if len(timed_out_rows):
        print(f"\n**{len(timed_out_rows)} row(s) hit the {args.wall_clock_timeout}s hard "
              f"wall-clock timeout and were killed** -- their Elapsed_s is a lower bound "
              f"(approximately the timeout value), not the true search time. If this "
              f"happens at every spare value including large ones, that argues against "
              f"the placement-count-scales-with-spare theory and toward the dense-pair "
              f"interaction graph's own symmetry being the dominant cost regardless of "
              f"device occupancy -- worth checking directly (e.g. by trying a "
              f"differently-shaped, less symmetric interaction graph) rather than assumed.")

    hit_cap_rows = df[df["HitCap"]]
    if len(hit_cap_rows):
        print(f"\n**{len(hit_cap_rows)} row(s) hit the {args.max_placements}-placement cap** "
              f"-- their Elapsed_s reflects only enumerating up to the cap, not the true "
              f"total placement count. See NPlacementsFound=None for these rows; the cap "
              f"itself, not a null, is the informative result.")


if __name__ == "__main__":
    main()
