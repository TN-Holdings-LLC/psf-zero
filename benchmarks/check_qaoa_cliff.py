"""check_qaoa_cliff.py -- Addendum 135.

Does QAOA (MaxCut, p=1) hit Qiskit's VF2Layout failure region, and does it
get there sooner as the problem graph gets denser? Reuses the callback
method from check_vf2_cliff_version.py (Addendum 127) to isolate VF2Layout's
own time from total transpile time.

Usage:
    python check_qaoa_cliff.py
"""
from __future__ import annotations

import csv
import time

import networkx as nx
import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

BASIS = ["rz", "sx", "x", "cx"]
SEEDS = (0, 1, 2)
DENSITIES = (0.1, 0.2, 0.3, 0.4, 0.5)
DEVICES = [
    ("8x8 grid", CouplingMap.from_grid(8, 8), 64),
    ("heavy_hex_d5", None, 57),  # coupling map built below, same construction as Addendum 104/106
]


def heavy_hex_edges(d):
    """Same construction as this project's own Addendum 104/106 heavy-hex
    instances: IBM's own heavy-hex lattice via rustworkx's generator,
    re-expressed as a plain edge list for Qiskit's CouplingMap.

    NOTE: an earlier version of this function passed bidirectional=False to
    heavy_hex_graph(); the installed rustworkx does not accept that keyword
    at all (confirmed by the actual TypeError raised, not assumed from
    documentation, which described a different version's signature). Calling
    heavy_hex_graph(d) with no extra arguments avoids relying on that
    keyword's existence either way, and edges are made explicitly symmetric
    here in plain Python so CouplingMap gets an undirected graph regardless
    of which convention the installed rustworkx version's own PyGraph uses
    internally.
    """
    import rustworkx as rx
    graph = rx.generators.heavy_hex_graph(d)
    edges = list(graph.edge_list())
    symmetric = set(edges) | {(b, a) for a, b in edges}
    return sorted(symmetric)


def qaoa_circuit(graph: nx.Graph, n: int):
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    gamma = 0.7  # fixed angle; this experiment does not optimize QAOA, only
                # tests whether its own fixed interaction graph compiles
    for a, b in graph.edges():
        qc.rzz(gamma, a, b)
    for q in range(n):
        qc.rx(0.5, q)
    return qc


def run_one(qc, cm, seed):
    record = {"vf2_s": None, "stop_reason": None}

    def callback(**kwargs):
        if kwargs["pass_"].name() == "VF2Layout":
            record["vf2_s"] = kwargs["time"]
            reason = kwargs["property_set"].get("VF2Layout_stop_reason")
            record["stop_reason"] = getattr(reason, "name", str(reason))

    t0 = time.perf_counter()
    transpile(qc, coupling_map=cm, basis_gates=BASIS, optimization_level=3,
              seed_transpiler=seed, callback=callback)
    record["total_s"] = time.perf_counter() - t0
    return record


def main():
    print(f"Qiskit {qiskit.__version__}\n")

    hh_edges = heavy_hex_edges(5)
    hh_n = max(max(e) for e in hh_edges) + 1
    print(f"heavy_hex_d5: n={hh_n} qubits, {len(hh_edges)} edges "
          f"(reused unchanged from Addendum 104/106)\n")
    DEVICES[1] = ("heavy_hex_d5", CouplingMap(hh_edges), hh_n)

    rows = []
    for dev_label, cm, n in DEVICES:
        print("=" * 100)
        print(f"Device: {dev_label} (n={n})")
        print("=" * 100)
        for d in DENSITIES:
            for seed in SEEDS:
                graph = nx.gnp_random_graph(n, d, seed=seed)
                qc = qaoa_circuit(graph, n)
                rec = run_one(qc, cm, seed)
                vf2 = f"{rec['vf2_s']*1000:9.1f} ms" if rec["vf2_s"] is not None else "   not run "
                print(f"  d={d:.1f} seed {seed}: edges={graph.number_of_edges():4d} | "
                      f"VF2Layout {vf2} | stop {rec['stop_reason']} | "
                      f"total {rec['total_s']*1000:9.1f} ms", flush=True)
                rows.append(dict(device=dev_label, n=n, density=d, seed=seed,
                                 n_edges=graph.number_of_edges(), **rec))
        print()

    out = "qaoa_cliff_2026-09-22.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
