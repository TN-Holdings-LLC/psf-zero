"""microbench.py -- (2026-10-10, exploratory) the profile's suspects timed without the profiler, which inflates
small Python calls: building Qiskit's preset pass manager from a coupling map and basis (as psf_compile does on
every call), the same with the Target, and a whole warm compile of c33's default call, on FakeTorino.

    cd <psf-zero repository>;  python <this file>
"""
import contextlib
import io
import os
import statistics
import sys
import time
import warnings

REPO = os.path.abspath(os.environ.get("PSF_ZERO_REPO", os.getcwd()))
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
warnings.simplefilter("ignore")


def timed(f, n=15):
    f()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    import core_fix_c2_eval as H
    from qiskit.circuit.random import random_circuit
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    pc = H.load_module(os.path.join(REPO, "patches", "psf_compile_c33_2026-10-10", "psf_compile.py"), "psf_compile")
    pc.WARN_WITHOUT_TARGET = False
    be = FakeTorino()
    t = be.target
    cm = pc.prune_coupling_map(t.build_coupling_map(), t, 0.5)
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    lay = list(range(9))
    rows = [
        ("preset pass manager, level 1, map + basis (as psf_compile builds it)",
         lambda: generate_preset_pass_manager(1, coupling_map=cm, basis_gates=basis, seed_transpiler=0)),
        ("the same with an initial layout",
         lambda: generate_preset_pass_manager(1, coupling_map=cm, basis_gates=basis, seed_transpiler=0,
                                              initial_layout=lay)),
        ("preset pass manager, level 2, the backend (as Qiskit's user builds it)",
         lambda: generate_preset_pass_manager(2, backend=be, seed_transpiler=0)),
    ]
    qc = random_circuit(9, 12, max_operands=2, measure=True, seed=62)
    rows.append(("c33's default call, whole (9 qubits, warm)",
                 lambda: pc.compile_for_hardware(qc, backend=be, entangling_basis="cx", layout_search=True,
                                                 seed_transpiler=0)))
    rows.append(("Qiskit level 2, whole (the same circuit, warm)",
                 lambda: generate_preset_pass_manager(2, backend=be, seed_transpiler=0).run(qc)))
    try:
        import networkx as nx
        import rustworkx as rx
        g = nx.random_regular_graph(3, 60, seed=1)
        rg = rx.PyGraph()
        rg.add_nodes_from(range(60))
        rg.add_edges_from_no_data(list(g.edges()))
        rows.append(("networkx max_weight_matching (60 nodes, 3-regular)",
                     lambda: nx.max_weight_matching(g, maxcardinality=True)))
        rows.append(("rustworkx max_weight_matching (the same graph)",
                     lambda: rx.max_weight_matching(rg, max_cardinality=True)))
    except ImportError as exc:
        print("matching rows skipped:", exc)
    with contextlib.redirect_stdout(io.StringIO()):
        res = [(name, timed(f)) for name, f in rows]
    for name, v in res:
        print(f"{v * 1000:9.2f} ms  {name}")


if __name__ == "__main__":
    main()
