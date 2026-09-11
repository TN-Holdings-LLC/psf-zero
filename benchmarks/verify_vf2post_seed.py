"""Does Qiskit's own VF2 fail for ordering reasons?

`VF2Layout` calls `with_vf2pp_ordering()` unconditionally and exposes no
`id_order` equivalent, but `shuffle_seed` permutes the coupling graph's node
indices before the search. Same map, same circuit, same call limit: only the
node order changes.
"""
import csv, platform, sys, time
import qiskit
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import VF2Layout
from qiskit.converters import circuit_to_dag
from phase3_v5_spare_qubits import build_dense_pair_blocks_circuit

ROWS, COLS, N_QUBITS, GATES_PER_PAIR = 6, 7, 42, 20
CALL_LIMIT, SEEDS = 3_000_000, 30

env = {
    "platform": platform.platform(),
    "processor": platform.processor(),
    "python": sys.version.split()[0],
    "qiskit": qiskit.__version__,
}
print(env)

cmap = CouplingMap.from_grid(ROWS, COLS)
dag = circuit_to_dag(build_dense_pair_blocks_circuit(N_QUBITS, GATES_PER_PAIR))

rows = []
for seed in range(SEEDS):
    pass_ = VF2Layout(coupling_map=cmap, seed=seed, call_limit=CALL_LIMIT)
    t0 = time.perf_counter()
    pass_.run(dag)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    reason = str(pass_.property_set.get("VF2Layout_stop_reason"))
    print(f"Seed {seed:>3}: {reason:<45} {elapsed_ms:9.2f} ms")
    rows.append({"seed": seed, "stop_reason": reason, "elapsed_ms": round(elapsed_ms, 3),
                 "rows": ROWS, "cols": COLS, "circuit_qubits": N_QUBITS,
                 "call_limit": CALL_LIMIT, **env})

found = sum(r["stop_reason"].endswith("SOLUTION_FOUND")
            and not r["stop_reason"].endswith("NO_SOLUTION_FOUND") for r in rows)
print(f"\nSOLUTION_FOUND: {found}/{SEEDS}")

with open("vf2_seed_scan_2026-09-11.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
