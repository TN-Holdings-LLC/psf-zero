import time
import csv
import platform
import sys
import time

import qiskit
from qiskit import transpile
from qiskit.transpiler import CouplingMap, Target
from qiskit.transpiler.passes import VF2PostLayout
from qiskit.converters import circuit_to_dag
from phase3_v5_spare_qubits import build_dense_pair_blocks_circuit

BASIS = ["rz", "sx", "x", "cx"]
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
target = Target.from_configuration(
    basis_gates=BASIS, coupling_map=cmap, num_qubits=cmap.size()
)

qc = build_dense_pair_blocks_circuit(N_QUBITS, GATES_PER_PAIR)
routed = transpile(
    qc, coupling_map=cmap, basis_gates=BASIS,
    optimization_level=1, seed_transpiler=0,
)
dag = circuit_to_dag(routed)

rows = []
for seed in range(SEEDS):
    p = VF2PostLayout(
        target=target, seed=seed, call_limit=CALL_LIMIT, strict_direction=False
    )
    t0 = time.perf_counter()
    p.run(dag)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    reason = str(p.property_set.get("VF2PostLayout_stop_reason"))
    print(f"Seed {seed:>3}: {reason:<52} {elapsed_ms:9.2f} ms")
    rows.append({
        "seed": seed, "stop_reason": reason, "elapsed_ms": round(elapsed_ms, 3),
        "rows": ROWS, "cols": COLS, "circuit_qubits": N_QUBITS,
        "call_limit": CALL_LIMIT, "pass": "VF2PostLayout", **env,
    })

from collections import Counter
print("\n" + "\n".join(f"{k}: {v}" for k, v in Counter(r["stop_reason"] for r in rows).items()))

with open("vf2post_seed_scan_2026-09-11.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
