"""W1 timing probe: a18's item 30 check on a handed-on input that is one PauliEvolutionGate / QAOA gate (12 qubits),
against the same check on the input unrolled first (as c57's item 93 does). Offline."""
import os, sys, time, contextlib, io
NAMES = sys.argv[1:]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "w1_handoff.py")).read().split("import time  # noqa: E402")[0])
import time
from qiskit import transpile
T, F = V.device("FakeFez")
for name in NAMES:
    qc = V.build(name)
    with contextlib.redirect_stdout(io.StringIO()):
        out = PC.compile_for_hardware(qc, target=T, basis_gates=["cz", "rz", "sx", "x"], entangling_basis="cx",
                                      layout_search=True, seed_transpiler=0)
    t0 = time.perf_counter(); u = H_un = V.H._unrolled(qc); v = AI._exact(u, out); t1 = time.perf_counter() - t0
    print(f"{name}: item 30 on the unrolled input: {v} in {t1:.1f} s", flush=True)
    t0 = time.perf_counter(); v = AI._exact(qc, out); print(f"{name}: item 30 as a18 calls it: {v} in {time.perf_counter() - t0:.1f} s", flush=True)
