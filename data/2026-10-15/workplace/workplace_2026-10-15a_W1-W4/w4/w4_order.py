"""W4 side probe: c57's in-scope output when commuting two-qubit gates are written in another order (same unitary:
within each Trotter step the ZZ terms commute; the edge loop is run in reverse). Reported only."""
import os, sys
sys.argv = [sys.argv[0], "probe_out"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "w4_orient.py")).read().split("def main():")[0])
from qiskit.quantum_info import Operator
def edge_reversed(qc):
    """Reverse each maximal run of consecutive commuting ZZ-type two-qubit gates (rzz, cz, ZZ PauliEvolution)."""
    out = qc.copy_empty_like(); run = []
    def flush():
        for ins in reversed(run): out.append(ins)
        run.clear()
    for ins in qc.data:
        if len(ins.qubits) == 2 and ins.operation.name in ("rzz", "cz", "PauliEvolution"):
            run.append(ins)
        else:
            flush(); out.append(ins)
    flush()
    return out
T, F = V.device("FakeFez")
for test in ("chain_rzz_n12", "chain_rzz_n16", "ring_rzz_n12", "hh_rzz_n16", "chain_cz_n12", "chain_cz_n16", "chain_pe_n12"):
    b = build(test); r = edge_reversed(b)
    for tk, tg in (("T", T), ("F", F)):
        a = compile_with(V.M57, b, tg); c = compile_with(V.M57, r, tg)
        print(f"{test:14s} {tk}: as written {V.metrics(a, tg)['dur_us']:.3f} us, runs reversed {V.metrics(c, tg)['dur_us']:.3f} us, identical {V.sig(a) == V.sig(c)}", flush=True)
