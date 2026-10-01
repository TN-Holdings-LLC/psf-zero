"""Exploratory: release vs candidate compile_for_hardware on larger circuits (Kingston, cz basis, layout_search=True).
Same layout module for both (put first on PYTHONPATH). Times: this sandbox, median of 3."""
import sys, io, contextlib, warnings, importlib.util, time, statistics
sys.path.insert(0, '.'); sys.path.insert(0, '../gh_latest/benchmarks')
warnings.simplefilter("ignore")
from qiskit.circuit.random import random_circuit
from qiskit_ibm_runtime import fake_provider
from physeq import phys_equiv
import circuit_family_sweep as F
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
rel, cand = load(sys.argv[1], "rel"), load(sys.argv[2], "cand")
tgt = fake_provider.FakeKingston().target; cm = tgt.build_coupling_map(); nat = ["cz", "rz", "sx", "x"]
tq = lambda c: sum(1 for i in c.data if len(i.qubits) == 2)
cases = []
for nq, d, s in [(10, 20, 1), (12, 30, 2), (20, 20, 3), (40, 20, 4), (60, 20, 5), (100, 10, 6)]:
    cases.append((f"random_circuit {nq}q d{d}", random_circuit(nq, d, max_operands=2, seed=s)))
for nq in (50, 100, 156):
    cases.append((f"dense_pair_blocks {nq}q", F.build_dense_pair_blocks_circuit(nq, seed=0)))
for fam in ("k_chains", "random_regular", "ghz_star", "linear_chain"):
    if fam in F.CIRCUIT_FAMILIES:
        cases.append((f"{fam} 60q", F.build_circuit_from_family(60, fam, seed=0)[0]))
for name, qc in cases:
    row = [name]
    outs = []
    for m in (rel, cand):
        ts = []
        for _ in range(3):
            with contextlib.redirect_stdout(io.StringIO()):
                m._CX_CORE_CACHE.clear(); t = time.perf_counter()
                o = m.compile_for_hardware(qc, coupling_map=cm, basis_gates=nat, entangling_basis="cx", layout_search=True, seed_transpiler=0)
                ts.append(time.perf_counter() - t)
        outs.append(o); row += [f"{m.VERSION}: 2q {tq(o)} {statistics.median(ts)*1000:.0f}ms"]
    if qc.num_qubits <= 12:
        row += ["equiv " + str([phys_equiv(qc, o)[0] for o in outs])]
    print(" | ".join(row), flush=True)
