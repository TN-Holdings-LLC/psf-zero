"""steiner6_chem_probe.py -- chemistry probe; BENCHPRESS=<clone> python benchmarks/steiner6_chem_probe.py
<test substrings...>: Qiskit's synthesis without wiring constraints (level 3, all-to-all) vs the naive ladder count, vs
STEINER6 before translation."""
import os, sys, json, warnings
warnings.simplefilter("ignore")
REPO = os.getcwd()  # the psf-zero repository root
sys.path[:0] = [os.path.join(REPO, "data", "2026-10-05", "workplace", "depth1"), os.path.join(REPO, "benchmarks"), REPO]
import steiner3 as S3, bp_mock as B
from qiskit import transpile
BP = os.environ["BENCHPRESS"]  # a Benchpress clone
for name in sys.argv[1:]:
    t = [t for t in S3.tests(BP) if name in t[1]][0]
    qc, _ = B.build(BP, t[2], t[3])
    evo = [i for i in qc.data if i.operation.name == "PauliEvolution"][0]
    op = evo.operation.operator
    naive = sum(2 * (sum(1 for ch in lb if ch != "I") - 1) for lb in op.paulis.to_labels() if sum(1 for ch in lb if ch != "I") > 1)
    free = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=3, seed_transpiler=0)
    l1 = transpile(qc, basis_gates=["cx", "rz", "sx", "x"], optimization_level=1, seed_transpiler=0)
    print(name, "terms", len(op), "naive ladder", naive, "| all-to-all L1", l1.count_ops().get("cx", 0), "L3", free.count_ops().get("cx", 0))
