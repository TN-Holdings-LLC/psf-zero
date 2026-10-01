# Compare release vs candidate compile() on random circuits, counting CX AFTER basis translation
# (the input may contain cu3/ch/... which count 1 as written but several CX on hardware).
import sys, io, contextlib, time, statistics, importlib.util, warnings
from qiskit.circuit.random import random_circuit
from qiskit import transpile
from qiskit.quantum_info import Operator
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m
rel = load(sys.argv[1], "rel"); cand = load(sys.argv[2], "cand")
cxc = lambda c: transpile(c, basis_gates=["cx","rz","sx","x"], optimization_level=0).count_ops().get("cx", 0)
for nq,depth,seed in [(10,20,1),(20,20,2),(40,20,3),(60,30,4),(5,40,5),(8,40,6)]:
    qc = random_circuit(nq, depth, max_operands=2, seed=seed)
    row=[nq,depth,"in_cx",cxc(qc)]
    for m in (rel,cand):
        ts=[]
        for _ in range(3):
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m._CX_CORE_CACHE.clear(); t=time.perf_counter(); c=m.compile(qc, entangling_basis="cx"); ts.append(time.perf_counter()-t)
        row += [m.VERSION, "%.1fms"%(statistics.median(ts)*1000), "cx", cxc(c)]
        if nq<=10: row += ["eq", Operator(c).equiv(Operator(qc))]
    print(*row)
