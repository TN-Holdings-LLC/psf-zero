"""Exploratory: which post-routing passes close the gap to L2/L3 on full coupling?"""
import sys, io, contextlib, warnings, random, time
sys.path.insert(0,'.'); warnings.simplefilter("ignore")
src=open("eqtest.py").read()
exec(src.split("rng = random.Random")[0])
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import CommutativeCancellation, Optimize1qGatesDecomposition, ConsolidateBlocks, Collect2qBlocks, UnitarySynthesis, InverseCancellation
c2 = mods[2]
def psf_round(c, comm):
    passes = []
    if comm: passes += [CommutativeCancellation(basis_gates=nat)]
    c = PassManager(passes).run(c) if passes else c
    r = c2._post_routing_resynthesis(c, nat, True, 1e-5, "keep")
    return r
rng = random.Random(7); T={}
for k in range(60):
    n = rng.choice([3, 4, 5]); qc = rand_dense(n, rng.randint(6, 20), rng)
    full = CouplingMap.from_full(n)
    with contextlib.redirect_stdout(io.StringIO()):
        a = c2.compile_for_hardware(qc, coupling_map=full, basis_gates=nat, entangling_basis="cx", seed_transpiler=0)
    T.setdefault("c2",0); T["c2"]+=tq(a)
    for comm in (False, True):
        c = a
        for it in range(4):
            c2c = psf_round(c, comm); c2c._layout = a._layout
            if tq(c2c) >= tq(c) and it>0: break
            c = c2c
        T.setdefault(f"loop comm={comm}",0); T[f"loop comm={comm}"]+=tq(c)
    T.setdefault("L2",0); T["L2"]+=tq(transpile(qc, coupling_map=full, basis_gates=nat, optimization_level=2, seed_transpiler=0))
print(T)
