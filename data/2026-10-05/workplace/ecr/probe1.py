import sys, warnings, contextlib, io, time
warnings.simplefilter("ignore"); sys.path.insert(0, ".")
import psf_compile as pc
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import fake_provider as fp
for d in ("FakeBrussels", "FakeStrasbourg", "FakeOsaka", "FakeSherbrooke"):
    try:
        be = getattr(fp, d)()
    except Exception as e:
        print(d, "n/a", e); continue
    t = be.target
    ops = sorted(t.operation_names)
    two = [g for g in ("ecr", "cz", "cx") if g in ops]
    errs = [p.error for q, p in t[two[0]].items() if p is not None and p.error is not None]
    print(d, t.num_qubits, ops, "2q:", two, "failed(>=0.5):", sum(e >= 0.5 for e in errs), "of", len(errs))
    qc = QuantumCircuit(4); qc.h(0); [qc.cx(i, i + 1) for i in range(3)]; qc.cx(3, 0); qc.measure_all()
    basis = [g for g in ops if g in ("cx", "cz", "ecr", "rz", "sx", "x")]
    for kw in (dict(), dict(final_resynthesis="select", compare_level3=True, compare_floor=True, candidate_score="hybrid")):
        try:
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                out = pc.compile_for_hardware(qc, coupling_map=t.build_coupling_map(), basis_gates=basis, entangling_basis="cx",
                                              layout_search=True, seed_transpiler=0, target=t, placement_refine=True, **kw)
            off = [(i.operation.name, tuple(out.find_bit(b).index for b in i.qubits)) for i in out.data
                   if i.operation.name not in ("barrier", "measure") and (i.operation.name not in t.operation_names or
                   tuple(out.find_bit(b).index for b in i.qubits) not in t[i.operation.name])]
            print("   ", "full" if kw else "base", dict(out.count_ops()), "off-target", len(off), off[:3], "%.2fs" % (time.perf_counter() - t0))
        except Exception as e:
            print("   ", "full" if kw else "base", "ERROR", type(e).__name__, str(e)[:300])
