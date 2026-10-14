"""R4: on FakeTorino, why does item 91 not change c54's output for hhI_n16 and ring12_tfim (C54-VAL)?"""
import sys, os, contextlib, io, tempfile, importlib.util, collections
sys.argv = ["c54_val.py", tempfile.mkdtemp() + "/c54"]
spec = importlib.util.spec_from_file_location("cv", "benchmarks/c54_val.py"); CV = importlib.util.module_from_spec(spec)
sys.modules["cv"] = CV; spec.loader.exec_module(CV)
P = CV.M54
from qiskit_ibm_runtime import fake_provider
for dev in ("FakeTorino", "FakeFez"):
    T, F = CV.device(dev)
    cz = collections.Counter(round(p.duration * 1e9) for q, p in T["cz"].items() if p is not None)
    sx = collections.Counter(round(p.duration * 1e9) for q, p in T["sx"].items() if p is not None)
    print(dev, "cz ns:", dict(cz), "sx ns:", dict(sx))
    for name in ("hhI_n16", "ring12_tfim", "hhI_n20"):
        qc = CV.build(name)
        for tk, tg in (("T", T), ("F", F)):
            g0 = dict(P.GRAPH_STATS); k0 = (P.CHAIN_STATS["kept"], P.CHAIN_STATS["kept_default"])
            raw = P._graph_brickwork(qc, tg, 0.5, 0)
            with contextlib.redirect_stdout(io.StringIO()):
                own = P._compile_for_hardware_c51(qc, target=tg, entangling_basis="cx", layout_search=True, seed_transpiler=0)
                out = P.compile_for_hardware(qc, target=tg, entangling_basis="cx", layout_search=True, seed_transpiler=0)
            m = lambda c: CV.metrics(c, tg)
            print(f"  {name:12s} {tk} | item 91 raw {m(raw)['dur_us']:6.3f} ({m(raw)['twoq']}) | c53's own {m(own)['dur_us']:6.3f} ({m(own)['twoq']}) | returned {m(out)['dur_us']:6.3f} | kept/kept_default +{P.CHAIN_STATS['kept'] - k0[0]}/+{P.CHAIN_STATS['kept_default'] - k0[1]}", flush=True)
print("--- couplers used on FakeTorino (F)")
T, F = CV.device("FakeTorino")
for name in ("hhI_n16", "ring12_tfim"):
    qc = CV.build(name)
    g0 = dict(P.GRAPH_STATS)
    raw = P._graph_brickwork(qc, F, 0.5, 0)
    with contextlib.redirect_stdout(io.StringIO()):
        own = P._compile_for_hardware_c51(qc, target=F, entangling_basis="cx", layout_search=True, seed_transpiler=0)
    for lbl, c in (("item 91", raw), ("c53", own)):
        pos = {q: i for i, q in enumerate(c.qubits)}
        used = collections.Counter()
        for ins in c.data:
            if len(ins.qubits) == 2:
                qs = tuple(pos[q] for q in ins.qubits)
                p = F[ins.operation.name].get(qs) or F[ins.operation.name].get(qs[::-1])
                used[round(p.duration * 1e9)] += 1
        n1 = sum(1 for i in c.data if len(i.qubits) == 1 and i.operation.name in ("sx", "rx", "x"))
        print(f"  {name:12s} {lbl:8s} 2q gates by length (ns): {dict(used)}  1q pulses {n1}")
    print("   GRAPH_STATS common_length +", P.GRAPH_STATS["common_length"] - g0["common_length"])
