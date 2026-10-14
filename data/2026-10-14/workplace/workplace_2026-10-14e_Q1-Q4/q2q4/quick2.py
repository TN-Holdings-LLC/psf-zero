import os, sys, time, contextlib, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_common import *  # noqa
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeFez
NEW = compiler(os.environ.get("PSF_NEW92", "psf_compile_c53_91_92.py"), "c53_9192")
C53 = compiler(REPO + "/patches/psf_compile_c53_2026-10-14/psf_compile.py", "c53")
b = FakeFez(); T = b.target; F = fractional(T)
def xxz(n, steps, measure=True):
    import random
    rng = random.Random(3)
    terms = [(lab(n, {q: p, (q + 1) % n: p}), (1.0 if p != "Z" else 0.6) * rng.choice([-1, 1])) for q in range(n) for p in "XYZ"]
    terms += [(lab(n, {q: "Z"}), rng.uniform(-1, 1)) for q in range(n)]
    return evo(n, terms, 0.7, steps, measure=measure)
cases = []
for n in (4, 6, 8, 11):
    cases.append((f"ring ising n={n} s=4", ising(n, ring(n), 4, signs=True), n))
cases.append(("ring xxz n=8 s=2", xxz(8, 2), 8))
cases.append(("ring ising n=6 s=4 unmeasured", ising(6, ring(6), 4, signs=True).remove_final_measurements(inplace=False), 6))
for name, qc, n in cases:
    for tn, tg in (("T", T), ("F", F)):
        row = []
        for lb, M in (("new", NEW), ("c53", C53)):
            with contextlib.redirect_stdout(io.StringIO()):
                o = M.compile_for_hardware(qc, target=tg, seed_transpiler=0)
            d, two, nr = dur_2q(o, tg)
            if any(i.operation.name == "measure" for i in qc.data):
                e = err_dist(o, qc, n)
            else:
                e = 0.0 if NEW._implements(W2.unrolled(qc), o) else 1.0
            row.append(f"{lb} {d:6.2f}us 2q {two:4d} err {e:.0e}")
        print(f"{name:30s} {tn} | " + " | ".join(row), flush=True)
    print("   ", {k: v for k, v in NEW.RING_STATS.items() if v}, NEW.CHAIN_STATS["kept"], NEW.CHAIN_STATS["kept_default"])
