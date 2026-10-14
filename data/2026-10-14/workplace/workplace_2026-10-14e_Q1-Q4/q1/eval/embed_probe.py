import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q1_common import *  # noqa
NEW = compiler(os.environ.get("PSF_NEW", REPO + "/patches/psf_compile_c53_91_2026-10-14/psf_compile.py"), "c53_91")
from qiskit_ibm_runtime.fake_provider import FakeFez, FakeMarrakesh
for dn, B in (("FakeFez", FakeFez), ("FakeMarrakesh", FakeMarrakesh)):
    T = B().target
    K, kadj, fadj, _ = NEW._chain_lattice(T, 0.5)
    ok = 0
    t0 = time.perf_counter()
    for s in range(20):
        e = tree(16, 100 + s)
        if NEW._graph_place(16, e, K, kadj) is not None: ok += 1
    print(dn, "random max-degree-3 trees n=16 embeddable:", ok, "/ 20", f"{time.perf_counter()-t0:.1f}s", "|K|", len(K))
    for n in (40, 80, len(K)):
        e = hh_patch(T, n, 3)
        t0 = time.perf_counter(); m = NEW._graph_place(n, e, K, fadj); m2 = m or NEW._graph_place(n, e, K, kadj)
        print(f"  hh patch n={n} edges {len(e)}: fadj {'yes' if m else 'no'} kadj {'yes' if m2 else 'no'} {time.perf_counter()-t0:.1f}s")
