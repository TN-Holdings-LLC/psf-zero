"""Home check (2026-10-15): the workplace's W4 tests (three orientations of each symmetric two-qubit gate) on prototype
c59, FakeFez, T and F. Reports, per cell, each orientation's duration (us) and compile time (ms), or REFUSED
(OutOfScopeError), and whether the three orientations gave one circuit. Offline; nothing is chosen.
Run from the repository root:
    python data/2026-10-15/workplace/home_checks/c59_orient_check.py data/2026-10-15/workplace/workplace_2026-10-15a_W1-W4/w4
"""
import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import time
import warnings

warnings.simplefilter("ignore")
W = os.path.abspath(sys.argv[1])
sys.argv = ["w4_orient.py", tempfile.mkdtemp()]
os.environ["C57_97"] = os.path.join(W, "c57_97", "psf_compile.py")
spec = importlib.util.spec_from_file_location("w4o", os.path.join(W, "w4_orient.py"))
W4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W4)
V = W4.V
P = V.load("patches/psf_compile_c59_2026-10-15/psf_compile.py", "psf_c59_orient")
P.WARN_WITHOUT_TARGET = False
t, f = V.device("FakeFez")
from c29_identity import value_sig  # noqa: E402

for name in W4.TESTS:
    qc0 = W4.build(name)
    for tn, tg in (("T", t), ("F", f)):
        res = []
        for how in W4.ORIENT:
            qc = W4.orient(qc0, how, name)
            s = time.perf_counter()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    out = P.compile_for_hardware(qc, target=tg, seed_transpiler=0)
                dt = (time.perf_counter() - s) * 1000
                res.append((how, round(P._duration_s(out, tg) * 1e6, 3), value_sig(out), round(dt)))
            except P.OutOfScopeError as e:
                res.append((how, "REFUSED", str(e)[:50], 0))
        sigs = {r[2] for r in res}
        print(name, tn, [(r[1], r[3]) for r in res], "one circuit" if len(sigs) == 1 else f"{len(sigs)} circuits")
