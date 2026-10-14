import sys, time, json
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *  # noqa
from qiskit_ibm_runtime.fake_provider import FakeFez
NEW = compiler(os.environ.get("PSF_NEW", REPO + "/patches/psf_compile_c51_87_88_2026-10-14/psf_compile.py"), "c51_87_88")
C51 = compiler(REPO + "/patches/psf_compile_c51_2026-10-14/psf_compile.py", "c51")
b = FakeFez(); T = b.target; F = fractional(T)
cases = [("tfim n=12 s=2", W2.tfim_pauli(12, 2), W2.tfim_brickwork(12, 2), 12),
         ("tfim n=12 s=10", W2.tfim_pauli(12, 10), W2.tfim_brickwork(12, 10), 12),
         ("heis n=10 s=4", heis(10, 4), None, 10),
         ("tfim-signs n=10 s=4", tfim_signs(10, 4), None, 10)]
for name, qc, bw, n in cases:
    for lab_, M in (("new", NEW), ("c51", C51)):
        t0 = time.perf_counter()
        out = M.compile_for_hardware(qc, target=F, seed_transpiler=0)
        s = time.perf_counter() - t0
        d, two, nr = dur_2q(out, F)
        print(f"{name:22s} {lab_:4s} {d:6.2f}us 2q {two:4d} rzz {nr:4d} range {rzz_ok(out)} err {err_dist(out, qc, n):.1e} {s:.1f}s", flush=True)
    print("   new stats", {k: v for k, v in NEW.CHAIN_STATS.items() if v})
    if bw is not None:
        raw, fold = qk3f_bw(bw, F)
        d, two, nr = dur_2q(fold, F)
        print(f"{name:22s} QK3F-BW {d:6.2f}us 2q {two:4d} rzz {nr} raw-range {rzz_ok(raw)} err {err_dist(fold, qc, n):.1e}")
