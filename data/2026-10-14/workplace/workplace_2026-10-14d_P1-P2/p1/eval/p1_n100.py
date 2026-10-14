"""p1_n100.py -- correctness of items 87-88's output at n = 100 (W2's cells): all one- and two-point Z correlations
(<Z_i>, <Z_i Z_i+1>) against the logical circuit, Aer MPS. python p1_n100.py <out dir>"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *  # noqa
from qiskit_ibm_runtime import fake_provider
NEW = compiler(os.environ.get("PSF_NEW", os.path.join(REPO, "patches", "psf_compile_c51_87_88_2026-10-14", "psf_compile.py")),
               "c51_87_88")
fh = open(os.path.join(sys.argv[1], "p1_n100_rows.jsonl"), "w")
for dev in ("FakeFez", "FakeMarrakesh"):
    F = fractional(getattr(fake_provider, dev)().target)
    for s in (2, 4, 8, 10):
        qc = W2.tfim_pauli(100, s)
        t0 = time.perf_counter()
        out = NEW.compile_for_hardware(qc, target=F, seed_transpiler=0)
        e = err_dist(out, qc, 100)
        r = dict(device=dev, n=100, steps=s, err=e, in_range=rzz_ok(out), isa=bool(NEW._isa_ok(out, F)),
                 secs=round(time.perf_counter() - t0, 1))
        fh.write(json.dumps(r) + "\n"); fh.flush(); print(r, flush=True)
