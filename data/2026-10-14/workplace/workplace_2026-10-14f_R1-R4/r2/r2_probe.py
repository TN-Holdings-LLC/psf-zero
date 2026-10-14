"""R2: where do the ~1e-7 differences of ordinary outputs come from? c48's _tfim(12, 4, 8350) ring on FakeTorino (6.1e-7
per home) and C52-VAL's tfim_n20_r3 on FakeFez (1.5e-8). Every transpile call inside c54 is wrapped; for the call
whose output moves the distribution, the level-3 passes are bisected with a callback, and the changed 2q blocks are
decomposed to see whether the Weyl snap (infidelity <= 1e-9) is the cause."""
import sys, os, contextlib, io, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import *  # noqa
P = load(REPO + "/patches/psf_compile_c54_2026-10-14/psf_compile.py", "psf_compile"); P.WARN_WITHOUT_TARGET = False
sys.path.insert(0, REPO + "/patches/psf_compile_c54_2026-10-14")
from test_c48_on_c49_for_c54 import _tfim
from qiskit_ibm_runtime import fake_provider
sys.path.insert(0, os.path.join(REPO, "data", "2026-10-12", "c48_dev"))
import gens
P.EXACT_CHECK = False
cases = [("ring12 FakeTorino", _tfim(12, 4, 8350), "FakeTorino"), ("tfim_n20_r3 FakeFez", gens.build("tfim_n20_r3"), "FakeFez")]
def maxdiff(a, b):
    pa, pb = W2.probs(a), W2.probs(b)
    return float(abs(pa - pb).max())
orig = P.transpile
for name, qc, dev in cases:
    b = getattr(fake_provider, dev)()
    calls = []
    def wrapped(circ, *a, **k):
        out = orig(circ, *a, **k)
        try:
            if not isinstance(circ, list) and any(i.operation.name == "measure" for i in circ.data):
                d = maxdiff(out, circ)
                calls.append((d, k.get("optimization_level"), k.get("routing_method")))
                if d > 1e-10:
                    pickle.dump((circ, k), open(f"bad_in_{dev}.pkl", "wb"))
        except Exception as ex:
            calls.append(("check failed", repr(ex)[:60]))
        return out
    P.transpile = wrapped
    with contextlib.redirect_stdout(io.StringIO()):
        o = P.compile_for_hardware(qc, backend=b, entangling_basis="cx", layout_search=True, seed_transpiler=0)
    P.transpile = orig
    print(name, "final max |dp|", maxdiff(o, W2.unrolled(qc)), "transpile calls (max|dp|, level, routing):", calls, flush=True)
