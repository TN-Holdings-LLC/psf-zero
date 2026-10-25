"""Checks a25 against a24 (item 49), from the PSF-Zero root with rust/psfbind built (ring r3):

    TOQB=<TOQB root> python patches/psf_ai_compile_a25_2026-10-25/check_a25.py

The 272 ring cases of check_a24 (FakeFez, FakeTorino, FakeMarrakesh, FakeKingston; n = 4-20; L = 1, 2, 4, 8; TOQB's
model circuit; `compile_for_model_circuit` and `model_template(...).bind`), compared as check_a24 does (every
operation's name, qubits, clbits and angles as a float's hex, the global phase, the initial and final layout; a
refusal by its exception type). PSF_RUST_TEMPLATE=check: every Rust template is also built by the Python and any
difference stops the check (item 46).

A. Without PSF_RING_ZIP: a25 identical to a24 on every call.
B. PSF_RING_ZIP=1, even n: a25 identical to a24 on every call.
C. PSF_RING_ZIP=1, odd n: a25 builds where a24 builds (the same refusals otherwise); the schedule chosen is counted
   per ring size; where the zipper is chosen, the template has (5n - 7)/2 CZ per layer.
"""
import contextlib
import importlib.util
import io
import os
import sys
import warnings
from collections import Counter

import numpy as np
from qiskit_ibm_runtime import fake_provider

warnings.simplefilter("ignore")
os.environ["PSF_RUST_TEMPLATE"] = "check"
HERE = os.path.dirname(os.path.abspath(__file__))
C24 = importlib.util.spec_from_file_location("check_a24", os.path.join(HERE, "..", "psf_ai_compile_a24_2026-10-22",
                                                                       "check_a24.py"))
CK = importlib.util.module_from_spec(C24)
C24.loader.exec_module(CK)
A24 = "patches/psf_ai_compile_a24_2026-10-22/psf_ai_compile.py"
A25 = "patches/psf_ai_compile_a25_2026-10-25/psf_ai_compile.py"


def main():
    sys.path.insert(0, os.environ["TOQB"])
    from toqb.task_loop import angle_list, model_circuit, n_params
    with contextlib.redirect_stdout(io.StringIO()):
        rel = CK.load("psf_compile.py", "psf_compile")
        rel.WARN_WITHOUT_TARGET = False
        a24, a25 = CK.load(A24, "ai24"), CK.load(A25, "ai25")
    print("a25", a25.AI_COMPILE_VERSION.split()[0], "| a24", a24.AI_COMPILE_VERSION.split()[0], "| psf_compile",
          rel.VERSION, "| psfbind", a25._psfbind().version)
    rng = np.random.default_rng(7)
    cases = []
    for dev in CK.DEVICES:
        t = getattr(fake_provider, dev)().target
        for n in range(4, 21):
            for L in (1, 2, 4, 8):
                qc = model_circuit(angle_list(rng.uniform(-3, 3, n_params(n, L)), rng.uniform(-1, 1, n), n, L,
                                              np.pi / 2), n, L)
                cases.append((dev, t, n, L, qc))
    ok = True
    for part, zip_on in (("A (no switch)", False), ("B/C (PSF_RING_ZIP=1)", True)):
        if zip_on:
            os.environ["PSF_RING_ZIP"] = "1"
        else:
            os.environ.pop("PSF_RING_ZIP", None)
        cnt, diffs, sched, czbad = Counter(), [], Counter(), []
        for dev, t, n, L, qc in cases:
            for kind in ("recompile", "template"):
                o24, o25 = CK.run(a24, kind, qc, t), CK.run(a25, kind, qc, t)
                if not zip_on or n % 2 == 0:
                    if CK.same(o24, o25):
                        cnt["identical"] += 1
                    else:
                        cnt["different"] += 1
                        diffs.append((dev, n, L, kind))
                else:
                    if isinstance(o24, tuple) or o24 is None:
                        good = CK.same(o24, o25)
                        cnt["odd: same refusal" if good else "odd: refusal differs"] += 1
                        if not good:
                            diffs.append((dev, n, L, kind, "refusal"))
                    elif isinstance(o25, tuple) or o25 is None:
                        cnt["odd: a25 refused where a24 built"] += 1
                        diffs.append((dev, n, L, kind, o25))
                    else:
                        cnt["odd: built"] += 1
            if zip_on and n % 2 == 1:
                with contextlib.redirect_stdout(io.StringIO()):
                    try:
                        tp = a25.model_template(qc, t)
                    except Exception as e:  # noqa: BLE001
                        tp = None
                        if "item 46" in str(e):
                            raise
                if tp is not None:
                    sched[(n, tp.schedule)] += 1
                    if tp.schedule == "zip":
                        ncz = sum(1 for i in tp.circuit.data if i.operation.name == "cz")
                        if ncz != L * (5 * n - 7) // 2:
                            czbad.append((dev, n, L, ncz))
        print(part, dict(cnt), "| differences:", diffs[:10])
        if zip_on:
            print("odd n, schedule chosen per ring size:", sorted(sched.items()), "| zipper CZ count off:", czbad)
        ok = ok and not diffs and not czbad
    print("A25_STATS", a25.A24_STATS)
    print("RESULT", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
