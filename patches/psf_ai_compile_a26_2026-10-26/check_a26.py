"""Checks a26 against a25 (item 50), from the PSF-Zero root with rust/psfbind built (ring r3):

    TOQB=<TOQB root> python patches/psf_ai_compile_a26_2026-10-26/check_a26.py

The 272 ring cases of check_a24 (FakeFez, FakeTorino, FakeMarrakesh, FakeKingston; n = 4-20; L = 1, 2, 4, 8; TOQB's
model circuit; `compile_for_model_circuit` and `model_template(...).bind`), plus the same on FakeSherbrooke (Eagle,
ECR: no fold), compared as check_a24 does; PSF_RUST_TEMPLATE=check.

A. PSF_RING_FOLD=0: a26 identical to a25 on every call.
B. Default: where item 50 folds (a FoldTemplate), the output equals the logical model on every clbit (statevector,
   P(1) within 1e-9), the default call equals the template's binding, and the pendant schedule has n + (n - m)
   two-qubit gates per layer (the holes' count is reported against n + 2 (m - n)); every other call is identical to
   a25's. The schedule chosen is counted per ring size.
"""
import contextlib
import importlib.util
import io
import os
import sys
import warnings
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import fake_provider

warnings.simplefilter("ignore")
os.environ["PSF_RUST_TEMPLATE"] = "check"
HERE = os.path.dirname(os.path.abspath(__file__))
C24 = importlib.util.spec_from_file_location("check_a24", os.path.join(HERE, "..", "psf_ai_compile_a24_2026-10-22",
                                                                       "check_a24.py"))
CK = importlib.util.module_from_spec(C24)
C24.loader.exec_module(CK)
A25 = "patches/psf_ai_compile_a25_2026-10-25/psf_ai_compile.py"
A26 = "patches/psf_ai_compile_a26_2026-10-26/psf_ai_compile.py"
DEVICES = CK.DEVICES + ("FakeSherbrooke",)


def p1_all(c):
    act = sorted({c.find_bit(q).index for i in c.data for q in i.qubits if i.operation.name not in ("barrier", "delay")})
    idx = {q: k for k, q in enumerate(act)}
    s, meas = QuantumCircuit(len(act)), {}
    for i in c.data:
        nm = i.operation.name
        if nm in ("barrier", "delay"):
            continue
        if nm == "measure":
            meas[c.find_bit(i.clbits[0]).index] = idx[c.find_bit(i.qubits[0]).index]
            continue
        s.append(i.operation, [idx[c.find_bit(q).index] for q in i.qubits])
    sv = Statevector(s)
    return [float(sv.probabilities([meas[k]])[1]) for k in sorted(meas)]


def main():
    sys.path.insert(0, os.environ["TOQB"])
    from toqb.task_loop import angle_list, model_circuit, n_params
    with contextlib.redirect_stdout(io.StringIO()):
        rel = CK.load("psf_compile.py", "psf_compile")
        rel.WARN_WITHOUT_TARGET = False
        a25, a26 = CK.load(A25, "ai25"), CK.load(A26, "ai26")
    print("a26", a26.AI_COMPILE_VERSION.split()[0], "| a25", a25.AI_COMPILE_VERSION.split()[0], "| psf_compile",
          rel.VERSION, "| psfbind", a26._psfbind().version)
    rng = np.random.default_rng(7)
    cases = []
    for dev in DEVICES:
        t = getattr(fake_provider, dev)().target
        for n in range(4, 21):
            for L in (1, 2, 4, 8):
                qc = model_circuit(angle_list(rng.uniform(-3, 3, n_params(n, L)), rng.uniform(-1, 1, n), n, L,
                                              np.pi / 2), n, L)
                cases.append((dev, t, n, L, qc))
    ok = True
    os.environ["PSF_RING_FOLD"] = "0"
    cnt, diffs = Counter(), []
    for dev, t, n, L, qc in cases:
        for kind in ("recompile", "template"):
            if CK.same(CK.run(a25, kind, qc, t), CK.run(a26, kind, qc, t)):
                cnt["identical"] += 1
            else:
                cnt["different"] += 1
                diffs.append((dev, n, L, kind))
    print("A (PSF_RING_FOLD=0)", dict(cnt), "| differences:", diffs[:10])
    ok = ok and not diffs
    os.environ.pop("PSF_RING_FOLD")
    cnt, diffs, sched, bad, holes_extra = Counter(), [], Counter(), [], Counter()
    for dev, t, n, L, qc in cases:
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                tp = a26.model_template(qc, t)
            except Exception as e:  # noqa: BLE001
                if "item 46" in str(e):
                    raise
                tp = None
        if isinstance(tp, a26.FoldTemplate):
            sched[(dev, n, tp.schedule, tp.cycle)] += 1
            c = tp.bind(qc)
            d = CK.run(a26, "recompile", qc, t)
            if not CK.same(c, d):
                bad.append((dev, n, L, "default call differs from the template"))
            err = max(abs(a - b) for a, b in zip(p1_all(c), p1_all(qc)))
            if err > 1e-9:
                bad.append((dev, n, L, f"P(1) differs by {err:.1e}"))
            q2 = sum(1 for i in c.data if len(i.qubits) == 2)
            if tp.schedule == "pend" and q2 != L * (2 * n - tp.cycle):
                bad.append((dev, n, L, f"pend count {q2}"))
            if tp.schedule == "holes":
                holes_extra[q2 - L * (n + 2 * (tp.cycle - n))] += 1
            cnt["folded, checked"] += 1
        else:
            for kind in ("recompile", "template"):
                if CK.same(CK.run(a25, kind, qc, t), CK.run(a26, kind, qc, t)):
                    cnt["not folded, identical to a25"] += 1
                else:
                    cnt["not folded, different"] += 1
                    diffs.append((dev, n, L, kind))
    print("B (default)", dict(cnt), "| problems:", bad[:10], "| differences:", diffs[:10])
    print("holes: two-qubit gates minus L (n + 2 (m - n)):", dict(holes_extra))
    per = Counter()
    for (dev, n, s, m), k in sched.items():
        per[(dev, s, m)] += 1
    print("folded per device, schedule, cycle (ring sizes):", sorted(per.items()))
    print("ring sizes folded:", sorted({(n, s) for (dev, n, s, m) in sched}))
    print("A26_STATS", a26.A26_STATS)
    ok = ok and not diffs and not bad
    print("RESULT", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
