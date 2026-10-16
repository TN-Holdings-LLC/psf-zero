"""Checks c65 against its base (item 102): the same output, bit for bit, and the time, alternating in one process.

    TOQB=<TOQB root> python patches/psf_compile_c65_2026-10-16/check_c65.py <base psf_compile.py> <c65 psf_compile.py>

Run from the PSF-Zero root. Calls: C63-VAL's 19 tests and c55_val's 28 (47 distinct), on FakeFez, FakeMarrakesh,
FakeKingston, FakeTorino and the Marrakesh snapshot, T and F targets, `compile_for_hardware(qc, target=)` and
`compile_template(qc, target).bind(qc)`; TOQB's ring model circuits (n = 4-20, L = 1, 2, 4, 8, four devices) through
a23's `compile_for_model_circuit` and `model_template(...).bind`. Compared per call: every operation's name, qubits,
clbits and angles (a float's hex), the global phase, the initial and the final layout; a refusal by its exception
type. Time: base and c65 alternately, 5 rounds, the median per call (after one warm-up of each).
"""
import contextlib
import hashlib
import importlib.util
import io
import os
import runpy
import shutil
import statistics
import sys
import tempfile
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
BASE, C65 = sys.argv[1], sys.argv[2]
_tmp = tempfile.mkdtemp() + "/imp"
sys.argv = ["c55_val", _tmp]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join("benchmarks", "c55_val.py"), run_name="check_c65_import")
shutil.rmtree(_tmp, ignore_errors=True)
sys.path[:0] = [os.path.join("data", "2026-10-15", "workplace", "workplace_2026-10-15b_W1-W2", "w30"), "benchmarks"]
import c60_tests as TT  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


with contextlib.redirect_stdout(io.StringIO()):
    M0 = load(BASE, "psf_compile")
    M1 = load(C65, "psf_compile_c65")
    M0.WARN_WITHOUT_TARGET = M1.WARN_WITHOUT_TARGET = False
    AI = load(os.path.join("patches", "psf_ai_compile_a23_2026-10-16", "psf_ai_compile.py"), "ai23")


def fingerprint(c):
    if c is None or isinstance(c, tuple):
        return repr(c)
    q = {b: i for i, b in enumerate(c.qubits)}
    k = {b: i for i, b in enumerate(c.clbits)}
    ops = tuple((ci.operation.name, tuple(q[x] for x in ci.qubits), tuple(k[x] for x in ci.clbits),
                 tuple(str(p) if hasattr(p, "parameters") else float(p).hex() for p in ci.operation.params))
                for ci in c.data)
    lay = getattr(c, "_layout", None)
    il = None if lay is None else tuple(lay.initial_index_layout(filter_ancillas=False))
    fl = None if lay is None or lay.final_layout is None else tuple(lay.final_index_layout())
    gp = c.global_phase
    return hashlib.sha256(repr((ops, str(gp) if hasattr(gp, "parameters") else float(gp).hex(), il, fl,
                                c.num_qubits, c.num_clbits)).encode()).hexdigest()


def once(fn, m):
    AI.pc = m
    s = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            out = fn(m)
    except Exception as e:  # noqa: BLE001 - a refusal compares by its type
        out = ("error", type(e).__name__)
    return out, time.perf_counter() - s


def main():
    jobs = []
    names = list(dict.fromkeys(g["TESTS"] + ("hhtree_heis_n16", "ghz_n12") + tuple(
        ("chain_cz_n12", "chain_cz_n16", "ring_cz_n12", "hh_cz_n16", "hh_cz_n20", "cx_chain_n12", "cx_chain_n16"))))
    for d in g["DEVICES"]:
        t, ft = g["device"](d)
        for name in names:
            try:
                qc = g["build"](name)
            except Exception:  # noqa: BLE001 - the workplace's W1 tests live in c60_tests
                qc = TT.build(name)
            for lab, tg in (("T", t), ("F", ft)):
                jobs.append((f"{d}|{name}|{lab}|cfh", lambda m, qc=qc, tg=tg: m.compile_for_hardware(qc, target=tg)))
                jobs.append((f"{d}|{name}|{lab}|tpl", lambda m, qc=qc, tg=tg: m.compile_template(qc, tg).bind(qc)))
    sys.path.insert(0, os.environ["TOQB"])
    from qiskit_ibm_runtime import fake_provider
    from toqb.task_loop import angle_list, model_circuit, n_params
    rng = np.random.default_rng(7)
    for d in ("FakeFez", "FakeTorino", "FakeMarrakesh", "FakeKingston"):
        t = getattr(fake_provider, d)().target
        cm = t.build_coupling_map()
        basis = [x for x in ("cz", "rz", "sx", "x") if x in t.operation_names]
        for n in range(4, 21):
            for L in (1, 2, 4, 8):
                qc = model_circuit(angle_list(rng.uniform(-3, 3, n_params(n, L)), rng.uniform(-1, 1, n), n, L,
                                              np.pi / 2), n, L)
                jobs.append((f"{d}|ring{n}|L{L}|rec",
                             lambda m, qc=qc, t=t, cm=cm, basis=basis: AI.compile_for_model_circuit(qc, cm, basis,
                                                                                                    target=t)))
                jobs.append((f"{d}|ring{n}|L{L}|tpl",
                             lambda m, qc=qc, t=t: (lambda tp: None if tp is None else tp.bind(qc))(
                                 AI.model_template(qc, t))))
    diff, rows = [], []
    for key, fn in jobs:
        o0, _ = once(fn, M0)
        o1, _ = once(fn, M1)
        if fingerprint(o0) != fingerprint(o1):
            diff.append(key)
        a, b = [], []
        for r in range(5):
            for m, acc in ((M0, a), (M1, b)) if r % 2 == 0 else ((M1, b), (M0, a)):
                acc.append(once(fn, m)[1])
        rows.append((key, statistics.median(a), statistics.median(b)))
    print(f"calls {len(jobs)}: identical {len(jobs) - len(diff)}, different {len(diff)} {diff[:10]}")
    for part, sel in (("C63-VAL and c55_val tests", lambda k: "|L" not in k), ("TOQB rings", lambda k: "|L" in k)):
        rs = [r for r in rows if sel(r[0])]
        A, B = sum(r[1] for r in rs), sum(r[2] for r in rs)
        rat = sorted(r[2] / r[1] for r in rs)
        print(f"{part}: {len(rs)} calls, total {A:.2f} s -> {B:.2f} s ({100 * (B / A - 1):+.1f}%); per call median "
              f"{statistics.median(rat):.3f}, 10% {rat[len(rat) // 10]:.3f}, 90% {rat[9 * len(rat) // 10]:.3f}, "
              f"min {rat[0]:.3f}, max {rat[-1]:.3f}")
    print("item 102 decisions (second file):", getattr(M1, "S96_FORMULA_STATS", None))
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())
