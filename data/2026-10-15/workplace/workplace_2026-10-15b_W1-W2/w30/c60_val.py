"""C60-VAL (workplace, 2026-10-15): prototype c60 (item 99: native two-qubit gate circuits on an embeddable graph,
by formula) against the release (2026-10-14.1 = c57), Qiskit level 3 and c59. Offline; no real device.

Part A: item 99's 21 tests (c60_tests.py) on C57-VAL's 5 devices, without (T) and with (F) fractional gates. Arms
(seed 0): C60 (called twice), C57 (the release's default call as C57-VAL calls it), QK3 (generate_preset_pass_manager
level 3 on the Target; on F folded by c57's `_fold`, as c57 hands it on).
Part B (regression): C57-VAL's 28 tests and W4's 10 gate-by-gate tests (as written) on the same 5 devices, T and F,
through c59 and c60: identical output where c59 builds; the same refusal where c59 refuses.
Part C (reported only): W4's tests in three orientations and with each run of commuting gates written in reverse
order, through c60, on FakeFez T and F.

    python c60_val.py <out dir> [A|B|C ...]   (run from the repository root; refuses an existing out dir)"""
import contextlib
import io
import json
import os
import runpy
import shutil
import sys
import tempfile
import time
import warnings

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
PARTS = sys.argv[2:] or ["A", "B", "C"]
if os.path.exists(OUT):
    raise SystemExit(f"STOP: {OUT} exists (a run never overwrites another)")
_tmp = tempfile.mkdtemp() + "/c55imp"
sys.argv = ["c55_val", _tmp]
g = runpy.run_path(os.path.join("benchmarks", "c55_val.py"), run_name="c60_val_import")  # C57-VAL's helpers
shutil.rmtree(_tmp, ignore_errors=True)
os.makedirs(OUT)
sys.path.insert(0, HERE)
import c60_tests as TT  # noqa: E402

C60_FILE = os.environ.get("C60_FILE", os.path.join(HERE, "c60", "psf_compile.py"))
C59_FILE = os.path.join("patches", "psf_compile_c59_2026-10-15", "psf_compile.py")
REL_FILE = "psf_compile.py"
M60 = g["load"](C60_FILE, "psf_compile_c60")
M59 = g["load"](C59_FILE, "psf_compile_c59")
REL = g["load"](REL_FILE, "psf_compile_rel")
for m in (M60, M59, REL):
    m.WARN_WITHOUT_TARGET = False
from qiskit.transpiler import generate_preset_pass_manager  # noqa: E402



SMOKE = os.environ.get("C60_SMOKE") == "1"  # a quick check of the script itself: FakeFez, two tests per part
DEVICES = ("FakeFez",) if SMOKE else g["DEVICES"]
A_TESTS = TT.TESTS[:2] if SMOKE else TT.TESTS


def call(mod, qc, tg):
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.compile_for_hardware(qc, target=tg, entangling_basis="cx", layout_search=True, seed_transpiler=0)


def timed(f):
    s = time.perf_counter()
    try:
        return f(), None, time.perf_counter() - s
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:160]}", time.perf_counter() - s


def describe(c, qc, tg):
    fe, fq = M60._failed_elements(tg, 0.5)
    return dict(sig=g["sig"](c), exact=g["exactness"](qc, c), uses_failed=bool(M60._uses_failed(c, fe, fq)),
                **g["metrics"](c, tg))


def part_a(fh):
    for d in DEVICES:
        t, ft = g["device"](d)
        for name in A_TESTS:
            qc = TT.build(name)
            for lab, tg in (("T", t), ("F", ft)):
                row = dict(part="A", device=d, test=name, target=lab)
                c, err, secs = timed(lambda: call(M60, qc, tg))
                row["C60"] = dict(error=err, secs=round(secs, 4))
                if c is not None:
                    row["C60"].update(describe(c, qc, tg))
                    c2, _, _ = timed(lambda: call(M60, qc, tg))
                    row["C60"]["sig_repeat"] = None if c2 is None else g["sig"](c2)
                c, err, secs = timed(lambda: call(REL, qc, tg))
                row["C57"] = dict(error=err, secs=round(secs, 3))
                if c is not None:
                    row["C57"].update(describe(c, qc, tg))

                def qk3():
                    o = generate_preset_pass_manager(3, target=tg, seed_transpiler=0).run(qc)
                    if "rzz" in tg.operation_names and not REL._rzz_in_range(o):
                        o = REL._fold(o, tg)
                    return o
                c, err, secs = timed(qk3)
                row["QK3"] = dict(error=err, secs=round(secs, 3))
                if c is not None:
                    row["QK3"].update(describe(c, qc, tg))
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                f = lambda a: f"{row[a].get('dur_us', float('nan')):6.3f}"  # noqa: E731
                print(f"A {d[-12:]:12s} {name:14s} {lab} C60 {f('C60')} ({row['C60']['secs']*1000:6.1f} ms) "
                      f"C57 {f('C57')} QK3 {f('QK3')} exact {row['C60'].get('exact')}", flush=True)


def w4_build(name):
    sys.argv = ["w4_orient.py", tempfile.mkdtemp()]
    ns = runpy.run_path(os.path.join(HERE, "w4_orient.py"), run_name="w4_import")
    return ns


def part_b(fh):
    w4 = w4_build(None)
    tests = [("C57VAL", n, lambda n=n: g["build"](n)) for n in g["TESTS"]]
    tests += [("W4", n, lambda n=n: w4["build"](n)) for n in w4["TESTS"]]
    if SMOKE:
        tests = [x for x in tests if x[1] in ("tfim_n12", "qaoa_n12_p2", "chain_rzz_n12", "chain_cz_n12")]
    for d in DEVICES:
        t, ft = g["device"](d)
        for src, name, b in tests:
            qc = b()
            for lab, tg in (("T", t), ("F", ft)):
                c59, e59, s59 = timed(lambda: call(M59, qc, tg))
                c60, e60, s60 = timed(lambda: call(M60, qc, tg))
                row = dict(part="B", source=src, device=d, test=name, target=lab, c59_error=e59, c60_error=e60,
                           c59_secs=round(s59, 4), c60_secs=round(s60, 4),
                           c59_sig=None if c59 is None else g["sig"](c59), c60_sig=None if c60 is None else g["sig"](c60))
                if c59 is None and c60 is not None:
                    row["c60_new"] = describe(c60, qc, tg)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                st = ("identical" if row["c59_sig"] and row["c59_sig"] == row["c60_sig"] else
                      "both refuse" if c59 is None and c60 is None else
                      "NEW (c59 refused)" if c59 is None else "DIFFERENT")
                print(f"B {d[-12:]:12s} {src:6s} {name:16s} {lab} {st}", flush=True)


def part_c(fh):
    w4 = w4_build(None)
    order_ns = {}
    src = open(os.path.join(HERE, "w4_order.py")).read().split("T, F = V.device")[0]
    src = src.split("from qiskit.quantum_info import Operator")[1]
    exec(src, order_ns)
    t, ft = g["device"]("FakeFez")
    for name in (w4["TESTS"][:2] if SMOKE else w4["TESTS"]):
        base = w4["build"](name)
        for lab, tg in (("T", t), ("F", ft)):
            durs, sigs = {}, {}
            for o in w4["ORIENT"]:
                c, err, _ = timed(lambda: call(M60, w4["orient"](base, o, name), tg))
                durs[o] = None if c is None else g["metrics"](c, tg)["dur_us"]
                sigs[o] = None if c is None else g["sig"](c)
            c, err, _ = timed(lambda: call(M60, order_ns["edge_reversed"](base), tg))
            durs["runs_reversed"] = None if c is None else g["metrics"](c, tg)["dur_us"]
            sigs["runs_reversed"] = None if c is None else g["sig"](c)
            row = dict(part="C", test=name, target=lab, dur=durs, distinct_outputs=len({s for s in sigs.values() if s}))
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(f"C {name:16s} {lab} {durs} distinct {row['distinct_outputs']}", flush=True)


head = dict(c60_sha256=g["sha"](C60_FILE), c59_sha256=g["sha"](C59_FILE), release_sha256=g["sha"](REL_FILE),
            tests_sha256=g["sha"](os.path.join(HERE, "c60_tests.py")), script_sha256=g["sha"](os.path.abspath(__file__)),
            versions=[M60.VERSION[:40], M59.VERSION[:40], REL.VERSION[:40]])
json.dump(head, open(os.path.join(OUT, "c60_val_head.json"), "w"), indent=1)
with open(os.path.join(OUT, "c60_val_rows.jsonl"), "a") as fh:
    for p in PARTS:
        dict(A=part_a, B=part_b, C=part_c)[p](fh)
print("C60-VAL DONE")
