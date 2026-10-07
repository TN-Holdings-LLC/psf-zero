"""release_10071_diag.py -- why benchmarks/test_release_2026_10_03_2.py::test_compare_returns_lower_estimate_exact_and_safe
failed on FakeAuckland with release 2026-10-07.1 (Addendum 397). Repeats that test's four circuits with the release,
2026-10-06.4 (kept copy) and candidate c19 (item 46 only), and prints the two estimates, their relative difference
and which circuit each returns. Exploratory; changes nothing.

    python benchmarks/release_10071_diag.py
"""
import contextlib
import importlib.util
import io
import os
import sys
import warnings

warnings.simplefilter("ignore")
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for p in (os.path.join(REPO, "benchmarks"), REPO):
    sys.path.insert(0, p)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


import core_fix_c2_eval as H  # noqa: E402
sys.modules["psf_smart_layout"] = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psl_diag")
T = load("benchmarks/test_release_2026_10_03_2.py", "t10032_diag")  # its circuits and helpers only
MODS = {"2026-10-07.1": "psf_compile.py",
        "2026-10-06.4": "patches/psf_compile_release_2026-10-06.4/psf_compile.py",
        "c19": "patches/psf_compile_c19_2026-10-07/psf_compile.py"}
mods = {k: H.load_module(os.path.join(REPO, v), "diag_" + k.replace(".", "_").replace("-", "_")) for k, v in MODS.items()}
prev = H.load_module(os.path.join(REPO, "patches", "psf_compile_c8_2026-10-03", "psf_compile.py"), "diag_c8")
for dev in ("FakeAuckland",):
    tgt = T.backend(dev).target
    kw = T.kw_for(tgt)
    for label, qc in (("ring4", T.ring(4, seed=1)), ("ring6", T.ring(6, seed=2)), ("chain5", T.chain(5, seed=3)),
                      ("xxz_ring", T.xxz_ring(seed=4))):
        with contextlib.redirect_stdout(io.StringIO()):
            a = prev.compile_for_hardware(qc, target=tgt, placement_refine=True, final_resynthesis="select", **kw)
            b = T.l3(qc, tgt)
        print(f"{dev} {label}: level 3 acceptable {mods['2026-10-07.1']._acceptable(b, tgt, 0.5)}")
        for k, m in mods.items():
            with contextlib.redirect_stdout(io.StringIO()):
                c = m.compile_for_hardware(qc, target=tgt, placement_refine=True, final_resynthesis="select",
                                           compare_level3=True, **kw)
            ea, eb = m.excitation_cost(a, tgt), m.excitation_cost(b, tgt)
            rel = (ea - eb) / max(abs(ea), 1e-300)
            print(f"   {k:13s} ea {ea!r} eb {eb!r} (ea-eb)/ea {rel:.3e} | returns "
                  f"{'level 3' if T.sig(c) == T.sig(b) else 'own' if T.sig(c) == T.sig(a) else 'other'}")
