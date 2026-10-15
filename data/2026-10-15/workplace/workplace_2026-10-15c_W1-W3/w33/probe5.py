"""Feasibility: c62 (item 101) against c60 on FakeFez, the 12 C58-VAL in-scope tests, T and F."""
# Path note (2026-10-15c): run from the repository root (bundle q); C61-VAL, c61 and c60 are read from the
# 2026-10-15b delivery inside the repository (the original run used the workplace copies).
import os as _os
_B = _os.path.join("data", "2026-10-15", "workplace", "workplace_2026-10-15b_W1-W2")
_os.environ.setdefault("C61_FILE", _os.path.join(_B, "w31", "c61", "psf_compile.py"))
_os.environ.setdefault("C60_FILE", _os.path.join(_B, "w30", "c60", "psf_compile.py"))
_os.environ.setdefault("C60_TESTS_DIR", _os.path.join(_B, "w30"))
import sys, os, runpy, tempfile, warnings, time, contextlib, io
warnings.simplefilter("ignore")
sys.argv = [sys.argv[0], tempfile.mkdtemp() + "/x"]
exec(open(os.path.join(_B, "w31", "c61_val.py")).read().split("head = dict(only=")[0])
M62 = g["load"](os.path.join(os.path.dirname(os.path.abspath(__file__)), "c62", "psf_compile.py"), "psf_compile_c62"); M62.WARN_WITHOUT_TARGET = False
from c29_identity import value_sig
def call(M, qc, tg):
    with contextlib.redirect_stdout(io.StringIO()):
        return M.compile_for_hardware(qc, target=tg, entangling_basis="cx", layout_search=True, seed_transpiler=0)
for d in ("FakeFez",):
    t, ft = g["device"](d)
    for name in FORMULA:
        qc = build(name)
        for lab, tg in (("T", t), ("F", ft)):
            s = time.perf_counter(); a = call(M62, qc, tg); ta = time.perf_counter() - s
            b = call(M60, qc, tg)
            ma, mb = g["metrics"](a, tg), g["metrics"](b, tg)
            ex = fast_exact(qc, a) if ma["active"] <= 20 else None
            print(f"{name:16s} {lab} same {value_sig(a) == value_sig(b)!s:5s} c62 {ma['dur_us']:7.3f}/{ma['twoq']:4d} c60 {mb['dur_us']:7.3f}/{mb['twoq']:4d} ratio {ma['dur_us']/mb['dur_us']:.3f} exact {ex} isa {ma['isa']} {ta*1000:.0f} ms", flush=True)
print(M62.CZ101_STATS)
