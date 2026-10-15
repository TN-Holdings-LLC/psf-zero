"""W2 probe: bind time of c63's planned binding against c61's, same-sign draws, FakeFez T and F (single process)."""
# Path note (2026-10-15c): run from the repository root (bundle q); C61-VAL, c61 and c60 are read from the
# 2026-10-15b delivery inside the repository (the original run used the workplace copies).
import os as _os
_B = _os.path.join("data", "2026-10-15", "workplace", "workplace_2026-10-15b_W1-W2")
_os.environ.setdefault("C61_FILE", _os.path.join(_B, "w31", "c61", "psf_compile.py"))
_os.environ.setdefault("C60_FILE", _os.path.join(_B, "w30", "c60", "psf_compile.py"))
_os.environ.setdefault("C60_TESTS_DIR", _os.path.join(_B, "w30"))
import sys, os, warnings, time, statistics, random, runpy, tempfile, importlib.util
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], tempfile.mkdtemp() + "/x"]
exec(open(os.path.join(_B, "w31", "c61_val.py")).read().split("head = dict(only=")[0])
M63 = g["load"](os.environ.get("C63_FILE", os.path.join(HERE, "c63", "psf_compile.py")), "psf_compile_c63"); M63.WARN_WITHOUT_TARGET = False
only = sys.argv[1:] or None
t, ft = g["device"]("FakeFez")
for name in FORMULA + NATIVE:
    qc = build(name)
    for lab, tg in (("T", t), ("F", ft)):
        out = []
        for M in (M61, M63):
            tpl = M.compile_template(qc, tg); rng = random.Random(7)
            vals = [tpl.values_of(randomize(qc, rng, same=True)) for _ in range(60)]
            for v in vals[:5]: tpl.bind_values(v)
            ts = []
            for v in vals:
                s = time.perf_counter(); tpl.bind_values(v); ts.append(time.perf_counter() - s)
            out.append(statistics.median(ts) * 1000)
        print(f"{name:16s} {lab} c61 {out[0]:6.3f} ms  c63 {out[1]:6.3f} ms", flush=True)
