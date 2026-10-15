# Path note (2026-10-15c): run from the repository root (bundle q); C61-VAL, c61 and c60 are read from the
# 2026-10-15b delivery inside the repository (the original run used the workplace copies).
import os as _os
_B = _os.path.join("data", "2026-10-15", "workplace", "workplace_2026-10-15b_W1-W2")
_os.environ.setdefault("C61_FILE", _os.path.join(_B, "w31", "c61", "psf_compile.py"))
_os.environ.setdefault("C60_FILE", _os.path.join(_B, "w30", "c60", "psf_compile.py"))
_os.environ.setdefault("C60_TESTS_DIR", _os.path.join(_B, "w30"))
import sys, os, warnings, random, tempfile
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], tempfile.mkdtemp() + "/x"]
exec(open(os.path.join(_B, "w31", "c61_val.py")).read().split("head = dict(only=")[0])
M63 = g["load"](os.path.join(HERE, "c63", "psf_compile.py"), "psf_compile_c63"); M63.WARN_WITHOUT_TARGET = False
t, ft = g["device"]("FakeFez")
qc = build("hhtree_heis_n16"); tpl = M63.compile_template(qc, t); gen = tpl._gen
print("blocks", [ (len(b[2]), b[4] is not None) for b in gen.prog if b[0]=="block"][:20])
from collections import Counter
print(Counter((len(b[2]), b[4] is not None) for b in gen.prog if b[0]=="block"))
p = tpl._planned
rng = random.Random(1)
for k in range(4):
    items = list(gen.items(tpl.values_of(randomize(qc, rng, same=True))))
    diffs = []
    for idx, cls, pos, m, pin in p["runs"]:
        U = M63._T100_ONE
        for i in idx: U = M63._n99_mul(items[i][2], U)
        c, ang = tpl._run(U, pin)
        if c != cls: diffs.append((idx, cls, c, [round(a,4) for a in ang]))
    print(len(diffs), diffs[:4])
