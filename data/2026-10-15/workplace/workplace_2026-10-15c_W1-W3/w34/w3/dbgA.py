import sys, os, warnings, random, runpy, tempfile
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = ["x", tempfile.mkdtemp() + "/c"]
g = runpy.run_path(os.path.join("benchmarks", "c55_val.py"), run_name="imp")
M = g["load"](os.path.join(HERE, "c63a", "psf_compile.py"), "c63a"); M.WARN_WITHOUT_TARGET = False
ns = runpy.run_path(os.path.join(HERE, "..", "c63_val.py").replace("c63_val.py", "w3/w3_walk.py"), run_name="no") if False else None
t, ft = g["device"]("FakeFez")
qc = g["build"]("tfim_n12"); tpl = M.compile_template(qc, ft); gen = tpl._gen; v0 = tpl.values_of(qc)
from qiskit.quantum_info import Operator
import numpy as np
rng = random.Random(5)
for k in range(6):
    v = [x + rng.gauss(0, 0.8) for x in v0]
    items = list(gen.items(v))
    a = tpl._batch(items, tpl._planned)
    sh, full = M._t63_emit(iter(items), gen.phys, ft, tpl.use_rx, tpl.use_x)
    # compare batch slots with emit's slots on the variable positions
    p = tpl._planned
    rem = p["remap"]
    import math
    w = lambda x: (x + math.pi) % (2 * math.pi) - math.pi
    diff = max(abs(w(a[rem[i]] - full[i])) for i in rem) if a else None
    rz = [i for i, x in enumerate(sh) if False]
    fixed_bad = 0
    print(k, "batch None" if a is None else f"max slot diff {diff:.2e}", "shape same", sh == tpl._shape)

# fixed slots vs emit
items = list(gen.items(v)); sh, full = M._t63_emit(iter(items), gen.phys, ft, tpl.use_rx, tpl.use_x)
w = lambda x: (x + 3.141592653589793) % (6.283185307179586) - 3.141592653589793
print("fixed diffs", max(abs(w(full[i] - x)) for i, x in tpl._planned["fixed"].items()))
print("n fixed", len(tpl._planned["fixed"]), "runs", len(tpl._planned["runs"]))
