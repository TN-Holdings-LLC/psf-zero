"""fuse_run_diag.py -- exploratory, after FUSE's scored run (2026-10-07): for every circuit of the run whose c19 output
differed from release 2026-10-06.4's, rebuild it (FUSE's own generator and seed), compile it with both, and print what
each chose, the two outputs' shapes, their estimates under both modules, and whether both implement the input. Also
prints FUSE's per-device n = 20 medians (F4).

    cd <repo> && python <this file> <run dir>
"""
import contextlib
import glob
import io
import json
import os
import statistics
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
RUN = sys.argv[1]
REPO = os.getcwd()
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
import core_fix_c2_eval as H  # noqa: E402

H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile")
c19 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c19_2026-10-07", "psf_compile.py"), "psf_compile_c19")
import fuse_eval as FE  # noqa: E402
from qiskit_ibm_runtime import fake_provider  # noqa: E402

STATS = ("COMPARE_STATS", "RESYNTH_STATS", "EXACT_STATS", "PRUNE_STATS")
for p in sorted(glob.glob(os.path.join(RUN, "fuse_*.json"))):
    if p.endswith("_smoke.json"):
        continue
    r = json.load(open(p))
    dev = r["meta"]["device"]
    n20 = [x["t_c19"] / x["t_rel"] for x in r["rows"] if x["n"] == 20]
    print(f"{dev}: n=20 median ratio {statistics.median(n20):.3f}; ratios {[round(v, 2) for v in sorted(n20)]}; "
          f"release times {[round(x['t_rel'], 3) for x in r['rows'] if x['n'] == 20]}")
    diffs = [k for k, x in enumerate(r["rows"]) if not x.get("identical", True)]
    if not diffs:
        continue
    t = getattr(fake_provider, dev)().target
    cm = t.build_coupling_map()
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x")]
    circs = FE.circuits(dev, False)
    for k in diffs:
        name, n, measured, qc, bare = circs[k]
        print(f"\n== {dev} circuit {k}: {name} n={n} measured={measured} (seed {82_000_000 + k})")
        outs = {}
        for mname, mod in (("release", rel), ("c19", c19)):
            before = {s: dict(getattr(mod, s)) for s in STATS}
            with contextlib.redirect_stdout(io.StringIO()):
                out = mod.compile_for_hardware(qc, coupling_map=cm, basis_gates=basis, entangling_basis="cx",
                                               layout_search=True, target=t, seed_transpiler=0, **FE.RECOMMENDED)
            moved = {s: {kk: getattr(mod, s)[kk] - v for kk, v in d.items() if getattr(mod, s)[kk] != v}
                     for s, d in before.items()}
            outs[mname] = out
            print(f"{mname}: moved {moved}")
            print(f"   ops {dict(out.count_ops())}")
            print(f"   initial {list(out.layout.initial_index_layout(filter_ancillas=True))}")
        print("identical now:", FE.full_sig(outs["release"]) == FE.full_sig(outs["c19"]))
        for oname, o in outs.items():
            vals = [f"{m}: hybrid {mod.hybrid_cost(o, t)!r} excitation {mod.excitation_cost(o, t)!r}"
                    for m, mod in (("rel", rel), ("c19", c19))]
            print(f"output of {oname}: " + " | ".join(vals))
        a, b = rel.hybrid_cost(outs["release"], t), rel.hybrid_cost(outs["c19"], t)
        print(f"release's own hybrid estimates of the two outputs differ by {abs(a - b) / max(abs(a), abs(b)):.2e} "
              f"(relative; the tie band is 1e-12)")
        print("implements (release check):", rel._implements(bare, outs["release"]), rel._implements(bare, outs["c19"]))
