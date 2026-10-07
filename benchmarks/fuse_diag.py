"""fuse_diag.py -- exploratory (2026-10-07): why FUSE's smoke circuit 3 on FakeKingston (ring, 16 qubits, measured,
seed 82,500,003) came out different from release 2026-10-06.4 under candidate c19. Recompiles it with both and prints
the counters each moved, the two outputs' placements and two-qubit counts, and each output's estimates under both
modules.

    cd <repo> && python <this file>
"""
import contextlib
import io
import os
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
REPO = os.getcwd()
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
import core_fix_c2_eval as H  # noqa: E402

H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile")
c19 = H.load_module(os.path.join(REPO, "patches", "psf_compile_c19_2026-10-07", "psf_compile.py"), "psf_compile_c19")
import skip_eval  # noqa: E402
from qiskit_ibm_runtime import fake_provider  # noqa: E402

RECOMMENDED = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True,
                   candidate_score="hybrid")
t = fake_provider.FakeKingston().target
cm = t.build_coupling_map()
basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x")]
qc = skip_eval.family_circuit("ring", 16, np.random.default_rng(82_500_003))
qc.measure_all()
STATS = ("COMPARE_STATS", "RESYNTH_STATS", "EXACT_STATS", "PRUNE_STATS", "SKIP_STATS")


def run(mod):
    before = {s: dict(getattr(mod, s)) for s in STATS if hasattr(mod, s)}
    with contextlib.redirect_stdout(io.StringIO()):
        out = mod.compile_for_hardware(qc, coupling_map=cm, basis_gates=basis, entangling_basis="cx",
                                       layout_search=True, target=t, seed_transpiler=0, **RECOMMENDED)
    moved = {s: {k: getattr(mod, s)[k] - v for k, v in d.items() if getattr(mod, s)[k] != v} for s, d in before.items()}
    return out, moved


for name, mod in (("release", rel), ("c19", c19)):
    out, moved = run(mod)
    globals()["out_" + name] = out
    print(f"{name}: moved {moved}")
    print(f"   2q {sum(1 for i in out.data if len(i.qubits) == 2)}, ops {dict(out.count_ops())}")
    print(f"   initial {list(out.layout.initial_index_layout(filter_ancillas=True))}")
    print(f"   final   {list(out.layout.final_index_layout(filter_ancillas=True))}")
for oname in ("release", "c19"):
    o = globals()["out_" + oname]
    vals = []
    for mname, mod in (("rel", rel), ("c19", c19)):
        vals.append(f"{mname}: hybrid {mod.hybrid_cost(o, t)!r} excitation {mod.excitation_cost(o, t)!r}")
    print(f"output of {oname}: " + " | ".join(vals))
print("same instructions apart from layout:",
      [(i.operation.name, [out_release.find_bit(q).index for q in i.qubits]) for i in out_release.data] ==
      [(i.operation.name, [out_c19.find_bit(q).index for q in i.qubits]) for i in out_c19.data])
print("implements (release check):", rel._implements(qc, out_release), rel._implements(qc, out_c19))
