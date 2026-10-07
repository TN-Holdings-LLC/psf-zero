"""c20_profile.py -- exploratory profile (not pre-registered, 2026-10-07): where candidate c20's recommended call still
spends its time at 16 qubits, after items 46-47. Same circuits as cliff16_profile.py (SKIP's first seed of the pauli
and qft cells at 16 qubits, measured). For each: the cumulative time of the module's own functions, and the 20
functions (any module, numpy included) with the most time of their own.

    cd <repo> && python benchmarks/c20_profile.py [--module patches/psf_compile_c20_2026-10-07/psf_compile.py]
                                                  [--device FakeTorino] [--out FILE]
"""
import argparse
import contextlib
import cProfile
import io
import os
import pstats
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
REPO = os.getcwd()
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]

OWN = ("compile_for_hardware", "_recompile_pruned", "_resynthesise", "_select_resynthesis", "_final_resynthesis",
       "_resynthesis_candidate", "_compare_level3", "_implements", "_same_action", "_ops_of", "_fuse_ops",
       "_apply_ops", "excitation_cost", "hybrid_cost", "readout_cost", "_populations", "_embed_1q", "_free_diagonal",
       "_choose", "_choose_lazy", "_acceptable", "floor_aware_target", "compile", "smart_vf2_layout",
       "_refine_placement")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default=os.path.join("patches", "psf_compile_c20_2026-10-07", "psf_compile.py"))
    ap.add_argument("--device", default="FakeTorino")
    ap.add_argument("--out")
    a = ap.parse_args()
    import core_fix_c2_eval as H
    import skip_eval as S
    from qiskit_ibm_runtime import fake_provider
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    mod = H.load_module(os.path.join(REPO, a.module), "psf_compile_profiled")
    t = getattr(fake_provider, a.device)().target
    cm = t.build_coupling_map()
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x")]
    lines = [f"module {mod.VERSION}, core {mod.CORE_VERSION}, device {a.device}"]
    cell = {(f, n): i for i, (f, n) in enumerate((f, n) for f in S.FAMILIES for n in (10, 16, 17, 99))}
    for fam in ("pauli", "qft"):
        qc = S.family_circuit(fam, 16, np.random.default_rng(81_000_000 + 3 * cell[(fam, 16)]))
        qc.measure_all()
        pr = cProfile.Profile()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            pr.enable()
            mod.compile_for_hardware(qc, coupling_map=cm, basis_gates=basis, entangling_basis="cx",
                                     layout_search=True, target=t, seed_transpiler=0, **S.RECOMMENDED)
            pr.disable()
        wall = time.perf_counter() - t0
        st = pstats.Stats(pr).stats
        own = sorted(((ct, nc, name) for (fn, _, name), (cc, nc, tt, ct, _c) in st.items()
                      if name in OWN and "psf_" in os.path.basename(fn)), reverse=True)
        top = sorted(((tt, nc, f"{name} ({os.path.basename(fn)}:{line})") for (fn, line, name), (cc, nc, tt, ct, _c)
                      in st.items()), reverse=True)[:20]
        block = [f"\n{fam} n=16: wall {wall:.2f} s", "  own functions, cumulative s / calls"]
        block += [f"  {ct:10.2f} {nc:7d}  {name}" for ct, nc, name in own]
        block += ["  most time of their own, s / calls"]
        block += [f"  {tt:10.2f} {nc:7d}  {name}" for tt, nc, name in top]
        print("\n".join(block), flush=True)
        lines += block
    if a.out:
        open(a.out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
