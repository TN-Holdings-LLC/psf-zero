"""cliff16_profile.py -- exploratory profile (not pre-registered, 2026-10-06): where the recommended call of the release
spends its time on 16-qubit circuits (SKIP, Addendum 380: pauli 31.5 s and qft 20.2 s median at 16 qubits against
0.35 s and 0.33 s at 17). Uses SKIP's own circuit builders and seeds, so the circuits are SKIP's.

    cd <repo> && python cliff16_profile.py [--device FakeTorino] [--out profile.txt]

For each of four circuits (pauli and qft at 16 and 17 qubits, SKIP's first seed of each cell) it runs the recommended
call once under cProfile and prints the cumulative time of the release's own functions and of Qiskit's transpile.
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

FUNCS = ("compile_for_hardware", "_recompile_pruned", "_resynthesise", "_select_resynthesis", "_final_resynthesis",
         "_compare_level3", "_implements", "_same_action", "_ops_of", "_apply_ops", "excitation_cost", "pauli_cost",
         "hybrid_cost", "kraus_cost", "_choose", "_acceptable", "floor_aware_target", "transpile", "compile",
         "smart_vf2_layout", "_refine_placement")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="FakeTorino")
    ap.add_argument("--out", default="cliff16_profile.txt")
    a = ap.parse_args()
    import skip_eval as S
    from qiskit_ibm_runtime import fake_provider
    rel, _c17, _RE = None, None, None
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile")
    t = getattr(fake_provider, a.device)().target
    cm = t.build_coupling_map()
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x")]
    lines = [f"release {rel.VERSION}, device {a.device}"]
    # SKIP's seed of the first circuit of each (family, n) cell: cells are ordered family-major, sizes (10, 16, 17, L)
    cell = {(f, n): i for i, (f, n) in enumerate((f, n) for f in S.FAMILIES for n in (10, 16, 17, 99))}
    for fam, n in (("pauli", 16), ("pauli", 17), ("qft", 16), ("qft", 17)):
        rng = np.random.default_rng(81_000_000 + 3 * cell[(fam, n)])
        qc = S.family_circuit(fam, n, rng)
        qc.measure_all()
        for k in rel.EXACT_STATS:
            rel.EXACT_STATS[k] = 0
        pr = cProfile.Profile()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            pr.enable()
            rel.compile_for_hardware(qc, coupling_map=cm, basis_gates=basis, entangling_basis="cx",
                                     layout_search=True, target=t, seed_transpiler=0, **S.RECOMMENDED)
            pr.disable()
        wall = time.perf_counter() - t0
        st = pstats.Stats(pr)
        rows = []
        for (fn, line, name), (cc, nc, tt, ct, callers) in st.stats.items():
            if name in FUNCS and ("psf_compile" in fn or "psf_smart_layout" in fn or name == "transpile"):
                rows.append((ct, nc, name, os.path.basename(fn)))
        rows.sort(reverse=True)
        lines.append(f"\n{fam} n={n}: wall {wall:.2f} s; EXACT_STATS {dict(rel.EXACT_STATS)}; "
                     f"SKIP_STATS {getattr(rel, 'SKIP_STATS', None)}")
        lines.append(f"  {'cumulative s':>12} {'calls':>6}  function")
        for ct, nc, name, fn in rows[:16]:
            lines.append(f"  {ct:12.2f} {nc:6d}  {name} ({fn})")
        print("\n".join(lines[-(len(rows[:16]) + 2):]), flush=True)
    open(a.out, "w").write("\n".join(lines) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
