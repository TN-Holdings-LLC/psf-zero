"""c22_timing.py -- exploratory (not pre-registered; 2026-10-07): the recommended call's time with candidate c22
(item 49) against candidate c21 (items 46-48) on SKIP's four families at 12, 14 and 16 qubits on one device, measured,
alternating which module runs first. Prints per circuit both times, whether the outputs are identical, and how many
item 39 checks each made; then the median ratio per size. Made from c20_timing.py by renaming the two modules.

    cd <repo> && python benchmarks/c22_timing.py [--device FakeTorino] [--out FILE.json]
"""
import argparse
import contextlib
import io
import json
import os
import statistics
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
REPO = os.getcwd()
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
RECOMMENDED = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True,
                   candidate_score="hybrid")


def sig(c):
    """Every instruction with its qubits, clbits and parameters, the global phase and both layouts."""
    return [[i.operation.name, [c.find_bit(q).index for q in i.qubits], [c.find_bit(b).index for b in i.clbits],
             [repr(p) for p in i.operation.params]] for i in c.data] + \
        [repr(c.global_phase), list(c.layout.initial_index_layout(filter_ancillas=True)),
         list(c.layout.final_index_layout(filter_ancillas=True))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="FakeTorino")
    ap.add_argument("--out")
    a = ap.parse_args()
    import core_fix_c2_eval as H
    import skip_eval
    from qiskit_ibm_runtime import fake_provider
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    mods = {"c21": H.load_module(os.path.join(REPO, "patches", "psf_compile_c21_2026-10-07", "psf_compile.py"),
                                 "psf_compile_c21"),
            "c22": H.load_module(os.path.join(REPO, "patches", "psf_compile_c22_2026-10-07", "psf_compile.py"),
                                 "psf_compile_c22")}
    print({k: m.VERSION for k, m in mods.items()}, "core", mods["c22"].CORE_VERSION)
    t = getattr(fake_provider, a.device)().target
    cm = t.build_coupling_map()
    basis = [g for g in ("cx", "cz", "rz", "sx", "x") if g in t.operation_names]

    def call(m, qc):
        c0 = mods[m].EXACT_STATS["checked"]
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            o = mods[m].compile_for_hardware(qc, coupling_map=cm, basis_gates=basis, entangling_basis="cx",
                                             layout_search=True, target=t, seed_transpiler=0, **RECOMMENDED)
        return o, time.perf_counter() - t0, mods[m].EXACT_STATS["checked"] - c0

    call("c21", skip_eval.family_circuit("ring", 4, np.random.default_rng(1)))  # warm-up
    call("c22", skip_eval.family_circuit("ring", 4, np.random.default_rng(1)))
    rows, k = [], 0
    for n in (12, 14, 16):
        for fam in ("ring", "brick", "pauli", "qft"):
            for rep in range(2):
                qc = skip_eval.family_circuit(fam, n, np.random.default_rng(47_500_000 + k))
                if rep:
                    qc.measure_all()
                order = ("c21", "c22") if k % 2 == 0 else ("c22", "c21")
                res = {m: call(m, qc) for m in order}
                same = sig(res["c21"][0]) == sig(res["c22"][0])
                r = dict(n=n, family=fam, measured=bool(rep), t21=round(res["c21"][1], 3), t22=round(res["c22"][1], 3),
                         checks21=res["c21"][2], checks22=res["c22"][2], identical=bool(same))
                rows.append(r)
                print(f"n={n:2d} {fam:5s} m={rep} c21 {r['t21']:7.2f} s  c22 {r['t22']:7.2f} s  "
                      f"ratio {r['t22'] / r['t21']:.2f}  checks {r['checks21']}->{r['checks22']}  identical {same}",
                      flush=True)
                k += 1
    for n in (12, 14, 16):
        rs = [x["t22"] / x["t21"] for x in rows if x["n"] == n]
        tot21 = sum(x["t21"] for x in rows if x["n"] == n)
        tot22 = sum(x["t22"] for x in rows if x["n"] == n)
        print(f"n={n}: median ratio {statistics.median(rs):.3f} (range {min(rs):.2f}-{max(rs):.2f}); "
              f"total c21 {tot21:.1f} s, c22 {tot22:.1f} s")
    print("all identical:", all(x["identical"] for x in rows))
    if a.out:
        json.dump(dict(device=a.device, versions={k: m.VERSION for k, m in mods.items()}, rows=rows),
                  open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
