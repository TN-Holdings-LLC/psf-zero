"""probe_v4_fallbacks.py -- exploratory (home, 2026-09-28): what are the 15 blocks that
still fall back from the fixed Rust core in part E of long_loop_100k_v4.py (Addendum
244), and which of them needed the exact rebuild (changelog item 23)?

Recreates part E's inputs exactly (target seed 7 with every rzz angle 0; one
normal(0, 0.5) draw per lap from seed 202, in lap order), compiles only the 15 laps
that fell back, and records for every block the core rejected: its capture position
and circuit pair, the core's error, the block's Weyl coordinates (Qiskit,
unspecialized), their distances to the chamber's special loci, the pair's
interaction angles, and the GUARD_STATS change of that compile. It also checks that
the other 30,000 - 15 laps are not needed: the lap list is read from the v4 CSV.

    python -u benchmarks/probe_v4_fallbacks.py data/long_loop_E_2026-09-28_v4.csv 2>&1 | tee probe_v4_fallbacks.txt
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "benchmarks"), os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from qiskit.synthesis import TwoQubitWeylDecomposition  # noqa: E402
from qiskit.transpiler import CouplingMap  # noqa: E402

import loop_endurance as le  # noqa: E402
import psf_compile as pc  # noqa: E402

E_LAPS = 30000
# capture position -> circuit pair, as mapped by the workplace (map_pairs.py, Addendum 237 context)
POS_PAIR = {0: (0, 1), 1: (2, 3), 2: (1, 2), 3: (4, 5), 4: (3, 4), 5: (6, 7), 6: (5, 6), 7: (8, 9),
            8: (7, 8), 9: (10, 11), 10: (9, 10)}
OUT_JSON = "probe_v4_fallbacks_2026-09-28.json"


def fallback_laps(path):
    with open(path, encoding="utf-8") as f:
        return [int(r["lap"]) for r in csv.DictReader(f) if int(r["fallbacks"] or 0) > 0]


def special_distances(a, b, c):
    """Distances of Weyl coordinates to the loci Qiskit's specialization snaps to."""
    q = np.pi / 4
    return {"a-b": abs(a - b), "b-|c|": abs(b - abs(c)), "|c|": abs(c), "pi/4-a": abs(q - a), "b": abs(b)}


def main():
    print("LOADED", pc.__file__, pc.VERSION, le.normalized_sha256(pc.__file__), "| CORE_VERSION", pc.CORE_VERSION)
    print("SCRIPT", os.path.abspath(__file__), le.normalized_sha256(os.path.abspath(__file__)))
    laps = fallback_laps(sys.argv[1])
    print(f"fallback laps in {sys.argv[1]}: {len(laps)} -> {laps}")
    rng_e = np.random.default_rng(7)
    tt = rng_e.uniform(-np.pi, np.pi, le.E_NPARAMS)
    tt[8::15] = 0.0
    rng_t = np.random.default_rng(202)
    want = set(laps)
    thetas = {}
    for lap in range(1, E_LAPS + 1):
        th = tt + rng_t.normal(0.0, 0.5, le.E_NPARAMS)
        if lap in want:
            thetas[lap] = th
    cmap = CouplingMap.from_line(le.E_QUBITS)

    cap = []
    orig_checked, orig_plain = pc._CORE_CHECKED, pc.geometric_decompose

    def wrap(f):
        def g(u_r, u_i):
            u = np.array(u_r) + 1j * np.array(u_i)
            try:
                r = f(u_r, u_i)
                cap.append((u, "ok", ""))
                return r
            except Exception as e:  # noqa: BLE001 -- recorded, re-raised
                cap.append((u, type(e).__name__, str(e)[:200]))
                raise
        return g

    rebuilt_inputs = []
    orig_rebuild = pc._exact_rebuild

    def rebuild_probe(u):
        rebuilt_inputs.append(np.array(u))
        return orig_rebuild(u)

    if orig_checked is not None:
        pc._CORE_CHECKED = wrap(orig_checked)
    pc.geometric_decompose = wrap(orig_plain)
    pc._exact_rebuild = rebuild_probe
    records = []
    try:
        for lap in laps:
            cap.clear()
            rebuilt_inputs.clear()
            before = dict(pc.GUARD_STATS)
            qc = le.e_circuit(thetas[lap])
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("ignore")
                pc._CX_CORE_CACHE.clear()
                pc.compile_for_hardware(qc, coupling_map=cmap, basis_gates=["cx", "rz", "sx", "x"],
                                        entangling_basis="cx", initial_layout=list(range(le.E_QUBITS)),
                                        on_unsupported="keep", seed_transpiler=0, block_gate_floor=8)
            delta = {k: pc.GUARD_STATS[k] - before[k] for k in pc.GUARD_STATS
                     if isinstance(pc.GUARD_STATS[k], int) and pc.GUARD_STATS[k] != before[k]}
            for pos, (u, tag, msg) in enumerate(cap):
                if tag == "ok":
                    continue
                pair = POS_PAIR.get(pos)
                k = le.E_BLOCKS.index(pair) if pair in le.E_BLOCKS else None
                p = thetas[lap][15 * k:15 * k + 15] if k is not None else None
                w = TwoQubitWeylDecomposition(u, fidelity=None)
                a, b, c = float(w.a), float(w.b), float(w.c)
                rebuilt_here = any(np.allclose(u, r, atol=1e-14) for r in rebuilt_inputs)
                rec = dict(lap=lap, position=pos, pair=pair, layer=2 if pos in (2, 4, 6, 8, 10) else 1,
                           error=tag, message=msg, weyl=[a, b, c], special=special_distances(a, b, c),
                           rxx_ryy_rzz=[float(x) for x in p[6:9]] if p is not None else None,
                           rebuilt=rebuilt_here, guard_delta=delta)
                records.append(rec)
                sd = rec["special"]
                near = min(sd, key=sd.get)
                print(f"lap {lap:5d} pos {pos:2d} pair {pair} layer {rec['layer']}: {tag} | Weyl "
                      f"({a:+.6f}, {b:+.6f}, {c:+.6e}) nearest special {near}={sd[near]:.2e} | "
                      f"rebuilt {rebuilt_here} | guard {delta}", flush=True)
            if not any(t != "ok" for _, t, _ in cap):
                print(f"lap {lap:5d}: no core failure reproduced (guard {delta})", flush=True)
    finally:
        pc._CORE_CHECKED, pc.geometric_decompose, pc._exact_rebuild = orig_checked, orig_plain, orig_rebuild

    print("\n=== summary ===")
    print(f"core failures reproduced: {len(records)} (v4 log: 15)")
    for key in ("error", "layer", "pair"):
        vals = {}
        for r in records:
            vals[str(r[key])] = vals.get(str(r[key]), 0) + 1
        print(f"by {key}: {vals}")
    print(f"rebuilt exactly: {sum(r['rebuilt'] for r in records)} (v4 log: 4)")
    for name in ("a-b", "b-|c|", "|c|", "pi/4-a", "b"):
        xs = sorted(r["special"][name] for r in records)
        if xs:
            print(f"distance to {name:7s}: min {xs[0]:.2e}, median {xs[len(xs) // 2]:.2e}, max {xs[-1]:.2e}")
    json.dump(records, open(OUT_JSON, "w"), indent=1, default=str)
    print(f"wrote {OUT_JSON}\nDONE")


if __name__ == "__main__":
    main()
