"""a12_seeds.py -- exploratory (not pre-registered), 2026-10-06: the AI front end a12 calls the release for every start
point with seeds (0, 1, 2, 3) (fewer for the extra start points). What does it lose if it uses fewer seeds?

Circuits: MODEL-RO2's model-style generator (ai10_eval2), 2 per (family, n) cell, 32 circuits, exploratory seeds
78,000,000 + k, each with measure_all(). Per circuit and device: a12 with seeds (0, 1, 2, 3) (its default), (0, 1)
and (0,); the time of each, whether the output is the default's circuit, and the output's state-aware estimate
(a12.state_aware_cost, the score a12 itself chooses by) relative to the default's.

    python a12_seeds.py --out DIR --device FakeTorino [--device ...]
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
REPO = os.getcwd()
WORK = os.path.join(REPO, "data", "2026-10-05", "workplace")
for p in (os.path.join(REPO, "benchmarks"), REPO):
    sys.path.insert(0, p)
VARIANTS = {"s4": (0, 1, 2, 3), "s2": (0, 1), "s1": (0,)}


def sig(c):
    return [[i.operation.name, [c.find_bit(q).index for q in i.qubits], [repr(p) for p in i.operation.params]]
            for i in c.data] + [list(c.layout.initial_index_layout(filter_ancillas=True)),
                                list(c.layout.final_index_layout(filter_ancillas=True))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", action="append", required=True)
    a = ap.parse_args()
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile")
    a12 = H.load_module(os.path.join(REPO, "benchmarks", "psf_ai_compile.py"), "psf_ai_compile_a12_seeds")
    E2 = H.load_module(os.path.join(WORK, "model_ro2", "ai10_eval2.py"), "ai10_eval2_seeds")
    assert rel.VERSION == "2026-10-06.1" and a12.AI_COMPILE_VERSION == "2026-10-06.a12"
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    from qiskit_ibm_runtime import fake_provider
    circs, k = [], 0
    for name, ns in E2.FAMILIES:
        for n in ns:
            for _ in range(2):
                rng = np.random.default_rng(78_000_000 + k)
                k += 1
                qc = E2._model_style(E2._family(name, n, rng), rng)
                for q in range(n):
                    qc.append(UnitaryGate(random_unitary(2, seed=int(rng.integers(2**31)))), [q])
                qc.measure_all()
                circs.append((f"{name}{n}", qc))
    rows = []
    for dev in a.device:
        be = getattr(fake_provider, dev)()
        t = be.target
        cm = t.build_coupling_map()
        basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x")]
        for name, qc in circs:
            with contextlib.redirect_stdout(io.StringIO()):
                a12.compile_for_model_circuit(qc, cm, basis, target=t)  # warm-up
            rec = dict(device=dev, name=name)
            outs = {}
            for v, seeds in VARIANTS.items():
                t0 = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    o = a12.compile_for_model_circuit(qc, cm, basis, target=t, seeds=seeds)
                rec[f"t_{v}"] = time.perf_counter() - t0
                rec[f"cost_{v}"] = a12.state_aware_cost(o, t)
                outs[v] = o
            for v in ("s2", "s1"):
                rec[f"same_{v}"] = sig(outs[v]) == sig(outs["s4"])
            rows.append(rec)
        rs = [r for r in rows if r["device"] == dev]
        line = f"{dev}: {len(rs)} circuits, median s4 {np.median([r['t_s4'] for r in rs]):.3f} s"
        for v in ("s2", "s1"):
            tr = np.median([r[f"t_{v}"] / r["t_s4"] for r in rs])
            same = sum(r[f"same_{v}"] for r in rs)
            cr = [r[f"cost_{v}"] / r["cost_s4"] for r in rs if r["cost_s4"] > 0]
            worse = sum(1 for x in cr if x > 1 + 1e-9)
            better = sum(1 for x in cr if x < 1 - 1e-9)
            line += (f" | {v}: time {tr:.2f} x, same circuit {same}/{len(rs)}, estimate worse {worse} / better {better}, "
                     f"mean estimate ratio {np.mean(cr):.4f}, worst {max(cr):.3f}")
        print(line, flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(rows, open(os.path.join(a.out, "a12_seeds.json"), "w"))


if __name__ == "__main__":
    main()
