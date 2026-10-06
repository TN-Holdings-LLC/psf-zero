"""a12_dups.py -- exploratory (not pre-registered), 2026-10-06: how many of the AI front end a12's release calls and
polish calls (benchmarks/psf_ai_compile.py) repeat an earlier call of the same compile with exactly the same input?

Circuits: MODEL-RO2's model-style generator (ai10_eval2._family, _model_style; Addendum 352), 2 per (family, n) cell,
32 circuits, at new exploratory seeds 75,000,000 + k, each with measure_all(). Devices given on the command line.
For every circuit a11 runs once with its phases timed by wrapping (the code itself is not changed):
  cfh       psf_compile.compile_for_hardware calls from the start x seed loop and the level-3-layout candidates
  polish    polish() of each candidate
  l3        the two Qiskit level-3 transpiles (layout candidates and a7's output)
  place     best_placement_state_aware() of the top candidates (state-aware re-placement)
  check     psf_compile._implements (a9's check of level 3's output) and _keep_direction (a11's backstop)
  other     the rest (start-point preparation, sorting)
and whether the final circuit is PSF-Zero's or level 3's output.

    python a11_profile.py --out DIR --device FakeTorino [--device FakeKingston ...]
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
import warnings
from collections import Counter, defaultdict

import numpy as np

warnings.simplefilter("ignore")
REPO = os.getcwd()
WORK = os.path.join(REPO, "data", "2026-10-05", "workplace")
for p in (os.path.join(REPO, "benchmarks"), REPO):
    sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", action="append", required=True)
    ap.add_argument("--per-cell", type=int, default=2)
    a = ap.parse_args()
    import core_fix_c2_eval as H
    lay = H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile")  # registered before anything imports it
    a11 = H.load_module(os.path.join(REPO, "benchmarks", "psf_ai_compile.py"), "psf_ai_compile_profile")
    E2 = H.load_module(os.path.join(WORK, "model_ro2", "ai10_eval2.py"), "ai10_eval2_profile")
    assert rel.VERSION == "2026-10-06.1" and a11.AI_COMPILE_VERSION == "2026-10-06.a12"
    import qiskit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    from qiskit_ibm_runtime import fake_provider

    circs, k = [], 0
    for name, ns in E2.FAMILIES:
        for n in ns:
            for _ in range(a.per_cell):
                rng = np.random.default_rng(77_000_000 + k)
                k += 1
                qc = E2._model_style(E2._family(name, n, rng), rng)
                for q in range(n):
                    qc.append(UnitaryGate(random_unitary(2, seed=int(rng.integers(2**31)))), [q])
                qc.measure_all()
                circs.append((f"{name}{n}", qc))

    T = defaultdict(float)
    C = Counter()
    phase = []   # stack of phase names, so nested calls are charged once (to the outer phase)

    def timed(name, f, label=None):
        def w(*args, **kw):
            if phase:
                return f(*args, **kw)
            phase.append(name)
            t = time.perf_counter()
            try:
                return f(*args, **kw)
            finally:
                T[name] += time.perf_counter() - t
                C[name] += 1
                phase.pop()
        return w

    # wrap: release calls, polish, placement, checks, level-3 transpiles; and record each call's exact input
    seen = {}

    def keyed(name, f):
        def w(circ, *args, **kw):
            if phase != [name]:  # only the top-level call of this phase (timed() has pushed it)
                return f(circ, *args, **kw)
            try:
                k = (name, repr([(i.operation.name, [circ.find_bit(q).index for q in i.qubits],
                                  [repr(p) if not hasattr(p, "shape") else p.tobytes() for p in i.operation.params])
                                 for i in circ.data]), repr(circ.global_phase), repr(args),
                     repr(sorted((x, repr(y)) for x, y in kw.items() if x not in ("target", "coupling_map"))))
            except Exception:
                k = None
            t = time.perf_counter()
            out = f(circ, *args, **kw)
            dt = time.perf_counter() - t
            if k is not None:
                if k in seen:
                    T["dup_" + name] += dt
                    C["dup_" + name] += 1
                seen[k] = True
            return out
        return w

    orig_cfh = rel.compile_for_hardware
    rel.compile_for_hardware = timed("cfh", keyed("cfh", orig_cfh))
    a11.polish = timed("polish", keyed("polish", a11.polish))
    a11.best_placement_state_aware = timed("place", a11.best_placement_state_aware)
    rel._implements = timed("check", rel._implements)
    a11._keep_direction = timed("check", a11._keep_direction)
    qiskit.transpile = timed("l3", qiskit.transpile)

    rows = []
    for dev in a.device:
        be = getattr(fake_provider, dev)()
        t = be.target
        cm = t.build_coupling_map()
        basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x")]
        for name, qc in circs:
            T.clear()
            C.clear()
            seen.clear()
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                out, info = a11.compile_for_model_circuit(qc, cm, basis, target=t, return_info=True)
            total = time.perf_counter() - t0
            rec = dict(device=dev, name=name, n=qc.num_qubits, total=round(total, 4),
                       phases={p: round(v, 4) for p, v in T.items()}, calls=dict(C),
                       tried=[list(map(str, x)) for x in info.get("tried", [])], chosen=info.get("chosen"),
                       best=list(info.get("best") or []))
            rec["phases"]["other"] = round(total - sum(v for p, v in T.items() if not p.startswith("dup_")), 4)
            rows.append(rec)
            print(f"{dev} {name}: {total:.3f} s  " + "  ".join(f"{p} {v:.3f}" for p, v in rec["phases"].items())
                  + f"  cfh calls {C['cfh']}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(rows=rows, qiskit=qiskit.__version__), open(os.path.join(a.out, "a12_dups.json"), "w"))
    print()
    for dev in a.device:
        rs = [r for r in rows if r["device"] == dev]
        tot = sum(r["total"] for r in rs)
        ph = Counter()
        for r in rs:
            ph.update(r["phases"])
        calls = sum(r["calls"].get("cfh", 0) for r in rs)
        dups = sum(r["calls"].get("dup_cfh", 0) for r in rs)
        pd = sum(r["calls"].get("dup_polish", 0) for r in rs)
        pc_ = sum(r["calls"].get("polish", 0) for r in rs)
        print(f"{dev}: {len(rs)} circuits, median {np.median([r['total'] for r in rs]):.3f} s, total {tot:.1f} s; "
              + ", ".join(f"{p} {100 * v / tot:.0f}%" for p, v in ph.most_common() if not p.startswith("dup_"))
              + f"; release calls per circuit {calls / len(rs):.1f}, repeats {dups} ({100 * ph['dup_cfh'] / tot:.0f}% of time)"
              + f"; polish repeats {pd} of {pc_} ({100 * ph['dup_polish'] / tot:.0f}% of time)")


if __name__ == "__main__":
    main()
