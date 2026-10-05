"""exact_eval.py -- pre-registered home test EXACT (2026-10-05): candidates psf_compile 2026-10-05.c12 (item 39,
equivalence check of Qiskit-made circuits) and psf_ai_compile 2026-10-05.a9 (item 14, the same for level 3's output in
the AI front end). Does the fix make the recommended calls exact on the near-boundary workloads where release
2026-10-04.1 was found wrong (Addendum 340), without changing anything where Qiskit's circuits are exact?

Part X (exactness), devices FakeAuckland, FakeHanoiV2, FakeAlgiers, FakeGeneva (cx) and FakeTorino, FakeKingston (cz);
  circuits from B17's generator code (benchmarks/b17_practice_eval.py, copied unchanged except the seed base, which is
  B17's + 2,000,000), n = 4 and 6, each with a seeded random single-qubit layer prepended (Addendum 340):
    X1 W1 near   Heisenberg Trotter cells (dt, r) = (1e-3, 1e-4), (1e-3, 1e-5), (1e-2, 1e-5): 8 seeds each
    X2 W1 ctrl   (dt, r) = (0.1, 1.0): 4 seeds
    X3 W2        four explicit near-boundary two-qubit unitaries between random single-qubit layers: 20
    X4 W3        the same with Haar two-qubit unitaries: 8
    X5 W4        small-angle ansatz, s = 1e-4 and 1e-3: 4 each
  (128 circuits per device). Arms:
    RPSF  release, target + placement_refine=True (2026-10-02.2's call: item 17's guard only)
    R41   release 2026-10-04.1's recommended call (+ final_resynthesis="select", compare_level3, compare_floor,
          candidate_score="hybrid")
    C12   the candidate with the same call
    L3T   Qiskit level 3 with the Target, approximation_degree 1.0
    A8    the adopted AI front end (with the release)
    A9    the candidate front end (with c12 registered as psf_compile)
  Metric: noiseless output-state infidelity on the touched qubits after undoing the final layout (the workplace
  probe's `state_infid`, Qiskit Statevector and partial_trace, independent of item 39's checks); wrong if > 1e-6.
Part Y (no change where Qiskit is exact), HOLD6's F circuits (benchmarks/hold6_eval.py, all 1,506 per device, nine
  devices): R41 and C12 compiled with the recommended call and compared instruction by instruction; C12's output
  checked for exactness as in X; A8 and A9 compared on every tenth circuit.

  python exact_eval.py x --repo <repo> --out <dir> --device <d> --arm <a> [--smoke]
  python exact_eval.py y --repo <repo> --out <dir> --device <d> [--smoke]
  python exact_eval.py score --out <dir>
"""
import argparse
import contextlib
import glob
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
X_DEVICES = ("FakeAuckland", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva", "FakeTorino", "FakeKingston")
CX = ("FakeAuckland", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva")
Y_DEVICES = ("FakeAuckland", "FakeTorino", "FakeKingston", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva", "FakeFez",
             "FakeMarrakesh", "FakeAachen")
X_ARMS = ("RPSF", "R41", "C12", "L3T", "A8", "A9")
CELLS = ("X1", "X2", "X3", "X4", "X5")
CAND = os.path.join("patches", "psf_compile_c12_2026-10-05", "psf_compile.py")
CAND_AI = os.path.join("patches", "psf_ai_compile_a9_2026-10-05", "psf_ai_compile.py")
WORKLOADS = ("W1", "W2", "W3", "W4")             # as in b17_practice_eval
SEED_SHIFT = 2_000_000
WRONG = 1e-6
FULL = dict(placement_refine=True, final_resynthesis="select", compare_level3=True, compare_floor=True,
            candidate_score="hybrid")


def norm_sha(path):
    with open(path, "rb") as f:
        txt = f.read().decode("utf-8").replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in txt.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def b17_circuit(workload, n, k_or_s, dt=None, r=None, sd=None):
    """One circuit of b17_practice_eval.circuits' code, with the seed base shifted by SEED_SHIFT. `k_or_s` is the
    generator's own counter (W1, W4) or seed (W2, W3)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator, random_unitary
    base = 100_000 * (WORKLOADS.index(workload) + 1) + 10_000 * n + SEED_SHIFT
    rng = np.random.default_rng(base + k_or_s)
    qc = QuantumCircuit(n)
    if workload == "W1":
        jx, jy = rng.uniform(0.5, 1.5, 2)
        h = rng.uniform(-1, 1, n)
        for _ in range(4):
            for start in (0, 1):
                for i in range(start, n - 1, 2):
                    qc.rxx(2 * jx * dt, i, i + 1)
                    qc.ryy(2 * jy * dt, i, i + 1)
                    qc.rzz(2 * r * jx * dt, i, i + 1)
            for i in range(n):
                qc.rz(2 * h[i] * dt, i)
    elif workload in ("W2", "W3"):
        for blk in range(4):
            for q in range(n):
                qc.append(UnitaryGate(random_unitary(2, seed=int(rng.integers(2**31)))), [q])
            p, q = (int(v) for v in rng.choice(n, 2, replace=False))
            if workload == "W3":
                u = random_unitary(4, seed=int(rng.integers(2**31))).data
            else:
                a, b = sorted(rng.uniform(0, math.pi / 4, 2), reverse=True)
                c = 10 ** rng.uniform(-10, -4)
                core = QuantumCircuit(2)
                core.rxx(-2 * a, 0, 1)
                core.ryy(-2 * b, 0, 1)
                core.rzz(-2 * c, 0, 1)
                k1 = random_unitary(2, seed=int(rng.integers(2**31))).tensor(random_unitary(2, seed=int(rng.integers(2**31))))
                k2 = random_unitary(2, seed=int(rng.integers(2**31))).tensor(random_unitary(2, seed=int(rng.integers(2**31))))
                u = (k1 @ Operator(core) @ k2).data
            qc.append(UnitaryGate(u), [p, q])
    else:  # W4: b17's small-angle ansatz
        for _ in range(3):
            for q in range(n):
                qc.ry(float(rng.normal(0, sd)), q)
                qc.rz(float(rng.normal(0, sd)), q)
            for q in range(n - 1):
                qc.cx(q, q + 1)
    return qc


def x_circuits(smoke):
    """Yields (cell, params, circuit with its random layer)."""
    k = 0
    for n in (4, 6):
        cells = [("X1", "W1", dict(dt=1e-3, r=1e-4), 8), ("X1", "W1", dict(dt=1e-3, r=1e-5), 8),
                 ("X1", "W1", dict(dt=1e-2, r=1e-5), 8), ("X2", "W1", dict(dt=0.1, r=1.0), 4),
                 ("X3", "W2", {}, 20), ("X4", "W3", {}, 8), ("X5", "W4", dict(sd=1e-4), 4), ("X5", "W4", dict(sd=1e-3), 4)]
        for cell, wl, prm, size in cells:
            for s in range(1 if smoke else size):
                k += 1
                qc = b17_circuit(wl, n, k + (50_000 if smoke else 0), **prm)   # k: a running counter, so every circuit has its own seed
                yield cell, dict(n=n, workload=wl, seed=s, **prm), with_prep(qc, 9_000_000 + k + (500_000 if smoke else 0))


def with_prep(qc, seed):
    """The workplace probe's construction (Addendum 340)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    rng = np.random.default_rng(seed)
    out = QuantumCircuit(qc.num_qubits)
    for q in range(qc.num_qubits):
        out.append(UnitaryGate(random_unitary(2, seed=int(rng.integers(2**31)))), [q])
    out.compose(qc, inplace=True)
    return out


def state_infid(qc, out):
    """The workplace probe's check (Addendum 340), independent of item 39's code."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity
    ideal = Statevector(qc)
    active = sorted({out.find_bit(b).index for ins in out.data for b in ins.qubits})
    fin = out.layout.final_index_layout(filter_ancillas=True) if out.layout is not None else list(range(qc.num_qubits))
    active = sorted(set(active) | set(fin))
    idx = {p: i for i, p in enumerate(active)}
    red = QuantumCircuit(len(active))
    for ins in out.data:
        if ins.operation.name in ("barrier", "measure", "delay"):
            continue
        red.append(ins.operation, [idx[out.find_bit(b).index] for b in ins.qubits])
    sv = Statevector(red)
    keep = [idx[p] for p in fin]
    trace_out = [i for i in range(len(active)) if i not in keep]
    rho = partial_trace(sv, trace_out) if trace_out else sv
    order = sorted(keep)
    perm = [order.index(k) for k in keep]
    rho = DensityMatrix(rho)
    n = len(keep)
    t = rho.data.reshape([2] * (2 * n))
    row_axes = [n - 1 - perm[v] for v in range(n)][::-1]
    col_axes = [2 * n - 1 - perm[v] for v in range(n)][::-1]
    t = np.transpose(t, row_axes + col_axes).reshape(2 ** n, 2 ** n)
    return float(1 - state_fidelity(DensityMatrix(t), ideal))


def sig(c):
    return [(i.operation.name, [c.find_bit(q).index for q in i.qubits], [float(p) for p in i.operation.params])
            for i in c.data]


def load(args, release_as_pc):
    """Loads the release, the candidate, the layout search and the front ends. release_as_pc chooses which module
    the front ends see as `psf_compile` ("rel" or "c12")."""
    sys.path.insert(0, os.path.join(args.repo, "benchmarks"))
    sys.path.insert(0, args.repo)
    import core_fix_c2_eval as H
    lay = H.load_module(os.path.join(args.repo, "benchmarks", "psf_smart_layout.py"), "psl_c2")
    sys.modules["psf_smart_layout"] = lay
    c12 = H.load_module(os.path.join(args.repo, CAND), "psf_compile_c12")
    rel = H.load_module(os.path.join(args.repo, "psf_compile.py"), "psf_compile_rel")
    if rel.VERSION != "2026-10-04.1":
        raise SystemExit("STOP: psf_compile.py is %s, not release 2026-10-04.1" % rel.VERSION)
    sys.modules["psf_compile"] = c12 if release_as_pc == "c12" else rel
    a8 = H.load_module(os.path.join(args.repo, "benchmarks", "psf_ai_compile.py"), "psf_ai_compile_a8")
    a9 = H.load_module(os.path.join(args.repo, CAND_AI), "psf_ai_compile_a9")
    return H, rel, c12, a8, a9, lay


def meta_for(args, rel, c12, a8, a9, lay, extra):
    import psf_zero_core
    import qiskit
    return dict(release=rel.VERSION, c12=c12.VERSION, a8=a8.AI_COMPILE_VERSION, a9=a9.AI_COMPILE_VERSION,
                layout=lay.LAYOUT_VERSION, core=getattr(psf_zero_core, "CORE_VERSION", None), qiskit=qiskit.__version__,
                git_head=subprocess.run(["git", "-C", args.repo, "rev-parse", "--short=7", "HEAD"], capture_output=True,
                                        text=True).stdout.strip(),
                sha=dict(script=norm_sha(os.path.abspath(__file__)), release=norm_sha(os.path.join(args.repo, "psf_compile.py")),
                         c12=norm_sha(os.path.join(args.repo, CAND)), a9=norm_sha(os.path.join(args.repo, CAND_AI)),
                         a8=norm_sha(os.path.join(args.repo, "benchmarks", "psf_ai_compile.py"))),
                device=args.device, smoke=args.smoke, started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra)


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **k)


def run_x(args):
    from qiskit import transpile
    from qiskit_ibm_runtime import fake_provider
    H, rel, c12, a8, a9, lay = load(args, "c12" if args.arm == "A9" else "rel")
    tgt = getattr(fake_provider, args.device)().target
    cm = tgt.build_coupling_map()
    nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
    base = dict(coupling_map=cm, basis_gates=nat, entangling_basis="cx", layout_search=True, seed_transpiler=0, target=tgt)
    f = {"RPSF": lambda qc: rel.compile_for_hardware(qc, placement_refine=True, **base),
         "R41": lambda qc: rel.compile_for_hardware(qc, **FULL, **base),
         "C12": lambda qc: c12.compile_for_hardware(qc, **FULL, **base),
         "L3T": lambda qc: transpile(qc, target=tgt, optimization_level=3, seed_transpiler=0, approximation_degree=1.0),
         "A8": lambda qc: a8.compile_for_model_circuit(qc, cm, nat, target=tgt),
         "A9": lambda qc: a9.compile_for_model_circuit(qc, cm, nat, target=tgt)}[args.arm]
    meta = meta_for(args, rel, c12, a8, a9, lay, dict(part="x", arm=args.arm))
    rows, t0 = [], time.time()
    for cell, params, qc in x_circuits(args.smoke):
        t1 = time.perf_counter()
        try:
            out = quiet(f, qc)
            err = None
        except Exception as e:  # recorded, scored as a compile error
            out, err = None, repr(e)[:200]
        tc = time.perf_counter() - t1
        row = dict(cell=cell, params=params, compile_s=tc, error=err)
        if out is not None:
            row.update(infid=state_infid(qc, out), two_q=sum(1 for i in out.data if len(i.qubits) == 2))
        rows.append(row)
    stats = dict(exact=dict(c12.EXACT_STATS), l3t_check=dict(a9.L3T_CHECK_STATS))
    name = "exact_x_%s_%s%s.json" % (args.device, args.arm, "_smoke" if args.smoke else "")
    json.dump(dict(meta=meta, rows=rows, stats=stats, wall_s=time.time() - t0), open(os.path.join(args.out, name), "w"))
    print("wrote %s: %d circuits, %d wrong, %.0f s" % (name, len(rows), sum(1 for r in rows if r.get("infid", 1) > WRONG),
                                                      time.time() - t0), flush=True)


def run_y(args):
    from qiskit_ibm_runtime import fake_provider
    H, rel, c12, a8, a9, lay = load(args, "rel")
    gen = H.load_module(os.path.join(args.repo, "benchmarks", "hold6_eval.py"), "hold6_eval")
    tgt = getattr(fake_provider, args.device)().target
    cm = tgt.build_coupling_map()
    nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
    base = dict(coupling_map=cm, basis_gates=nat, entangling_basis="cx", layout_search=True, seed_transpiler=0, target=tgt)
    meta = meta_for(args, rel, c12, a8, a9, lay, dict(part="y"))
    rows, t0, j = [], time.time(), 0
    for fam in ("F1", "F2", "F3", "F4", "F5", "F6"):
        for i, (params, qc) in enumerate(gen.family(fam, args.smoke)):
            t1 = time.perf_counter()
            r = quiet(rel.compile_for_hardware, qc, **FULL, **base)
            t2 = time.perf_counter()
            c = quiet(c12.compile_for_hardware, qc, **FULL, **base)
            t3 = time.perf_counter()
            row = dict(family=fam, index=i, n=qc.num_qubits, same=sig(r) == sig(c), r41_s=t2 - t1, c12_s=t3 - t2,
                       c12_infid=state_infid(qc, c))
            if j % 10 == 0:
                # a8 with the release, a9 with c12 (as after adoption)
                a8.pc = rel
                x = quiet(a8.compile_for_model_circuit, qc, cm, nat, target=tgt)
                a9.pc = c12
                y = quiet(a9.compile_for_model_circuit, qc, cm, nat, target=tgt)
                row.update(ai_same=sig(x) == sig(y), a9_infid=state_infid(qc, y))
            j += 1
            rows.append(row)
        print("  %s %s: %d rows, %.0f s" % (args.device, fam, len(rows), time.time() - t0), flush=True)
    stats = dict(exact=dict(c12.EXACT_STATS), l3t_check=dict(a9.L3T_CHECK_STATS))
    name = "exact_y_%s%s.json" % (args.device, "_smoke" if args.smoke else "")
    json.dump(dict(meta=meta, rows=rows, stats=stats, wall_s=time.time() - t0), open(os.path.join(args.out, name), "w"))
    print("wrote %s: %d circuits, %d differ, %.0f s" % (name, len(rows), sum(1 for r in rows if not r["same"]),
                                                       time.time() - t0), flush=True)


def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def score(args):
    X, Y, smoke = {}, {}, None
    for p in sorted(glob.glob(os.path.join(args.out, "exact_x_*.json"))):
        r = json.load(open(p))
        X[(r["meta"]["device"], r["meta"]["arm"])] = r
        smoke = r["meta"]["smoke"]
    for p in sorted(glob.glob(os.path.join(args.out, "exact_y_*.json"))):
        r = json.load(open(p))
        Y[r["meta"]["device"]] = r
    missing = [(d, a) for d in X_DEVICES for a in X_ARMS if (d, a) not in X] + [d for d in Y_DEVICES if d not in Y]
    errs = sum(1 for r in X.values() for x in r["rows"] if x["error"])
    L = [f"# exact score{' (SMOKE -- not a result)' if smoke else ''}", "",
         f"P0: {'PASS' if not missing and not errs else 'FAIL'} -- files {len(X)} of {len(X_DEVICES) * len(X_ARMS)} (X), "
         f"{len(Y)} of {len(Y_DEVICES)} (Y); missing {missing[:5]}; compile errors {errs}", ""]

    def wrong(d, a, cells=CELLS):
        rs = [x for x in X.get((d, a), {"rows": []})["rows"] if x["cell"] in cells]
        return sum(1 for x in rs if x.get("infid", 1.0) > WRONG), len(rs), max((x.get("infid", 1.0) for x in rs), default=0.0)

    L += ["| device | cell | " + " | ".join(X_ARMS) + " |", "|---|---|" + "---|" * len(X_ARMS)]
    for d in X_DEVICES:
        for c in CELLS:
            L.append(f"| {d}{' (cx)' if d in CX else ''} | {c} | " + " | ".join(
                "%d/%d%s" % (w, n, " (%.1e)" % m if w else "") for w, n, m in (wrong(d, a, (c,)) for a in X_ARMS)) + " |")
    e1 = sum(wrong(d, "C12")[0] for d in X_DEVICES)
    e2 = sum(wrong(d, "A9")[0] for d in X_DEVICES)
    e3v = {d: wrong(d, "R41", ("X3",))[0] for d in CX}
    e3 = verdict(sum(v >= 1 for v in e3v.values()) >= 3, all(v == 0 for v in e3v.values()))
    e4 = sum(wrong(d, "R41")[0] for d in X_DEVICES if d not in CX)
    e5 = sum(wrong(d, "RPSF")[0] for d in X_DEVICES)
    same = {d: np.mean([x["same"] for x in Y[d]["rows"]]) for d in Y_DEVICES if d in Y}
    e6 = verdict(all(v >= 0.995 for v in same.values()) and len(same) == len(Y_DEVICES), any(v < 0.98 for v in same.values()))
    ais = {d: np.mean([x["ai_same"] for x in Y[d]["rows"] if "ai_same" in x]) for d in Y_DEVICES if d in Y}
    e7 = verdict(all(v >= 0.99 for v in ais.values()) and len(ais) == len(Y_DEVICES), any(v < 0.95 for v in ais.values()))
    e8n = sum(1 for d in Y for x in Y[d]["rows"] if x["c12_infid"] > WRONG)
    e8 = verdict(e8n == 0 and len(Y) == len(Y_DEVICES), e8n > 0)
    mr = float(np.median([x["r41_s"] for d in Y for x in Y[d]["rows"]])) if Y else float("nan")
    mc = float(np.median([x["c12_s"] for d in Y for x in Y[d]["rows"]])) if Y else float("nan")
    e9 = verdict(mc <= 1.2 * mr, mc > 2 * mr)
    fmt = lambda dct: ", ".join(f"{k.replace('Fake', '')} {v:.4f}" for k, v in dct.items())
    L += ["", "## Predictions", "",
          f"- E1 (C12 never wrong, part X): **{verdict(e1 == 0, e1 > 0)}** ({e1} wrong)",
          f"- E2 (A9 never wrong, part X): **{verdict(e2 == 0, e2 > 0)}** ({e2} wrong)",
          f"- E3 (the defect reproduces: R41 wrong on W2 on at least 3 of 4 cx devices): **{e3}** ({e3v})",
          f"- E4 (cz devices unaffected: R41 never wrong on FakeTorino, FakeKingston): **{verdict(e4 == 0, e4 > 0)}** ({e4} wrong)",
          f"- E5 (the guarded call RPSF never wrong): **{verdict(e5 == 0, e5 > 0)}** ({e5} wrong)",
          f"- E6 (C12 identical to R41 on >= 99.5% of HOLD6's F circuits on every device): **{e6}** ({fmt(same)})",
          f"- E7 (A9 identical to A8 on >= 99% of the sampled F circuits on every device): **{e7}** ({fmt(ais)})",
          f"- E8 (C12 exact on every HOLD6 F circuit): **{e8}** ({e8n} wrong)",
          f"- E9 (median compile time C12 <= 1.2 x R41 on the F circuits): **{e9}** ({mc:.3f} s vs {mr:.3f} s)"]
    L += ["", "Reported: A8 wrong by device " + str({d: wrong(d, "A8")[:2] for d in X_DEVICES}),
          "Reported: L3T wrong by device " + str({d: wrong(d, "L3T")[:2] for d in X_DEVICES}),
          "Reported: C12 EXACT_STATS by device (part X) " + str({d: X[(d, "C12")]["stats"]["exact"] for d in X_DEVICES if (d, "C12") in X}),
          "Reported: A9 L3T_CHECK_STATS by device (part X) " + str({d: X[(d, "A9")]["stats"]["l3t_check"] for d in X_DEVICES if (d, "A9") in X}),
          "Reported: C12 EXACT_STATS by device (part Y) " + str({d: Y[d]["stats"]["exact"] for d in Y}),
          "Reported: A9 exact on the sampled F circuits: %d wrong" % sum(1 for d in Y for x in Y[d]["rows"] if x.get("a9_infid", 0) > WRONG)]
    txt = "\n".join(L) + "\n"
    open(os.path.join(args.out, "score.md"), "w").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("x", "y", "score"))
    ap.add_argument("--repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=sorted(set(X_DEVICES) | set(Y_DEVICES)))
    ap.add_argument("--arm", choices=X_ARMS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dict(x=run_x, y=run_y, score=score)[args.part](args)


if __name__ == "__main__":
    main()
