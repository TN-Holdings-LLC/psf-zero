"""hold6_eval.py -- pre-registered home test HOLD6 (2026-10-04): candidate psf_compile 2026-10-04.c11 (choice by
`hybrid_cost`, changelog item 38) and candidate psf_ai_compile 2026-10-04.a8 (target-aware path above 8 qubits, its
item 13), on fresh circuits.

Circuits (new seeds):
  F1-F6  HOLD's six families with HOLD5's generator code and per-cell sizes (1,506 per device), seed base 90,000,000
         + ...; at most 11 touched qubits simulated
  W1-W6  WIDE's generator code at 9-10 logical qubits: W1 rings n 10 (L 2, 4), W2 QAOA n 10 (p 1, 2), W3 XXZ n 10
         (open, periodic), W4 brickwork n 9 and 10, W5 GHZ n 9 and 10, W6 QFT n 9; 12 per family (72 per device),
         seed base 95,000,000 + ...; at most 12 touched qubits simulated
  The smoke run uses 1 circuit per cell and its own seeds.
Arms:
  R3   release 2026-10-03.3 as recommended: cx devices .2's call + compare_floor=True, candidate_score="pauli";
       cz devices .2's call (target, placement_refine, final_resynthesis="select", compare_level3=True)
  C11  the candidate with compare_floor=True, candidate_score="hybrid" (+ .2's call), on every device
  A7   the adopted AI front end 2026-10-02.a7 with the target
  A8   the candidate front end 2026-10-04.a8 with the target (W families only)
  L3T  Qiskit transpile with the Target, optimization_level 3, approximation_degree 1.0
Devices: HOLD's nine. Metric as in GAP. Noiseless P0 check with Aer's statevector method.

  python hold6_eval.py run --repo <repo> --out <dir> --device <d> --arm <a> --family <F1..F6|W1..W6> [--smoke]
  python hold6_eval.py score --out <dir>
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
ARMS = ("R3", "C11", "A7", "A8", "L3T")
F_ARMS = ("R3", "C11", "A7", "L3T")
CX = ("FakeAuckland", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva")
CAND = os.path.join("patches", "psf_compile_c11_2026-10-04", "psf_compile.py")
CAND_AI = os.path.join("patches", "psf_ai_compile_a8_2026-10-04", "psf_ai_compile.py")
SEEN = ("FakeAuckland", "FakeTorino", "FakeKingston")
NEW = ("FakeHanoiV2", "FakeAlgiers", "FakeGeneva", "FakeFez", "FakeMarrakesh", "FakeAachen")
DEVICES = SEEN + NEW
CZ = ("FakeTorino", "FakeKingston", "FakeFez", "FakeMarrakesh", "FakeAachen")
FAMILIES = ("F1", "F2", "F3", "F4", "F5", "F6")
W_FAMILIES = ("W1", "W2", "W3", "W4", "W5", "W6")
SIZES = dict(F1=72, F2=120, F3=150, F4=90, F5=48, F6=40)        # per cell; twice GAP's per-cell sizes for F1-F5
SMOKE_SIZES = dict(F1=1, F2=1, F3=1, F4=1, F5=1, F6=1)
BASE = 90_000_000
MAX_ACTIVE = 11
W_SIZES = dict(W1=6, W2=6, W3=6, W4=6, W5=6, W6=12)          # per (n, variant)
W_SMOKE_SIZES = dict(W1=1, W2=1, W3=1, W4=1, W5=1, W6=1)
W_BASE = 95_000_000
W_MAX_ACTIVE = 12


def norm_sha(path):
    with open(path, "rb") as f:
        txt = f.read().decode("utf-8").replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in txt.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def regular3(n, rng):
    """A random simple 3-regular graph on n nodes (configuration model with rejection); gap_eval.regular3, copied."""
    while True:
        stubs = np.repeat(np.arange(n), 3)
        rng.shuffle(stubs)
        edges = {tuple(sorted((int(stubs[i]), int(stubs[i + 1])))) for i in range(0, len(stubs), 2)}
        if len(edges) == len(stubs) // 2 and all(a != b for a, b in edges):
            return sorted(edges)


def family(fam, smoke):
    """Yields (params, circuit). F1-F5: gap_eval.family's code with a new seed base and larger sizes; F6: QFT."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    size = (SMOKE_SIZES if smoke else SIZES)[fam]
    base = BASE + 1_000_000 * (FAMILIES.index(fam) + 1) + (500_000 if smoke else 0)
    k = 0

    def rng_next():
        nonlocal k
        k += 1
        return np.random.default_rng(base + k)

    if fam == "F1":
        for n in (4, 6):
            for L in (2, 4, 6):
                for s in range(size):
                    rng = rng_next()
                    x, v = rng.uniform(-1, 1, n), rng.uniform(-math.pi, math.pi, (L, n, 2))
                    fin = rng.uniform(-math.pi, math.pi, n)
                    qc = QuantumCircuit(n)
                    for l in range(L):
                        for q in range(n):
                            qc.ry(math.pi * float(x[q]), q)
                        for q in range(n):
                            qc.ry(float(v[l, q, 0]), q)
                            qc.rz(float(v[l, q, 1]), q)
                        for q in range(n):
                            qc.cz(q, (q + 1) % n)
                    for q in range(n):
                        qc.ry(float(fin[q]), q)
                    yield dict(n=n, L=L, seed=s), qc
    elif fam == "F2":
        n = 6
        for p in (1, 2):
            for s in range(size):
                rng = rng_next()
                edges = regular3(n, rng)
                gam, bet = rng.uniform(0, math.pi, p), rng.uniform(0, math.pi, p)
                qc = QuantumCircuit(n)
                qc.h(range(n))
                for j in range(p):
                    for a, b in edges:
                        qc.rzz(2 * float(gam[j]), a, b)
                    for q in range(n):
                        qc.rx(2 * float(bet[j]), q)
                yield dict(n=n, p=p, seed=s), qc
    elif fam == "F3":
        n = 6
        for bc in ("o", "p"):
            for s in range(size):
                rng = rng_next()
                jx, jy = rng.uniform(0.5, 1.5, 2)
                jz = rng.uniform(0.2, 1.0) * jx
                h = rng.uniform(-1, 1, n)
                bonds = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if bc == "p" else [])
                qc = QuantumCircuit(n)
                for _ in range(4):
                    for parity in (0, 1):
                        for a, b in bonds:
                            if a % 2 == parity:
                                qc.rxx(2 * jx * 0.1, a, b)
                                qc.ryy(2 * jy * 0.1, a, b)
                                qc.rzz(2 * jz * 0.1, a, b)
                    for q in range(n):
                        qc.rz(2 * float(h[q]) * 0.1, q)
                yield dict(n=n, bc=bc, seed=s), qc
    elif fam == "F4":
        for n in (4, 5, 6):
            for s in range(size):
                rng = rng_next()
                qc = QuantumCircuit(n)
                for _ in range(n):
                    perm = rng.permutation(n)
                    for i in range(0, n - 1, 2):
                        u = random_unitary(4, seed=int(rng.integers(2**31)))
                        qc.append(UnitaryGate(u), [int(perm[i]), int(perm[i + 1])])
                yield dict(n=n, seed=s), qc
    elif fam == "F5":
        for n in (4, 6, 8):
            for s in range(size):
                rng = rng_next()
                qc = QuantumCircuit(n)
                qc.h(0)
                for q in range(n - 1):
                    qc.cx(q, q + 1)
                for q in range(n):
                    qc.ry(float(rng.uniform(-math.pi, math.pi)), q)
                    qc.rz(float(rng.uniform(-math.pi, math.pi)), q)
                yield dict(n=n, seed=s), qc
    elif fam == "F6":
        for n in (4, 5, 6):
            for s in range(size):
                rng = rng_next()
                qc = QuantumCircuit(n)
                for q in range(n):
                    qc.ry(float(rng.uniform(-math.pi, math.pi)), q)
                    qc.rz(float(rng.uniform(-math.pi, math.pi)), q)
                for j in range(n):
                    qc.h(j)
                    for m in range(j + 1, n):
                        qc.cp(math.pi / 2 ** (m - j), m, j)
                for i in range(n // 2):
                    qc.swap(i, n - 1 - i)
                yield dict(n=n, seed=s), qc


def family_w(fam, smoke):
    """Yields (params, circuit) for W1-W6: wide_eval.family's code at 9-10 logical qubits, own seed base."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    size = (W_SMOKE_SIZES if smoke else W_SIZES)[fam]
    base = W_BASE + 1_000_000 * (W_FAMILIES.index(fam) + 1) + (500_000 if smoke else 0)
    k = 0

    def rng_next():
        nonlocal k
        k += 1
        return np.random.default_rng(base + k)

    if fam == "W1":
        for n in (10,):
            for L in (2, 4):
                for s in range(size):
                    rng = rng_next()
                    x, v = rng.uniform(-1, 1, n), rng.uniform(-math.pi, math.pi, (L, n, 2))
                    fin = rng.uniform(-math.pi, math.pi, n)
                    qc = QuantumCircuit(n)
                    for l in range(L):
                        for q in range(n):
                            qc.ry(math.pi * float(x[q]), q)
                        for q in range(n):
                            qc.ry(float(v[l, q, 0]), q)
                            qc.rz(float(v[l, q, 1]), q)
                        for q in range(n):
                            qc.cz(q, (q + 1) % n)
                    for q in range(n):
                        qc.ry(float(fin[q]), q)
                    yield dict(n=n, L=L, seed=s), qc
    elif fam == "W2":
      for n in (10,):
        for p in (1, 2):
            for s in range(size):
                rng = rng_next()
                edges = regular3(n, rng)
                gam, bet = rng.uniform(0, math.pi, p), rng.uniform(0, math.pi, p)
                qc = QuantumCircuit(n)
                qc.h(range(n))
                for j in range(p):
                    for a, b in edges:
                        qc.rzz(2 * float(gam[j]), a, b)
                    for q in range(n):
                        qc.rx(2 * float(bet[j]), q)
                yield dict(n=n, p=p, seed=s), qc
    elif fam == "W3":
      for n in (10,):
        for bc in ("o", "p"):
            for s in range(size):
                rng = rng_next()
                jx, jy = rng.uniform(0.5, 1.5, 2)
                jz = rng.uniform(0.2, 1.0) * jx
                h = rng.uniform(-1, 1, n)
                bonds = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if bc == "p" else [])
                qc = QuantumCircuit(n)
                for _ in range(4):
                    for parity in (0, 1):
                        for a, b in bonds:
                            if a % 2 == parity:
                                qc.rxx(2 * jx * 0.1, a, b)
                                qc.ryy(2 * jy * 0.1, a, b)
                                qc.rzz(2 * jz * 0.1, a, b)
                    for q in range(n):
                        qc.rz(2 * float(h[q]) * 0.1, q)
                yield dict(n=n, bc=bc, seed=s), qc
    elif fam == "W4":
        for n in (9, 10):
            for s in range(size):
                rng = rng_next()
                qc = QuantumCircuit(n)
                for _ in range(n):
                    perm = rng.permutation(n)
                    for i in range(0, n - 1, 2):
                        u = random_unitary(4, seed=int(rng.integers(2**31)))
                        qc.append(UnitaryGate(u), [int(perm[i]), int(perm[i + 1])])
                yield dict(n=n, seed=s), qc
    elif fam == "W5":
        for n in (9, 10):
            for s in range(size):
                rng = rng_next()
                qc = QuantumCircuit(n)
                qc.h(0)
                for q in range(n - 1):
                    qc.cx(q, q + 1)
                for q in range(n):
                    qc.ry(float(rng.uniform(-math.pi, math.pi)), q)
                    qc.rz(float(rng.uniform(-math.pi, math.pi)), q)
                yield dict(n=n, seed=s), qc
    elif fam == "W6":
        for n in (9,):
            for s in range(size):
                rng = rng_next()
                qc = QuantumCircuit(n)
                for q in range(n):
                    qc.ry(float(rng.uniform(-math.pi, math.pi)), q)
                    qc.rz(float(rng.uniform(-math.pi, math.pi)), q)
                for j in range(n):
                    qc.h(j)
                    for m in range(j + 1, n):
                        qc.cp(math.pi / 2 ** (m - j), m, j)
                for i in range(n // 2):
                    qc.swap(i, n - 1 - i)
                yield dict(n=n, seed=s), qc


def run(args):
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime import fake_provider
    sys.path.insert(0, os.path.join(args.repo, "benchmarks"))
    sys.path.insert(0, args.repo)
    import core_fix_c2_eval as H
    rel = H.load_module(os.path.join(args.repo, "psf_compile.py"), "psf_compile")
    if rel.VERSION != "2026-10-03.3":
        raise SystemExit("STOP: psf_compile.py is %s, not release 2026-10-03.3" % rel.VERSION)
    c11 = H.load_module(os.path.join(args.repo, CAND), "psf_compile_c11")
    lay = H.load_module(os.path.join(args.repo, "benchmarks", "psf_smart_layout.py"), "psl_c2")
    sys.modules["psf_smart_layout"] = lay
    a7 = H.load_module(os.path.join(args.repo, "benchmarks", "psf_ai_compile.py"), "psf_ai_compile")
    a8 = H.load_module(os.path.join(args.repo, CAND_AI), "psf_ai_compile_a8")
    wset = args.family.startswith("W")
    if args.arm == "A8" and not wset:
        raise SystemExit("STOP: arm A8 runs on the W families only")
    import psf_zero_core
    import qiskit
    backend = getattr(fake_provider, args.device)()
    tgt = backend.target
    cm = tgt.build_coupling_map()
    nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
    g2 = next(g for g in ("cz", "ecr", "cx") if g in tgt.operation_names)
    failed = {tuple(q) for q, p in tgt[g2].items() if p is not None and p.error is not None and p.error >= 0.5}
    failed_q = {q[0] for q, p in tgt["sx"].items() if p is not None and p.error is not None and p.error >= 0.5}
    opts = dict(method="density_matrix", max_parallel_threads=1, max_parallel_experiments=1)
    sims = {"noisy": AerSimulator(noise_model=NoiseModel.from_backend(backend), **opts),
            "ideal": AerSimulator(**dict(opts, method="statevector"))}   # the P0 check only
    meta = dict(release=rel.VERSION, c11=c11.VERSION, a7=a7.AI_COMPILE_VERSION, a8=a8.AI_COMPILE_VERSION,
                layout=lay.LAYOUT_VERSION,
                core=getattr(psf_zero_core, "CORE_VERSION", None), qiskit=qiskit.__version__, python=sys.version.split()[0],
                git_head=subprocess.run(["git", "-C", args.repo, "rev-parse", "--short=7", "HEAD"], capture_output=True,
                                        text=True).stdout.strip(),
                sha=dict(script=norm_sha(os.path.abspath(__file__)),
                         release=norm_sha(os.path.join(args.repo, "psf_compile.py")),
                         c11=norm_sha(os.path.join(args.repo, CAND)),
                         a8=norm_sha(os.path.join(args.repo, CAND_AI)),
                         a7=norm_sha(os.path.join(args.repo, "benchmarks", "psf_ai_compile.py"))),
                device=args.device, arm=args.arm, family=args.family, smoke=args.smoke, n_qubits=tgt.num_qubits,
                two_q_gate=g2, failed_edges=len(failed), failed_qubits=len(failed_q),
                started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print("META", json.dumps(meta), flush=True)

    def off_target(out):
        bad = 0
        for ins in out.data:
            name = ins.operation.name
            if name in ("barrier", "measure", "delay"):
                continue
            q = tuple(out.find_bit(b).index for b in ins.qubits)
            if name not in tgt.operation_names or q not in tgt[name]:
                bad += 1
        return bad

    t0 = time.time()
    rows, circs, keep = [], [], []
    for params, qc in (family_w if wset else family)(args.family, args.smoke):
        n = qc.num_qubits
        chosen = None
        t1 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if args.arm == "L3T":
                out = transpile(qc, target=tgt, optimization_level=3, seed_transpiler=0, approximation_degree=1.0)
            elif args.arm in ("A7", "A8"):
                out, info = (a7 if args.arm == "A7" else a8).compile_for_model_circuit(qc, cm, nat, target=tgt,
                                                                                       return_info=True)
                chosen = info.get("chosen", info.get("path"))
            else:
                rec = dict(target=tgt, placement_refine=True, final_resynthesis="select", compare_level3=True)
                if args.arm == "R3":
                    mod, kw = rel, (dict(rec, compare_floor=True, candidate_score="pauli") if args.device in CX else rec)
                else:
                    mod, kw = c11, dict(rec, compare_floor=True, candidate_score="hybrid")
                before = dict(mod.COMPARE_STATS)
                out = mod.compile_for_hardware(qc, coupling_map=cm, basis_gates=nat, entangling_basis="cx",
                                               layout_search=True, seed_transpiler=0, **kw)
                moved = [k for k, v in mod.COMPARE_STATS.items() if v > before[k]]
                chosen = ",".join(sorted(moved))
        tc = time.perf_counter() - t1
        fin = list(out.layout.final_index_layout(filter_ancillas=True)[:n])
        idx = [tuple(out.find_bit(q).index for q in ins.qubits) for ins in out.data]
        active = {i for t in idx for i in t}
        row = dict(params=params, n=n, compile_s=tc, two_q=sum(1 for t in idx if len(t) == 2), depth=out.depth(),
                   active=len(active), failed_uses=sum(1 for t in idx if len(t) == 2 and (t in failed or t[::-1] in failed)),
                   failed_q_uses=sum(1 for t in idx if any(i in failed_q for i in t)), off_target=off_target(out),
                   failed_dir_uses=sum(1 for t in idx if len(t) == 2 and t in failed),
                   chosen=chosen)
        rows.append(row)
        if len(active) > (W_MAX_ACTIVE if wset else MAX_ACTIVE):
            row["too_wide"] = True
            continue
        c = out.copy()
        c.remove_final_measurements(inplace=True)
        c.save_density_matrix(qubits=fin)
        circs.append(c)
        keep.append((row, Statevector(qc).data))
    for which, key in (("noisy", "infid"), ("ideal", "infid_ideal")):
        if not circs:
            break
        res = sims[which].run(circs).result()
        for j, (row, psi) in enumerate(keep):
            rho = np.asarray(res.data(j)["density_matrix"])
            row[key] = float(1 - np.real(np.conj(psi) @ rho @ psi))
    name = f"hold6_{args.device}_{args.arm}_{args.family}{'_smoke' if args.smoke else ''}.json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(dict(meta=meta, rows=rows, wall_s=time.time() - t0), f)
    print(f"wrote {name}: {len(rows)} circuits, {len(circs)} simulated, {time.time() - t0:.0f} s", flush=True)


def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def score(args):
    D, smoke = {}, None
    for p in sorted(glob.glob(os.path.join(args.out, "hold6_*.json"))):
        r = json.load(open(p))
        m = r["meta"]
        smoke = m["smoke"]
        D[(m["device"], m["arm"], m["family"])] = r["rows"]
    want = [(d, a, f) for d in DEVICES for a in F_ARMS for f in FAMILIES] + \
           [(d, a, f) for d in DEVICES for a in ARMS for f in W_FAMILIES]
    missing = [k for k in want if k not in D]
    fr = [x for (d, a, f), v in D.items() if f in FAMILIES for x in v]
    wr = [x for (d, a, f), v in D.items() if f in W_FAMILIES for x in v]
    fw, ww = sum(1 for x in fr if x.get("too_wide")), sum(1 for x in wr if x.get("too_wide"))
    p0i = max((x["infid_ideal"] for x in fr + wr if "infid_ideal" in x), default=1.0)
    p0 = not missing and p0i <= 1e-6 and fw <= 0.05 * max(len(fr), 1) and ww <= 0.30 * max(len(wr), 1)
    L = [f"# hold6 score{' (SMOKE -- not a result)' if smoke else ''}", "",
         f"P0: {'PASS' if p0 else 'FAIL'} -- files {len(D)} of {len(want)} (missing {missing[:5]}); noiseless "
         f"infidelity max {p0i:.1e} (<= 1e-6); too wide {fw} of {len(fr)} in F (<= 5%), {ww} of {len(wr)} in W (<= 30%)"]

    def sel(d, a, f, bc=None):
        return [x for x in D.get((d, a, f), []) if bc is None or x["params"].get("bc") == bc]

    def ratio(d, cells, a, b):
        xs, ys = [], []
        for f, bc in cells:
            for x, y in zip(sel(d, a, f, bc), sel(d, b, f, bc)):
                if "infid" in x and "infid" in y:
                    xs.append(x["infid"])
                    ys.append(y["infid"])
        return float(np.mean(xs) / max(np.mean(ys), 1e-12)) if xs else float("nan")

    FA = [(f, None) for f in FAMILIES]
    WA = [(f, None) for f in W_FAMILIES]
    CELLS = [("F1", None), ("F2", None), ("F3", "o"), ("F3", "p"), ("F4", None), ("F5", None), ("F6", None)]
    WCELLS = [("W1", None), ("W2", None), ("W3", "o"), ("W3", "p"), ("W4", None), ("W5", None), ("W6", None)]
    fmt = lambda dct: ", ".join(f"{k.replace('Fake', '')} {v:.4f}" for k, v in dct.items())
    L += ["", "F families (HOLD sizes):", "",
          "| device | C11/R3 | C11/L3T | R3/L3T | A7/C11 | F3 open C11/R3 | F5 C11/R3 |", "|---|---|---|---|---|---|---|"]
    for d in DEVICES:
        L.append(f"| {d}{' (cx)' if d in CX else ''} | {ratio(d, FA, 'C11', 'R3'):.4f} | {ratio(d, FA, 'C11', 'L3T'):.4f} | "
                 f"{ratio(d, FA, 'R3', 'L3T'):.4f} | {ratio(d, FA, 'A7', 'C11'):.4f} | "
                 f"{ratio(d, [('F3', 'o')], 'C11', 'R3'):.4f} | {ratio(d, [('F5', None)], 'C11', 'R3'):.4f} |")
    L += ["", "W families (9-10 qubits):", "",
          "| device | C11/R3 | A8/A7 | A8/L3T | A7/L3T | R3/L3T | A8/R3 |", "|---|---|---|---|---|---|---|"]
    for d in DEVICES:
        L.append(f"| {d}{' (cx)' if d in CX else ''} | {ratio(d, WA, 'C11', 'R3'):.4f} | {ratio(d, WA, 'A8', 'A7'):.4f} | "
                 f"{ratio(d, WA, 'A8', 'L3T'):.4f} | {ratio(d, WA, 'A7', 'L3T'):.4f} | {ratio(d, WA, 'R3', 'L3T'):.4f} | "
                 f"{ratio(d, WA, 'A8', 'R3'):.4f} |")
    for title, cells in (("cell C11/R3 (F)", CELLS), ("cell C11/R3 (W)", WCELLS)):
        L += ["", f"| {title} | " + " | ".join(d.replace("Fake", "") for d in DEVICES) + " |",
              "|---" * (len(DEVICES) + 1) + "|"]
        for f, bc in cells:
            L.append(f"| {f}{bc or ''} | " + " | ".join(f"{ratio(d, [(f, bc)], 'C11', 'R3'):.3f}" for d in DEVICES) + " |")

    h1v = {d: ratio(d, FA, "C11", "R3") for d in DEVICES}
    h1 = verdict(sum(v <= 1.00 for v in h1v.values()) >= 8, any(v > 1.01 for v in h1v.values()))
    h2v = {d: h1v[d] for d in CX}
    h2 = verdict(all(v < 1.00 for v in h2v.values()), sum(v > 1.003 for v in h2v.values()) >= 2)
    h3v = {d: ratio(d, [("F3", "o")], "C11", "R3") for d in CX}
    h3 = verdict(sum(v <= 1.00 for v in h3v.values()) >= 3, sum(v > 1.01 for v in h3v.values()) >= 2)
    h4v = {d: ratio(d, [("F5", None)], "C11", "R3") for d in CX}
    h4 = verdict(all(v <= 1.01 for v in h4v.values()), any(v > 1.03 for v in h4v.values()))
    h5v = {d: h1v[d] for d in CZ}
    h5 = verdict(sum(v <= 1.00 for v in h5v.values()) >= 4, sum(v > 1.005 for v in h5v.values()) >= 2)
    nfd = sum(x["failed_dir_uses"] + x["failed_q_uses"] for (d, a, f), v in D.items() if a in ("C11", "A8") for x in v)
    noff = sum(x["off_target"] > 0 for (d, a, f), v in D.items() if a in ("C11", "A8") for x in v)
    h6 = verdict(nfd + noff == 0, nfd + noff > 0)
    medf = {a: float(np.median([x["compile_s"] for (d, aa, f), v in D.items() if aa == a and f in FAMILIES for x in v]
                               or [float("nan")])) for a in F_ARMS}
    h7 = verdict(medf["C11"] <= 3 * medf["R3"], medf["C11"] > 10 * medf["R3"])
    h8v = {d: ratio(d, FA, "C11", "L3T") for d in DEVICES}
    h8 = verdict(all(v <= 1.00 for v in h8v.values()), any(v > 1.02 for v in h8v.values()))
    h9v = {d: ratio(d, WA, "A8", "A7") for d in DEVICES}
    h9 = verdict(sum(v <= 0.95 for v in h9v.values()) >= 7, sum(v > 1.00 for v in h9v.values()) >= 2)
    h10v = {d: ratio(d, WA, "A8", "L3T") for d in DEVICES}
    h10 = verdict(sum(v <= 1.00 for v in h10v.values()) >= 7, sum(v > 1.03 for v in h10v.values()) >= 3)
    h11v = {d: ratio(d, WA, "C11", "R3") for d in DEVICES}
    h11 = verdict(sum(v <= 1.00 for v in h11v.values()) >= 7, sum(v > 1.02 for v in h11v.values()) >= 2)
    L += ["", "## Predictions", "",
          f"- H1 (F: C11/R3 <= 1.00 on at least 8 of 9 devices): **{h1}** ({fmt(h1v)})",
          f"- H2 (F, cx devices: C11/R3 < 1.00 on all 4): **{h2}** ({fmt(h2v)})",
          f"- H3 (F3 open, cx devices: C11/R3 <= 1.00 on at least 3 of 4): **{h3}** ({fmt(h3v)})",
          f"- H4 (F5, cx devices: C11/R3 <= 1.01 on all 4): **{h4}** ({fmt(h4v)})",
          f"- H5 (F, cz devices: C11/R3 <= 1.00 on at least 4 of 5): **{h5}** ({fmt(h5v)})",
          f"- H6 (C11 and A8 never use a failed direction or qubit, nor an off-target instruction): **{h6}** "
          f"({nfd} failed uses, {noff} circuits off target)",
          f"- H7 (F: median compile time C11 <= 3 x R3): **{h7}** ({medf['C11']:.3f} s vs {medf['R3']:.3f} s)",
          f"- H8 (F: C11/L3T <= 1.00 on all 9 devices): **{h8}** ({fmt(h8v)})",
          f"- H9 (W: A8/A7 <= 0.95 on at least 7 of 9 devices): **{h9}** ({fmt(h9v)})",
          f"- H10 (W: A8/L3T <= 1.00 on at least 7 of 9 devices): **{h10}** ({fmt(h10v)})",
          f"- H11 (W: C11/R3 <= 1.00 on at least 7 of 9 devices): **{h11}** ({fmt(h11v)})"]
    chosen = {}
    for (d, a, f), v in D.items():
        if a in ("C11", "R3"):
            for x in v:
                k = (a, f[0], x.get("chosen"))
                chosen[k] = chosen.get(k, 0) + 1
    medw = {a: float(np.median([x["compile_s"] for (d, aa, f), v in D.items() if aa == a and f in W_FAMILIES for x in v]
                               or [float("nan")])) for a in ARMS}
    nfc = {a: sum(x["failed_uses"] + x["failed_q_uses"] for (d, aa, f), v in D.items() if aa == a for x in v) for a in ARMS}
    nfdir = {a: sum(x["failed_dir_uses"] + x["failed_q_uses"] for (d, aa, f), v in D.items() if aa == a for x in v) for a in ARMS}
    offt = {a: sum(x["off_target"] > 0 for (d, aa, f), v in D.items() if aa == a for x in v) for a in ARMS}
    toow = {a: sum(1 for (d, aa, f), v in D.items() if aa == a for x in v if x.get("too_wide")) for a in ARMS}
    L += ["", "Reported: choices (COMPARE_STATS counters that moved), by arm and set: " + str(dict(sorted(chosen.items(), key=str))),
          "Reported: median compile s, F " + ", ".join(f"{a} {v:.3f}" for a, v in medf.items()) + "; W " +
          ", ".join(f"{a} {v:.3f}" for a, v in medw.items()),
          f"Reported: failed uses by coupler {nfc}; by direction {nfdir}",
          "Reported: circuits with an off-target instruction, by arm: " + ", ".join(f"{a} {v}" for a, v in offt.items()),
          "Reported: too wide by arm: " + ", ".join(f"{a} {v}" for a, v in toow.items())]
    txt = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "score.md"), "w") as f:
        f.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("run", "score"))
    ap.add_argument("--repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=DEVICES)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--family", choices=FAMILIES + W_FAMILIES)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    (run if args.part == "run" else score)(args)


if __name__ == "__main__":
    main()
