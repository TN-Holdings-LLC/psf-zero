"""hold4_eval.py -- pre-registered home test (2026-10-03): candidate psf_compile 2026-10-03.c9 (choice against Qiskit
level 3 by excitation_cost, changelog item 36) on fresh held-out circuits. Release 2026-10-03.1 still trails Qiskit L3T
on periodic chains on cx devices, rings on cz devices and QFT (Addendum 324); Addendum 326 traced those to synthesis
on routed circuits, placement and routing, and found that choosing per circuit between the release's circuit and
level 3's by excitation_cost matched the better of the two closely. Does it hold on new circuits, on every family?

Circuits (held out again): hold_eval's families F1-F6 (GAP's five, with gap_eval.family's code, plus QFT), the same
  per-cell sizes (1,506 circuits per device), and a new seed base 50,000,000 + ... (HOLD3 used 40,000,000 + ...,
  HOLD2 30,000,000 + ..., HOLD 20,000,000 + ..., GAP 1,000,000-5,500,000). The smoke run uses 1 circuit per cell and
  its own seeds.
Arms (release psf_compile 2026-10-03.1 and AI front end 2026-10-02.a7 as adopted; candidate c9):
  R3   release compile_for_hardware(cx, layout_search=True, seed 0, target, placement_refine=True,
       final_resynthesis="select") -- the release as its README recommends
  C9   candidate, the same call plus compare_level3=True
  A7   benchmarks/psf_ai_compile.py (a7) with the target
  L3T  Qiskit transpile with the Target, optimization_level 3, approximation_degree 1.0
Devices: as in HOLD (Addendum 318): FakeAuckland, FakeHanoiV2, FakeAlgiers, FakeGeneva (cx); FakeTorino, FakeKingston,
  FakeFez, FakeMarrakesh, FakeAachen (cz). Metric as in GAP. Failed-element uses are counted both ways: by coupler
  (either direction reported failed, as in HOLD) and by direction (the gate's own direction reported failed).

  python hold4_eval.py run --repo <repo> --out <dir> --device <d> --arm <a> --family <F1..F6> [--smoke]
  python hold4_eval.py score --out <dir>
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
ARMS = ("R3", "C9", "A7", "L3T")
CX = ("FakeAuckland", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva")
CAND = os.path.join("patches", "psf_compile_c9_2026-10-03", "psf_compile.py")
SEEN = ("FakeAuckland", "FakeTorino", "FakeKingston")
NEW = ("FakeHanoiV2", "FakeAlgiers", "FakeGeneva", "FakeFez", "FakeMarrakesh", "FakeAachen")
DEVICES = SEEN + NEW
CZ = ("FakeTorino", "FakeKingston", "FakeFez", "FakeMarrakesh", "FakeAachen")
FAMILIES = ("F1", "F2", "F3", "F4", "F5", "F6")
SIZES = dict(F1=72, F2=120, F3=150, F4=90, F5=48, F6=40)        # per cell; twice GAP's per-cell sizes for F1-F5
SMOKE_SIZES = dict(F1=1, F2=1, F3=1, F4=1, F5=1, F6=1)
BASE = 50_000_000
MAX_ACTIVE = 11


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
    c9 = H.load_module(os.path.join(args.repo, CAND), "psf_compile_c9")
    lay = H.load_module(os.path.join(args.repo, "benchmarks", "psf_smart_layout.py"), "psl_c2")
    sys.modules["psf_smart_layout"] = lay
    a7 = H.load_module(os.path.join(args.repo, "benchmarks", "psf_ai_compile.py"), "psf_ai_compile")
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
    sims = {"noisy": AerSimulator(noise_model=NoiseModel.from_backend(backend), **opts), "ideal": AerSimulator(**opts)}
    meta = dict(release=rel.VERSION, c9=c9.VERSION, a7=a7.AI_COMPILE_VERSION, layout=lay.LAYOUT_VERSION,
                core=getattr(psf_zero_core, "CORE_VERSION", None), qiskit=qiskit.__version__, python=sys.version.split()[0],
                git_head=subprocess.run(["git", "-C", args.repo, "rev-parse", "--short=7", "HEAD"], capture_output=True,
                                        text=True).stdout.strip(),
                sha=dict(script=norm_sha(os.path.abspath(__file__)),
                         release=norm_sha(os.path.join(args.repo, "psf_compile.py")),
                         c9=norm_sha(os.path.join(args.repo, CAND)),
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
    for params, qc in family(args.family, args.smoke):
        n = qc.num_qubits
        chosen = None
        t1 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if args.arm == "L3T":
                out = transpile(qc, target=tgt, optimization_level=3, seed_transpiler=0, approximation_degree=1.0)
            elif args.arm == "A7":
                out, info = a7.compile_for_model_circuit(qc, cm, nat, target=tgt, return_info=True)
                chosen = info.get("chosen", info.get("path"))
            else:
                mod, kw = {"R3": (rel, dict(target=tgt, placement_refine=True, final_resynthesis="select")),
                           "C9": (c9, dict(target=tgt, placement_refine=True, final_resynthesis="select",
                                           compare_level3=True))}[args.arm]
                before = dict(c9.COMPARE_STATS)
                out = mod.compile_for_hardware(qc, coupling_map=cm, basis_gates=nat, entangling_basis="cx",
                                               layout_search=True, seed_transpiler=0, **kw)
                if args.arm == "C9":
                    moved = [k for k, v in c9.COMPARE_STATS.items() if v > before[k]]
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
        if len(active) > MAX_ACTIVE:
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
    name = f"hold4_{args.device}_{args.arm}_{args.family}{'_smoke' if args.smoke else ''}.json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(dict(meta=meta, rows=rows, wall_s=time.time() - t0), f)
    print(f"wrote {name}: {len(rows)} circuits, {len(circs)} simulated, {time.time() - t0:.0f} s", flush=True)


def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def score(args):
    D, smoke = {}, None
    for p in sorted(glob.glob(os.path.join(args.out, "hold4_*.json"))):
        r = json.load(open(p))
        m = r["meta"]
        smoke = m["smoke"]
        D[(m["device"], m["arm"], m["family"])] = r["rows"]
    rows = [x for v in D.values() for x in v]
    missing = [(d, a, f) for d in DEVICES for a in ARMS for f in FAMILIES if (d, a, f) not in D]
    wide = sum(1 for x in rows if x.get("too_wide"))
    p0i = max((x["infid_ideal"] for x in rows if "infid_ideal" in x), default=1.0)
    p0 = not missing and p0i <= 1e-6 and wide <= 0.05 * max(len(rows), 1)
    L = [f"# hold4 score{' (SMOKE -- not a result)' if smoke else ''}", "",
         f"P0: {'PASS' if p0 else 'FAIL'} -- files {len(D)} of {len(DEVICES) * len(ARMS) * len(FAMILIES)} (missing "
         f"{missing}); noiseless infidelity max {p0i:.1e} (<= 1e-6); too wide {wide} of {len(rows)}"]

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

    ALL = [(f, None) for f in FAMILIES]
    CELLS = [("F1", None), ("F2", None), ("F3", "o"), ("F3", "p"), ("F4", None), ("F5", None), ("F6", None)]
    CHAIN = [("F3", "o"), ("F5", None)]
    F3O, F3P, F1 = [("F3", "o")], [("F3", "p")], [("F1", None)]
    L += ["", "| device | C9/R3 | C9/L3T | R3/L3T | A7/C9 | A7/L3T | F3 periodic C9/L3T | F3 periodic R3/L3T | "
          "F1 C9/L3T | F1 R3/L3T |", "|---|---|---|---|---|---|---|---|---|---|"]
    for d in DEVICES:
        L.append(f"| {d}{' (cx)' if d in CX else ''} | {ratio(d, ALL, 'C9', 'R3'):.3f} | {ratio(d, ALL, 'C9', 'L3T'):.3f} | "
                 f"{ratio(d, ALL, 'R3', 'L3T'):.3f} | {ratio(d, ALL, 'A7', 'C9'):.3f} | {ratio(d, ALL, 'A7', 'L3T'):.3f} | "
                 f"{ratio(d, F3P, 'C9', 'L3T'):.3f} | {ratio(d, F3P, 'R3', 'L3T'):.3f} | "
                 f"{ratio(d, F1, 'C9', 'L3T'):.3f} | {ratio(d, F1, 'R3', 'L3T'):.3f} |")
    for a, b in (("C9", "R3"), ("C9", "L3T"), ("A7", "C9")):
        L += ["", f"| cell {a}/{b} | " + " | ".join(d.replace("Fake", "") for d in DEVICES) + " |",
              "|---" * (len(DEVICES) + 1) + "|"]
        for f, bc in CELLS:
            L.append(f"| {f}{bc or ''} | " + " | ".join(f"{ratio(d, [(f, bc)], a, b):.3f}" for d in DEVICES) + " |")
    h1v = {d: ratio(d, ALL, "C9", "R3") for d in DEVICES}
    h1 = verdict(all(v <= 1.00 for v in h1v.values()), any(v > 1.02 for v in h1v.values()))
    h2v = {d: ratio(d, ALL, "C9", "L3T") for d in DEVICES}
    h2 = verdict(sum(v <= 1.00 for v in h2v.values()) >= 7, sum(v > 1.03 for v in h2v.values()) >= 3)
    cl = [ratio(d, [c], "C9", "L3T") for d in DEVICES for c in CELLS]
    shl = sum(v <= 1.02 for v in cl) / len(cl)
    h3 = verdict(shl >= 0.90, shl < 0.70)
    h4v = {d: ratio(d, F3P, "C9", "L3T") for d in CX}
    h4 = verdict(all(v <= 1.03 for v in h4v.values()), any(v >= 1.10 for v in h4v.values()))
    h5v = {d: ratio(d, F1, "C9", "L3T") for d in DEVICES if d not in CX}
    h5 = verdict(all(v <= 1.02 for v in h5v.values()), any(v >= 1.05 for v in h5v.values()))
    nfd = sum(x["failed_dir_uses"] + x["failed_q_uses"] for (d, aa, f), v in D.items() if aa == "C9" for x in v)
    noff = sum(x["off_target"] > 0 for (d, aa, f), v in D.items() if aa == "C9" for x in v)
    h6 = verdict(nfd + noff == 0, nfd + noff > 0)
    med = {a: float(np.median([x["compile_s"] for (d, aa, f), v in D.items() if aa == a for x in v])) for a in ARMS}
    h7 = verdict(med["C9"] <= 4 * med["R3"], med["C9"] > 10 * med["R3"])
    h8v = {d: ratio(d, ALL, "A7", "C9") for d in DEVICES}
    h8 = verdict(sum(v >= 0.96 for v in h8v.values()) >= 7, sum(v < 0.93 for v in h8v.values()) >= 3)
    right = n_diff = 0
    for d in DEVICES:
        for f in FAMILIES:
            for xr, xl, x9 in zip(sel(d, "R3", f), sel(d, "L3T", f), sel(d, "C9", f)):
                if not all("infid" in x for x in (xr, xl, x9)) or abs(xr["infid"] - xl["infid"]) <= 1e-12:
                    continue
                n_diff += 1
                right += abs(x9["infid"] - min(xr["infid"], xl["infid"])) <= 1e-12
    acc = right / max(n_diff, 1)
    h9 = verdict(acc >= 0.75, acc < 0.55)
    chosen = {}
    for (d, aa, f), v in D.items():
        if aa == "C9":
            for x in v:
                chosen[x.get("chosen")] = chosen.get(x.get("chosen"), 0) + 1
    fmt = lambda dct: ", ".join(f"{k.replace('Fake', '')} {v:.3f}" for k, v in dct.items())
    nfc = {a: sum(x["failed_uses"] + x["failed_q_uses"] for (d, aa, f), v in D.items() if aa == a for x in v) for a in ARMS}
    nfdir = {a: sum(x["failed_dir_uses"] + x["failed_q_uses"] for (d, aa, f), v in D.items() if aa == a for x in v) for a in ARMS}
    offt = {a: sum(x["off_target"] > 0 for (d, aa, f), v in D.items() if aa == a for x in v) for a in ARMS}
    L += ["", "## Predictions", "",
          f"- H1 (C9/R3 <= 1.00 on every device): **{h1}** ({fmt(h1v)})",
          f"- H2 (C9/L3T <= 1.00 on at least 7 of 9 devices): **{h2}** ({fmt(h2v)})",
          f"- H3 (C9/L3T <= 1.02 in >= 90% of the {len(cl)} cell-device pairs): **{h3}** ({shl:.3f})",
          f"- H4 (cx devices, F3 periodic: C9/L3T <= 1.03 on all 4): **{h4}** ({fmt(h4v)})",
          f"- H5 (cz devices, F1: C9/L3T <= 1.02 on all 5): **{h5}** ({fmt(h5v)})",
          f"- H6 (C9 never uses a failed direction or qubit, nor an off-target instruction): **{h6}** "
          f"({nfd} failed uses, {noff} circuits off target)",
          f"- H7 (median compile time C9 <= 4 x R3): **{h7}** (R3 {med['R3']:.3f} s, C9 {med['C9']:.3f} s)",
          f"- H8 (A7/C9 >= 0.96 on at least 7 of 9 devices): **{h8}** ({fmt(h8v)})",
          f"- H9 (where R3 and L3T differ, C9 has the lower measured infidelity of the two in >= 75%): **{h9}** "
          f"({right} of {n_diff}, {acc:.3f})"]
    L.append("\nReported: C9 choices (COMPARE_STATS counters that moved): " + str(chosen))
    L.append("\nReported: median compile s " + ", ".join(f"{a} {v:.3f}" for a, v in med.items()))
    L.append("Reported: failed uses by coupler " + str(nfc) + "; by direction " + str(nfdir))
    L.append("Reported: circuits with an off-target instruction, by arm: " + ", ".join(f"{a} {v}" for a, v in offt.items()))
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
    ap.add_argument("--family", choices=FAMILIES)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    (run if args.part == "run" else score)(args)


if __name__ == "__main__":
    main()
