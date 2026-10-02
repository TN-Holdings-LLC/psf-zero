"""hold_eval.py -- pre-registered home test (2026-10-03): do the two adoptions of 2026-10-02 hold on new circuits and new
devices? Release psf_compile 2026-10-02.2 with placement_refine (Addenda 309-311) and the AI front end a7 (Addenda
315-317) were both chosen and judged on the GAP circuits and on FakeAuckland, FakeTorino and FakeKingston. This test
uses circuits none of them has seen and six devices no compiler test has used.

Circuits (held out): the five GAP families (Addendum 300; generator code copied verbatim from gap_eval.family) with a
  new seed base (20,000,000 + ...; GAP used 1,000,000-5,500,000) and twice GAP's per-cell sizes, plus a new
  family F6 (the quantum Fourier transform, written with h, cp and final swaps, on a random product state; n = 4, 5,
  6). 1,506 circuits per device (13,554 per arm over nine devices). The smoke run uses 1 circuit per cell and its
  own seed base.
Arms (release psf_compile 2026-10-02.2, AI front end 2026-10-02.a7, core and layout unchanged):
  C3   compile_for_hardware(cx, layout_search=True, seed 0, target)                       (placement_refine off)
  C5   the same call plus placement_refine=True
  A7   benchmarks/psf_ai_compile.py (a7) compile_for_model_circuit(..., target)
  L3T  Qiskit transpile with the Target, optimization_level 3, approximation_degree 1.0
Devices: seen -- FakeAuckland (cx), FakeTorino, FakeKingston (cz); new -- FakeHanoiV2, FakeAlgiers, FakeGeneva (cx,
  27 qubits), FakeFez, FakeMarrakesh, FakeAachen (cz, 156 qubits). Metric as in GAP: infidelity 1 - <psi|rho|psi> of
  the final-layout qubits under NoiseModel.from_backend (density matrix).

  python hold_eval.py run --repo <repo> --out <dir> --device <d> --arm <a> --family <F1..F6> [--smoke]
  python hold_eval.py score --out <dir>
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
ARMS = ("C3", "C5", "A7", "L3T")
SEEN = ("FakeAuckland", "FakeTorino", "FakeKingston")
NEW = ("FakeHanoiV2", "FakeAlgiers", "FakeGeneva", "FakeFez", "FakeMarrakesh", "FakeAachen")
DEVICES = SEEN + NEW
CZ = ("FakeTorino", "FakeKingston", "FakeFez", "FakeMarrakesh", "FakeAachen")
FAMILIES = ("F1", "F2", "F3", "F4", "F5", "F6")
SIZES = dict(F1=72, F2=120, F3=150, F4=90, F5=48, F6=40)        # per cell; twice GAP's per-cell sizes for F1-F5
SMOKE_SIZES = dict(F1=1, F2=1, F3=1, F4=1, F5=1, F6=1)
BASE = 20_000_000
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
    meta = dict(release=rel.VERSION, a7=a7.AI_COMPILE_VERSION, layout=lay.LAYOUT_VERSION,
                core=getattr(psf_zero_core, "CORE_VERSION", None), qiskit=qiskit.__version__, python=sys.version.split()[0],
                git_head=subprocess.run(["git", "-C", args.repo, "rev-parse", "--short=7", "HEAD"], capture_output=True,
                                        text=True).stdout.strip(),
                sha=dict(script=norm_sha(os.path.abspath(__file__)),
                         release=norm_sha(os.path.join(args.repo, "psf_compile.py")),
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
                kw = dict(target=tgt) if args.arm == "C3" else dict(target=tgt, placement_refine=True)
                out = rel.compile_for_hardware(qc, coupling_map=cm, basis_gates=nat, entangling_basis="cx",
                                               layout_search=True, seed_transpiler=0, **kw)
        tc = time.perf_counter() - t1
        fin = list(out.layout.final_index_layout(filter_ancillas=True)[:n])
        idx = [tuple(out.find_bit(q).index for q in ins.qubits) for ins in out.data]
        active = {i for t in idx for i in t}
        row = dict(params=params, n=n, compile_s=tc, two_q=sum(1 for t in idx if len(t) == 2), depth=out.depth(),
                   active=len(active), failed_uses=sum(1 for t in idx if len(t) == 2 and (t in failed or t[::-1] in failed)),
                   failed_q_uses=sum(1 for t in idx if any(i in failed_q for i in t)), off_target=off_target(out),
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
    name = f"hold_{args.device}_{args.arm}_{args.family}{'_smoke' if args.smoke else ''}.json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(dict(meta=meta, rows=rows, wall_s=time.time() - t0), f)
    print(f"wrote {name}: {len(rows)} circuits, {len(circs)} simulated, {time.time() - t0:.0f} s", flush=True)


def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def score(args):
    D, smoke = {}, None
    for p in sorted(glob.glob(os.path.join(args.out, "hold_*.json"))):
        r = json.load(open(p))
        m = r["meta"]
        smoke = m["smoke"]
        D[(m["device"], m["arm"], m["family"])] = r["rows"]
    rows = [x for v in D.values() for x in v]
    missing = [(d, a, f) for d in DEVICES for a in ARMS for f in FAMILIES if (d, a, f) not in D]
    wide = sum(1 for x in rows if x.get("too_wide"))
    p0i = max((x["infid_ideal"] for x in rows if "infid_ideal" in x), default=1.0)
    p0 = not missing and p0i <= 1e-6 and wide <= 0.05 * max(len(rows), 1)
    L = [f"# hold score{' (SMOKE -- not a result)' if smoke else ''}", "",
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
    L += ["", "| device | C5/C3 | C5/L3T | A7/L3T | A7/C5 | C3/L3T | chains C5/L3T | A7 chose L3T |",
          "|---|---|---|---|---|---|---|---|"]
    for d in DEVICES:
        nl3 = sum(x.get("chosen") == "L3T" for f in FAMILIES for x in sel(d, "A7", f))
        L.append(f"| {d}{' (new)' if d in NEW else ''} | {ratio(d, ALL, 'C5', 'C3'):.3f} | {ratio(d, ALL, 'C5', 'L3T'):.3f} | "
                 f"{ratio(d, ALL, 'A7', 'L3T'):.3f} | {ratio(d, ALL, 'A7', 'C5'):.3f} | {ratio(d, ALL, 'C3', 'L3T'):.3f} | "
                 f"{ratio(d, CHAIN, 'C5', 'L3T'):.3f} | {nl3} |")
    L += ["", "| cell | " + " | ".join(d.replace("Fake", "") for d in DEVICES) + " |", "|---" * (len(DEVICES) + 1) + "|"]
    for f, bc in CELLS:
        for a, b in (("C5", "C3"), ("A7", "L3T")):
            L.append(f"| {f}{bc or ''} {a}/{b} | " + " | ".join(f"{ratio(d, [(f, bc)], a, b):.3f}" for d in DEVICES) + " |")
    h1v = {d: ratio(d, ALL, "C5", "C3") for d in DEVICES}
    h1 = verdict(all(v <= 1.00 for v in h1v.values()), any(v > 1.02 for v in h1v.values()))
    cellv = [ratio(d, [c], "C5", "C3") for d in DEVICES for c in CELLS]
    share = sum(v <= 1.00 for v in cellv) / len(cellv)
    h2 = verdict(share >= 0.90, share < 0.75)
    h3v = {d: ratio(d, CHAIN, "C5", "L3T") for d in CZ}
    h3 = verdict(all(v <= 1.05 for v in h3v.values()), any(v >= 1.15 for v in h3v.values()))
    h4v = {d: ratio(d, ALL, "A7", "L3T") for d in DEVICES}
    h4 = verdict(all(v <= 1.00 for v in h4v.values()), any(v > 1.05 for v in h4v.values()))
    h5v = {d: ratio(d, ALL, "A7", "C5") for d in DEVICES}
    h5 = verdict(all(v <= 1.00 for v in h5v.values()), any(v > 1.03 for v in h5v.values()))
    nf = {a: sum(x["failed_uses"] + x["failed_q_uses"] for (d, aa, f), v in D.items() if aa == a for x in v) for a in ARMS}
    h6 = verdict(nf["C5"] + nf["A7"] == 0, nf["C5"] + nf["A7"] > 0)
    h7v = {d: ratio(d, [("F6", None)], "A7", "L3T") for d in DEVICES}
    h7 = verdict(all(v <= 1.05 for v in h7v.values()), any(v > 1.20 for v in h7v.values()))
    med = {a: float(np.median([x["compile_s"] for (d, aa, f), v in D.items() if aa == a for x in v])) for a in ARMS}
    fmt = lambda dct: ", ".join(f"{k.replace('Fake', '')} {v:.3f}" for k, v in dct.items())
    offt = {a: sum(x["off_target"] > 0 for (d, aa, f), v in D.items() if aa == a for x in v) for a in ARMS}
    L += ["", "## Predictions", "",
          f"- H1 (C5/C3 <= 1.00 on every device): **{h1}** ({fmt(h1v)})",
          f"- H2 (C5/C3 <= 1.00 in >= 90% of the {len(cellv)} cell-device pairs): **{h2}** ({share:.3f})",
          f"- H3 (chains, cz devices: C5/L3T <= 1.05): **{h3}** ({fmt(h3v)})",
          f"- H4 (A7/L3T <= 1.00 on every device): **{h4}** ({fmt(h4v)})",
          f"- H5 (A7/C5 <= 1.00 on every device): **{h5}** ({fmt(h5v)})",
          f"- H6 (C5 and A7 never use a failed coupler or qubit): **{h6}** ({nf})",
          f"- H7 (new family F6: A7/L3T <= 1.05 on every device): **{h7}** ({fmt(h7v)})"]
    L.append("\nReported: median compile s " + ", ".join(f"{a} {v:.3f}" for a, v in med.items()))
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
