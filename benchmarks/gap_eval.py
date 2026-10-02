"""gap_eval.py -- pre-registered home test (2026-10-02): a map of where the PSF-Zero release (C2) trails, ties or
beats error-aware Qiskit level 3 (L3T) and psf_ai_compile a5 (A5), by circuit family, on three fake devices under
Aer noise. It turns the two improvement targets found in Addendum 297 (routing of cycles on heavy-hex,
readout-blind placement) into measured gaps over a wider set of circuits.

Families (inputs |0...0>; every circuit depends only on its own seed):
  F1 ring ansatz: the QML model family (data re-uploading, CZ ring) on n = 4 and 6 qubits, L = 2, 4, 6 layers,
     random parameters and inputs; 36 per (n, L).
  F2 QAOA MaxCut on random 3-regular graphs, n = 6, p = 1 and 2, random angles; 60 per p.
  F3 XYZ-Heisenberg Trotter chain, n = 6, dt = 0.1, 4 steps, open ("o") and periodic ("p") boundary; 75 each.
  F4 quantum-volume-style layers of Haar SU(4) on random pairs, n = 4, 5, 6, depth n; 45 per n.
  F5 GHZ chain plus a random single-qubit layer, n = 4, 6, 8; 24 per n.
Arms: C2 (release psf_compile 2026-10-01.1, entangling_basis="cx", layout_search=True), A5 (psf_ai_compile a5 with
  the Target), L3T (Qiskit transpile with the Target, optimization_level=3) -- as in Addenda 290-297.
Devices: FakeAuckland, FakeTorino, FakeKingston. Noise: NoiseModel.from_backend, AerSimulator(density_matrix).
Metric: infidelity 1 - <psi|rho|psi> of the final-layout qubits' state against the ideal output psi.
  A compiled circuit touching more than 11 physical qubits is not simulated (recorded as too wide).

  python gap_eval.py run --repo <repo> --out <dir> --device <d> --arm <a> --family <F> [--smoke]
  python gap_eval.py score --out <dir>
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
ARMS = ("C2", "A5", "L3T")
DEVICES = ("FakeAuckland", "FakeTorino", "FakeKingston")
FAMILIES = ("F1", "F2", "F3", "F4", "F5")
MAX_ACTIVE = 11
SIZES = dict(F1=36, F2=60, F3=75, F4=45, F5=24)
SMOKE_SIZES = dict(F1=1, F2=1, F3=1, F4=1, F5=1)


def norm_sha(path):
    with open(path, "rb") as f:
        txt = f.read().decode("utf-8").replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in txt.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- families
def regular3(n, rng):
    """A random simple 3-regular graph on n nodes (configuration model with rejection)."""
    while True:
        stubs = np.repeat(np.arange(n), 3)
        rng.shuffle(stubs)
        edges = {tuple(sorted((int(stubs[i]), int(stubs[i + 1])))) for i in range(0, len(stubs), 2)}
        if len(edges) == len(stubs) // 2 and all(a != b for a, b in edges):
            return sorted(edges)


def family(fam, smoke):
    """Yields (params, circuit)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    size = (SMOKE_SIZES if smoke else SIZES)[fam]
    base = 1_000_000 * (FAMILIES.index(fam) + 1) + (500_000 if smoke else 0)
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


# ---------------------------------------------------------------- compilers and simulation
class Stack:
    def __init__(self, repo):
        sys.path.insert(0, os.path.join(repo, "benchmarks"))
        sys.path.insert(0, repo)
        import core_fix_c2_eval as H
        self.H = H
        self.c2 = H.load_module(os.path.join(repo, "psf_compile.py"), "psf_compile")
        self.lay = H.load_module(os.path.join(repo, "benchmarks", "psf_smart_layout.py"), "psl_c2")
        sys.modules["psf_smart_layout"] = self.lay
        self.a5 = H.load_module(os.path.join(repo, "benchmarks", "psf_ai_compile.py"), "psf_ai_compile")
        import psf_zero_core
        import qiskit
        import qiskit_aer
        self.meta = dict(c2=self.c2.VERSION, layout=self.lay.LAYOUT_VERSION, a5=self.a5.AI_COMPILE_VERSION,
                         core=getattr(psf_zero_core, "CORE_VERSION", None), qiskit=qiskit.__version__,
                         aer=qiskit_aer.__version__, python=sys.version.split()[0],
                         git_head=subprocess.run(["git", "-C", repo, "rev-parse", "--short=7", "HEAD"],
                                                 capture_output=True, text=True).stdout.strip(),
                         sha=dict(script=norm_sha(os.path.abspath(__file__)),
                                  c2=norm_sha(os.path.join(repo, "psf_compile.py")),
                                  a5=norm_sha(os.path.join(repo, "benchmarks", "psf_ai_compile.py"))))

    def device(self, name):
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel
        from qiskit_ibm_runtime import fake_provider
        backend = getattr(fake_provider, name)()
        tgt = backend.target
        opts = dict(method="density_matrix", max_parallel_threads=1, max_parallel_experiments=1)
        self.d = dict(tgt=tgt, cm=tgt.build_coupling_map(),
                      nat=[g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names],
                      ideal=AerSimulator(**opts), noisy=AerSimulator(noise_model=NoiseModel.from_backend(backend), **opts))

    def compile(self, arm, qc):
        from qiskit import transpile
        d = self.d
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if arm == "C2":
                return self.c2.compile_for_hardware(qc, coupling_map=d["cm"], basis_gates=d["nat"],
                                                    entangling_basis="cx", layout_search=True, seed_transpiler=0)
            if arm == "A5":
                return self.a5.compile_for_model_circuit(qc, d["cm"], d["nat"], target=d["tgt"])
            return transpile(qc, target=d["tgt"], optimization_level=3, seed_transpiler=0)

    def readout(self, q):
        try:
            m = self.d["tgt"]["measure"][(q,)]
            return float(m.error) if m is not None and m.error is not None else 0.0
        except Exception:
            return 0.0


def run(args):
    from qiskit.quantum_info import Statevector
    st = Stack(args.repo)
    st.device(args.device)
    meta = dict(st.meta, device=args.device, arm=args.arm, family=args.family, smoke=args.smoke,
                started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print("META", json.dumps(meta), flush=True)
    t0 = time.time()
    rows, sims, keep = [], [], []
    for params, qc in family(args.family, args.smoke):
        n = qc.num_qubits
        t1 = time.perf_counter()
        out = st.compile(args.arm, qc)
        tc = time.perf_counter() - t1
        fin = list(out.layout.final_index_layout(filter_ancillas=True)[:n])
        active = sorted({out.find_bit(q).index for ins in out.data for q in ins.qubits})
        row = dict(params=params, n=n, compile_s=tc, two_q=sum(1 for i in out.data if i.operation.num_qubits == 2),
                   depth=out.depth(), active=len(active), ro_mean=float(np.mean([st.readout(q) for q in fin])))
        rows.append(row)
        if len(active) > MAX_ACTIVE:
            row["too_wide"] = True
            continue
        c = out.copy()
        c.remove_final_measurements(inplace=True)
        c.save_density_matrix(qubits=fin)
        sims.append(c)
        keep.append((row, Statevector(qc).data))
    for which, key in (("noisy", "infid"), ("ideal", "infid_ideal")):
        if not sims:
            break
        res = st.d[which].run(sims).result()
        for j, (row, psi) in enumerate(keep):
            rho = np.asarray(res.data(j)["density_matrix"])
            row[key] = float(1 - np.real(np.conj(psi) @ rho @ psi))
    out_name = f"gap_{args.device}_{args.arm}_{args.family}{'_smoke' if args.smoke else ''}.json"
    with open(os.path.join(args.out, out_name), "w") as f:
        json.dump(dict(meta=meta, rows=rows, wall_s=time.time() - t0), f)
    print(f"wrote {out_name}: {len(rows)} circuits, {len(sims)} simulated, {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------- scoring (json only)
def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def score(args):
    data = {}
    smoke = None
    for p in sorted(glob.glob(os.path.join(args.out, "gap_*.json"))):
        r = json.load(open(p))
        m = r["meta"]
        smoke = m["smoke"]
        data[(m["device"], m["arm"], m["family"])] = r["rows"]
    L = [f"# gap score{' (SMOKE -- not a result)' if smoke else ''}", ""]
    missing = [(d, a, f) for d in DEVICES for a in ARMS for f in FAMILIES if (d, a, f) not in data]
    rows_all = [r for v in data.values() for r in v]
    wide = sum(1 for r in rows_all if r.get("too_wide"))
    p0i = max((r["infid_ideal"] for r in rows_all if "infid_ideal" in r), default=1.0)
    p0 = not missing and p0i <= 1e-9 and wide <= 0.05 * max(len(rows_all), 1)
    L.append(f"P0: {'PASS' if p0 else 'FAIL'} -- files {len(data)} of 45 (missing {missing}); noiseless infidelity max "
             f"{p0i:.1e}; too wide {wide} of {len(rows_all)}")

    def sel(d, a, f, pred=lambda r: True):
        return [r for r in data.get((d, a, f), []) if pred(r)]

    def paired(d, f, pred, a="C2", b="L3T"):
        """Rows of arms a and b for the same circuits (same order), both simulated."""
        ra, rb = sel(d, a, f, pred), sel(d, b, f, pred)
        return [(x, y) for x, y in zip(ra, rb) if "infid" in x and "infid" in y]

    subsets = dict(cyc=[("F1", lambda r: True), ("F3", lambda r: r["params"]["bc"] == "p")],
                   chain=[("F3", lambda r: r["params"]["bc"] == "o"), ("F5", lambda r: True)])
    L += ["", "## Mean infidelity / mean two-qubit count by family and device", "",
          "| family | device | C2 | A5 | L3T | C2/L3T infidelity | C2 2q > L3T 2q |", "|---|---|---|---|---|---|---|"]
    fams = [("F1", None), ("F2", None), ("F3o", ("F3", "o")), ("F3p", ("F3", "p")), ("F4", None), ("F5", None)]
    for name, spec in fams:
        f = spec[0] if spec else name
        pred = (lambda r, bc=spec[1]: r["params"]["bc"] == bc) if spec else (lambda r: True)
        for d in DEVICES:
            cells = []
            for a in ARMS:
                rs = [r for r in sel(d, a, f, pred) if "infid" in r]
                cells.append(f"{np.mean([r['infid'] for r in rs]):.4f} / {np.mean([r['two_q'] for r in rs]):.1f}"
                             if rs else "-")
            pr = paired(d, f, pred)
            ratio = (np.mean([x["infid"] for x, _ in pr]) / max(np.mean([y["infid"] for _, y in pr]), 1e-12)) if pr else float("nan")
            more = (sum(x["two_q"] > y["two_q"] for x, y in pr) / len(pr)) if pr else float("nan")
            L.append(f"| {name} | {d} | " + " | ".join(cells) + f" | {ratio:.3f} | {more:.2f} |")

    def frac_more(d, subset):
        pr = [p for f, pred in subsets[subset] for p in paired(d, f, pred)]
        return (sum(x["two_q"] > y["two_q"] for x, y in pr) / len(pr)) if pr else float("nan")

    def frac_le(d, subset):
        pr = [p for f, pred in subsets[subset] for p in paired(d, f, pred)]
        return (sum(x["two_q"] <= y["two_q"] for x, y in pr) / len(pr)) if pr else float("nan")

    def ratio(d, subset_or_fams, a="C2", b="L3T"):
        pr = [p for f, pred in subset_or_fams for p in paired(d, f, pred, a, b)]
        return (np.mean([x["infid"] for x, _ in pr]) / max(np.mean([y["infid"] for _, y in pr]), 1e-12)) if pr else float("nan")

    heron = ("FakeTorino", "FakeKingston")
    h1 = verdict(all(frac_more(d, "cyc") >= 0.80 for d in heron), any(frac_more(d, "cyc") <= 0.50 for d in heron))
    h2 = verdict(all(frac_le(d, "chain") >= 0.90 for d in DEVICES), any(frac_le(d, "chain") < 0.70 for d in DEVICES))
    h3 = verdict(all(ratio(d, subsets["cyc"]) >= 1.10 for d in heron), any(ratio(d, subsets["cyc"]) <= 1.0 for d in heron))
    f4 = [("F4", lambda r: True)]
    h4 = verdict(all(ratio(d, f4) <= 1.05 for d in DEVICES), any(ratio(d, f4) > 1.20 for d in DEVICES))
    allf = [(f, lambda r: True) for f in FAMILIES]
    r5 = [ratio(d, allf, "A5", "L3T") for d in DEVICES]
    h5 = verdict(sum(x <= 1.0 for x in r5) >= 2, sum(x > 1.05 for x in r5) >= 2)
    rch = {d: ratio(d, subsets["chain"]) for d in DEVICES}
    h6 = verdict(all(v >= 1.10 for v in rch.values()), any(v <= 1.0 for v in rch.values()))
    fm = {d: frac_more(d, "cyc") for d in heron}
    fl = {d: frac_le(d, "chain") for d in DEVICES}
    rc = {d: ratio(d, subsets["cyc"]) for d in heron}
    r4 = {d: ratio(d, f4) for d in DEVICES}
    fmt = lambda dct, nd: ", ".join(f"{d} {v:.{nd}f}" for d, v in dct.items())
    L += ["", "## Predictions", "",
          f"- H1 (cycles on Heron: C2 uses more 2q gates than L3T in >= 80% of F1 + F3p circuits): **{h1}** ({fmt(fm, 2)})",
          f"- H2 (chains: C2 2q <= L3T 2q in >= 90% of F3o + F5 circuits, every device): **{h2}** ({fmt(fl, 2)})",
          f"- H3 (cycles on Heron: mean infidelity C2 >= 1.10 x L3T): **{h3}** ({fmt(rc, 3)})",
          f"- H4 (dense random F4: C2 <= 1.05 x L3T mean infidelity, every device): **{h4}** ({fmt(r4, 3)})",
          f"- H5 (A5 <= L3T mean infidelity over all families on >= 2 of 3 devices): **{h5}** "
          f"({fmt(dict(zip(DEVICES, r5)), 3)})",
          f"- H6 (added after the smoke run; chains, where 2q counts are equal: mean infidelity C2 >= 1.10 x L3T on "
          f"every device): **{h6}** ({fmt(rch, 3)})"]
    ro = {(d, a): np.mean([r["ro_mean"] for f in FAMILIES for r in sel(d, a, f)]) for d in DEVICES for a in ARMS}
    cs = {a: np.median([r["compile_s"] for d in DEVICES for f in FAMILIES for r in sel(d, a, f)]) for a in ARMS}
    L += ["", "Reported: mean readout error of the final-layout qubits " +
          "; ".join(f"{d} " + ", ".join(f"{a} {ro[(d, a)]:.4f}" for a in ARMS) for d in DEVICES),
          "Reported: median compile s " + ", ".join(f"{a} {cs[a]:.3f}" for a in ARMS)]
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
