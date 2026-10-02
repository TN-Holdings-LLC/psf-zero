"""b17_practice_eval.py -- pre-registered home test (2026-10-02): does Qiskit issue #17057 (wrong ZSX/CX synthesis
near the two-CX boundary) bite in practice through transpile(), and does the PSF-Zero release's guard
(psf_compile changelog item 17) keep its output exact on the same workloads?

Workloads (n = 4 and 6 qubits each):
  W1 XYZ-Heisenberg Trotter chain, 4 steps: per bond rxx(2 Jx dt) ryy(2 Jy dt) rzz(2 Jz dt), even/odd bonds,
     plus rz(2 h_i dt) fields. Jx, Jy ~ U[0.5, 1.5], h_i ~ U[-1, 1], Jz = r * Jx with
     r in {0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1}, dt in {1e-3, 1e-2, 0.1}; 50 seeds per cell.
  W2 four near-boundary two-qubit unitaries on random qubit pairs (a, b ~ U[0, pi/4], c log-uniform in
     [1e-10, 1e-4], random local unitaries), between layers of random single-qubit unitaries; 500 circuits.
  W3 the same with Haar-random two-qubit unitaries; 500 circuits.
  W4 hardware-efficient ansatz, 3 layers of ry(t) rz(t) and a cx ladder, t ~ N(0, s), s in {1e-4, 1e-3, 1e-2};
     150 circuits per s.
Compilers (coupling map: a line of n qubits):
  QK1, QK2, QK3   transpile(basis_gates=[cx, rz, sx, x], optimization_level=1/2/3, seed_transpiler=0)
  QK3CZ           transpile(basis_gates=[cz, rz, sx, x], optimization_level=3)        -- control
  QK3U            transpile(basis_gates=[cx, u], optimization_level=3)                -- control
  PSF             release psf_compile.compile_for_hardware(entangling_basis="cx", basis [cx, rz, sx, x],
                  layout_search=True, seed_transpiler=0)
  PSFNG           the same module loaded separately with USE_CX_GUARD = False         -- positive control
Metric: 1 - F_avg between Operator(original) and Operator.from_circuit(compiled); a failure is > 1e-6.

  python b17_practice_eval.py run --repo <repo> --out <dir> --n 4|6 --workload W1|W2|W3|W4 [--smoke]
  python b17_practice_eval.py score --out <dir>
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
COMPILERS = ("QK1", "QK2", "QK3", "QK3CZ", "QK3U", "PSF", "PSFNG")
WORKLOADS = ("W1", "W2", "W3", "W4")
FAIL = 1e-6
R_VALUES = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0)
DT_VALUES = (1e-3, 1e-2, 0.1)
S_VALUES = (1e-4, 1e-3, 1e-2)
SIZES = dict(W1=50, W2=500, W3=500, W4=150)     # seeds per cell (W1), circuits (W2, W3), per s (W4)
SMOKE_SIZES = dict(W1=1, W2=3, W3=3, W4=2)


def norm_sha(path):
    with open(path, "rb") as f:
        txt = f.read().decode("utf-8").replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in txt.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- workloads
def circuits(workload, n, smoke):
    """Yields (params dict, QuantumCircuit). Every circuit depends only on its own seed."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    size = (SMOKE_SIZES if smoke else SIZES)[workload]
    base = 100_000 * (WORKLOADS.index(workload) + 1) + 10_000 * n + (5_000 if smoke else 0)
    if workload == "W1":
        k = 0
        for dt in DT_VALUES:
            for r in R_VALUES:
                for s in range(size):
                    rng = np.random.default_rng(base + k)
                    k += 1
                    jx, jy = rng.uniform(0.5, 1.5, 2)
                    h = rng.uniform(-1, 1, n)
                    qc = QuantumCircuit(n)
                    for _ in range(4):
                        for start in (0, 1):
                            for i in range(start, n - 1, 2):
                                qc.rxx(2 * jx * dt, i, i + 1)
                                qc.ryy(2 * jy * dt, i, i + 1)
                                qc.rzz(2 * r * jx * dt, i, i + 1)
                        for i in range(n):
                            qc.rz(2 * h[i] * dt, i)
                    yield dict(dt=dt, r=r, seed=s), qc
    elif workload in ("W2", "W3"):
        for s in range(size):
            rng = np.random.default_rng(base + s)
            qc = QuantumCircuit(n)
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
                    from qiskit.quantum_info import Operator
                    k1 = random_unitary(2, seed=int(rng.integers(2**31))).tensor(
                        random_unitary(2, seed=int(rng.integers(2**31))))
                    k2 = random_unitary(2, seed=int(rng.integers(2**31))).tensor(
                        random_unitary(2, seed=int(rng.integers(2**31))))
                    u = (k1 @ Operator(core) @ k2).data
                qc.append(UnitaryGate(u), [p, q])
            yield dict(seed=s), qc
    elif workload == "W4":
        k = 0
        for sd in S_VALUES:
            for s in range(size):
                rng = np.random.default_rng(base + k)
                k += 1
                qc = QuantumCircuit(n)
                for _ in range(3):
                    for q in range(n):
                        qc.ry(float(rng.normal(0, sd)), q)
                        qc.rz(float(rng.normal(0, sd)), q)
                    for q in range(n - 1):
                        qc.cx(q, q + 1)
                yield dict(s=sd, seed=s), qc


# ---------------------------------------------------------------- compilers
class Compilers:
    def __init__(self, repo):
        sys.path.insert(0, os.path.join(repo, "benchmarks"))
        sys.path.insert(0, repo)
        import core_fix_c2_eval as H
        self.psf = H.load_module(os.path.join(repo, "psf_compile.py"), "psf_compile")
        self.psfng = H.load_module(os.path.join(repo, "psf_compile.py"), "psf_compile_noguard")
        self.psfng.USE_CX_GUARD = False
        lay = H.load_module(os.path.join(repo, "benchmarks", "psf_smart_layout.py"), "psl_c2")
        sys.modules["psf_smart_layout"] = lay
        import psf_zero_core
        import qiskit
        self.meta = dict(psf=self.psf.VERSION, layout=lay.LAYOUT_VERSION,
                         core=getattr(psf_zero_core, "CORE_VERSION", None), qiskit=qiskit.__version__,
                         python=sys.version.split()[0],
                         git_head=subprocess.run(["git", "-C", repo, "rev-parse", "--short=7", "HEAD"],
                                                 capture_output=True, text=True).stdout.strip(),
                         sha=dict(script=norm_sha(os.path.abspath(__file__)),
                                  psf=norm_sha(os.path.join(repo, "psf_compile.py"))))

    def compile(self, name, qc):
        from qiskit import transpile
        from qiskit.transpiler import CouplingMap
        cm = CouplingMap.from_line(qc.num_qubits)
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if name.startswith("QK"):
                level = int(name[2])
                basis = {"QK3CZ": ["cz", "rz", "sx", "x"], "QK3U": ["cx", "u"]}.get(name, ["cx", "rz", "sx", "x"])
                return transpile(qc, basis_gates=basis, coupling_map=cm, optimization_level=level,
                                 seed_transpiler=0)
            mod = self.psf if name == "PSF" else self.psfng
            return mod.compile_for_hardware(qc, coupling_map=cm, basis_gates=["cx", "rz", "sx", "x"],
                                            entangling_basis="cx", layout_search=True, seed_transpiler=0)


def infidelity(qc, out):
    from qiskit.quantum_info import Operator, average_gate_fidelity
    op = Operator.from_circuit(out) if out.layout is not None else Operator(out)
    return float(1 - average_gate_fidelity(op, Operator(qc)))


def run(args):
    C = Compilers(args.repo)
    meta = dict(C.meta, n=args.n, workload=args.workload, smoke=args.smoke,
                started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print("META", json.dumps(meta), flush=True)
    name = f"b17_{args.workload}_n{args.n}{'_smoke' if args.smoke else ''}.jsonl"
    path = os.path.join(args.out, name)
    t0 = time.time()
    k = 0
    with open(path + ".tmp", "w") as f:
        f.write(json.dumps(dict(meta=meta)) + "\n")
        for params, qc in circuits(args.workload, args.n, args.smoke):
            row = dict(params=params, res={})
            for comp in COMPILERS:
                g0 = dict(C.psf.GUARD_STATS) if comp == "PSF" else None
                t1 = time.perf_counter()
                try:
                    out = C.compile(comp, qc)
                    tc = time.perf_counter() - t1
                    r = dict(inf=infidelity(qc, out), t=tc,
                             two_q=sum(1 for ins in out.data if ins.operation.num_qubits == 2))
                    if g0 is not None:
                        r["guard_rejected"] = C.psf.GUARD_STATS["zsx_rejected"] - g0["zsx_rejected"]
                except Exception as ex:
                    r = dict(error=f"{type(ex).__name__}: {ex}"[:200], t=time.perf_counter() - t1)
                row["res"][comp] = r
            f.write(json.dumps(row) + "\n")
            k += 1
            if k % 50 == 0:
                print(f"{k} circuits ({time.time() - t0:.0f} s)", flush=True)
    os.replace(path + ".tmp", path)
    print(f"wrote {name}: {k} circuits, {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------- scoring (json only)
def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def load(out):
    data = {}
    smoke = None
    for p in sorted(glob.glob(os.path.join(out, "b17_W*_n*.jsonl"))):
        with open(p) as f:
            lines = [json.loads(x) for x in f]
        m = lines[0]["meta"]
        smoke = m["smoke"]
        data[(m["workload"], m["n"])] = dict(meta=m, rows=lines[1:])
    return data, smoke


def score(args):
    data, smoke = load(args.out)
    L = [f"# b17 score{' (SMOKE -- not a result)' if smoke else ''}", ""]
    metas = [d["meta"] for d in data.values()]
    L.append("versions " + json.dumps({k: metas[0][k] for k in ("psf", "layout", "core", "qiskit", "python",
                                                                  "git_head")}) if metas else "no data")
    expected = [(w, n) for w in WORKLOADS for n in (4, 6)]
    missing = [k for k in expected if k not in data]
    errors = sum(1 for d in data.values() for r in d["rows"] for v in r["res"].values() if "error" in v)
    sizes = SMOKE_SIZES if smoke else SIZES
    want = dict(W1=sizes["W1"] * len(DT_VALUES) * len(R_VALUES), W2=sizes["W2"], W3=sizes["W3"],
                W4=sizes["W4"] * len(S_VALUES))
    short = [k for k, d in data.items() if len(d["rows"]) != want[k[0]]]

    def fails(w, comp, sel=lambda r: True):
        tot = bad = 0
        worst = 0.0
        for n in (4, 6):
            for r in data.get((w, n), {"rows": []})["rows"]:
                if not sel(r):
                    continue
                v = r["res"][comp]
                tot += 1
                if "error" in v or v["inf"] > FAIL:
                    bad += 1
                if "inf" in v:
                    worst = max(worst, v["inf"])
        return bad, tot, worst

    ctrl = [fails(w, c) for w in WORKLOADS for c in ("QK3CZ", "QK3U")]
    p0 = not missing and not short and errors == 0 and all(b == 0 for b, _, _ in ctrl)
    L.append(f"P0: {'PASS' if p0 else 'FAIL'} -- files {len(data)} of 8 (missing {missing}), short {short}, "
             f"compile errors {errors}, control failures {sum(b for b, _, _ in ctrl)}")
    L += ["", "## Failures (1 - F_avg > 1e-6) / circuits, worst 1 - F_avg", "",
          "| workload | " + " | ".join(COMPILERS) + " |", "|---|" + "---|" * len(COMPILERS)]
    tab = {}
    for w in WORKLOADS:
        cells = []
        for c in COMPILERS:
            b, t, wv = fails(w, c)
            tab[(w, c)] = (b, t, wv)
            cells.append(f"{b}/{t} ({wv:.1e})")
        L.append(f"| {w} | " + " | ".join(cells) + " |")
    L += ["", "## W1 by cell: QK3 failures / circuits (rows dt, columns r = Jz/Jx)", "",
          "| dt | " + " | ".join(f"{r:g}" for r in R_VALUES) + " |", "|---|" + "---|" * len(R_VALUES)]
    for dt in DT_VALUES:
        cells = []
        for r in R_VALUES:
            b, t, _ = fails("W1", "QK3", lambda x, dt=dt, r=r: x["params"]["dt"] == dt and x["params"]["r"] == r)
            cells.append(f"{b}/{t}")
        L.append(f"| {dt:g} | " + " | ".join(cells) + " |")
    g = sum(v["res"]["PSF"].get("guard_rejected", 0) for d in data.values() for v in d["rows"])
    L.append(f"\nPSF guard rejections of the ZSX decomposer (all workloads): {g}")
    rate = lambda w, c: tab[(w, c)][0] / max(tab[(w, c)][1], 1)
    h1 = verdict(rate("W2", "QK2") >= 0.10 and rate("W2", "QK3") >= 0.10,
                 tab[("W2", "QK2")][0] == 0 and tab[("W2", "QK3")][0] == 0)
    h2 = verdict(all(tab[("W3", c)][0] == 0 for c in ("QK1", "QK2", "QK3")),
                 any(tab[("W3", c)][0] > 0 for c in ("QK1", "QK2", "QK3")))
    h3 = verdict(all(tab[(w, "PSF")][0] == 0 for w in WORKLOADS), any(tab[(w, "PSF")][0] > 0 for w in WORKLOADS))
    h4 = verdict(rate("W2", "PSFNG") >= 0.01, tab[("W2", "PSFNG")][0] == 0)
    h5 = verdict(tab[("W1", "QK3")][0] >= 1, tab[("W1", "QK3")][0] == 0)
    L += ["", "## Predictions", "",
          f"- H1 (QK2 and QK3 fail on >= 10% of W2): **{h1}**",
          f"- H2 (QK1-3 never fail on Haar blocks, W3): **{h2}**",
          f"- H3 (the PSF release never fails, any workload): **{h3}**",
          f"- H4 (PSF with the guard off fails on >= 1% of W2): **{h4}**",
          f"- H5 (a realistic physics workload hits the bug: QK3 fails on >= 1 W1 circuit): **{h5}**"]
    t = {c: np.median([v["res"][c]["t"] for d in data.values() for v in d["rows"]]) for c in COMPILERS}
    tq = {c: np.mean([v["res"][c].get("two_q", np.nan) for d in data.values() for v in d["rows"]]) for c in COMPILERS}
    L += ["", "Reported: median compile s " + ", ".join(f"{c} {t[c]:.3f}" for c in COMPILERS),
          "Reported: mean two-qubit count " + ", ".join(f"{c} {tq[c]:.1f}" for c in COMPILERS)]
    txt = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "score.md"), "w") as f:
        f.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("run", "score"))
    ap.add_argument("--repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, choices=(4, 6))
    ap.add_argument("--workload", choices=WORKLOADS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    (run if args.part == "run" else score)(args)


if __name__ == "__main__":
    main()
