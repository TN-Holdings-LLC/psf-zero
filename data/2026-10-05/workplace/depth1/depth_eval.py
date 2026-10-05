"""depth_eval.py -- QML-2 "DEPTH", stage 1 (workplace sandbox, CPU only).

Question: on a noisy fake device, how does a data re-uploading quantum classifier trained on real data behave as it
is made deeper, and how much do compilers move that? Does fine-tuning through the compiler and the noise help?

Model (n qubits, L layers): per layer RY(pi/2 * x_q) on every qubit; RY(a) RZ(b) on every qubit; CZ ring
(0,1),(1,2),...,(n-1,0). Then RY(c) on every qubit. Output z = <Z_0>; prediction sign(z); loss MSE(y, z).
Data (bundled with scikit-learn, no download): breast_cancer (BC) and digits 3 vs 8 (D38); 80/20 stratified split,
standardised, PCA to n components, scaled to [-1, 1] on the training set.
Training (noiseless): Adam (lr 0.05, batch 64, 300 steps) on exact adjoint gradients (numpy statevector).
Arms: RPSF = psf_compile, target + placement_refine (2026-10-02.2's call); C12 = candidate 2026-10-05.c12 with the
recommended call (+ final_resynthesis="select", compare_level3, compare_floor, candidate_score="hybrid");
L3T = Qiskit level 3 with the Target, approximation_degree 1.0.
Noise: NoiseModel.from_backend restricted to the compiled circuit's touched qubits (same errors, renumbered),
Aer density matrix, z of the qubit carrying logical 0 read exactly; readout error of that physical qubit (Aer's
asymmetric assignment probabilities) and 4,000 shots x 20 repetitions applied in the score, with the same random
numbers in every arm.

  python depth_eval.py train   --dataset BC --n 6 --out DIR [--dry]
  python depth_eval.py deploy  --dataset BC --n 6 --device FakeAuckland --arm C12 --c12 PATH --out DIR [--dry]
  python depth_eval.py finetune --seed 1 --L 12 --c12 PATH --out DIR [--dry]
  python depth_eval.py score   --out DIR
"""
import argparse
import contextlib
import glob
import hashlib
import io
import json
import os
import sys
import time
import warnings
import zlib

import numpy as np

warnings.simplefilter("ignore")
DATASETS = ("BC", "D38")
NS = (4, 6)
LS = (1, 2, 4, 8, 12, 16)
DEVICES = ("FakeAuckland", "FakeTorino")
ARMS = ("RPSF", "C12", "L3T")
SHOTS, REPS = 4000, 20
SPLIT_SEED, INIT_SEED = 1, 1          # scored; the pilot used 0, the dry run uses 2
FT = dict(dataset="BC", n=6, device="FakeAuckland", arm="C12", steps=40, batch=16, a=0.3, c=0.1, A=5.0)  # a: 0.1 in the dry run, raised before the lock (disclosed)


def norm_sha(path):
    with open(path, "rb") as f:
        txt = f.read().decode("utf-8").replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in txt.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------ data
def data(name, n, split_seed):
    from sklearn.datasets import load_breast_cancer, load_digits
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    if name == "BC":
        X, y = load_breast_cancer(return_X_y=True)
        y = np.where(y == 1, 1.0, -1.0)
    elif name == "D38":
        X, y = load_digits(return_X_y=True)
        keep = (y == 3) | (y == 8)
        X, y = X[keep], np.where(y[keep] == 3, 1.0, -1.0)
    else:
        raise ValueError(name)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=split_seed)
    sc = StandardScaler().fit(Xtr)
    pca = PCA(n_components=n, random_state=0).fit(sc.transform(Xtr))
    Ztr, Zte = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
    lo, hi = Ztr.min(0), Ztr.max(0)
    f = lambda Z: np.clip(2 * (Z - lo) / (hi - lo) - 1, -1, 1)
    return f(Ztr), ytr, f(Zte), yte


# ------------------------------------------------------------------------------------------------ numpy model
def n_params(n, L):
    return 2 * n * L + n


def _ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2).astype(complex)


def _rz(t):
    e = np.exp(-0.5j * np.asarray(t, float))
    z = np.zeros(np.shape(t) + (2, 2), complex)
    z[..., 0, 0], z[..., 1, 1] = e, np.conj(e)
    return z


def _dry(t):   # d/dt RY(t)
    c, s = np.cos(t / 2), np.sin(t / 2)
    return 0.5 * np.stack([np.stack([-s, -c], -1), np.stack([c, -s], -1)], -2).astype(complex)


def _drz(t):
    e = np.exp(-0.5j * np.asarray(t, float))
    z = np.zeros(np.shape(t) + (2, 2), complex)
    z[..., 0, 0], z[..., 1, 1] = -0.5j * e, 0.5j * np.conj(e)
    return z


def _apply1(psi, U, q, n):
    """psi (B, 2,...,2) with axis 1 + (n-1-q) = qubit q (Qiskit order); U (2,2) or (B,2,2)."""
    ax = 1 + (n - 1 - q)
    psi = np.moveaxis(psi, ax, -1)
    psi = psi @ U.T if U.ndim == 2 else np.einsum("b...j,bij->b...i", psi, U)
    return np.moveaxis(psi, -1, ax)


def _ring_phase(n):
    idx = np.arange(2 ** n)
    bits = (idx[:, None] >> np.arange(n)[None, :]) & 1
    par = np.zeros(2 ** n, int)
    for a in range(n):
        par ^= bits[:, a] & bits[:, (a + 1) % n]
    return np.where(par == 1, -1.0, 1.0)


def _gates(theta, X, n, L):
    """The circuit as a list of ('u', U, q, k, kind) / ('cz',); k = parameter index or None."""
    g, k = [], 0
    for _ in range(L):
        for q in range(n):
            g.append(("u", _ry(np.pi / 2 * X[:, q]), q, None, None))
        for q in range(n):
            g.append(("u", _ry(theta[k]), q, k, "ry")); k += 1
            g.append(("u", _rz(theta[k]), q, k, "rz")); k += 1
        g.append(("cz",))
    for q in range(n):
        g.append(("u", _ry(theta[k]), q, k, "ry")); k += 1
    return g


def _z0_vec(n):
    return np.where((np.arange(2 ** n) & 1) == 0, 1.0, -1.0)


def forward(theta, X, n, L):
    B = X.shape[0]
    psi = np.zeros((B,) + (2,) * n, complex)
    psi[(slice(None),) + (0,) * n] = 1
    ph = _ring_phase(n)
    for g in _gates(theta, X, n, L):
        psi = (psi.reshape(B, -1) * ph).reshape(psi.shape) if g[0] == "cz" else _apply1(psi, g[1], g[2], n)
    return (np.abs(psi.reshape(B, -1)) ** 2) @ _z0_vec(n)


def loss_grad(theta, X, y, n, L):
    """MSE loss and its exact gradient (adjoint method)."""
    B = X.shape[0]
    gates = _gates(theta, X, n, L)
    ph = _ring_phase(n)
    psi = np.zeros((B,) + (2,) * n, complex)
    psi[(slice(None),) + (0,) * n] = 1
    for g in gates:
        psi = (psi.reshape(B, -1) * ph).reshape(psi.shape) if g[0] == "cz" else _apply1(psi, g[1], g[2], n)
    flat = psi.reshape(B, -1)
    z = (np.abs(flat) ** 2) @ _z0_vec(n)
    w = -2.0 * (y - z) / B                                   # dL/dz per sample
    lam = (flat * _z0_vec(n)[None, :] * w[:, None]).reshape(psi.shape)
    grad = np.zeros(len(theta))
    for g in reversed(gates):
        if g[0] == "cz":
            psi = (psi.reshape(B, -1) * ph).reshape(psi.shape)
            lam = (lam.reshape(B, -1) * ph).reshape(lam.shape)
            continue
        _, U, q, k, kind = g
        Ud = np.conj(np.swapaxes(U, -1, -2))
        psi = _apply1(psi, Ud, q, n)                          # state before the gate
        if k is not None:
            dU = _dry(theta[k]) if kind == "ry" else _drz(theta[k])
            mu = _apply1(psi, dU, q, n)
            grad[k] = 2.0 * np.real(np.sum(np.conj(lam) * mu))
        lam = _apply1(lam, Ud, q, n)
    return float(np.mean((y - z) ** 2)), grad


def train(Xtr, ytr, n, L, seed, steps=300, lr=0.05, batch=64):
    rng = np.random.default_rng(seed)
    th = rng.normal(0, 0.3, n_params(n, L))
    m = np.zeros_like(th); v = np.zeros_like(th)
    for t in range(1, steps + 1):
        bi = rng.choice(len(Xtr), min(batch, len(Xtr)), replace=False)
        _, g = loss_grad(th, Xtr[bi], ytr[bi], n, L)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        th = th - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return th


# ------------------------------------------------------------------------------------------------ circuits, compilers
def circuit(x, theta, n, L):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    k = 0
    for _ in range(L):
        for q in range(n):
            qc.ry(np.pi / 2 * x[q], q)
        for q in range(n):
            qc.ry(theta[k], q); k += 1
            qc.rz(theta[k], q); k += 1
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    for q in range(n):
        qc.ry(theta[k], q); k += 1
    return qc


def load_psf(path):
    import importlib.util
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))   # psf_smart_layout.py must sit next to it
    spec = importlib.util.spec_from_file_location("psf_compile", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["psf_compile"] = mod
    spec.loader.exec_module(mod)
    return mod


def compiler(arm, be, P):
    from qiskit import transpile
    t = be.target
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    base = dict(coupling_map=t.build_coupling_map(), basis_gates=basis, entangling_basis="cx", layout_search=True,
                seed_transpiler=0, target=t, placement_refine=True)
    full = dict(base, final_resynthesis="select", compare_level3=True, compare_floor=True, candidate_score="hybrid")
    f = {"RPSF": lambda qc: P.compile_for_hardware(qc, **base),
         "C12": lambda qc: P.compile_for_hardware(qc, **full),
         "L3T": lambda qc: transpile(qc, target=t, optimization_level=3, seed_transpiler=0,
                                     approximation_degree=1.0)}[arm]

    def run(qc):
        with contextlib.redirect_stdout(io.StringIO()):
            return f(qc)
    return run


class Noisy:
    """Density-matrix simulation of a compiled circuit on its touched qubits, with the device's noise model
    restricted to those qubits (same QuantumError objects, renumbered)."""

    def __init__(self, be):
        from qiskit_aer.noise import NoiseModel
        self.nm = NoiseModel.from_backend(be)
        self.cache = {}

    def reduced_model(self, active):
        key = tuple(active)
        if key in self.cache:
            return self.cache[key]
        from qiskit_aer.noise import NoiseModel
        idx = {p: i for i, p in enumerate(active)}
        m = NoiseModel(basis_gates=self.nm.basis_gates)
        for gate, d in self.nm._local_quantum_errors.items():
            for qs, err in d.items():
                if all(q in idx for q in qs):
                    m.add_quantum_error(err, gate, [idx[q] for q in qs])
        if len(self.cache) > 2000:
            self.cache.clear()
        self.cache[key] = m
        return m

    def readout(self, p):
        e = self.nm._local_readout_errors.get((p,))
        if e is None:
            return 0.0, 0.0
        pr = np.asarray(e.probabilities)
        return float(pr[0][1]), float(pr[1][0])          # P(1|0), P(0|1)

    @staticmethod
    def reduce(out):
        from qiskit import QuantumCircuit
        fin = list(out.layout.final_index_layout(filter_ancillas=True)) if out.layout is not None else \
            list(range(out.num_qubits))
        active = sorted({out.find_bit(b).index for i in out.data for b in i.qubits} | set(fin))
        idx = {p: i for i, p in enumerate(active)}
        red = QuantumCircuit(len(active))
        for i in out.data:
            if i.operation.name in ("barrier", "measure", "delay"):
                continue
            red.append(i.operation, [idx[out.find_bit(b).index] for b in i.qubits])
        return red, active, idx, fin

    def z(self, out, noisy=True):
        from qiskit_aer import AerSimulator
        red, active, idx, fin = self.reduce(out)
        c = red.copy()
        c.save_density_matrix(qubits=[idx[fin[0]]])
        sim = AerSimulator(method="density_matrix", noise_model=self.reduced_model(active) if noisy else None)
        rho = np.asarray(sim.run(c).result().data()["density_matrix"])
        return float(np.real(rho[0, 0] - rho[1, 1])), fin[0], len(active)

    def z_fullwidth(self, out):
        """Reference for P0: the whole device, Aer's own noise model, save_density_matrix (as qml_core)."""
        from qiskit_aer import AerSimulator
        fin = list(out.layout.final_index_layout(filter_ancillas=True))
        c = out.copy()
        c.save_density_matrix(qubits=[fin[0]])
        rho = np.asarray(AerSimulator(method="density_matrix", noise_model=self.nm).run(c).result()
                         .data()["density_matrix"])
        return float(np.real(rho[0, 0] - rho[1, 1]))


def shot_z(z, e01, e10, key):
    """REPS finite-shot estimates of z with readout error; random numbers depend on `key` only (not the arm)."""
    p0 = (1 + z) / 2
    p0m = min(1.0, max(0.0, p0 * (1 - e01) + (1 - p0) * e10))
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    u = rng.random((REPS, SHOTS))
    return 2.0 * np.mean(u < p0m, axis=1) - 1.0


# ------------------------------------------------------------------------------------------------ modes
def seeds(dry):
    return (2, 2) if dry else (SPLIT_SEED, INIT_SEED)


def theta_path(out, ds, n, L):
    return os.path.join(out, f"theta_{ds}_n{n}_L{L}.npy")


def meta(args, **kw):
    import qiskit, qiskit_aer, sklearn
    m = dict(script_sha=norm_sha(os.path.abspath(__file__)), dry=bool(args.dry), qiskit=qiskit.__version__,
             aer=qiskit_aer.__version__, sklearn=sklearn.__version__,
             started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if getattr(args, "c12", None):
        m["c12_sha"] = norm_sha(args.c12)
    m.update(kw)
    return m


def do_train(args):
    split, init = seeds(args.dry)
    Ls = (1, 4, 12) if args.dry else LS
    Xtr, ytr, Xte, yte = data(args.dataset, args.n, split)
    rows = []
    for L in Ls:
        t0 = time.perf_counter()
        th = train(Xtr, ytr, args.n, L, seed=init + 1000 * L, steps=60 if args.dry else 300)
        np.save(theta_path(args.out, args.dataset, args.n, L), th)
        z = forward(th, Xte, args.n, L)
        rows.append(dict(L=L, ideal_acc=float(np.mean(np.sign(z) == yte)), ideal_margin=float(np.mean(yte * z)),
                         train_loss=float(np.mean((ytr - forward(th, Xtr, args.n, L)) ** 2)),
                         s=round(time.perf_counter() - t0, 1)))
        print("TRAIN", json.dumps(rows[-1]), flush=True)
    json.dump(dict(meta=meta(args, mode="train", dataset=args.dataset, n=args.n, ntest=len(yte)), rows=rows),
              open(os.path.join(args.out, f"train_{args.dataset}_n{args.n}.json"), "w"), indent=1)


def do_deploy(args):
    from qiskit_ibm_runtime import fake_provider as fp
    split, _ = seeds(args.dry)
    Ls = (1, 4, 12) if args.dry else LS
    Xtr, ytr, Xte, yte = data(args.dataset, args.n, split)
    if args.dry:
        Xte, yte = Xte[:12], yte[:12]
    be = getattr(fp, args.device)()
    P = load_psf(args.c12)
    comp = compiler(args.arm, be, P)
    sim = Noisy(be)
    rows = []
    for L in Ls:
        th = np.load(theta_path(args.out, args.dataset, args.n, L))
        zid = forward(th, Xte, args.n, L)
        for i, (x, yv) in enumerate(zip(Xte, yte)):
            qc = circuit(x, th, args.n, L)
            t0 = time.perf_counter()
            out = comp(qc)
            tc = time.perf_counter() - t0
            z0, p, touched = sim.z(out, noisy=False)
            zn, _, _ = sim.z(out, noisy=True)
            e01, e10 = sim.readout(p)
            zs = shot_z(zn, e01, e10, f"{args.dataset}|{args.n}|{args.device}|{L}|{i}")
            row = dict(L=L, i=i, y=float(yv), z_ideal=float(zid[i]), z_compiled_noiseless=z0, z_noisy=zn,
                       shot_acc=float(np.mean(np.sign(zs) == yv)), shot_flip=float(np.mean(np.sign(zs) != np.sign(zid[i]))),
                       e01=e01, e10=e10, n2q=sum(1 for g in out.data if len(g.qubits) == 2), touched=touched,
                       compile_s=round(tc, 4))
            if i < 2:   # P0: reduced simulation equals the whole-device simulation
                row["z_fullwidth"] = sim.z_fullwidth(out)
            rows.append(row)
        sel = [r for r in rows if r["L"] == L]
        print("DEPLOY", json.dumps(dict(L=L, acc=np.mean([np.sign(r["z_noisy"]) == r["y"] for r in sel]),
                                         margin=np.mean([r["y"] * r["z_noisy"] for r in sel]),
                                         shot_acc=np.mean([r["shot_acc"] for r in sel]),
                                         n2q=np.mean([r["n2q"] for r in sel]))), flush=True)
    name = f"deploy_{args.dataset}_n{args.n}_{args.device}_{args.arm}.json"
    json.dump(dict(meta=meta(args, mode="deploy", dataset=args.dataset, n=args.n, device=args.device, arm=args.arm,
                             psf_version=P.VERSION), rows=rows),
              open(os.path.join(args.out, name), "w"))


def evaluate_deployed(th, X, y, n, L, comp, sim, tag):
    zid = forward(th, X, n, L)
    res = []
    for i, x in enumerate(X):
        out = comp(circuit(x, th, n, L))
        zn, p, _ = sim.z(out)
        e01, e10 = sim.readout(p)
        zs = shot_z(zn, e01, e10, f"{tag}|{i}")
        res.append((zn, float(np.mean(np.sign(zs) == y[i])), float(zid[i])))
    zn = np.array([r[0] for r in res])
    return dict(acc=float(np.mean(np.sign(zn) == y)), margin=float(np.mean(y * zn)),
                shot_acc=float(np.mean([r[1] for r in res])),
                ideal_acc=float(np.mean(np.sign(zid) == y)), ideal_margin=float(np.mean(y * zid)))


def do_finetune(args):
    """From the noiselessly trained theta*: SPSA for FT['steps'] steps either through the compiler and the noise
    (FTN) or noiselessly (FT0, control, same random numbers); then deploy the test set through the compiler."""
    from qiskit_ibm_runtime import fake_provider as fp
    split, _ = seeds(args.dry)
    n, L, ds = FT["n"], args.L, FT["dataset"]
    Xtr, ytr, Xte, yte = data(ds, n, split)
    steps = 3 if args.dry else FT["steps"]
    if args.dry:
        Xte, yte = Xte[:12], yte[:12]
    be = getattr(fp, FT["device"])()
    P = load_psf(args.c12)
    comp = compiler(FT["arm"], be, P)
    sim = Noisy(be)
    th0 = np.load(theta_path(args.out, ds, n, L))
    result = dict(DEP=evaluate_deployed(th0, Xte, yte, n, L, comp, sim, f"ft|{ds}|{n}|{L}"))
    print("FT DEP", json.dumps(result["DEP"]), flush=True)
    for kind in ("FT0", "FTN"):
        rng = np.random.default_rng(10_000 * args.seed + L)
        th = th0.copy()
        curve = []
        for k in range(1, steps + 1):
            ak = FT["a"] / (k + FT["A"]) ** 0.602
            ck = FT["c"] / k ** 0.101
            bi = rng.choice(len(Xtr), FT["batch"], replace=False)
            delta = rng.choice([-1.0, 1.0], size=len(th))
            losses = []
            for sgn in (1, -1):
                tp = th + sgn * ck * delta
                if kind == "FT0":
                    z = forward(tp, Xtr[bi], n, L)
                else:
                    z = np.array([sim.z(comp(circuit(Xtr[j], tp, n, L)))[0] for j in bi])
                losses.append(float(np.mean((ytr[bi] - z) ** 2)))
            th = th - ak * (losses[0] - losses[1]) / (2 * ck) * delta
            curve.append(dict(step=k, loss_plus=losses[0], loss_minus=losses[1]))
        result[kind] = evaluate_deployed(th, Xte, yte, n, L, comp, sim, f"ft|{ds}|{n}|{L}")
        result[kind + "_curve"] = curve
        print("FT", kind, json.dumps(result[kind]), flush=True)
    json.dump(dict(meta=meta(args, mode="finetune", L=L, seed=args.seed, **FT), result=result),
              open(os.path.join(args.out, f"finetune_L{L}_s{args.seed}.json"), "w"), indent=1)


# ------------------------------------------------------------------------------------------------ score
def verdict(ok, bad):
    return "CONFIRMED" if ok else ("REFUTED" if bad else "AMBIGUOUS")


def do_score(args):
    D = {}
    for p in glob.glob(os.path.join(args.out, "deploy_*.json")):
        d = json.load(open(p))
        m = d["meta"]
        D[(m["dataset"], m["n"], m["device"], m["arm"])] = d["rows"]
    T = {}
    for p in glob.glob(os.path.join(args.out, "train_*.json")):
        d = json.load(open(p))
        T[(d["meta"]["dataset"], d["meta"]["n"])] = {r["L"]: r for r in d["rows"]}
    F = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.out, "finetune_*.json")))]
    dry = any(d for d in [json.load(open(p))["meta"]["dry"] for p in glob.glob(os.path.join(args.out, "*.json"))])
    Ls = sorted({r["L"] for rows in D.values() for r in rows})
    L = [f"# DEPTH stage 1 score{' (DRY RUN -- not a result)' if dry else ''}", ""]

    def cell(ds, n, dev, arm, Lv, key):
        rows = [r for r in D[(ds, n, dev, arm)] if r["L"] == Lv]
        if key == "acc":
            return float(np.mean([np.sign(r["z_noisy"]) == r["y"] for r in rows]))
        if key == "margin":
            return float(np.mean([r["y"] * r["z_noisy"] for r in rows]))
        if key == "shot_acc":
            return float(np.mean([r["shot_acc"] for r in rows]))
        if key == "flip":
            return float(np.mean([np.sign(r["z_noisy"]) != np.sign(r["z_ideal"]) for r in rows]))
        if key == "n2q":
            return float(np.mean([r["n2q"] for r in rows]))

    # P0
    n_files = len(D)
    exp_files = len(DATASETS) * len(NS) * len(DEVICES) * len(ARMS)
    exact = max(abs(r["z_compiled_noiseless"] - r["z_ideal"]) for rows in D.values() for r in rows)
    fw = [abs(r["z_fullwidth"] - r["z_noisy"]) for rows in D.values() for r in rows if "z_fullwidth" in r]
    ft_ok = len(F) == (4 if not dry else len(F))
    p0 = n_files == exp_files and exact <= 1e-6 and max(fw) <= 1e-9 and ft_ok
    L += [f"P0: {'PASS' if p0 else 'FAIL'} -- deploy files {n_files}/{exp_files}; max |z_compiled_noiseless - z_ideal| "
          f"{exact:.2e} (<= 1e-6); max |reduced - whole-device| {max(fw):.2e} over {len(fw)} circuits (<= 1e-9); "
          f"finetune files {len(F)}", ""]
    # tables
    for dev in DEVICES:
        L += [f"## {dev}", "", "| dataset | n | L | ideal acc / margin | " + " | ".join(
            f"{a} acc / shot acc / margin / flip / 2q" for a in ARMS) + " |",
              "|" + "---|" * (4 + len(ARMS))]
        for ds in DATASETS:
            for n in NS:
                for Lv in Ls:
                    t = T[(ds, n)][Lv]
                    L.append(f"| {ds} | {n} | {Lv} | {t['ideal_acc']:.3f} / {t['ideal_margin']:.3f} | " + " | ".join(
                        f"{cell(ds, n, dev, a, Lv, 'acc'):.3f} / {cell(ds, n, dev, a, Lv, 'shot_acc'):.3f} / "
                        f"{cell(ds, n, dev, a, Lv, 'margin'):.3f} / {cell(ds, n, dev, a, Lv, 'flip'):.3f} / "
                        f"{cell(ds, n, dev, a, Lv, 'n2q'):.0f}" for a in ARMS) + " |")
        L.append("")
    res = {}
    A = "FakeAuckland"
    # H1: on FakeAuckland, n = 6, the deployed margin has an interior maximum: margin at the deepest L below 0.8 x the
    # best margin over L, for every dataset and arm. REFUTED if the deepest L has the largest margin for any.
    ok1 = bad1 = True; bad1 = False; det = []
    for ds in DATASETS:
        for a in ARMS:
            ms = [cell(ds, 6, A, a, Lv, "margin") for Lv in Ls]
            det.append(f"{ds}/{a}: deepest {ms[-1]:.3f}, best {max(ms):.3f} at L={Ls[int(np.argmax(ms))]}")
            ok1 &= ms[-1] < 0.8 * max(ms)
            bad1 |= int(np.argmax(ms)) == len(Ls) - 1
    res["H1"] = verdict(ok1, bad1)
    L.append(f"- H1 (noise-limited depth, margin; FakeAuckland, n=6): **{res['H1']}** ({'; '.join(det)})")
    # H2: on FakeAuckland, n = 6, the shot-based accuracy at the deepest L is at least 2 test points below the best over
    # L, for at least one dataset with every arm. REFUTED if the deepest L is the best (or tied best) for every
    # dataset and arm.
    det = []; okd = []; badd = True
    for ds in DATASETS:
        nt = len([r for r in D[(ds, 6, A, ARMS[0])] if r["L"] == Ls[0]])
        oks = []
        for a in ARMS:
            sa = [cell(ds, 6, A, a, Lv, "shot_acc") for Lv in Ls]
            oks.append(max(sa) - sa[-1] >= 2.0 / nt - 1e-12)
            badd &= sa[-1] >= max(sa) - 1e-12
            det.append(f"{ds}/{a}: deepest {sa[-1]:.3f}, best {max(sa):.3f}")
        okd.append(all(oks))
    res["H2"] = verdict(any(okd), badd)
    L.append(f"- H2 (noise-limited depth, shot accuracy; FakeAuckland, n=6): **{res['H2']}** ({'; '.join(det)})")
    # H3-H5: pooled over datasets, n and L, per device
    det3, det4, det5 = [], [], []
    ok3 = ok4 = ok5 = True; bad3 = bad4 = bad5 = False
    for dev in DEVICES:
        def pool(a, key):
            return float(np.mean([cell(ds, n, dev, a, Lv, key) for ds in DATASETS for n in NS for Lv in Ls]))
        mR, mC, mL = pool("RPSF", "margin"), pool("C12", "margin"), pool("L3T", "margin")
        fR, fC = pool("RPSF", "flip"), pool("C12", "flip")
        det3.append(f"{dev} C12-RPSF {mC - mR:+.4f}"); det4.append(f"{dev} C12-L3T {mC - mL:+.4f}")
        det5.append(f"{dev} C12 {fC:.4f} vs RPSF {fR:.4f}")
        ok3 &= mC - mR >= 0.005; bad3 |= mC - mR < 0
        ok4 &= abs(mC - mL) <= 0.01; bad4 |= mC < mL - 0.02
        ok5 &= fC <= fR; bad5 |= fC > fR + 0.01
    res["H3"] = verdict(ok3, bad3); res["H4"] = verdict(ok4, bad4); res["H5"] = verdict(ok5, bad5)
    L.append(f"- H3 (C12 keeps more margin than RPSF, pooled, every device): **{res['H3']}** ({'; '.join(det3)})")
    L.append(f"- H4 (C12 level with L3T, pooled margin, every device): **{res['H4']}** ({'; '.join(det4)})")
    L.append(f"- H5 (C12 flips no more predictions than RPSF, pooled, every device): **{res['H5']}** ({'; '.join(det5)})")
    # H6: fine-tuning through the noise (FTN) at L = 12 raises the deployed margin over DEP by >= 0.02 (mean of seeds),
    # and more than the noiseless control FT0 does. REFUTED if FTN < DEP - 0.01.
    f12 = [f["result"] for f in F if f["meta"]["L"] == 12]
    f4 = [f["result"] for f in F if f["meta"]["L"] == 4]
    if f12:
        dN = float(np.mean([r["FTN"]["margin"] - r["DEP"]["margin"] for r in f12]))
        d0 = float(np.mean([r["FT0"]["margin"] - r["DEP"]["margin"] for r in f12]))
        res["H6"] = verdict(dN >= 0.02 and dN > d0, dN < -0.01)
        L.append(f"- H6 (fine-tuning through the noise helps at L=12): **{res['H6']}** (FTN-DEP {dN:+.4f}, FT0-DEP {d0:+.4f})")
    for tag, fs in (("L=4", f4), ("L=12", f12)):
        for r in fs:
            L.append(f"  - {tag}: DEP {r['DEP']}, FT0 {r['FT0']}, FTN {r['FTN']}")
    # gate for stage 2
    gate_peak = res["H1"] == "CONFIRMED" or res["H2"] == "CONFIRMED"
    big = []
    for dev in DEVICES:
        for ds in DATASETS:
            for n in NS:
                nt = len([r for r in D[(ds, n, dev, "C12")] if r["L"] == Ls[0]])
                for Lv in Ls:
                    if abs(cell(ds, n, dev, "C12", Lv, "shot_acc") - cell(ds, n, dev, "RPSF", Lv, "shot_acc")) >= 2.0 / nt - 1e-12:
                        big.append(f"{dev}/{ds}/n{n}/L{Lv}")
    gate_diff = res["H3"] == "CONFIRMED" or len(big) > 0
    L += ["", f"GATE (stage 2): {'GO' if (gate_peak and gate_diff) else 'NO-GO'} -- peak {gate_peak}, compiler "
          f"difference {gate_diff} (cells with |shot acc C12 - RPSF| >= 2 test points: {', '.join(big) or 'none'})"]
    L += ["", "SUMMARY " + json.dumps(res)]
    txt = "\n".join(L)
    open(os.path.join(args.out, "score.md"), "w").write(txt + "\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "deploy", "finetune", "score"])
    ap.add_argument("--dataset"); ap.add_argument("--n", type=int); ap.add_argument("--device"); ap.add_argument("--arm")
    ap.add_argument("--seed", type=int); ap.add_argument("--L", type=int)
    ap.add_argument("--c12"); ap.add_argument("--out", required=True); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"train": do_train, "deploy": do_deploy, "finetune": do_finetune, "score": do_score}[a.mode](a)


if __name__ == "__main__":
    main()
