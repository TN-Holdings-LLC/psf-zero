"""qml_home2_eval.py -- pre-registered home test, part 2 (2026-10-02): the QML test of Addendum 290/291, widened
where Addendum 291 was weak.

Q2W (learning, wide): the Addendum 290 model and data (4 qubits, 2 layers, teacher 23 rule, train seed 31, test
     seed 32), trained from scratch by SPSA (40 steps) with the compiler and the noisy simulator inside the loss,
     for 8 seeds (41-48) on FakeAuckland, FakeTorino and FakeKingston, all four arms: 96 runs.
Q1D (keeping, deep): the same model family with 4 layers. theta* is trained without noise in numpy (Adam on
     finite-difference gradients); teacher rule: the first teacher seed >= 81 whose theta* reaches test accuracy
     >= 0.85. Two test sets: A, 16+16 points with |z_teacher| >= 0.25 (as before); B ("fragile"), 32+32 points
     with 0.01 <= |z_theta*| <= 0.10, signed by theta*. Every circuit goes through every arm on the three devices.
     The score also applies the device's readout error on the measured qubit and samples shots
     (4,000 shots, 20 repetitions, the same random numbers for every arm).
Arms: REL (git 9131cee), C2 (release 2026-10-01.1), A5 (psf_ai_compile a5, with Target), L3T (Qiskit level 3
     with Target) -- exactly as in qml_home_eval.py.
Noise: NoiseModel.from_backend + AerSimulator(density_matrix), z read exactly from the density matrix of the
     final-layout qubits. Circuits are simulated in batches (one Aer job per loss evaluation).

  python qml_home2_eval.py q1 --repo <repo> --out <dir> [--smoke]
  python qml_home2_eval.py q2 --repo <repo> --out <dir> --arm REL|C2|A5|L3T --device <name> --seed <n> [--smoke]
  python qml_home2_eval.py score --out <dir>
"""
import argparse
import contextlib
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
N, RING = 4, [(0, 1), (1, 2), (2, 3), (3, 0)]
REL_COMMIT = "9131cee"
REL_SHA = {"psf_compile.py": "3616efc8b8a7bea184d6170fb7379d509d0bfec9828f4eb2f703dd4d26d8c60b",
           "benchmarks/psf_smart_layout.py": "a639efdef484379d23b4c0a52dffe557c47c30f639c41e4f8521ca608712d875"}
ARMS = ("REL", "C2", "A5", "L3T")
DEVICES = ("FakeAuckland", "FakeTorino", "FakeKingston")
SHOTS, REPS = 4000, 20
CFG = dict(q2=dict(L=2, teacher_from=21, train=31, test=32, theta=40, steps_theta=80, min_acc=0.9,
                   seeds=tuple(range(41, 49)), steps=40, devices=DEVICES),
           deep=dict(L=4, teacher_from=81, train=91, test=92, fragile=94, adam_seed=93, iters=400, min_acc=0.85,
                     n_fragile=32, devices=DEVICES))
SMOKE = dict(q2=dict(L=2, teacher_from=21, train=931, test=932, theta=40, steps_theta=80, min_acc=0.0,
                     seeds=(941,), steps=2, devices=DEVICES),
             deep=dict(L=4, teacher_from=81, train=991, test=992, fragile=994, adam_seed=993, iters=10, min_acc=0.0,
                       n_fragile=1, devices=DEVICES, n_test=1))


# ---------------------------------------------------------------- numpy model (noiseless reference)
def _ry(psi, q, t):
    t = np.broadcast_to(np.asarray(t, float), (psi.shape[0],))
    c, s = np.cos(t / 2), np.sin(t / 2)
    a = np.moveaxis(psi, q + 1, 1)
    a0, a1 = a[:, 0].copy(), a[:, 1].copy()
    sh = (-1,) + (1,) * (a0.ndim - 1)
    a[:, 0] = c.reshape(sh) * a0 - s.reshape(sh) * a1
    a[:, 1] = s.reshape(sh) * a0 + c.reshape(sh) * a1
    return np.moveaxis(a, 1, q + 1)


def _rz(psi, q, t):
    t = np.broadcast_to(np.asarray(t, float), (psi.shape[0],))
    a = np.moveaxis(psi, q + 1, 1).copy()
    sh = (-1,) + (1,) * (a.ndim - 2)
    a[:, 0] *= np.exp(-0.5j * t).reshape(sh)
    a[:, 1] *= np.exp(0.5j * t).reshape(sh)
    return np.moveaxis(a, 1, q + 1)


def _cz(psi, p, q):
    psi = psi.copy()
    idx = [slice(None)] * (N + 1)
    idx[p + 1] = 1
    idx[q + 1] = 1
    psi[tuple(idx)] *= -1
    return psi


def npar(L):
    return L * N * 2 + N


def model_z(X, v, L):
    """<Z> of qubit 0 for each row of X (numpy statevector; axis q+1 is qubit q)."""
    lay, fin = v[:L * N * 2].reshape(L, N, 2), v[L * N * 2:]
    psi = np.zeros((X.shape[0],) + (2,) * N, complex)
    psi[(slice(None),) + (0,) * N] = 1
    for l in range(L):
        for q in range(N):
            psi = _ry(psi, q, math.pi * X[:, q])
        for q in range(N):
            psi = _rz(_ry(psi, q, lay[l, q, 0]), q, lay[l, q, 1])
        for p, q in RING:
            psi = _cz(psi, p, q)
    for q in range(N):
        psi = _ry(psi, q, fin[q])
    pr = np.abs(psi) ** 2
    return pr[:, 0].sum(axis=(1, 2, 3)) - pr[:, 1].sum(axis=(1, 2, 3))


def teacher(seed, L):
    r = np.random.default_rng(seed)
    return np.concatenate([r.uniform(-math.pi, math.pi, L * N * 2), r.uniform(-math.pi, math.pi, N)])


def sample(npos, nneg, seed, f, lo, hi):
    """Balanced points with lo <= |f(x)| <= hi, the sign of f giving the class."""
    r = np.random.default_rng(seed)
    pos, neg = [], []
    while len(pos) < npos or len(neg) < nneg:
        x = r.uniform(-1, 1, (1, N))
        z = f(x)[0]
        if lo <= z <= hi and len(pos) < npos:
            pos.append(x[0])
        elif -hi <= z <= -lo and len(neg) < nneg:
            neg.append(x[0])
    return np.array(pos + neg), np.array([1.0] * npos + [-1.0] * nneg)


def spsa(seed, steps, f, n, log=None):
    """Standard SPSA; the random stream depends only on `seed` (common random numbers across arms)."""
    rs = np.random.default_rng(seed)
    v = rs.normal(0, 0.3, n)
    for k in range(steps):
        ak, ck = 0.6 / (k + 1 + 5) ** 0.602, 0.2 / (k + 1) ** 0.101
        d = rs.choice([-1.0, 1.0], n)
        fp, fm = f(v + ck * d), f(v - ck * d)
        v = v - ak * (fp - fm) / (2 * ck) * d
        if log is not None:
            log.append(dict(step=k + 1, f_plus=fp, f_minus=fm, t=time.time()))
    return v


def adam(seed, iters, f, n, h=1e-4, lr=0.05):
    """Adam on central finite-difference gradients (noiseless numpy training of the deep theta*)."""
    rs = np.random.default_rng(seed)
    v = rs.normal(0, 0.3, n)
    m, s = np.zeros(n), np.zeros(n)
    for k in range(1, iters + 1):
        g = np.zeros(n)
        for i in range(n):
            e = np.zeros(n)
            e[i] = h
            g[i] = (f(v + e) - f(v - e)) / (2 * h)
        m = 0.9 * m + 0.1 * g
        s = 0.999 * s + 0.001 * g * g
        v = v - lr * (m / (1 - 0.9 ** k)) / (np.sqrt(s / (1 - 0.999 ** k)) + 1e-8)
    return v


def problem_q2(c):
    L = c["L"]
    for ts in range(c["teacher_from"], c["teacher_from"] + 20):
        vt = teacher(ts, L)
        f = lambda X: model_z(X, vt, L)
        Xtr, ytr = sample(8, 8, c["train"], f, 0.25, 9)
        Xte, yte = sample(16, 16, c["test"], f, 0.25, 9)
        th = spsa(c["theta"], c["steps_theta"], lambda v: float(np.mean((ytr - model_z(Xtr, v, L)) ** 2)), npar(L))
        acc = float(np.mean(np.sign(model_z(Xte, th, L)) == yte))
        if acc >= c["min_acc"]:
            return dict(teacher_seed=ts, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, theta=th, theta_test_acc=acc)
    return None


def problem_deep(c):
    L = c["L"]
    nt = c.get("n_test", 16)
    for ts in range(c["teacher_from"], c["teacher_from"] + 20):
        vt = teacher(ts, L)
        f = lambda X: model_z(X, vt, L)
        Xtr, ytr = sample(32, 32, c["train"], f, 0.25, 9)
        Xte, yte = sample(nt, nt, c["test"], f, 0.25, 9)
        th = adam(c["adam_seed"], c["iters"], lambda v: float(np.mean((ytr - model_z(Xtr, v, L)) ** 2)), npar(L))
        acc = float(np.mean(np.sign(model_z(Xte, th, L)) == yte))
        print(f"deep teacher {ts}: theta* test accuracy {acc:.3f}", flush=True)
        if acc >= c["min_acc"]:
            nf = c["n_fragile"]
            Xb, sb = sample(nf, nf, c["fragile"], lambda X: model_z(X, th, L), 0.01, 0.10)
            return dict(teacher_seed=ts, Xte=Xte, yte=yte, Xb=Xb, sb=sb, theta=th, theta_test_acc=acc)
    return None


# ---------------------------------------------------------------- qiskit side
def build(x, v, L):
    from qiskit import QuantumCircuit
    lay, fin = v[:L * N * 2].reshape(L, N, 2), v[L * N * 2:]
    qc = QuantumCircuit(N)
    for l in range(L):
        for q in range(N):
            qc.ry(math.pi * float(x[q]), q)
        for q in range(N):
            qc.ry(float(lay[l, q, 0]), q)
            qc.rz(float(lay[l, q, 1]), q)
        for p, q in RING:
            qc.cz(p, q)
    for q in range(N):
        qc.ry(float(fin[q]), q)
    return qc


def logical_z(x, v, L):
    from qiskit.quantum_info import Pauli, Statevector
    return float(np.real(Statevector(build(x, v, L)).expectation_value(Pauli("I" * (N - 1) + "Z"))))


def norm_sha(path):
    with open(path, "rb") as f:
        txt = f.read().decode("utf-8").replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in txt.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


class Stack:
    """Loads the four arms once per process; compiles one circuit at a time and simulates in batches."""

    def __init__(self, repo, out):
        sys.path.insert(0, os.path.join(repo, "benchmarks"))
        sys.path.insert(0, repo)
        import core_fix_c2_eval as H
        self.H = H
        rel_dir = os.path.join(out, "rel_9131cee")
        os.makedirs(rel_dir, exist_ok=True)
        paths = {}
        for rel, want in REL_SHA.items():
            dst = os.path.join(rel_dir, os.path.basename(rel))
            if not os.path.exists(dst):
                txt = subprocess.run(["git", "-C", repo, "show", f"{REL_COMMIT}:{rel}"], capture_output=True,
                                     check=True).stdout
                tmp = f"{dst}.{os.getpid()}.tmp"
                with open(tmp, "wb") as f:
                    f.write(txt)
                os.replace(tmp, dst)
            if norm_sha(dst) != want:
                sys.exit(f"STOP: {dst} is not the expected release file")
            paths[rel] = dst
        self.rel = H.load_module(paths["psf_compile.py"], "psfc_rel")
        self.lay_rel = H.load_module(paths["benchmarks/psf_smart_layout.py"], "psl_rel")
        self.c2 = H.load_module(os.path.join(repo, "psf_compile.py"), "psf_compile")
        self.lay = H.load_module(os.path.join(repo, "benchmarks", "psf_smart_layout.py"), "psl_c2")
        sys.modules["psf_smart_layout"] = self.lay
        self.a5 = H.load_module(os.path.join(repo, "benchmarks", "psf_ai_compile.py"), "psf_ai_compile")
        import psf_zero_core
        import qiskit
        import qiskit_aer
        self.meta = dict(rel=self.rel.VERSION, rel_layout=self.lay_rel.LAYOUT_VERSION, c2=self.c2.VERSION,
                         layout=self.lay.LAYOUT_VERSION, a5=self.a5.AI_COMPILE_VERSION,
                         core=getattr(psf_zero_core, "CORE_VERSION", None), qiskit=qiskit.__version__,
                         aer=qiskit_aer.__version__, python=sys.version.split()[0],
                         git_head=subprocess.run(["git", "-C", repo, "rev-parse", "--short=7", "HEAD"],
                                                 capture_output=True, text=True).stdout.strip(),
                         sha=dict(script=norm_sha(os.path.abspath(__file__)),
                                  c2=norm_sha(os.path.join(repo, "psf_compile.py")),
                                  layout=norm_sha(os.path.join(repo, "benchmarks", "psf_smart_layout.py")),
                                  a5=norm_sha(os.path.join(repo, "benchmarks", "psf_ai_compile.py"))))
        self.dev = {}

    def device(self, name):
        if name not in self.dev:
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import NoiseModel
            from qiskit_ibm_runtime import fake_provider
            backend = getattr(fake_provider, name)()
            tgt = backend.target
            nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
            opts = dict(method="density_matrix", max_parallel_threads=1, max_parallel_experiments=1)
            self.dev[name] = dict(tgt=tgt, cm=tgt.build_coupling_map(), nat=nat, ideal=AerSimulator(**opts),
                                  noisy=AerSimulator(noise_model=NoiseModel.from_backend(backend), **opts))
        return self.dev[name]

    def compile(self, arm, qc, dname):
        from qiskit import transpile
        d = self.device(dname)
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if arm == "REL":
                sys.modules["psf_smart_layout"] = self.lay_rel
                return self.rel.compile_for_hardware(qc, coupling_map=d["cm"], basis_gates=d["nat"],
                                                     entangling_basis="cx", layout_search=True, seed_transpiler=0)
            if arm == "C2":
                sys.modules["psf_smart_layout"] = self.lay
                return self.c2.compile_for_hardware(qc, coupling_map=d["cm"], basis_gates=d["nat"],
                                                    entangling_basis="cx", layout_search=True, seed_transpiler=0)
            if arm == "A5":
                sys.modules["psf_smart_layout"] = self.lay
                return self.a5.compile_for_model_circuit(qc, d["cm"], d["nat"], target=d["tgt"])
            if arm == "L3T":
                return transpile(qc, target=d["tgt"], optimization_level=3, seed_transpiler=0)
        raise ValueError(arm)

    def run_points(self, arm, circuits, dname, ideal_check=False):
        """Compile each circuit, simulate all of them in one Aer job; returns one row per circuit."""
        d = self.device(dname)
        rows, sims, fins = [], [], []
        for qc in circuits:
            t0 = time.perf_counter()
            out = self.compile(arm, qc, dname)
            tc = time.perf_counter() - t0
            fin = list(out.layout.final_index_layout(filter_ancillas=True)[:N])
            c = out.copy()
            c.remove_final_measurements(inplace=True)
            c.save_density_matrix(qubits=fin)
            try:
                m = d["tgt"]["measure"][(fin[0],)]
            except Exception:
                m = None
            rows.append(dict(two_q=self.H.two_q(out), compile_s=tc, q0=int(fin[0]),
                             ro_err=float(m.error) if m is not None and m.error is not None else 0.0))
            sims.append(c)
            fins.append(fin)
        sign = np.array([1.0 if (i & 1) == 0 else -1.0 for i in range(2 ** N)])  # qubits=fin: fin[0] is the LSB
        for which, key in (("noisy", "z"),) + ((("ideal", "z_ideal"),) if ideal_check else ()):
            res = d[which].run(sims).result()
            for j in range(len(sims)):
                rho = np.asarray(res.data(j)["density_matrix"])
                rows[j][key] = float(np.real(np.diag(rho)) @ sign)
        return rows


# ---------------------------------------------------------------- runs
def q1(args):
    cfg = SMOKE if args.smoke else CFG
    cd, cq = cfg["deep"], cfg["q2"]
    t0 = time.time()
    P = problem_deep(cd)
    PQ = problem_q2(cq)
    st = Stack(args.repo, args.out)
    meta = dict(st.meta, part="q1", smoke=args.smoke, cfg=cfg, deep_teacher_seed=P["teacher_seed"],
                deep_theta_test_acc=P["theta_test_acc"], q2_teacher_seed=PQ["teacher_seed"],
                q2_theta_test_acc=PQ["theta_test_acc"], problem_s=time.time() - t0)
    print("META", json.dumps(meta), flush=True)
    L = cd["L"]
    th = P["theta"]
    pts = [("A", i, x, float(y)) for i, (x, y) in enumerate(zip(P["Xte"], P["yte"]))] + \
          [("B", i, x, float(s)) for i, (x, s) in enumerate(zip(P["Xb"], P["sb"]))]
    X = np.array([p[2] for p in pts])
    zn = model_z(X, th, L)
    p0a = float(np.max(np.abs(zn - np.array([logical_z(x, th, L) for x in X]))))
    XQ = np.vstack([PQ["Xtr"], PQ["Xte"]])
    p0a_q2 = float(np.max(np.abs(model_z(XQ, PQ["theta"], cq["L"]) -
                                 np.array([logical_z(x, PQ["theta"], cq["L"]) for x in XQ]))))
    print("P0a numpy vs Statevector", p0a, p0a_q2, flush=True)
    rows = []
    for dname in cd["devices"]:
        for arm in ARMS:
            t1 = time.time()
            rs = st.run_points(arm, [build(p[2], th, L) for p in pts], dname, ideal_check=True)
            for p, r, z in zip(pts, rs, zn):
                r.update(device=dname, arm=arm, set=p[0], i=p[1], y=p[3], z_logical=float(z))
                rows.append(r)
            print(f"Q1 {dname} {arm}: {len(rs)} circuits, {time.time() - t1:.0f} s", flush=True)
    ideal = {}
    for s in cq["seeds"]:
        v = spsa(s, cq["steps"], lambda w: float(np.mean((PQ["ytr"] - model_z(PQ["Xtr"], w, cq["L"])) ** 2)),
                 npar(cq["L"]))
        zt = model_z(PQ["Xte"], v, cq["L"])
        ideal[str(s)] = dict(train_loss=float(np.mean((PQ["ytr"] - model_z(PQ["Xtr"], v, cq["L"])) ** 2)),
                             test_acc=float(np.mean(np.sign(zt) == PQ["yte"])),
                             test_margin=float(np.mean(PQ["yte"] * zt)))
    res = dict(meta=meta, p0a=p0a, p0a_q2=p0a_q2, theta=th.tolist(), theta_test_acc=P["theta_test_acc"],
               q2_theta_test_acc=PQ["theta_test_acc"], ideal_q2=ideal, rows=rows, wall_s=time.time() - t0)
    with open(os.path.join(args.out, "q1.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("wrote q1.json", flush=True)


def q2(args):
    cfg = SMOKE if args.smoke else CFG
    c = cfg["q2"]
    if args.seed not in c["seeds"]:
        sys.exit(f"STOP: seed {args.seed} is not one of {c['seeds']}")
    L = c["L"]
    P = problem_q2(c)
    st = Stack(args.repo, args.out)
    meta = dict(st.meta, part="q2", smoke=args.smoke, arm=args.arm, device=args.device, seed=args.seed,
                teacher_seed=P["teacher_seed"])
    print("META", json.dumps(meta), flush=True)
    Xtr, ytr, Xte, yte = P["Xtr"], P["ytr"], P["Xte"], P["yte"]
    stats = dict(compiles=0, compile_s=0.0, two_q=[])

    def noisy_rows(X, v):
        rs = st.run_points(args.arm, [build(x, v, L) for x in X], args.device)
        stats["compiles"] += len(rs)
        stats["compile_s"] += sum(r["compile_s"] for r in rs)
        stats["two_q"] += [r["two_q"] for r in rs]
        return rs

    log = []
    t0 = time.time()
    v = spsa(args.seed, c["steps"],
             lambda w: float(np.mean((ytr - np.array([r["z"] for r in noisy_rows(Xtr, w)])) ** 2)), npar(L), log)
    rtr, rte = noisy_rows(Xtr, v), noisy_rows(Xte, v)
    ztr, zte = np.array([r["z"] for r in rtr]), np.array([r["z"] for r in rte])
    res = dict(meta=meta, log=log, wall_s=time.time() - t0, v=v.tolist(),
               train_loss=float(np.mean((ytr - ztr) ** 2)),
               test_acc=float(np.mean(np.sign(zte) == yte)), test_margin=float(np.mean(yte * zte)),
               noiseless_test_acc=float(np.mean(np.sign(model_z(Xte, v, L)) == yte)),
               test_rows=[dict(z=r["z"], ro_err=r["ro_err"], y=float(y)) for r, y in zip(rte, yte)],
               compiles=stats["compiles"], compile_s=stats["compile_s"],
               two_q_median=float(np.median(stats["two_q"])), two_q_max=int(max(stats["two_q"])))
    name = f"q2_{args.arm}_{args.device}_{args.seed}.json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(res, f, indent=1)
    print("Q2", json.dumps({k: res[k] for k in ("train_loss", "test_acc", "test_margin", "noiseless_test_acc",
                                                 "compiles", "compile_s", "wall_s")}), flush=True)
    print("wrote", name, flush=True)


# ---------------------------------------------------------------- scoring (numpy only)
def shots_z(z, ro_err, key):
    """REPS shot estimates of z: readout flip with probability ro_err, SHOTS shots; uniforms from `key` only, so
    every arm sees the same random numbers for the same point."""
    p0 = (1 + z) / 2
    p0 = p0 * (1 - ro_err) + (1 - p0) * ro_err
    u = np.random.default_rng(key).random((REPS, SHOTS))
    return 2 * (u < p0).sum(axis=1) / SHOTS - 1


def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def score(args):
    q = json.load(open(os.path.join(args.out, "q1.json")))
    smoke = q["meta"]["smoke"]
    cfg = SMOKE if smoke else CFG
    rows = q["rows"]
    L = [f"# qml_home2_eval score{' (SMOKE -- not a result)' if smoke else ''}", "",
         f"deep teacher {q['meta']['deep_teacher_seed']} (theta* test accuracy {q['theta_test_acc']:.3f}); "
         f"Q2W teacher {q['meta']['q2_teacher_seed']} (theta* {q['q2_theta_test_acc']:.3f}); versions "
         + json.dumps({k: q['meta'][k] for k in ('rel', 'c2', 'layout', 'a5', 'core', 'qiskit', 'aer')}), ""]
    p0b = max(abs(r["z_ideal"] - r["z_logical"]) for r in rows)
    q2rows = {}
    for f in sorted(os.listdir(args.out)):
        if f.startswith("q2_") and f.endswith(".json"):
            r = json.load(open(os.path.join(args.out, f)))
            q2rows[(r["meta"]["arm"], r["meta"]["device"], r["meta"]["seed"])] = r
    c2cfg = cfg["q2"]
    missing = [(a, d, s) for d in c2cfg["devices"] for a in ARMS for s in c2cfg["seeds"] if (a, d, s) not in q2rows]
    p0 = (q["p0a"] <= 1e-9 and q["p0a_q2"] <= 1e-9 and p0b <= 1e-6 and q["theta_test_acc"] >= cfg["deep"]["min_acc"]
          and q["q2_theta_test_acc"] >= c2cfg["min_acc"] and not missing)
    L.append(f"P0: {'PASS' if p0 else 'FAIL'} -- numpy vs Statevector {q['p0a']:.1e} / {q['p0a_q2']:.1e}; compiled "
             f"noiseless vs logical {p0b:.1e}; Q2W runs {len(q2rows)}, missing {len(missing)}")

    # Q1D
    devs = list(cfg["deep"]["devices"])
    m = {}
    for d in devs:
        for a in ARMS:
            ra = [r for r in rows if r["device"] == d and r["arm"] == a and r["set"] == "A"]
            rb = [r for r in rows if r["device"] == d and r["arm"] == a and r["set"] == "B"]
            di = devs.index(d)
            sa = np.array([np.mean(np.sign(shots_z(r["z"], r["ro_err"], [2002, di, 0, r["i"]])) == r["y"]) for r in ra])
            sb = np.array([np.mean(np.sign(shots_z(r["z"], r["ro_err"], [2002, di, 1, r["i"]])) != r["y"]) for r in rb])
            m[(d, a)] = dict(acc=float(np.mean([np.sign(r["z"]) == r["y"] for r in ra])),
                             margin=float(np.mean([r["y"] * r["z"] for r in ra])),
                             shot_acc=float(np.mean(sa)),
                             flip=float(np.mean([np.sign(r["z"]) != r["y"] for r in rb])),
                             shot_flip=float(np.mean(sb)),
                             tq=float(np.median([r["two_q"] for r in ra + rb])),
                             cs=float(np.median([r["compile_s"] for r in ra + rb])))
    ideal_margin = float(np.mean([r["y"] * r["z_logical"] for r in rows if r["set"] == "A" and r["arm"] == "REL"
                                  and r["device"] == devs[0]]))
    L += ["", f"## Q1D: deep theta* through each compiler (noiseless mean margin on A {ideal_margin:.4f})", "",
          "| device | arm | A acc (exact) | A margin | A acc (shots) | B flips (exact) | B flips (shots) | median 2q | "
          "compile s |", "|---|---|---|---|---|---|---|---|---|"]
    for d in devs:
        for a in ARMS:
            x = m[(d, a)]
            L.append(f"| {d} | {a} | {x['acc']:.3f} | {x['margin']:.4f} | {x['shot_acc']:.3f} | {x['flip']:.3f} | "
                     f"{x['shot_flip']:.3f} | {x['tq']:g} | {x['cs']:.3f} |")
    h1 = verdict(all(m[(d, 'C2')]['shot_flip'] <= m[(d, 'REL')]['shot_flip'] + 0.01 and
                     m[(d, 'C2')]['margin'] >= m[(d, 'REL')]['margin'] - 0.005 for d in devs),
                 any(m[(d, 'C2')]['shot_flip'] > m[(d, 'REL')]['shot_flip'] + 0.03 or
                     m[(d, 'C2')]['margin'] < m[(d, 'REL')]['margin'] - 0.02 for d in devs))
    g2 = sum(m[(d, 'A5')]['shot_flip'] <= m[(d, 'L3T')]['shot_flip'] + 0.01 and
             m[(d, 'A5')]['margin'] >= m[(d, 'L3T')]['margin'] - 0.005 for d in devs)
    b2 = sum(m[(d, 'A5')]['shot_flip'] > m[(d, 'L3T')]['shot_flip'] + 0.03 or
             m[(d, 'A5')]['margin'] < m[(d, 'L3T')]['margin'] - 0.02 for d in devs)
    h2 = verdict(g2 >= 2, b2 >= 2)
    heron = [d for d in devs if d in ("FakeTorino", "FakeKingston")]
    h3 = verdict(all(m[(d, 'L3T')]['margin'] >= m[(d, 'C2')]['margin'] + 0.005 for d in heron),
                 any(m[(d, 'L3T')]['margin'] < m[(d, 'C2')]['margin'] for d in heron))
    h4 = verdict(all(m[("FakeAuckland", a)]['flip'] >= 0.05 for a in ARMS),
                 all(m[("FakeAuckland", a)]['flip'] < 0.01 for a in ARMS))
    L += ["", f"- H1 (C2 vs REL: B shot flips <= REL + 0.01 and A margin >= REL - 0.005, all devices): **{h1}**",
          f"- H2 (A5 vs L3T, same test, >= 2 of 3 devices): **{h2}** ({g2} of {len(devs)})",
          f"- H3 (L3T keeps more A margin than C2 by >= 0.005 on Torino and Kingston): **{h3}**",
          f"- H4 (the fragile set sees noise: exact B flips >= 0.05 for every arm on Auckland): **{h4}**"]

    # Q2W
    ideal = q["ideal_q2"]
    idl = float(np.mean([ideal[str(s)]["test_acc"] for s in c2cfg["seeds"]]))
    L += ["", "## Q2W: learning with the compiler in the loop (mean over seeds)", "",
          f"IDEAL (noiseless, same seeds): mean test accuracy {idl:.3f}; per seed "
          + ", ".join(f"{s}: {ideal[str(s)]['test_acc']:.3f}" for s in c2cfg["seeds"]), "",
          "| device | arm | noisy test acc | shot test acc | noisy train loss | noiseless acc of result | runs | "
          "compile s per run | wall s per run | median 2q |", "|---|---|---|---|---|---|---|---|---|---|"]
    mean = {}
    for d in c2cfg["devices"]:
        di = list(c2cfg["devices"]).index(d)
        for a in ARMS:
            rs = [q2rows[(a, d, s)] for s in c2cfg["seeds"] if (a, d, s) in q2rows]
            if len(rs) != len(c2cfg["seeds"]):
                continue
            sh = [np.mean([np.mean(np.sign(shots_z(t["z"], t["ro_err"], [3003, di, r["meta"]["seed"], i])) == t["y"])
                           for i, t in enumerate(r["test_rows"])]) for r in rs]
            mv = dict(acc=float(np.mean([r["test_acc"] for r in rs])), shot=float(np.mean(sh)),
                      loss=float(np.mean([r["train_loss"] for r in rs])),
                      nl=float(np.mean([r["noiseless_test_acc"] for r in rs])),
                      cs=float(np.mean([r["compile_s"] for r in rs])), ws=float(np.mean([r["wall_s"] for r in rs])),
                      tq=float(np.median([r["two_q_median"] for r in rs])))
            mean[(d, a)] = mv
            L.append(f"| {d} | {a} | {mv['acc']:.3f} | {mv['shot']:.3f} | {mv['loss']:.4f} | {mv['nl']:.3f} | "
                     f"{len(rs)} | {mv['cs']:.0f} | {mv['ws']:.0f} | {mv['tq']:g} |")
    if missing:
        L.append(f"\nmissing Q2W runs: {missing}")
    keys = [(d, a) for d in c2cfg["devices"] for a in ARMS]
    h5 = verdict(not missing and all(mean[k]["acc"] >= idl - 0.05 for k in keys),
                 bool(missing) or any(mean[k]["acc"] < idl - 0.15 for k in keys if k in mean))
    if missing:
        h6 = "AMBIGUOUS"
    else:
        lo = lambda d, a: mean[(d, a)]["loss"]
        h6 = verdict(all(lo(d, "C2") <= lo(d, "REL") + 0.005 and lo(d, "A5") <= lo(d, "L3T") + 0.005
                         for d in c2cfg["devices"]),
                     sum(lo(d, "C2") > lo(d, "REL") + 0.02 for d in c2cfg["devices"]) >= 2
                     or sum(lo(d, "A5") > lo(d, "L3T") + 0.02 for d in c2cfg["devices"]) >= 2)
    L += ["", f"- H5 (every arm's mean noisy test accuracy >= IDEAL - 0.05, every device): **{h5}**",
          f"- H6 (mean train loss C2 <= REL + 0.005 and A5 <= L3T + 0.005, every device): **{h6}**"]
    txt = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "score.md"), "w") as f:
        f.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("q1", "q2", "score"))
    ap.add_argument("--repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--device", choices=DEVICES)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    {"q1": q1, "q2": q2, "score": score}[args.part](args)


if __name__ == "__main__":
    main()
