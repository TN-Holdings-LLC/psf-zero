"""qml_home_eval.py -- pre-registered home test (2026-10-01): does a small quantum classifier keep its accuracy,
and can it learn, when every circuit it runs goes through a given compiler onto a noisy fake device?

Independent of the workplace QML test of the same day (its design and results were not seen).

Model: 4 qubits, 2 layers, data re-uploading. Layer l: RY(pi*x_q) on every qubit, then RY(a_lq) RZ(b_lq) on every
qubit, then CZ on the ring (0,1),(1,2),(2,3),(3,0); after the layers RY(c_q) on every qubit. Output z = <Z> of
logical qubit 0; prediction sign(z); loss mean (y - z)^2. 20 parameters. Heavy-hex devices have no 4-cycle, so the
ring has to be routed.
Data: teacher-student. A teacher is the same model with parameters drawn from a seed; labels are sign(z_teacher),
points with |z_teacher| < 0.25 are skipped, classes balanced. Train 8+8 points (seed 31), test 16+16 (seed 32).
Teacher rule: the first teacher seed >= 21 for which noiseless SPSA (seed 40, 80 steps) reaches test accuracy
>= 0.9 (numpy only; no compiler, no noise).
Arms: REL  psf_compile 2026-09-28.1 + psf_smart_layout 2026-09-26.m1 (git 9131cee), compile_for_hardware
      C2   psf_compile 2026-10-01.1 + psf_smart_layout 2026-10-01.1 (the release), compile_for_hardware
      A5   benchmarks/psf_ai_compile.py 2026-10-01.a5, compile_for_model_circuit(..., target=<device Target>)
      L3T  Qiskit transpile(target=<device Target>, optimization_level=3, seed_transpiler=0)
      All PSF-Zero arms use entangling_basis="cx", layout_search=True, seed_transpiler=0, and the installed core.
Noise: qiskit_aer NoiseModel.from_backend(<fake device>), AerSimulator(method="density_matrix"); z is read exactly
from the density matrix of the final-layout qubits (no shot noise, no readout error).
Q1 (keep): theta* = noiseless SPSA (seed 40, 80 steps). The 32 test circuits through every arm on FakeAuckland,
    FakeTorino and FakeKingston; noisy accuracy and mean margin y*z.
Q2 (learn): SPSA from scratch with the compiler and the noisy simulator inside the loss (seeds 41 and 42, 40 steps,
    FakeAuckland and FakeTorino); the same seed gives the same initial point and the same perturbations in every
    arm. IDEAL = the same SPSA run on the noiseless numpy model.

  python qml_home_eval.py q1 --repo <repo> --out <dir> [--smoke]
  python qml_home_eval.py q2 --repo <repo> --out <dir> --arm REL|C2|A5|L3T --device <name> --seed <n> [--smoke]
  python qml_home_eval.py score --out <dir>
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
N, L, RING = 4, 2, [(0, 1), (1, 2), (2, 3), (3, 0)]
NPAR = L * N * 2 + N
MARGIN = 0.25
REL_COMMIT = "9131cee"
REL_SHA = {"psf_compile.py": "3616efc8b8a7bea184d6170fb7379d509d0bfec9828f4eb2f703dd4d26d8c60b",
           "benchmarks/psf_smart_layout.py": "a639efdef484379d23b4c0a52dffe557c47c30f639c41e4f8521ca608712d875"}
ARMS = ("REL", "C2", "A5", "L3T")
Q1_DEVICES = ("FakeAuckland", "FakeTorino", "FakeKingston")
Q2_DEVICES = ("FakeAuckland", "FakeTorino")
SEEDS = dict(teacher_from=21, train=31, test=32, theta=40, q2=(41, 42), steps_theta=80, steps_q2=40,
             q2_devices=Q2_DEVICES)
SMOKE = dict(teacher_from=21, train=931, test=932, theta=40, q2=(941,), steps_theta=80, steps_q2=2,
             q2_devices=("FakeAuckland",))


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


def unflat(v):
    return v[:L * N * 2].reshape(L, N, 2), v[L * N * 2:]


def model_z(X, v):
    """<Z> of qubit 0 for each row of X (numpy statevector; axis q+1 is qubit q)."""
    lay, fin = unflat(v)
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


def teacher(seed):
    r = np.random.default_rng(seed)
    return np.concatenate([r.uniform(-math.pi, math.pi, L * N * 2), r.uniform(-math.pi, math.pi, N)])


def balanced(npos, nneg, seed, vt):
    r = np.random.default_rng(seed)
    pos, neg = [], []
    while len(pos) < npos or len(neg) < nneg:
        x = r.uniform(-1, 1, (1, N))
        z = model_z(x, vt)[0]
        if z >= MARGIN and len(pos) < npos:
            pos.append(x[0])
        elif z <= -MARGIN and len(neg) < nneg:
            neg.append(x[0])
    return np.array(pos + neg), np.array([1.0] * npos + [-1.0] * nneg)


def spsa(seed, steps, f, log=None):
    """Standard SPSA; the random stream depends only on `seed`, not on f (common random numbers)."""
    rs = np.random.default_rng(seed)
    v = rs.normal(0, 0.3, NPAR)
    for k in range(steps):
        ak, ck = 0.6 / (k + 1 + 5) ** 0.602, 0.2 / (k + 1) ** 0.101
        d = rs.choice([-1.0, 1.0], NPAR)
        fp, fm = f(v + ck * d), f(v - ck * d)
        v = v - ak * (fp - fm) / (2 * ck) * d
        if log is not None:
            log.append(dict(step=k + 1, f_plus=fp, f_minus=fm, t=time.time()))
    return v


def problem(cfg):
    ts = cfg["teacher_from"]
    while True:
        vt = teacher(ts)
        Xtr, ytr = balanced(8, 8, cfg["train"], vt)
        Xte, yte = balanced(16, 16, cfg["test"], vt)
        th = spsa(cfg["theta"], cfg["steps_theta"], lambda v: float(np.mean((ytr - model_z(Xtr, v)) ** 2)))
        acc = float(np.mean(np.sign(model_z(Xte, th)) == yte))
        if acc >= 0.9 or ts >= cfg["teacher_from"] + 19:
            return dict(teacher_seed=ts, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, theta=th, theta_test_acc=acc)
        ts += 1


# ---------------------------------------------------------------- qiskit side
def build(x, v):
    from qiskit import QuantumCircuit
    lay, fin = unflat(v)
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


def norm_sha(path):
    with open(path, "rb") as f:
        txt = f.read().decode("utf-8").replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in txt.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


class Stack:
    """Loads the four arms once per process and compiles + simulates one circuit at a time."""

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
            if not os.path.exists(dst):  # several processes may start together: write a private file, then rename
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
            cm = tgt.build_coupling_map()
            nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
            self.dev[name] = dict(tgt=tgt, cm=cm, nat=nat,
                                  ideal=AerSimulator(method="density_matrix"),
                                  noisy=AerSimulator(method="density_matrix",
                                                     noise_model=NoiseModel.from_backend(backend)))
        return self.dev[name]

    def compile(self, arm, qc, dname):
        from qiskit import transpile
        d = self.device(dname)
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if arm == "REL":
                sys.modules["psf_smart_layout"] = self.lay_rel
                out = self.rel.compile_for_hardware(qc, coupling_map=d["cm"], basis_gates=d["nat"],
                                                    entangling_basis="cx", layout_search=True, seed_transpiler=0)
            elif arm == "C2":
                sys.modules["psf_smart_layout"] = self.lay
                out = self.c2.compile_for_hardware(qc, coupling_map=d["cm"], basis_gates=d["nat"],
                                                   entangling_basis="cx", layout_search=True, seed_transpiler=0)
            elif arm == "A5":
                sys.modules["psf_smart_layout"] = self.lay
                out = self.a5.compile_for_model_circuit(qc, d["cm"], d["nat"], target=d["tgt"])
            elif arm == "L3T":
                out = transpile(qc, target=d["tgt"], optimization_level=3, seed_transpiler=0)
            else:
                raise ValueError(arm)
        return out

    def z(self, out, dname, which="noisy"):
        sim = self.device(dname)[which]
        fin = list(out.layout.final_index_layout(filter_ancillas=True)[:N])
        c = out.copy()
        c.remove_final_measurements(inplace=True)
        c.save_density_matrix(qubits=fin)
        rho = np.asarray(sim.run(c).result().data()["density_matrix"])
        diag = np.real(np.diag(rho))
        sign = np.array([1.0 if (i & 1) == 0 else -1.0 for i in range(2 ** N)])  # qubits=fin: fin[0] is the LSB
        return float(diag @ sign)

    def run_point(self, arm, x, v, dname, ideal_check=False):
        qc = build(x, v)
        t0 = time.perf_counter()
        out = self.compile(arm, qc, dname)
        tc = time.perf_counter() - t0
        row = dict(two_q=self.H.two_q(out), compile_s=tc, z=self.z(out, dname))
        if ideal_check:
            row["z_ideal"] = self.z(out, dname, "ideal")
        return row


def logical_z(x, v):
    from qiskit.quantum_info import Pauli, Statevector
    return float(np.real(Statevector(build(x, v)).expectation_value(Pauli("I" * (N - 1) + "Z"))))


# ---------------------------------------------------------------- runs
def q1(args):
    cfg = SMOKE if args.smoke else SEEDS
    P = problem(cfg)
    st = Stack(args.repo, args.out)
    meta = dict(st.meta, part="q1", smoke=args.smoke, cfg=cfg, teacher_seed=P["teacher_seed"],
                theta_test_acc=P["theta_test_acc"])
    print("META", json.dumps(meta), flush=True)
    th = P["theta"]
    X = np.vstack([P["Xtr"], P["Xte"]])
    zn = model_z(X, th)
    zl = [logical_z(x, th) for x in X]
    p0a = float(np.max(np.abs(zn - np.array(zl))))
    print("P0a numpy vs Statevector max diff", p0a, flush=True)
    Xte, yte = P["Xte"], P["yte"]
    if args.smoke:
        Xte, yte = Xte[[0, -1]], yte[[0, -1]]
    devices = Q1_DEVICES[:1] if args.smoke else Q1_DEVICES
    rows = []
    for dname in devices:
        for arm in ARMS:
            for i, (x, y) in enumerate(zip(Xte, yte)):
                r = st.run_point(arm, x, th, dname, ideal_check=True)
                r.update(device=dname, arm=arm, i=i, y=float(y), z_logical=float(model_z(x[None, :], th)[0]))
                rows.append(r)
                print("Q1", json.dumps(r), flush=True)
    res = dict(meta=meta, p0a=p0a, theta=th.tolist(), theta_test_acc=P["theta_test_acc"],
               ideal_q2={str(s): ideal_q2(P, s, cfg["steps_q2"]) for s in cfg["q2"]}, rows=rows)
    with open(os.path.join(args.out, "q1.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("wrote q1.json", flush=True)


def ideal_q2(P, seed, steps):
    Xtr, ytr, Xte, yte = P["Xtr"], P["ytr"], P["Xte"], P["yte"]
    v = spsa(seed, steps, lambda w: float(np.mean((ytr - model_z(Xtr, w)) ** 2)))
    return dict(train_loss=float(np.mean((ytr - model_z(Xtr, v)) ** 2)),
                test_acc=float(np.mean(np.sign(model_z(Xte, v)) == yte)),
                test_margin=float(np.mean(yte * model_z(Xte, v))))


def q2(args):
    cfg = SMOKE if args.smoke else SEEDS
    if args.seed not in cfg["q2"]:
        sys.exit(f"STOP: seed {args.seed} is not one of {cfg['q2']}")
    P = problem(cfg)
    st = Stack(args.repo, args.out)
    meta = dict(st.meta, part="q2", smoke=args.smoke, arm=args.arm, device=args.device, seed=args.seed,
                teacher_seed=P["teacher_seed"])
    print("META", json.dumps(meta), flush=True)
    Xtr, ytr, Xte, yte = P["Xtr"], P["ytr"], P["Xte"], P["yte"]
    stats = dict(compiles=0, compile_s=0.0, two_q=[])

    def noisy_z(X, v):
        zs = []
        for x in X:
            r = st.run_point(args.arm, x, v, args.device)
            stats["compiles"] += 1
            stats["compile_s"] += r["compile_s"]
            stats["two_q"].append(r["two_q"])
            zs.append(r["z"])
        return np.array(zs)

    log = []
    t0 = time.time()
    v = spsa(args.seed, cfg["steps_q2"], lambda w: float(np.mean((ytr - noisy_z(Xtr, w)) ** 2)), log)
    ztr, zte = noisy_z(Xtr, v), noisy_z(Xte, v)
    res = dict(meta=meta, log=log, wall_s=time.time() - t0, v=v.tolist(),
               train_loss=float(np.mean((ytr - ztr) ** 2)),
               test_acc=float(np.mean(np.sign(zte) == yte)), test_margin=float(np.mean(yte * zte)),
               noiseless_test_acc=float(np.mean(np.sign(model_z(Xte, v)) == yte)),
               compiles=stats["compiles"], compile_s=stats["compile_s"],
               two_q_median=float(np.median(stats["two_q"])), two_q_max=int(max(stats["two_q"])))
    name = f"q2_{args.arm}_{args.device}_{args.seed}.json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(res, f, indent=1)
    print("Q2", json.dumps({k: res[k] for k in ("train_loss", "test_acc", "test_margin", "noiseless_test_acc",
                                                 "compiles", "compile_s", "wall_s")}), flush=True)
    print("wrote", name, flush=True)


# ---------------------------------------------------------------- scoring
def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def score(args):
    q = json.load(open(os.path.join(args.out, "q1.json")))
    rows = q["rows"]
    smoke = q["meta"]["smoke"]
    cfg = SMOKE if smoke else SEEDS
    lines = [f"# qml_home_eval score{' (SMOKE -- not a result)' if smoke else ''}", "",
             f"teacher seed {q['meta']['teacher_seed']}; theta* noiseless test accuracy {q['theta_test_acc']:.3f}; "
             f"versions {json.dumps({k: q['meta'][k] for k in ('rel', 'c2', 'layout', 'a5', 'core', 'qiskit', 'aer')})}",
             ""]
    p0b = max(abs(r["z_ideal"] - r["z_logical"]) for r in rows)
    lines.append(f"P0: numpy vs Statevector {q['p0a']:.2e} (<= 1e-9); compiled noiseless vs logical {p0b:.2e} "
                 f"(<= 1e-6); theta* test accuracy {q['theta_test_acc']:.3f} (>= 0.9)")
    acc, mar = {}, {}
    for r in rows:
        k = (r["device"], r["arm"])
        acc.setdefault(k, []).append(float(np.sign(r["z"]) == r["y"]))
        mar.setdefault(k, []).append(r["y"] * r["z"])
    devices = sorted({r["device"] for r in rows}, key=lambda d: Q1_DEVICES.index(d))
    lines += ["", "## Q1: theta* through each compiler (noisy test accuracy / mean margin / median 2q)", "",
              "| device | " + " | ".join(ARMS) + " |", "|---|" + "---|" * len(ARMS)]
    for dname in devices:
        cells = []
        for arm in ARMS:
            k = (dname, arm)
            tq = np.median([r["two_q"] for r in rows if (r["device"], r["arm"]) == k])
            cells.append(f"{np.mean(acc[k]):.3f} / {np.mean(mar[k]):.4f} / {tq:g}")
        lines.append(f"| {dname} | " + " | ".join(cells) + " |")
    m = {k: float(np.mean(v)) for k, v in mar.items()}
    a = {k: float(np.mean(v)) for k, v in acc.items()}
    h1 = verdict(all(m[(d, "C2")] >= m[(d, "REL")] - 0.005 for d in devices),
                 any(m[(d, "C2")] < m[(d, "REL")] - 0.02 for d in devices))
    good = sum(m[(d, "A5")] >= m[(d, "L3T")] - 0.005 for d in devices)
    badn = sum(m[(d, "A5")] < m[(d, "L3T")] - 0.02 for d in devices)
    h2 = verdict(good >= 2, badn >= 2)
    low = [d for d in devices if d in ("FakeTorino", "FakeKingston")]
    base = q["theta_test_acc"]
    h3 = verdict(all(a[(d, arm)] >= base - 2 / 32 for d in low for arm in ARMS),
                 any(a[(d, arm)] < base - 6 / 32 for d in low for arm in ARMS))
    lines += ["", f"- H1 (C2 margin >= REL - 0.005 on every device): **{h1}**",
              f"- H2 (A5 margin >= L3T - 0.005 on >= 2 of 3 devices): **{h2}** ({good} of {len(devices)})",
              f"- H3 (noisy accuracy >= theta* - 2/32 for every arm on Torino and Kingston): **{h3}**"]
    q2rows = {}
    for f in sorted(os.listdir(args.out)):
        if f.startswith("q2_") and f.endswith(".json"):
            r = json.load(open(os.path.join(args.out, f)))
            q2rows[(r["meta"]["arm"], r["meta"]["device"], r["meta"]["seed"])] = r
    if q2rows:
        ideal = q["ideal_q2"]
        idl = float(np.mean([ideal[str(s)]["test_acc"] for s in cfg["q2"]]))
        lines += ["", "## Q2: learning with the compiler in the loop (mean over seeds)", "",
                  f"IDEAL (noiseless, same seeds): test accuracy {idl:.3f}, "
                  + ", ".join(f"seed {s}: {ideal[str(s)]['test_acc']:.3f} / loss {ideal[str(s)]['train_loss']:.3f}"
                              for s in cfg["q2"]), "",
                  "| device | arm | noisy test acc | noisy test margin | noisy train loss | noiseless acc of result "
                  "| compiles | compile s (home) | median 2q |", "|---|---|---|---|---|---|---|---|---|"]
        mean = {}
        missing = []
        for dname in cfg["q2_devices"]:
            for arm in ARMS:
                rs = [q2rows.get((arm, dname, s)) for s in cfg["q2"]]
                if any(r is None for r in rs):
                    missing.append((arm, dname))
                    continue
                mv = {k: float(np.mean([r[k] for r in rs])) for k in ("test_acc", "test_margin", "train_loss",
                                                                       "noiseless_test_acc", "compile_s")}
                mean[(dname, arm)] = mv
                lines.append(f"| {dname} | {arm} | {mv['test_acc']:.3f} | {mv['test_margin']:.4f} | "
                             f"{mv['train_loss']:.4f} | {mv['noiseless_test_acc']:.3f} | "
                             f"{sum(r['compiles'] for r in rs)} | {mv['compile_s']:.1f} | "
                             f"{np.median([r['two_q_median'] for r in rs]):g} |")
        if missing:
            lines.append(f"\nmissing Q2 runs (count as failures): {missing}")
        keys = [(d, arm) for d in cfg["q2_devices"] for arm in ARMS]
        h4 = verdict(not missing and all(mean[k]["test_acc"] >= idl - 0.10 for k in keys),
                     bool(missing) or any(mean[k]["test_acc"] < idl - 0.25 for k in keys if k in mean))
        if missing:
            h5 = "AMBIGUOUS"
        else:
            ok5 = all(mean[(d, "C2")]["train_loss"] <= mean[(d, "REL")]["train_loss"] + 0.01 and
                      mean[(d, "A5")]["train_loss"] <= mean[(d, "L3T")]["train_loss"] + 0.01 for d in cfg["q2_devices"])
            bad5 = (all(mean[(d, "C2")]["train_loss"] > mean[(d, "REL")]["train_loss"] + 0.05
                        for d in cfg["q2_devices"])
                    or all(mean[(d, "A5")]["train_loss"] > mean[(d, "L3T")]["train_loss"] + 0.05
                           for d in cfg["q2_devices"]))
            h5 = verdict(ok5, bad5)
        lines += ["", f"- H4 (every arm's noisy test accuracy >= IDEAL - 0.10): **{h4}**",
                  f"- H5 (train loss C2 <= REL + 0.01 and A5 <= L3T + 0.01 on both devices): **{h5}**"]
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "score.md"), "w") as f:
        f.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("q1", "q2", "score"))
    ap.add_argument("--repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--device", choices=Q1_DEVICES)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    {"q1": q1, "q2": q2, "score": score}[args.part](args)


if __name__ == "__main__":
    main()
