"""qml_home_rescore.py -- independent re-scoring of the home QML test (Addendum 290), written after the lock.

It does not import qml_home_eval.py. It re-implements the noiseless model with dense 16x16 matrices (a different
code path from the scored script's axis-wise statevector), rebuilds the teacher, the data, theta* and the IDEAL arm
from the pre-registered seeds, and re-derives P0 and H1-H5 from the raw rows of q1.json and q2_*.json with the
thresholds copied from the pre-registration text.

    python qml_home_rescore.py <out dir>

Prints a report and exits 0; it never edits the run directory.
"""
import glob
import json
import math
import os
import sys

import numpy as np

N, L = 4, 2
RING = [(0, 1), (1, 2), (2, 3), (3, 0)]
NPAR = L * N * 2 + N
ARMS = ("REL", "C2", "A5", "L3T")
Q1_DEVICES = ("FakeAuckland", "FakeTorino", "FakeKingston")
Q2_DEVICES = ("FakeAuckland", "FakeTorino")
Q2_SEEDS = (41, 42)
EXPECT = dict(script="0a30fb22b709420418ed130a18531810616f405b42768a73157587e545b3ca32", git_head="680cf08",
              teacher_seed=23, rel="2026-09-28.1", c2="2026-10-01.1", layout="2026-10-01.1", a5="2026-10-01.a5",
              core="2026-09-29.1")


# ---------------------------------------------------------------- independent dense-matrix model
I2 = np.eye(2)
Z = np.diag([1.0, -1.0])


def ry(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], complex)


def rz(t):
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])


def on(q, g):
    """Single-qubit gate g on qubit q of N; qubit 0 is the most significant tensor factor here."""
    m = np.ones((1, 1))
    for k in range(N):
        m = np.kron(m, g if k == q else I2)
    return m


def cz_diag():
    d = np.ones(2 ** N)
    for i in range(2 ** N):
        bits = [(i >> (N - 1 - k)) & 1 for k in range(N)]
        if sum(bits[p] & bits[q] for p, q in RING) % 2:
            d[i] = -1.0
    return d


CZR = cz_diag()
Z0 = on(0, Z).diagonal().real


def z_one(x, v):
    lay = v[:L * N * 2].reshape(L, N, 2)
    fin = v[L * N * 2:]
    psi = np.zeros(2 ** N, complex)
    psi[0] = 1.0
    for l in range(L):
        for q in range(N):
            psi = on(q, ry(math.pi * x[q])) @ psi
        for q in range(N):
            psi = on(q, rz(lay[l, q, 1]) @ ry(lay[l, q, 0])) @ psi
        psi = CZR * psi
    for q in range(N):
        psi = on(q, ry(fin[q])) @ psi
    return float(np.real(np.vdot(psi, Z0 * psi)))


def z_all(X, v):
    return np.array([z_one(x, v) for x in X])


# ---------------------------------------------------------------- the pre-registered problem (Addendum 290, sec. 2)
def teacher(seed):
    r = np.random.default_rng(seed)
    return np.concatenate([r.uniform(-math.pi, math.pi, L * N * 2), r.uniform(-math.pi, math.pi, N)])


def balanced(npos, nneg, seed, vt):
    r = np.random.default_rng(seed)
    pos, neg = [], []
    while len(pos) < npos or len(neg) < nneg:
        x = r.uniform(-1, 1, (1, N))[0]
        z = z_one(x, vt)
        if z >= 0.25 and len(pos) < npos:
            pos.append(x)
        elif z <= -0.25 and len(neg) < nneg:
            neg.append(x)
    return np.array(pos + neg), np.array([1.0] * npos + [-1.0] * nneg)


def spsa(seed, steps, f):
    rs = np.random.default_rng(seed)
    v = rs.normal(0, 0.3, NPAR)
    for k in range(steps):
        ak, ck = 0.6 / (k + 1 + 5) ** 0.602, 0.2 / (k + 1) ** 0.101
        d = rs.choice([-1.0, 1.0], NPAR)
        v = v - ak * (f(v + ck * d) - f(v - ck * d)) / (2 * ck) * d
    return v


def problem():
    for ts in range(21, 41):
        vt = teacher(ts)
        Xtr, ytr = balanced(8, 8, 31, vt)
        Xte, yte = balanced(16, 16, 32, vt)
        th = spsa(40, 80, lambda v: float(np.mean((ytr - z_all(Xtr, v)) ** 2)))
        acc = float(np.mean(np.sign(z_all(Xte, th)) == yte))
        if acc >= 0.9:
            return dict(ts=ts, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, th=th, acc=acc)
    return None


# ---------------------------------------------------------------- re-scoring
def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def main():
    out = sys.argv[1]
    rep, flags = [], []
    q = json.load(open(os.path.join(out, "q1.json")))
    q2 = {}
    for f in sorted(glob.glob(os.path.join(out, "q2_*.json"))):
        r = json.load(open(f))
        q2[(r["meta"]["arm"], r["meta"]["device"], r["meta"]["seed"])] = r

    # provenance: every file from the locked script, the lock commit and the release under test
    for name, meta in [("q1.json", q["meta"])] + [(f"q2 {k}", r["meta"]) for k, r in q2.items()]:
        for key in ("git_head", "teacher_seed", "rel", "c2", "layout", "a5", "core"):
            if meta.get(key) != EXPECT[key]:
                flags.append(f"{name}: {key} = {meta.get(key)!r}, expected {EXPECT[key]!r}")
        if meta["sha"]["script"] != EXPECT["script"]:
            flags.append(f"{name}: script hash {meta['sha']['script'][:8]}")
        if meta.get("smoke"):
            flags.append(f"{name}: smoke run")
    shas = {json.dumps(r["meta"]["sha"], sort_keys=True) for r in q2.values()} | {json.dumps(q["meta"]["sha"],
                                                                                            sort_keys=True)}
    rep.append(f"provenance: {1 + len(q2)} files, {len(shas)} distinct hash set(s), "
               f"{'no flags' if not flags else str(len(flags)) + ' FLAGS'}")

    # independent rebuild of the problem
    P = problem()
    rep.append(f"independent rebuild: teacher {P['ts']}, theta* test accuracy {P['acc']:.4f} "
               f"(file: {q['meta']['teacher_seed']}, {q['theta_test_acc']:.4f}); "
               f"max |theta* - file| {np.max(np.abs(P['th'] - np.array(q['theta']))):.1e}")
    if P["ts"] != q["meta"]["teacher_seed"] or abs(P["acc"] - q["theta_test_acc"]) > 1e-12:
        flags.append("theta* rebuild differs from the file")
    ideal = {}
    for s in Q2_SEEDS:
        v = spsa(s, 40, lambda w: float(np.mean((P["ytr"] - z_all(P["Xtr"], w)) ** 2)))
        ideal[s] = dict(acc=float(np.mean(np.sign(z_all(P["Xte"], v)) == P["yte"])),
                        loss=float(np.mean((P["ytr"] - z_all(P["Xtr"], v)) ** 2)))
        fi = q["ideal_q2"][str(s)]
        if abs(fi["test_acc"] - ideal[s]["acc"]) > 1e-12 or abs(fi["train_loss"] - ideal[s]["loss"]) > 1e-9:
            flags.append(f"IDEAL seed {s} differs: file {fi}, rebuild {ideal[s]}")
    idl = float(np.mean([ideal[s]["acc"] for s in Q2_SEEDS]))
    rep.append("IDEAL rebuild: " + ", ".join(f"seed {s} acc {ideal[s]['acc']:.4f} loss {ideal[s]['loss']:.4f}"
                                             for s in Q2_SEEDS) + f"; mean {idl:.4f}")

    # P0
    rows = q["rows"]
    yte = P["yte"]
    zl_rebuild = z_all(P["Xte"], P["th"])
    p0b = max(abs(r["z_ideal"] - r["z_logical"]) for r in rows)
    p0c = max(abs(r["z_logical"] - zl_rebuild[r["i"]]) for r in rows)
    ylab = max(abs(r["y"] - yte[r["i"]]) for r in rows)
    n_rows = len(rows)
    combos = {(r["device"], r["arm"], r["i"]) for r in rows}
    p0 = (q["p0a"] <= 1e-9 and p0b <= 1e-6 and P["acc"] >= 0.9 and len(q2) == 16 and n_rows == 384
          and len(combos) == 384 and p0c <= 1e-9 and ylab == 0)
    rep.append(f"P0: numpy vs Statevector {q['p0a']:.1e}; compiled noiseless vs logical {p0b:.1e}; "
               f"logical vs independent model {p0c:.1e}; labels match {ylab == 0}; Q1 rows {n_rows} "
               f"({len(combos)} distinct); Q2 files {len(q2)} -> {'PASS' if p0 else 'FAIL'}")

    # Q1
    acc, mar, tq = {}, {}, {}
    for r in rows:
        k = (r["device"], r["arm"])
        acc.setdefault(k, []).append(1.0 if np.sign(r["z"]) == r["y"] else 0.0)
        mar.setdefault(k, []).append(r["y"] * r["z"])
        tq.setdefault(k, []).append(r["two_q"])
    a = {k: float(np.mean(v)) for k, v in acc.items()}
    m = {k: float(np.mean(v)) for k, v in mar.items()}
    rep += ["", "Q1 (acc / mean margin / median 2q / max 2q):"]
    for d in Q1_DEVICES:
        rep.append(f"  {d:13s} " + "  ".join(f"{arm} {a[(d, arm)]:.3f}/{m[(d, arm)]:.4f}/"
                                             f"{np.median(tq[(d, arm)]):g}/{max(tq[(d, arm)])}" for arm in ARMS))
    rep.append("  C2 - REL margin: " + ", ".join(f"{d} {m[(d, 'C2')] - m[(d, 'REL')]:+.4f}" for d in Q1_DEVICES))
    rep.append("  A5 - L3T margin: " + ", ".join(f"{d} {m[(d, 'A5')] - m[(d, 'L3T')]:+.4f}" for d in Q1_DEVICES))
    h1 = verdict(all(m[(d, "C2")] >= m[(d, "REL")] - 0.005 for d in Q1_DEVICES),
                 any(m[(d, "C2")] < m[(d, "REL")] - 0.02 for d in Q1_DEVICES))
    good = sum(m[(d, "A5")] >= m[(d, "L3T")] - 0.005 for d in Q1_DEVICES)
    h2 = verdict(good >= 2, sum(m[(d, "A5")] < m[(d, "L3T")] - 0.02 for d in Q1_DEVICES) >= 2)
    low = ("FakeTorino", "FakeKingston")
    h3 = verdict(all(a[(d, arm)] >= P["acc"] - 2 / 32 for d in low for arm in ARMS),
                 any(a[(d, arm)] < P["acc"] - 6 / 32 for d in low for arm in ARMS))

    # Q2
    rep += ["", "Q2 per run (noisy test acc / noisy train loss / noiseless acc of result, file vs rebuild / compiles):"]
    mean = {}
    missing = [(arm, d, s) for d in Q2_DEVICES for arm in ARMS for s in Q2_SEEDS if (arm, d, s) not in q2]
    for d in Q2_DEVICES:
        for arm in ARMS:
            rs = [q2[(arm, d, s)] for s in Q2_SEEDS if (arm, d, s) in q2]
            for r in rs:
                v = np.array(r["v"])
                nl = float(np.mean(np.sign(z_all(P["Xte"], v)) == P["yte"]))
                if abs(nl - r["noiseless_test_acc"]) > 1e-12:
                    flags.append(f"Q2 {arm} {d} {r['meta']['seed']}: noiseless acc file {r['noiseless_test_acc']} "
                                 f"vs rebuild {nl}")
                exp_compiles = 40 * 2 * 16 + 48
                if r["compiles"] != exp_compiles or len(r["log"]) != 40:
                    flags.append(f"Q2 {arm} {d} {r['meta']['seed']}: {r['compiles']} compiles, {len(r['log'])} steps")
                rep.append(f"  {d:13s} {arm:4s} seed {r['meta']['seed']}: {r['test_acc']:.3f} / {r['train_loss']:.4f}"
                           f" / {r['noiseless_test_acc']:.3f} vs {nl:.3f} / {r['compiles']}, wall {r['wall_s']:.0f} s")
            if len(rs) == len(Q2_SEEDS):
                mean[(d, arm)] = {k: float(np.mean([r[k] for r in rs])) for k in ("test_acc", "train_loss")}
    rep += ["", "Q2 means over seeds (noisy test acc / noisy train loss); IDEAL mean acc %.4f:" % idl]
    for d in Q2_DEVICES:
        rep.append(f"  {d:13s} " + "  ".join(
            f"{arm} {mean[(d, arm)]['test_acc']:.3f}/{mean[(d, arm)]['train_loss']:.4f}" if (d, arm) in mean
            else f"{arm} missing" for arm in ARMS))
    keys = [(d, arm) for d in Q2_DEVICES for arm in ARMS]
    h4 = verdict(not missing and all(mean[k]["test_acc"] >= idl - 0.10 for k in keys),
                 bool(missing) or any(mean[k]["test_acc"] < idl - 0.25 for k in keys if k in mean))
    if missing:
        h5 = "AMBIGUOUS (runs missing)"
    else:
        lo = lambda d, arm: mean[(d, arm)]["train_loss"]
        h5 = verdict(all(lo(d, "C2") <= lo(d, "REL") + 0.01 and lo(d, "A5") <= lo(d, "L3T") + 0.01 for d in Q2_DEVICES),
                     all(lo(d, "C2") > lo(d, "REL") + 0.05 for d in Q2_DEVICES)
                     or all(lo(d, "A5") > lo(d, "L3T") + 0.05 for d in Q2_DEVICES))
        rep.append("  C2 - REL loss: " + ", ".join(f"{d} {lo(d, 'C2') - lo(d, 'REL'):+.4f}" for d in Q2_DEVICES)
                   + "; A5 - L3T loss: " + ", ".join(f"{d} {lo(d, 'A5') - lo(d, 'L3T'):+.4f}" for d in Q2_DEVICES))

    rep += ["", f"P0 {'PASS' if p0 else 'FAIL'}", f"H1 {h1}", f"H2 {h2} ({good} of 3)", f"H3 {h3}", f"H4 {h4}",
            f"H5 {h5}"]
    if missing:
        rep.append(f"missing Q2 runs: {missing}")
    rep += ["", "flags: " + ("none" if not flags else "")] + ["  " + f for f in flags]
    print("\n".join(rep))


if __name__ == "__main__":
    main()
