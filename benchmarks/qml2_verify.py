"""qml2_verify.py -- independent check of the home QML test part 2 (Addendum 296), written after the run.

Does not import qml_home2_eval.py. Re-implements the model with dense 16x16 matrices, rebuilds the Q2W data, the
IDEAL arm, the deep test sets A and B (from the teacher seed and the stored theta*), checks provenance, the compile
counts and the noiseless accuracy of every Q2W result, and re-derives P0 and H1-H6 from the raw rows.

    python qml2_verify.py <run dir>
"""
import glob, hashlib, json, math, os, sys
import numpy as np

N = 4
RING = [(0, 1), (1, 2), (2, 3), (3, 0)]
ARMS = ("REL", "C2", "A5", "L3T")
DEVS = ("FakeAuckland", "FakeTorino", "FakeKingston")
EXPECT = dict(git_head="4839966", script="9e12f5bf34c29542ebe608a0c672519a52a0039d796dff5fc19ea600a2a6b95f",
              c2="2026-10-01.1", rel="2026-09-28.1", a5="2026-10-01.a5", core="2026-09-29.1")
I2, Zm = np.eye(2), np.diag([1.0, -1.0])


def on(q, g):
    m = np.ones((1, 1))
    for k in range(N):
        m = np.kron(m, g if k == q else I2)
    return m


def ry(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], complex)


def rz(t):
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])


CZR = np.array([(-1.0) ** (sum(((i >> (N - 1 - p)) & 1) & ((i >> (N - 1 - q)) & 1) for p, q in RING) % 2)
                for i in range(2 ** N)])
Z0 = on(0, Zm).diagonal().real


def z1(x, v, L):
    lay, fin = v[:L * N * 2].reshape(L, N, 2), v[L * N * 2:]
    psi = np.zeros(2 ** N, complex)
    psi[0] = 1
    for l in range(L):
        for q in range(N):
            psi = on(q, ry(math.pi * x[q])) @ psi
        for q in range(N):
            psi = on(q, rz(lay[l, q, 1]) @ ry(lay[l, q, 0])) @ psi
        psi = CZR * psi
    for q in range(N):
        psi = on(q, ry(fin[q])) @ psi
    return float(np.real(np.vdot(psi, Z0 * psi)))


zs = lambda X, v, L: np.array([z1(x, v, L) for x in X])


def teacher(seed, L):
    r = np.random.default_rng(seed)
    return np.concatenate([r.uniform(-math.pi, math.pi, L * N * 2), r.uniform(-math.pi, math.pi, N)])


def sample(npos, nneg, seed, f, lo, hi):
    r = np.random.default_rng(seed)
    pos, neg = [], []
    while len(pos) < npos or len(neg) < nneg:
        x = r.uniform(-1, 1, (1, N))[0]
        z = f(x)
        if lo <= z <= hi and len(pos) < npos:
            pos.append(x)
        elif -hi <= z <= -lo and len(neg) < nneg:
            neg.append(x)
    return np.array(pos + neg), np.array([1.0] * npos + [-1.0] * nneg)


def spsa(seed, steps, f, n):
    rs = np.random.default_rng(seed)
    v = rs.normal(0, 0.3, n)
    for k in range(steps):
        ak, ck = 0.6 / (k + 1 + 5) ** 0.602, 0.2 / (k + 1) ** 0.101
        d = rs.choice([-1.0, 1.0], n)
        v = v - ak * (f(v + ck * d) - f(v - ck * d)) / (2 * ck) * d
    return v


def shots(z, e, key, S=4000, R=20):
    p0 = (1 + z) / 2
    p0 = p0 * (1 - e) + (1 - p0) * e
    return 2 * (np.random.default_rng(key).random((R, S)) < p0).sum(axis=1) / S - 1


def V(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


out = sys.argv[1]
q = json.load(open(os.path.join(out, "q1.json")))
q2 = {}
for p in glob.glob(os.path.join(out, "q2_*.json")):
    r = json.load(open(p))
    q2[(r["meta"]["arm"], r["meta"]["device"], r["meta"]["seed"])] = r
flags = []
for name, m in [("q1", q["meta"])] + [(str(k), r["meta"]) for k, r in q2.items()]:
    for k in ("git_head", "c2", "rel", "a5", "core"):
        if m.get(k) != EXPECT[k]:
            flags.append(f"{name} {k}={m.get(k)}")
    if m["sha"]["script"] != EXPECT["script"]:
        flags.append(f"{name} script hash")
    if m["smoke"]:
        flags.append(f"{name} smoke")
print(f"provenance: {1 + len(q2)} files, flags {len(flags)}")

# Q2W data and IDEAL
vt = teacher(23, 2)
Xtr, ytr = sample(8, 8, 31, lambda x: z1(x, vt, 2), 0.25, 9)
Xte, yte = sample(16, 16, 32, lambda x: z1(x, vt, 2), 0.25, 9)
ideal = {}
for s in range(41, 49):
    v = spsa(s, 40, lambda w: float(np.mean((ytr - zs(Xtr, w, 2)) ** 2)), 20)
    ideal[s] = float(np.mean(np.sign(zs(Xte, v, 2)) == yte))
    if abs(ideal[s] - q["ideal_q2"][str(s)]["test_acc"]) > 1e-12:
        flags.append(f"IDEAL {s}")
idl = float(np.mean(list(ideal.values())))
print("IDEAL rebuild:", {s: round(a, 4) for s, a in ideal.items()}, "mean", idl)
for k, r in q2.items():
    nl = float(np.mean(np.sign(zs(Xte, np.array(r["v"]), 2)) == yte))
    if abs(nl - r["noiseless_test_acc"]) > 1e-12 or r["compiles"] != 1328 or len(r["log"]) != 40:
        flags.append(f"Q2 {k}: nl {nl} vs {r['noiseless_test_acc']}, compiles {r['compiles']}, steps {len(r['log'])}")
    if [t["y"] for t in r["test_rows"]] != list(yte):
        flags.append(f"Q2 {k}: test labels")

# deep sets from teacher 84 and the stored theta*
th = np.array(q["theta"])
vt4 = teacher(q["meta"]["deep_teacher_seed"], 4)
XA, yA = sample(16, 16, 92, lambda x: z1(x, vt4, 4), 0.25, 9)
XB, sB = sample(32, 32, 94, lambda x: z1(x, th, 4), 0.01, 0.10)
zA, zB = zs(XA, th, 4), zs(XB, th, 4)
accA = float(np.mean(np.sign(zA) == yA))
rows = q["rows"]
dz = max(abs(r["z_logical"] - (zA if r["set"] == "A" else zB)[r["i"]]) for r in rows)
dy = max(abs(r["y"] - (yA if r["set"] == "A" else sB)[r["i"]]) for r in rows)
p0b = max(abs(r["z_ideal"] - r["z_logical"]) for r in rows)
print(f"deep: theta* acc on rebuilt A {accA:.4f} (file {q['theta_test_acc']:.4f}); logical z vs rebuild {dz:.1e}; "
      f"labels {dy}; compiled noiseless vs logical {p0b:.1e}; rows {len(rows)}")
p0 = (not flags and dz < 1e-9 and dy == 0 and p0b <= 1e-6 and accA >= 0.85 and len(q2) == 96 and len(rows) == 1152)
m = {}
for d in DEVS:
    di = DEVS.index(d)
    for a in ARMS:
        ra = [r for r in rows if r["device"] == d and r["arm"] == a and r["set"] == "A"]
        rb = [r for r in rows if r["device"] == d and r["arm"] == a and r["set"] == "B"]
        m[(d, a)] = dict(
            margin=np.mean([r["y"] * r["z"] for r in ra]),
            flip=np.mean([np.sign(r["z"]) != r["y"] for r in rb]),
            sflip=np.mean([np.mean(np.sign(shots(r["z"], r["ro_err"], [2002, di, 1, r["i"]])) != r["y"]) for r in rb]),
            ro=np.mean([r["ro_err"] for r in ra + rb]))
for d in DEVS:
    print(d, " ".join(f"{a}: margin {m[(d, a)]['margin']:.4f} flip {m[(d, a)]['flip']:.3f} shot-flip "
                      f"{m[(d, a)]['sflip']:.3f} ro {m[(d, a)]['ro']:.4f}" for a in ARMS))
h1 = V(all(m[(d, 'C2')]['sflip'] <= m[(d, 'REL')]['sflip'] + 0.01 and m[(d, 'C2')]['margin'] >= m[(d, 'REL')]['margin'] - 0.005 for d in DEVS),
       any(m[(d, 'C2')]['sflip'] > m[(d, 'REL')]['sflip'] + 0.03 or m[(d, 'C2')]['margin'] < m[(d, 'REL')]['margin'] - 0.02 for d in DEVS))
g = sum(m[(d, 'A5')]['sflip'] <= m[(d, 'L3T')]['sflip'] + 0.01 and m[(d, 'A5')]['margin'] >= m[(d, 'L3T')]['margin'] - 0.005 for d in DEVS)
b = sum(m[(d, 'A5')]['sflip'] > m[(d, 'L3T')]['sflip'] + 0.03 or m[(d, 'A5')]['margin'] < m[(d, 'L3T')]['margin'] - 0.02 for d in DEVS)
h2 = V(g >= 2, b >= 2)
h3 = V(all(m[(d, 'L3T')]['margin'] >= m[(d, 'C2')]['margin'] + 0.005 for d in DEVS[1:]),
       any(m[(d, 'L3T')]['margin'] < m[(d, 'C2')]['margin'] for d in DEVS[1:]))
h4 = V(all(m[("FakeAuckland", a)]['flip'] >= 0.05 for a in ARMS), all(m[("FakeAuckland", a)]['flip'] < 0.01 for a in ARMS))
acc = {(d, a): np.mean([q2[(a, d, s)]["test_acc"] for s in range(41, 49)]) for d in DEVS for a in ARMS}
loss = {(d, a): np.mean([q2[(a, d, s)]["train_loss"] for s in range(41, 49)]) for d in DEVS for a in ARMS}
h5 = V(all(v >= idl - 0.05 for v in acc.values()), any(v < idl - 0.15 for v in acc.values()))
h6 = V(all(loss[(d, 'C2')] <= loss[(d, 'REL')] + 0.005 and loss[(d, 'A5')] <= loss[(d, 'L3T')] + 0.005 for d in DEVS),
       sum(loss[(d, 'C2')] > loss[(d, 'REL')] + 0.02 for d in DEVS) >= 2 or sum(loss[(d, 'A5')] > loss[(d, 'L3T')] + 0.02 for d in DEVS) >= 2)
print("Q2W loss C2-REL:", {d: round(loss[(d, 'C2')] - loss[(d, 'REL')], 4) for d in DEVS},
      "A5-L3T:", {d: round(loss[(d, 'A5')] - loss[(d, 'L3T')], 4) for d in DEVS})
print(f"P0 {'PASS' if p0 else 'FAIL'}  H1 {h1}  H2 {h2} ({g} of 3)  H3 {h3}  H4 {h4}  H5 {h5}  H6 {h6}")
print("flags:", flags if flags else "none")
