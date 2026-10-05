"""pilot_depth.py -- workplace PILOT for the QML-2 "DEPTH" design (exploratory, dev data only, not a test).

Question: does a data re-uploading classifier on real data (sklearn breast_cancer, PCA to n features) show a
noise-limited optimal depth when deployed through a compiler onto a noisy fake device, and do compilers move it?
Dev data only: split seed 0, init seed 0. A scored stage must use other split/init seeds (disclosed).

Model (n qubits, L layers): per layer RY(pi/2 * x_q) on every qubit; RY(a) RZ(b) on every qubit; CZ ring. Then RY(c).
Output z = <Z_0>, prediction sign(z), loss MSE(y, z). Noiseless training: Adam on exact parameter-shift gradients
(numpy statevector, batched over data). Deployment: every test circuit compiled by each arm and simulated with
NoiseModel.from_backend (density matrix, touched qubits only, noise on the physical qubits); z read exactly.
"""
import contextlib, io, json, sys, time, warnings
import numpy as np
warnings.simplefilter("ignore")

N_Q = int(sys.argv[1]) if len(sys.argv) > 1 else 6
LS = [int(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 4, 8]
DEVICES = sys.argv[3].split(",") if len(sys.argv) > 3 else ["FakeAuckland", "FakeTorino"]
ARMS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["RPSF", "C12", "L3T"]
OUT = sys.argv[5] if len(sys.argv) > 5 else f"pilot_n{N_Q}.json"


# ---------------------------------------------------------------- data
def data(n, split_seed=0):
    from sklearn.datasets import load_breast_cancer
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X, y = load_breast_cancer(return_X_y=True)
    y = np.where(y == 1, 1.0, -1.0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=split_seed)
    sc = StandardScaler().fit(Xtr)
    pca = PCA(n_components=n, random_state=0).fit(sc.transform(Xtr))
    Ztr, Zte = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
    lo, hi = Ztr.min(0), Ztr.max(0)
    f = lambda Z: np.clip(2 * (Z - lo) / (hi - lo) - 1, -1, 1)
    return f(Ztr), ytr, f(Zte), yte


# ---------------------------------------------------------------- numpy model
def n_params(n, L):
    return 2 * n * L + n


def _ry(t):  # t: (...,) -> (..., 2, 2)
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)


def _rz(t):
    e = np.exp(-0.5j * t)
    z = np.zeros(np.shape(t) + (2, 2), complex)
    z[..., 0, 0], z[..., 1, 1] = e, np.conj(e)
    return z


def _apply1(psi, U, q, n):  # psi (B, 2,...,2) ; U (B,2,2) or (2,2); qubit q is axis 1 + (n-1-q) (Qiskit order)
    ax = 1 + (n - 1 - q)
    psi = np.moveaxis(psi, ax, -1)
    if U.ndim == 2:
        psi = psi @ U.T
    else:
        psi = np.einsum("b...j,bij->b...i", psi, U)
    return np.moveaxis(psi, -1, ax)


def _cz_ring_phase(n):
    idx = np.arange(2 ** n)
    bits = (idx[:, None] >> np.arange(n)[None, :]) & 1
    par = np.zeros(2 ** n, int)
    pairs = [(q, (q + 1) % n) for q in range(n)] if n > 2 else [(0, 1)]
    for a, b in pairs:
        par ^= bits[:, a] & bits[:, b]
    return np.where(par == 1, -1.0, 1.0)


def forward(theta, X, n, L):
    B = X.shape[0]
    psi = np.zeros((B,) + (2,) * n, complex)
    psi[(slice(None),) + (0,) * n] = 1
    ph = _cz_ring_phase(n).reshape((2,) * n)  # index order: axis k <-> bit (n-1-k): build matching tensor
    ph = ph.reshape(2 ** n)
    k = 0
    for l in range(L):
        for q in range(n):
            psi = _apply1(psi, _ry(np.pi / 2 * X[:, q]), q, n)
        for q in range(n):
            psi = _apply1(psi, _ry(np.array(theta[k])), q, n); k += 1
            psi = _apply1(psi, _rz(np.array(theta[k])), q, n); k += 1
        psi = (psi.reshape(B, -1) * ph[None, :]).reshape(psi.shape)
    for q in range(n):
        psi = _apply1(psi, _ry(np.array(theta[k])), q, n); k += 1
    p = np.abs(psi.reshape(B, -1)) ** 2
    idx = np.arange(2 ** n)
    z0 = np.where((idx & 1) == 0, 1.0, -1.0)
    return p @ z0


def train(Xtr, ytr, n, L, seed=0, steps=200, lr=0.05, batch=64):
    rng = np.random.default_rng(seed)
    th = rng.normal(0, 0.3, n_params(n, L))
    m = np.zeros_like(th); v = np.zeros_like(th)
    for t in range(1, steps + 1):
        bi = rng.choice(len(Xtr), batch, replace=False)
        Xb, yb = Xtr[bi], ytr[bi]
        g = np.zeros_like(th)
        z0 = forward(th, Xb, n, L)
        for j in range(len(th)):
            e = np.zeros_like(th); e[j] = np.pi / 2
            dz = (forward(th + e, Xb, n, L) - forward(th - e, Xb, n, L)) / 2
            g[j] = np.mean(-2 * (yb - z0) * dz)
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
        th -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return th


# ---------------------------------------------------------------- circuits, compilers, noise
def circuit(x, theta, n, L):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    k = 0
    for l in range(L):
        for q in range(n):
            qc.ry(np.pi / 2 * x[q], q)
        for q in range(n):
            qc.ry(theta[k], q); k += 1
            qc.rz(theta[k], q); k += 1
        for q in range(n):
            if n > 2 or q == 0:
                qc.cz(q, (q + 1) % n)
    for q in range(n):
        qc.ry(theta[k], q); k += 1
    return qc


def compilers(be):
    import psf_compile as P
    from qiskit import transpile
    t = be.target
    cm = t.build_coupling_map()
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    base = dict(coupling_map=cm, basis_gates=basis, entangling_basis="cx", layout_search=True, seed_transpiler=0,
                target=t, placement_refine=True)
    full = dict(base, final_resynthesis="select", compare_level3=True, compare_floor=True, candidate_score="hybrid")
    def q(f):
        def g(qc):
            with contextlib.redirect_stdout(io.StringIO()):
                return f(qc)
        return g
    return {"RPSF": q(lambda qc: P.compile_for_hardware(qc, **base)),
            "C12": q(lambda qc: P.compile_for_hardware(qc, **full)),
            "L3T": q(lambda qc: transpile(qc, target=t, optimization_level=3, seed_transpiler=0,
                                          approximation_degree=1.0))}


def main():
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime import fake_provider as fp
    from qiskit.quantum_info import DensityMatrix, SparsePauliOp
    Xtr, ytr, Xte, yte = data(N_Q)
    rows = {"meta": dict(n=N_Q, Ls=LS, devices=DEVICES, arms=ARMS, ntest=len(Xte)), "cells": []}
    thetas = {}
    for L in LS:
        t0 = time.perf_counter()
        import os
        cache = f"pilot_theta_n{N_Q}_L{L}.npy"
        th = np.load(cache) if os.path.exists(cache) else train(Xtr, ytr, N_Q, L, steps=150 if L <= 4 else 100)
        np.save(cache, th)
        z = forward(th, Xte, N_Q, L)
        acc = float(np.mean(np.sign(z) == yte)); mar = float(np.mean(yte * z))
        thetas[L] = th
        print(f"train L={L}: ideal test acc {acc:.3f} margin {mar:.3f} ({time.perf_counter() - t0:.0f} s)", flush=True)
        rows["cells"].append(dict(L=L, arm="IDEAL", device=None, acc=acc, margin=mar))
    for dname in DEVICES:
        be = getattr(fp, dname)()
        noise = NoiseModel.from_backend(be)
        sim = AerSimulator(method="density_matrix", noise_model=noise)
        C = compilers(be)
        for L in LS:
            th = thetas[L]
            for arm in ARMS:
                t0 = time.perf_counter()
                zs, n2q, touched = [], [], []
                for x in Xte:
                    qc = circuit(x, th, N_Q, L)
                    out = C[arm](qc)
                    fin = out.layout.final_index_layout(filter_ancillas=True)
                    active = sorted({out.find_bit(b).index for i in out.data for b in i.qubits} | set(fin))
                    c = out.copy()
                    c.save_density_matrix(qubits=[fin[0]])   # as qml_core.noisy_z0 (validated); see note in design doc
                    rho = np.asarray(sim.run(c).result().data()["density_matrix"])
                    ev = rho[0, 0] - rho[1, 1]
                    zs.append(float(np.real(ev)))
                    n2q.append(sum(1 for i in out.data if len(i.qubits) == 2)); touched.append(len(active))
                zs = np.array(zs)
                cell = dict(L=L, arm=arm, device=dname, acc=float(np.mean(np.sign(zs) == yte)),
                            margin=float(np.mean(yte * zs)), n2q=float(np.mean(n2q)), touched=float(np.mean(touched)),
                            s=round(time.perf_counter() - t0, 1))
                rows["cells"].append(cell)
                print(json.dumps(cell), flush=True)
                json.dump(rows, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
