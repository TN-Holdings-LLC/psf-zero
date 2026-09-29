"""battery.py -- exploratory: run one core build (package dir in argv[1]) over block
sets and save, per block, acceptance and the raw residual of the returned parameters
(Frobenius distance of psf_compile._reconstruct to the block), plus the raw returned
values for bit-identity comparisons. Sets: the 15 rejected v4 blocks; the 33,000 part-E
blocks captured on 2026-09-28 (blocks_fixed3000.npz); 20,000 random SU(4) (seed 11);
near-degenerate perturbations of CNOT, SWAP, iSWAP and identity (eps 1e-4..1e-8, 200 each)."""
import sys, json
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import numpy as np
import psf_zero_core as core
import psf_compile as pc
from qiskit.quantum_info import random_unitary
tag = sys.argv[3]

def sets():
    d = np.load("blocks15.npz"); yield "v4_15", list(d["mats"])
    yield "E33000", list(np.load(sys.argv[4])["U"])
    yield "rand20000", [random_unitary(4, seed=11 * 100000 + i).data for i in range(20000)]
    rng = np.random.default_rng(5)
    base = {"cnot": np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], complex),
            "swap": np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], complex),
            "iswap": np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], complex),
            "id": np.eye(4, dtype=complex)}
    out = []
    for name, g in base.items():
        for eps in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
            for _ in range(200):
                h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)); h = (h + h.conj().T) / 2
                w, v = np.linalg.eigh(h); p = v @ np.diag(np.exp(1j * eps * w)) @ v.conj().T
                out.append(g @ p)
    yield "neardeg4000", out

res = {}
for name, mats in sets():
    acc, resid, raw = [], [], []
    for u in mats:
        try:
            cartan, k1, k2, ph, infid = core.geometric_decompose_checked(u.real.tolist(), u.imag.tolist())
            r = float(np.linalg.norm(pc._reconstruct(cartan, k1, k2, ph) - u))
            acc.append(1); resid.append(r); raw.append([*cartan, *[x for t in k1 for x in t], *[x for t in k2 for x in t], ph])
        except Exception as e:
            acc.append(0); resid.append(float("nan")); raw.append(None)
    res[name] = {"acc": acc, "resid": resid, "raw": raw}
    a = np.array(acc); r = np.array(resid)[a == 1]
    if r.size:
        print(f"{tag} {name}: accepted {a.sum()}/{len(a)}; raw residual median {np.median(r):.2e} max {r.max():.2e} "
              f"count>1e-13 {(r > 1e-13).sum()} count>1e-10 {(r > 1e-10).sum()}", flush=True)
    else:
        print(f"{tag} {name}: accepted 0/{len(a)}", flush=True)
json.dump(res, open(f"battery_{tag}.json", "w"))
