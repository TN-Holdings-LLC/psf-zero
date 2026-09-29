"""core_eigen_route_check.py -- pre-registered (workplace, 2026-09-29): the candidate
Rust core CORE_VERSION 2026-09-29.1 (core changelog item 12: eigen-route fallback in
decompose_one) against the release core 2026-09-28.1, on fresh block sets.

Sets (all generated here from fixed seeds):
  RAND  50,000 Haar-random U(4) (qiskit random_unitary, seed 29_000_000 + i)
  NEAR  near-degenerate perturbations g . exp(i eps H) of CNOT, SWAP, iSWAP and the
        identity, eps in 1e-4 .. 1e-8, 250 each (numpy seed 29)
  TIE   10,000 blocks built in the magic basis with two phases whose |cos| nearly
        coincide while the phases themselves stay apart: theta = (+d1, -d2, t3, t4),
        d1, d2 ~ U(0.005, 0.05) (half of them shifted by pi), t3 ~ U(-pi, pi),
        t4 = -(sum of the others); U = Q O1 diag(exp(i theta)) O2 Q^dagger with
        Haar SO(4) O1, O2 (numpy seed 30)
  EFRESH the two-qubit blocks of 30,000 fresh part-E training circuits
        (loop_endurance.e_circuit at target theta (seed 7, rzz angles 0) plus
        normal(0, 0.5) noise from seed 303), consolidated as psf_compile.compile()
        does at block_gate_floor 8 (Collect2qBlocks with len > 8, then
        ConsolidateBlocks); cached in eigen_check_efresh.npz by the first collect
  V4    the 15 blocks the release core rejects in part E of the v4 run
        (blocks15.npz, captured by capture15.py)

Modes:
  collect <core_dir> <tag>   run geometric_decompose_checked from the core found in
                             <core_dir> (put first on sys.path) over every set; save
                             acceptance, the raw returned values and the raw residual
                             (Frobenius distance of psf_compile._reconstruct to the
                             block) to eigen_check_<tag>.npz
  score                      compare eigen_check_release.npz and
                             eigen_check_candidate.npz as pre-registered

    python -u core_eigen_route_check.py collect <release_core_dir>   release
    python -u core_eigen_route_check.py collect <candidate_core_dir> candidate
    python -u core_eigen_route_check.py score
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

Q = np.array([[1, 1j, 0, 0], [0, 0, 1j, 1], [0, 0, 1j, -1], [1, -1j, 0, 0]]) / np.sqrt(2)
SETS = ("RAND", "NEAR", "TIE", "EFRESH", "V4")


def haar_so4(rng):
    a = rng.normal(size=(4, 4))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def build_sets(here):
    from qiskit.quantum_info import random_unitary
    out = {"RAND": [random_unitary(4, seed=29_000_000 + i).data for i in range(50_000)]}
    rng = np.random.default_rng(29)
    base = {"cnot": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex),
            "swap": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], complex),
            "iswap": np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], complex),
            "id": np.eye(4, dtype=complex)}
    near = []
    for g in base.values():
        for eps in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
            for _ in range(250):
                h = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
                h = (h + h.conj().T) / 2
                w, v = np.linalg.eigh(h)
                near.append(g @ (v @ np.diag(np.exp(1j * eps * w)) @ v.conj().T))
    out["NEAR"] = near
    rng = np.random.default_rng(30)
    tie = []
    for i in range(10_000):
        d1, d2 = rng.uniform(0.005, 0.05, 2)
        shift = np.pi if i % 2 else 0.0
        t3 = rng.uniform(-np.pi, np.pi)
        th = np.array([shift + d1, shift - d2, t3, 0.0])
        th[3] = -th[:3].sum()
        um = haar_so4(rng) @ np.diag(np.exp(1j * th)) @ haar_so4(rng)
        tie.append(Q @ um @ Q.conj().T)
    out["TIE"] = tie
    out["EFRESH"] = efresh_blocks()
    out["V4"] = list(np.load(os.path.join(here, "blocks15.npz"))["mats"])
    return out


def efresh_blocks():
    cache = os.path.join(os.getcwd(), "eigen_check_efresh.npz")
    if os.path.exists(cache):
        return list(np.load(cache)["U"])
    import loop_endurance as le
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
    rng_e = np.random.default_rng(7)
    tt = rng_e.uniform(-np.pi, np.pi, le.E_NPARAMS)
    tt[8::15] = 0.0
    rng_t = np.random.default_rng(303)
    pm = PassManager([Collect2qBlocks(filter_fn=lambda dag, block: len(block) > 8),
                      ConsolidateBlocks(kak_basis_gate=None, force_consolidate=True)])
    blocks = []
    for _ in range(30_000):
        qc = pm.run(le.e_circuit(tt + rng_t.normal(0.0, 0.5, le.E_NPARAMS)))
        for inst in qc.data:
            if len(inst.qubits) == 2 and inst.operation.name == "unitary":
                blocks.append(inst.operation.to_matrix())
    np.savez(cache, U=np.array(blocks))
    return blocks


def collect(core_dir, tag):
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, core_dir)
    import psf_zero_core as core
    import psf_compile as pc
    print("CORE", core.__file__, "CORE_VERSION", getattr(core, "CORE_VERSION", None))
    print("LOADED psf_compile", pc.VERSION)
    sets = build_sets(here)
    save = {}
    for name in SETS:
        acc, raw, res = [], [], []
        for u in sets[name]:
            try:
                cartan, k1, k2, ph, _ = core.geometric_decompose_checked(u.real.tolist(), u.imag.tolist())
                vals = [*cartan, *(x for t in k1 for x in t), *(x for t in k2 for x in t), ph]
                acc.append(True); raw.append(vals)
                res.append(float(np.linalg.norm(pc._reconstruct(cartan, k1, k2, ph) - u)))
            except Exception:  # noqa: BLE001 -- a rejection is the recorded outcome
                acc.append(False); raw.append([np.nan] * 16); res.append(np.nan)
        save[name + "_acc"] = np.array(acc)
        save[name + "_raw"] = np.array(raw)
        save[name + "_res"] = np.array(res)
        a = np.array(acc)
        r = np.array(res)[a]
        print(f"{tag} {name}: accepted {a.sum()}/{a.size}; raw residual max "
              f"{(r.max() if r.size else float('nan')):.2e}", flush=True)
    save["core_version"] = np.array(str(getattr(core, "CORE_VERSION", None)))
    np.savez(os.path.join(os.getcwd(), f"eigen_check_{tag}.npz"), **save)
    print("wrote", f"eigen_check_{tag}.npz")


def score():
    rel = np.load("eigen_check_release.npz")
    cand = np.load("eigen_check_candidate.npz")

    def v(ok, bad):
        return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")
    print("=" * 78); print("SCORING (thresholds exactly as pre-registered)"); print("=" * 78)
    cv_r, cv_c = str(rel["core_version"]), str(cand["core_version"])
    sizes = {s: int(rel[s + "_acc"].size) for s in SETS}
    ok0 = (cv_r == "2026-09-28.1" and cv_c == "2026-09-29.1"
           and sizes["RAND"] == 50000 and sizes["NEAR"] == 5000 and sizes["TIE"] == 10000 and sizes["V4"] == 15
           and sizes["EFRESH"] >= 300000
           and all(cand[s + "_acc"].size == sizes[s] for s in SETS))
    print(f"C0: release core {cv_r}, candidate core {cv_c}, set sizes {sizes} -> {'passed' if ok0 else 'FAILED'}")
    if not ok0:
        print("C0 FAILED: nothing is scored.")
        return
    summary = {}
    for s in SETS:
        ar, ac = rel[s + "_acc"], cand[s + "_acc"]
        same = int(np.sum([np.array_equal(x, y) for x, y, a in zip(rel[s + "_raw"], cand[s + "_raw"], ar) if a]))
        summary[s] = dict(rel_acc=int(ar.sum()), cand_acc=int(ac.sum()), identical=same,
                          newly_acc=int(np.sum(~ar & ac)), newly_rej=int(np.sum(ar & ~ac)),
                          cand_res_max=float(np.nanmax(cand[s + "_res"])) if ac.any() else float("nan"),
                          new_res_max=float(np.nanmax(cand[s + "_res"][~ar & ac])) if np.any(~ar & ac) else 0.0)
        print(f"   {s}: {json.dumps(summary[s])}")
    idsets = ("RAND", "NEAR", "TIE", "EFRESH")
    e1_ok = all(summary[s]["identical"] == summary[s]["rel_acc"] for s in idsets)
    print(f"E1 bit-identical on every block the release core accepts (RAND, NEAR, TIE, EFRESH): "
          f"{[(s, summary[s]['identical'], summary[s]['rel_acc']) for s in idsets]} -> {v(e1_ok, not e1_ok)}")
    nr = sum(summary[s]["newly_rej"] for s in SETS)
    print(f"E2 no block newly rejected (all sets): {nr} -> {v(nr == 0, nr > 0)}")
    t = summary["EFRESH"]
    n = sizes["EFRESH"]
    rej_c = t["cand_acc"] < n
    print(f"E3 candidate accepts every EFRESH block, newly accepted ones with raw residual <= 1e-8: accepted "
          f"{t['cand_acc']}/{n}, newly accepted {t['newly_acc']}, their worst raw residual {t['new_res_max']:.2e}"
          f" (refuted if any rejected or > 1e-6) -> {v(not rej_c and t['new_res_max'] <= 1e-8, rej_c or t['new_res_max'] > 1e-6)}")
    rr = n - t["rel_acc"]
    print(f"E4 the release core rejects between 1 and 100 EFRESH blocks (v4 rate: 15 in about 330,000): {rr} -> "
          f"{v(1 <= rr <= 100, rr == 0 or rr > 1000)}")
    w = summary["V4"]
    print(f"E5 (control) candidate accepts all 15 V4 blocks: {w['cand_acc']}/15; worst raw residual "
          f"{w['cand_res_max']:.2e} -> {v(w['cand_acc'] == 15, w['cand_acc'] < 15)}")


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        here = os.path.dirname(os.path.abspath(__file__))
        repo = sys.argv[4] if len(sys.argv) > 4 else os.path.dirname(here)
        sys.path[:0] = [repo, os.path.join(repo, "benchmarks")]
        collect(sys.argv[2], sys.argv[3])
    else:
        score()
