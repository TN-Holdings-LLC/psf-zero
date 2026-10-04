"""hybrid_diag.py -- exploratory diagnosis (2026-10-04, not a test): would an estimate that keeps both amplitude
damping (as the release's `excitation_cost` counts it) and pure dephasing (as `pauli_cost` counts it) choose better
among release 2026-10-03.3's candidates than either?

Addendum 332 found that `pauli_cost` (Pauli-twirled thermal relaxation) chose well on GHZ chains (F5), where dephasing
decides, and lost up to 1.8% on XXZ chains (F3), where amplitude damping decides and `excitation_cost` had ranked
correctly. This script re-creates, for every scored HOLD5 circuit (in-sample: Addendum 332 scored them) on all nine
devices, the candidate set that the release builds with `compare_floor=True, compare_level3=True,
candidate_score="pauli"`:
  psf     the release's own circuit (item 35's "select")
  floor   the same pipeline re-placed on `floor_aware_target(target)`, if `_acceptable`
  level3  Qiskit level 3's circuit, if `_acceptable`
(captured by wrapping `_choose`, so the set is exactly the release's), simulates each distinct candidate as HOLD5 did,
and scores each candidate with four estimates:
  exc     the release's `excitation_cost` (item 35)
  pauli   the release's `pauli_cost` (item 37)
  hyb     per gate and qubit: duration / T1 x P(1) on the noiseless state just before the gate (amplitude damping, as
          `exc`), plus p_phi (1 - <Z>^2) just after it (pure dephasing, as the Z part of `pauli`) with
          p_phi = (1 - exp(-t / T_phi)) / 2 and 1 / T_phi = 1 / T2 - 1 / (2 T1) (T2 capped at 2 T1); plus the reported
          error above the thermal floor, times (d + 1) / d (as `pauli`)
  excz    `exc` plus the same pure-dephasing term (a second, cruder combination)
Choices are made as `_choose` makes them (lowest estimate; the release's circuit on ties or if any estimate is None).
Checks: the choice by `pauli` among all candidates must reproduce HOLD5's C10 rows, and the choice by `exc` between
psf and level3 HOLD5's C9 rows. a7's and L3T's results are taken from HOLD5's rows.

  python hybrid_diag.py run --repo <repo> --out <dir> --device <d> [--smoke]
  python hybrid_diag.py summary --out <dir>
"""
import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
DEVICES = ("FakeAuckland", "FakeTorino", "FakeKingston", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva", "FakeFez",
           "FakeMarrakesh", "FakeAachen")
CX = ("FakeAuckland", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva")
FAMILIES = ("F1", "F2", "F3", "F4", "F5", "F6")
HOLD5 = os.path.join("data", "2026-10-03", "hold5", "outputs")
SCORES = ("exc", "pauli", "hyb", "excz")
SKIP = ("barrier", "measure", "delay")
CHUNK = 60


def _thermal(qp, i, t):
    """(t1, t2 capped at 2 t1) for qubit i, or None when T1 or the duration is unknown."""
    p = qp[i] if i < len(qp) else None
    t1 = getattr(p, "t1", None) if p is not None else None
    if not t or not t1:
        return None
    t2 = getattr(p, "t2", None)
    return t1, (min(t2, 2 * t1) if t2 else 2 * t1)


def hybrid_terms(circ, target, max_qubits=16):
    """(hyb, dephasing term) for `circ` (see the module docstring); None if too wide or a gate has no matrix.
    Same state propagation as the release's `pauli_cost` (numpy tensor, axis j = j-th touched qubit, Qiskit's
    little-endian gate matrices)."""
    ops = [(ins.operation, tuple(circ.find_bit(b).index for b in ins.qubits)) for ins in circ.data
           if ins.operation.name not in SKIP]
    active = sorted({i for _, q in ops for i in q})
    if len(active) > max_qubits:
        return None
    k = max(len(active), 1)
    pos = {p: j for j, p in enumerate(active)}
    qp = getattr(target, "qubit_properties", None) or []
    psi = np.zeros((2,) * k, dtype=complex)
    psi[(0,) * k] = 1.0

    def rho1(ax):
        mm = np.moveaxis(psi, ax, 0).reshape(2, -1)
        return mm @ mm.conj().T

    damp = deph = rest = 0.0
    for op, q in ops:
        try:
            mat = np.asarray(op.to_matrix(), dtype=complex)
        except Exception:
            return None
        props = target[op.name].get(q, None) if op.name in target.operation_names else None
        axes = [pos[i] for i in q]
        m = len(axes)
        t = (props.duration or 0.0) if props is not None else 0.0
        if props is not None:
            for i, ax in zip(q, axes):
                th = _thermal(qp, i, t)
                if th:
                    damp += t / th[0] * float(rho1(ax)[1, 1].real)
        rev = axes[::-1]
        psi = np.tensordot(mat.reshape((2,) * (2 * m)), psi, axes=(list(range(m, 2 * m)), rev))
        psi = np.moveaxis(psi, list(range(m)), rev)
        if props is None:
            continue
        thermal_f = 1.0
        for i, ax in zip(q, axes):
            th = _thermal(qp, i, t)
            if not th:
                continue
            t1, t2 = th
            rate = max(1.0 / t2 - 1.0 / (2.0 * t1), 0.0)
            p_phi = (1.0 - math.exp(-t * rate)) / 2.0
            r = rho1(ax)
            ez = float((r[0, 0] - r[1, 1]).real)
            deph += p_phi * (1.0 - ez * ez)
            thermal_f *= (1.0 + 2.0 * math.exp(-t / t2) + math.exp(-t / t1)) / 4.0
        d = 2 ** m
        rest += max((props.error or 0.0) - (1.0 - (d * thermal_f + 1.0) / (d + 1.0)), 0.0) * (d + 1) / d
    return damp + deph + rest, deph


def choose(costs):
    """Index chosen as the release's `_choose` chooses: lowest cost, the first on ties or if any cost is None."""
    if any(c is None for c in costs):
        return 0
    return min(range(len(costs)), key=lambda j: (costs[j], j))


def key_of(out, n):
    return (tuple((i.operation.name, tuple(out.find_bit(b).index for b in i.qubits),
                   tuple(round(float(p), 12) for p in i.operation.params)) for i in out.data),
            tuple(out.layout.final_index_layout(filter_ancillas=True)[:n]))


def run(args):
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime import fake_provider
    sys.path.insert(0, os.path.join(args.repo, "benchmarks"))
    sys.path.insert(0, args.repo)
    import core_fix_c2_eval as H
    rel = H.load_module(os.path.join(args.repo, "psf_compile.py"), "psf_compile")
    if rel.VERSION != "2026-10-03.3":
        raise SystemExit("STOP: psf_compile.py is %s, not release 2026-10-03.3" % rel.VERSION)
    lay = H.load_module(os.path.join(args.repo, "benchmarks", "psf_smart_layout.py"), "psl_c2")
    sys.modules["psf_smart_layout"] = lay
    gen = H.load_module(os.path.join(args.repo, "benchmarks", "hold5_eval.py"), "hold5_eval")
    backend = getattr(fake_provider, args.device)()
    tgt = backend.target
    cm = tgt.build_coupling_map()
    nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
    opts = dict(max_parallel_threads=1, max_parallel_experiments=1)
    noisy = AerSimulator(method="density_matrix", noise_model=NoiseModel.from_backend(backend), **opts)
    ideal = AerSimulator(method="statevector", **opts)

    captured = {}
    orig_choose = rel._choose

    def spy(cands, target, score):
        captured["cands"] = list(cands)
        return orig_choose(cands, target, score)

    rel._choose = spy
    kw = dict(target=tgt, placement_refine=True, final_resynthesis="select", compare_level3=True,
              compare_floor=True, candidate_score="pauli")

    t0, rows = time.time(), []
    for fam in FAMILIES:
        h5 = {}
        for arm in ("C9", "C10", "A7", "L3T"):
            p = os.path.join(args.repo, HOLD5, "hold5_%s_%s_%s.json" % (args.device, arm, fam))
            h5[arm] = json.load(open(p))["rows"] if (os.path.exists(p) and not args.smoke) else None
        todo = []
        for j, (params, qc) in enumerate(gen.family(fam, args.smoke)):
            if args.smoke and j >= 2:
                break
            n = qc.num_qubits
            captured.clear()
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = rel.compile_for_hardware(qc, coupling_map=cm, basis_gates=nat, entangling_basis="cx",
                                               layout_search=True, seed_transpiler=0, **kw)
            cands = captured.get("cands") or [("psf", out)]
            keys = [key_of(c, n) for _, c in cands]
            row = dict(family=fam, index=j, params=params, n=n, names=[nm for nm, _ in cands],
                       same=[keys.index(kk) for kk in keys], chosen_pauli=next(
                           (i for i, (_, c) in enumerate(cands) if c is out), None), cands=[])
            for (nm, c), kk in zip(cands, keys):
                ht = hybrid_terms(c, tgt)
                e, pa = rel.excitation_cost(c, tgt), rel.pauli_cost(c, tgt)
                row["cands"].append(dict(
                    name=nm, exc=e, pauli=pa, hyb=None if ht is None else ht[0],
                    excz=None if (ht is None or e is None) else e + ht[1],
                    two_q=sum(1 for i in c.data if len(i.qubits) == 2), depth=c.depth()))
            for h, key in ((h5["C9"], "C9"), (h5["C10"], "C10"), (h5["A7"], "A7"), (h5["L3T"], "L3T")):
                if h is not None and j < len(h):
                    row["h5_" + key] = dict(infid=h[j].get("infid"), two_q=h[j]["two_q"],
                                            params_match=h[j]["params"] == params)
            sims = []
            for i, (nm, c) in enumerate(cands):
                if row["same"][i] != i:
                    continue
                cc = c.copy()
                cc.remove_final_measurements(inplace=True)
                cc.save_density_matrix(qubits=list(c.layout.final_index_layout(filter_ancillas=True)[:n]))
                sims.append((i, cc))
            todo.append((row, Statevector(qc).data, sims))
            if len(todo) >= CHUNK:
                _simulate(todo, noisy, ideal, rows)
                todo = []
        if todo:
            _simulate(todo, noisy, ideal, rows)
        print("  %s %s: %d circuits so far, %.0f s" % (args.device, fam, len(rows), time.time() - t0), flush=True)
    rel._choose = orig_choose
    name = "hybrid_%s%s.json" % (args.device, "_smoke" if args.smoke else "")
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(dict(device=args.device, version=rel.VERSION, smoke=args.smoke, rows=rows,
                       wall_s=time.time() - t0), f)
    print("wrote %s: %d circuits, %.0f s" % (name, len(rows), time.time() - t0), flush=True)


def _simulate(todo, noisy, ideal, rows):
    circs = [cc for _, _, sims in todo for _, cc in sims]
    res = {"infid": noisy.run(circs).result(), "ideal": ideal.run(circs).result()}
    k = 0
    for row, psi, sims in todo:
        got = {}
        for i, _ in sims:
            got[i] = {key: float(1 - np.real(np.conj(psi) @ np.asarray(r.data(k)["density_matrix"]) @ psi))
                      for key, r in res.items()}
            k += 1
        for i, c in enumerate(row["cands"]):
            c.update(got[row["same"][i]])
        rows.append(row)


def summary(args):
    L = ["# hybrid_diag summary (exploratory, not a test; in-sample on HOLD5's circuits)", "",
         "Ratios are mean infidelity relative to HOLD5's L3T. PICK_x chooses among the release's candidates (psf,",
         "floor, level3; as `_choose`) by estimate x; C10r = PICK_pauli (must equal HOLD5's C10), C9r = choice by exc",
         "between psf and level3 (must equal HOLD5's C9). ORC is the measured best candidate. 'rank x' = circuits",
         "where x's choice is the measured best.", ""]
    head = ("| device | circuits | C9r | C10r | A7 | PICK_exc | PICK_hyb | PICK_excz | ORC | rank exc | rank pauli | "
            "rank hyb | rank excz | C9 / C10 mismatches |")
    L += [head, "|" + "---|" * (head.count("|") - 1)]
    fam_tab = {}
    for d in DEVICES:
        p = os.path.join(args.out, "hybrid_%s.json" % d)
        if not os.path.exists(p):
            p = os.path.join(args.out, "hybrid_%s_smoke.json" % d)
            if not os.path.exists(p):
                continue
        R = json.load(open(p))
        rows = [r for r in R["rows"] if all("infid" in c for c in r["cands"])]

        def pick(r, score, names=None):
            idx = [i for i, c in enumerate(r["cands"]) if names is None or c["name"] in names]
            j = choose([r["cands"][i][score] for i in idx])
            return r["cands"][idx[j]]["infid"]

        def ref(r):
            h = r.get("h5_L3T")
            if h and h.get("infid") is not None:
                return h["infid"]
            lv = [c["infid"] for c in r["cands"] if c["name"] == "level3"]
            return lv[0] if lv else None

        rows = [r for r in rows if ref(r) is not None]

        def ratio(rs, f):
            return float(np.mean([f(r) for r in rs]) / max(np.mean([ref(r) for r in rs]), 1e-15)) if rs else float("nan")

        orc = lambda r: min(c["infid"] for c in r["cands"])
        rank = {s: sum(abs(pick(r, s) - orc(r)) <= 1e-12 for r in rows) for s in SCORES}
        mis9 = sum(1 for r in rows if r.get("h5_C9", {}).get("infid") is not None
                   and abs(pick(r, "exc", ("psf", "level3")) - r["h5_C9"]["infid"]) > 1e-9)
        mis10 = sum(1 for r in rows if r.get("h5_C10", {}).get("infid") is not None
                    and abs(pick(r, "pauli") - r["h5_C10"]["infid"]) > 1e-9)
        a7 = ratio(rows, lambda r: r["h5_A7"]["infid"]) if all(r.get("h5_A7", {}).get("infid") is not None
                                                                for r in rows) else float("nan")
        L.append("| %s%s | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %d | %d | %d | %d | %d / %d |" % (
            d, " (cx)" if d in CX else "", len(rows), ratio(rows, lambda r: pick(r, "exc", ("psf", "level3"))),
            ratio(rows, lambda r: pick(r, "pauli")), a7, ratio(rows, lambda r: pick(r, "exc")),
            ratio(rows, lambda r: pick(r, "hyb")), ratio(rows, lambda r: pick(r, "excz")), ratio(rows, orc),
            rank["exc"], rank["pauli"], rank["hyb"], rank["excz"], mis9, mis10))
        for fam in FAMILIES + ("F3o",):
            rs = [r for r in rows if r["family"] == fam or (fam == "F3o" and r["family"] == "F3"
                                                             and r["params"].get("bc") == "o")]
            fam_tab[(d, fam)] = {s: ratio(rs, lambda r, s=s: pick(r, s)) for s in SCORES}
            fam_tab[(d, fam)]["orc"] = ratio(rs, orc)
        noiseless = max((c["ideal"] for r in R["rows"] for c in r["cands"] if "ideal" in c), default=float("nan"))
        L[-1] += "" if noiseless <= 1e-6 else "  (noiseless max %.1e!)" % noiseless
    for a, b in (("hyb", "pauli"), ("hyb", "exc"), ("excz", "pauli")):
        L += ["", "PICK_%s / PICK_%s by family:" % (a, b), "",
              "| family | " + " | ".join(d.replace("Fake", "") for d in DEVICES) + " |",
              "|---" * (len(DEVICES) + 1) + "|"]
        for fam in FAMILIES + ("F3o",):
            L.append("| %s | " % fam + " | ".join(
                "%.3f" % (fam_tab[(d, fam)][a] / fam_tab[(d, fam)][b]) if (d, fam) in fam_tab else "-"
                for d in DEVICES) + " |")
    txt = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("run", "summary"))
    ap.add_argument("--repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=DEVICES)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    (run if args.part == "run" else summary)(args)


if __name__ == "__main__":
    main()
