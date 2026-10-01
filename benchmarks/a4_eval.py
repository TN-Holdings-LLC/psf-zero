"""a4_eval.py -- pre-registered evaluation of psf_ai_compile 2026-10-01.a4 (state-aware error model) under noisy simulation.

Arms (psf_compile / psf_smart_layout 2026-10-01.c2, the same Rust core):
  A1   psf_ai_compile 2026-10-01.a1 (no error information; only for the no-target identity check)
  A2   psf_ai_compile 2026-10-01.a2 with target=<device Target> (placement/selection by reported gate errors)
  A4   psf_ai_compile 2026-10-01.a4 with target=<device Target> (state-aware estimate from T1/T2, durations, errors)
  L3   Qiskit transpile(coupling_map, basis_gates, optimization_level=3, seed_transpiler=0) (no error information)
  L3T  Qiskit transpile(target=<device Target>, optimization_level=3, seed_transpiler=0) (error-aware)
Noise: qiskit_aer NoiseModel.from_backend(<fake device>) (gate depolarizing + thermal relaxation; no idle noise,
crosstalk or readout). Fidelity <psi|rho|psi> on the final-layout qubits, density_matrix method.
Inputs (new): random dense circuits, Python random seeds 9001-9040 (3-5 qubits), FakeAuckland, FakeTorino, FakeKingston.
Also: A4 without a target must give the same output as A1 (first 20 Auckland circuits).

  python a4_eval.py run --compile <c2> --layout <c2> --a1 <a1.py> --a2 <a2.py> --a4 <a4.py> [--out a4_raw.json] [--dry]
  python a4_eval.py score [--out a4_raw.json]
"""
import argparse
import contextlib
import io
import json
import os
import random
import statistics
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_fix_c2_eval as H  # noqa: E402

DEVICES = ("FakeAuckland", "FakeTorino", "FakeKingston")
DRY_OFFSET = 900000
ARMS = ("A2", "A4", "L3", "L3T")
# (confirm bound, refute bound); F6: (median confirm s, median refute s, max confirm s)
# F2/F3 are split by device: on FakeAuckland (where reported gate errors understate the decoherence limit on some
# qubits) a clear gain is predicted; on FakeTorino/FakeKingston only "not worse".
THR = {"F2a": (0.90, 1.00), "F2b": (1.00, 1.05), "F3a": (0.90, 1.00), "F3b": (1.00, 1.05),
       "F4": (0.90, 1.00), "F5": (1.03, 1.10), "F6": (0.8, 1.6, 4.0)}
AUCK = "FakeAuckland"


def run(args):
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime import fake_provider
    comp = H.load_module(args.compile, "psf_compile")
    lay = H.load_module(args.layout, "psf_smart_layout")
    a1 = H.load_module(args.a1, "psf_ai_compile_a1")
    a2 = H.load_module(args.a2, "psf_ai_compile_a2")
    a4 = H.load_module(args.a4, "psf_ai_compile_a4")
    import psf_zero_core
    off = DRY_OFFSET if args.dry else 0
    meta = dict(c2=comp.VERSION, layout=lay.LAYOUT_VERSION, a1=a1.AI_COMPILE_VERSION, a2=a2.AI_COMPILE_VERSION,
                a4=a4.AI_COMPILE_VERSION,
                core=getattr(psf_zero_core, "CORE_VERSION", None), dry=bool(args.dry),
                qiskit=__import__("qiskit").__version__, aer=__import__("qiskit_aer").__version__,
                sha={"c2": H.norm_sha(args.compile), "layout": H.norm_sha(args.layout), "a1": H.norm_sha(args.a1),
                     "a2": H.norm_sha(args.a2), "a4": H.norm_sha(args.a4), "script": H.norm_sha(os.path.abspath(__file__)),
                     "helpers": H.norm_sha(H.__file__)})
    print("META", json.dumps(meta), flush=True)
    raw = {"meta": meta, "R": [], "NT": []}

    def fidelity(qc, out, sim):
        n = qc.num_qubits
        fin = list(out.layout.final_index_layout(filter_ancillas=True)[:n])
        c = out.copy()
        c.save_density_matrix(qubits=fin)
        rho = np.asarray(sim.run(c).result().data()["density_matrix"])
        psi = Statevector(qc).data
        return float(np.real(psi.conj() @ rho @ psi))

    for bname in DEVICES:
        backend = getattr(fake_provider, bname)()
        tgt = backend.target
        cm = tgt.build_coupling_map()
        nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
        ideal = AerSimulator(method="density_matrix")
        noisy = AerSimulator(method="density_matrix", noise_model=NoiseModel.from_backend(backend))
        for seed in range(off + 9001, off + (9004 if args.dry else 9041)):
            rng = random.Random(seed)
            n = rng.choice([3, 4, 5])
            qc = H.rand_dense(n, rng.randint(6, 20), rng)
            outs, times = {}, {}
            with contextlib.redirect_stdout(io.StringIO()):
                t = time.perf_counter()
                outs["A2"] = a2.compile_for_model_circuit(qc, cm, nat, target=tgt)
                times["A2"] = time.perf_counter() - t
                t = time.perf_counter()
                outs["A4"] = a4.compile_for_model_circuit(qc, cm, nat, target=tgt)
                times["A4"] = time.perf_counter() - t
            outs["L3"] = transpile(qc, coupling_map=cm, basis_gates=nat, optimization_level=3, seed_transpiler=0)
            outs["L3T"] = transpile(qc, target=tgt, optimization_level=3, seed_transpiler=0)
            row = dict(device=bname, seed=seed, n=n)
            for k, o in outs.items():
                row[k] = dict(two_q=H.two_q(o), ideal=fidelity(qc, o, ideal), noisy=fidelity(qc, o, noisy),
                              s=round(times.get(k, 0.0), 4))
            raw["R"].append(row)
            print("R", json.dumps(row), flush=True)
            if bname == "FakeAuckland" and seed < off + 9021:
                with contextlib.redirect_stdout(io.StringIO()):
                    o_a1 = a1.compile_for_model_circuit(qc, cm, nat)
                    o_nt = a4.compile_for_model_circuit(qc, cm, nat)
                raw["NT"].append(dict(seed=seed, a1=H.digest(o_a1), a4_no_target=H.digest(o_nt)))
    with open(args.out, "w") as f:
        json.dump(raw, f, indent=1)
    print("wrote", args.out)


def score(args):
    raw = json.load(open(args.out))
    m = raw["meta"]
    c0 = (m["c2"] == "2026-10-01.c2" and m["layout"].startswith("2026-10-01.c2") and m["a1"] == "2026-10-01.a1"
          and m["a2"] == "2026-10-01.a2" and m["a4"] == "2026-10-01.a4")
    print("C0 versions:", "OK" if c0 else "MISMATCH", m["c2"], m["layout"], m["a1"], m["a2"], m["a4"], "core", m["core"])
    V = H.verdict
    res = {}
    R = raw["R"]
    bad = [(r["device"], r["seed"], k) for r in R for k in ARMS if r[k]["ideal"] < 1 - 1e-9]
    res["F1"] = V(not bad, bool(bad))
    print(f"F1 all outputs exact without noise: {len(bad)} failures -> {res['F1']}")
    ok = {k: True for k in ("F2a", "F2b", "F3a", "F3b", "F4", "F5")}
    badf = {k: False for k in ok}
    meds = {}
    for d in DEVICES:
        D = [r for r in R if r["device"] == d]
        mi = {k: sum(1 - r[k]["noisy"] for r in D) / len(D) for k in ARMS}
        tq = {k: sum(r[k]["two_q"] for r in D) for k in ARMS}
        meds[d] = statistics.median(r["A4"]["s"] for r in D)
        wins = sum((1 - r["A4"]["noisy"]) < (1 - r["L3T"]["noisy"]) for r in D)
        print(f"   {d}: mean infidelity " + " ".join(f"{k} {mi[k]:.4f}" for k in ARMS)
              + " | 2q " + " ".join(f"{k} {tq[k]}" for k in ARMS)
              + f" | A4/A2 {mi['A4'] / mi['A2']:.3f} A4/L3T {mi['A4'] / mi['L3T']:.3f} A4/L3 {mi['A4'] / mi['L3']:.3f}"
              + f" | A4 lower than L3T in {wins}/{len(D)} | A4 median {meds[d] * 1000:.0f} ms")
        k2, k3 = ("F2a", "F3a") if d == AUCK else ("F2b", "F3b")
        ok[k2] &= mi["A4"] <= THR[k2][0] * mi["A2"]
        badf[k2] |= mi["A4"] > THR[k2][1] * mi["A2"]
        ok[k3] &= mi["A4"] <= THR[k3][0] * mi["L3T"]
        badf[k3] |= mi["A4"] > THR[k3][1] * mi["L3T"]
        ok["F4"] &= mi["A4"] <= THR["F4"][0] * mi["L3"]
        badf["F4"] |= mi["A4"] > THR["F4"][1] * mi["L3"]
        ok["F5"] &= tq["A4"] <= THR["F5"][0] * tq["A2"]
        badf["F5"] |= tq["A4"] > THR["F5"][1] * tq["A2"]
    for k in ("F2a", "F2b", "F3a", "F3b", "F4", "F5"):
        res[k] = V(ok[k], badf[k])
    print(f"F2a Auckland: A4 <= {THR['F2a'][0]} x A2 (refuted above {THR['F2a'][1]} x) -> {res['F2a']}")
    print(f"F2b Torino, Kingston: A4 <= {THR['F2b'][0]} x A2 (refuted above {THR['F2b'][1]} x) -> {res['F2b']}")
    print(f"F3a Auckland: A4 <= {THR['F3a'][0]} x L3T (refuted above {THR['F3a'][1]} x) -> {res['F3a']}")
    print(f"F3b Torino, Kingston: A4 <= {THR['F3b'][0]} x L3T (refuted above {THR['F3b'][1]} x) -> {res['F3b']}")
    print(f"F4 A4 beats Qiskit L3 without errors (<= {THR['F4'][0]} x L3; refuted above {THR['F4'][1]} x) -> {res['F4']}")
    print(f"F5 A4 keeps the 2-qubit count (<= {THR['F5'][0]} x A2; refuted above {THR['F5'][1]} x) -> {res['F5']}")
    mx = max(r["A4"]["s"] for r in R)
    res["F6"] = V(all(v <= THR["F6"][0] for v in meds.values()) and mx <= THR["F6"][2],
                  any(v > THR["F6"][1] for v in meds.values()))
    print(f"F6 A4 time: medians {', '.join(f'{d} {v * 1000:.0f} ms' for d, v in meds.items())}, max {mx:.2f} s -> {res['F6']}")
    NT = raw["NT"]
    nb = [r for r in NT if r["a1"] != r["a4_no_target"]]
    res["F7"] = V(not nb, bool(nb))
    print(f"F7 A4 without a target identical to A1: {len(NT) - len(nb)}/{len(NT)} -> {res['F7']}")
    print("SUMMARY", json.dumps(res))
    print("DECISION:", "ALL CONFIRMED" if c0 and all(v == "CONFIRMED" for v in res.values()) else "NOT ALL CONFIRMED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "score"])
    ap.add_argument("--compile")
    ap.add_argument("--layout")
    ap.add_argument("--a1")
    ap.add_argument("--a2")
    ap.add_argument("--a4")
    ap.add_argument("--out", default="a4_raw.json")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    run(a) if a.mode == "run" else score(a)


if __name__ == "__main__":
    main()
