"""nf_eval.py -- pre-registered noisy-simulation fidelity comparison (workplace, 2026-10-01).

Do fewer 2-qubit gates give higher fidelity under the fake devices' own noise? Arms:
  REL  released stack: psf_compile 2026-09-28.1 + psf_smart_layout 2026-09-26.m1, compile_for_hardware(cx, layout_search)
  C2   adopted stack: psf_compile 2026-10-01.c2 + psf_smart_layout 2026-10-01.c2, same call
  A1   psf_ai_compile 2026-10-01.a1 on top of C2
  L3   Qiskit transpile(optimization_level=3, seed_transpiler=0)
Noise: qiskit_aer NoiseModel.from_backend(<fake device>) -- per-gate depolarizing + thermal relaxation from the
snapshot's error rates, T1/T2 and gate lengths. No idle (delay) noise, no crosstalk, no readout (not measured).
Fidelity: <psi|rho|psi>, psi = ideal output of the logical circuit from |0...0>, rho = density matrix of the compiled
circuit on its final-layout qubits (AerSimulator density_matrix, unused qubits truncated by Aer).
Inputs (new): random dense circuits, Python random seeds 7001-7040 (3-5 qubits), on FakeAuckland and FakeTorino;
the 9/30 model-written circuits (unique, FakeAuckland) as a secondary set.

  python nf_eval.py run --rel-compile <...> --rel-layout <...> --compile <c2> --layout <c2> --a1 <a1> \
         [--model-dir <data/2026-09-30>] [--v10-dir <dir with e2e_vllm_psf_v10.py>] [--out nf_raw.json] [--dry]
  python nf_eval.py score [--out nf_raw.json]
"""
import argparse
import contextlib
import glob
import io
import json
import math
import os
import random
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_fix_c2_eval as H  # noqa: E402

DEVICES = ("FakeAuckland", "FakeTorino")
DRY_OFFSET = 900000


def run(args):
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime import fake_provider
    rel = H.load_module(args.rel_compile, "psfc_rel")
    lay_rel = H.load_module(args.rel_layout, "psl_rel")
    comp = H.load_module(args.compile, "psf_compile")
    lay = H.load_module(args.layout, "psl_c2")
    a1 = H.load_module(args.a1, "psf_ai_compile_a1")
    import psf_zero_core
    off = DRY_OFFSET if args.dry else 0
    meta = dict(rel=rel.VERSION, rel_layout=lay_rel.LAYOUT_VERSION, c2=comp.VERSION, layout=lay.LAYOUT_VERSION,
                a1=a1.AI_COMPILE_VERSION, core=getattr(psf_zero_core, "CORE_VERSION", None), dry=bool(args.dry),
                qiskit=__import__("qiskit").__version__, aer=__import__("qiskit_aer").__version__,
                sha={"rel": H.norm_sha(args.rel_compile), "rel_layout": H.norm_sha(args.rel_layout),
                     "c2": H.norm_sha(args.compile), "layout": H.norm_sha(args.layout), "a1": H.norm_sha(args.a1),
                     "script": H.norm_sha(os.path.abspath(__file__)), "helpers": H.norm_sha(H.__file__)})
    print("META", json.dumps(meta), flush=True)
    raw = {"meta": meta, "R": [], "MODEL": []}

    def arms(qc, cm, nat):
        out = {}
        sys.modules["psf_smart_layout"] = lay_rel
        with contextlib.redirect_stdout(io.StringIO()):
            out["REL"] = rel.compile_for_hardware(qc, coupling_map=cm, basis_gates=nat, entangling_basis="cx",
                                                  layout_search=True, seed_transpiler=0)
        sys.modules["psf_smart_layout"] = lay
        out["C2"] = H.cfh(comp, qc, cm, nat)
        with contextlib.redirect_stdout(io.StringIO()):
            out["A1"] = a1.compile_for_model_circuit(qc, cm, nat)
        out["L3"] = transpile(qc, coupling_map=cm, basis_gates=nat, optimization_level=3, seed_transpiler=0)
        return out

    def fidelity(qc, out, sim):
        n = qc.num_qubits
        fin = list(out.layout.final_index_layout(filter_ancillas=True)[:n])
        c = out.copy()
        c.remove_final_measurements(inplace=True)
        c.save_density_matrix(qubits=fin)
        rho = np.asarray(sim.run(c).result().data()["density_matrix"])
        psi = Statevector(qc.remove_final_measurements(inplace=False)).data
        return float(np.real(psi.conj() @ rho @ psi))

    def evaluate(qc, cm, nat, sims, extra):
        outs = arms(qc, cm, nat)
        row = dict(extra, n=qc.num_qubits)
        for k, o in outs.items():
            row[k] = dict(two_q=H.two_q(o), one_q=sum(1 for i in o.data if len(i.qubits) == 1),
                          ideal=fidelity(qc, o, sims["ideal"]), noisy=fidelity(qc, o, sims["noisy"]))
        return row

    for bname in DEVICES:
        backend = getattr(fake_provider, bname)()
        tgt = backend.target
        cm = tgt.build_coupling_map()
        nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
        sims = {"ideal": AerSimulator(method="density_matrix"),
                "noisy": AerSimulator(method="density_matrix", noise_model=NoiseModel.from_backend(backend))}
        for seed in range(off + 7001, off + (7004 if args.dry else 7041)):
            rng = random.Random(seed)
            n = rng.choice([3, 4, 5])
            qc = H.rand_dense(n, rng.randint(6, 20), rng)
            row = evaluate(qc, cm, nat, sims, dict(device=bname, seed=seed))
            raw["R"].append(row)
            print("R", json.dumps(row), flush=True)
        if bname == "FakeAuckland" and args.model_dir and not args.dry:
            sys.path.insert(0, args.v10_dir)
            import e2e_vllm_psf_v10 as E
            from psf_pennylane_gpu_prototype import tape_to_qiskit
            seen = set()
            for f in sorted(glob.glob(os.path.join(args.model_dir, "**", "best_circuit.json"), recursive=True)):
                task = f.split(os.sep)[-2]
                if task not in E.TASKS:
                    continue
                spec = json.load(open(f))
                key = task + json.dumps(spec.get("gates"), sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                nq = E.TASKS[task][0]
                if nq > 8:
                    continue  # 27-qubit tilings: density matrix too large and a1 falls through anyway
                tape, _, _, _ = E.to_tape(spec, nq)
                qc, _ = tape_to_qiskit(tape, wire_order=list(range(nq)))
                row = evaluate(qc, cm, nat, sims, dict(device=bname, task=task))
                raw["MODEL"].append(row)
                print("MODEL", json.dumps(row), flush=True)
    with open(args.out, "w") as f:
        json.dump(raw, f, indent=1)
    print("wrote", args.out)


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def score(args):
    raw = json.load(open(args.out))
    m = raw["meta"]
    c0 = (m["rel"] == "2026-09-28.1" and m["rel_layout"].startswith("2026-09-26.m1") and m["c2"] == "2026-10-01.c2"
          and m["layout"].startswith("2026-10-01.c2") and m["a1"] == "2026-10-01.a1")
    rows_all = raw["R"] + raw["MODEL"]
    ideal_bad = [(r.get("seed", r.get("task")), k) for r in rows_all for k in ("REL", "C2", "A1", "L3")
                 if r[k]["ideal"] < 1 - 1e-9]
    print("C0 versions:", "OK" if c0 else "MISMATCH", m["rel"], m["rel_layout"], m["c2"], m["layout"], m["a1"],
          "| noiseless fidelity < 1-1e-9:", len(ideal_bad))
    V = H.verdict
    res = {}
    ok1, bad1, ok2, bad2, ok4, bad4 = True, False, True, False, True, False
    for d in DEVICES:
        R = [r for r in raw["R"] if r["device"] == d]
        inf = {k: [1 - r[k]["noisy"] for r in R] for k in ("REL", "C2", "A1", "L3")}
        mean = {k: sum(v) / len(v) for k, v in inf.items()}
        two = {k: sum(r[k]["two_q"] for r in R) for k in inf}
        diff = [r for r in R if r["A1"]["two_q"] != r["REL"]["two_q"]]
        win = sum(1 - r["A1"]["noisy"] < 1 - r["REL"]["noisy"] for r in diff) / max(1, len(diff))
        print(f"   {d} (n={len(R)}): mean infidelity REL {mean['REL']:.4f} C2 {mean['C2']:.4f} A1 {mean['A1']:.4f} "
              f"L3 {mean['L3']:.4f} | 2q sums REL {two['REL']} C2 {two['C2']} A1 {two['A1']} L3 {two['L3']} | "
              f"A1 better than REL in {win:.0%} of {len(diff)} circuits with different 2q counts")
        ok1 &= mean["A1"] < mean["REL"] and win >= 0.70
        bad1 |= mean["A1"] >= mean["REL"]
        ok2 &= mean["A1"] <= 1.05 * mean["L3"]
        bad2 |= mean["A1"] > 1.15 * mean["L3"]
        ok4 &= mean["A1"] <= mean["C2"]
        bad4 |= mean["A1"] > 1.02 * mean["C2"]
    res["N1"] = V(ok1, bad1)
    res["N2"] = V(ok2, bad2)
    # N3: within one circuit, between two arms with different 2-qubit counts, does the one with fewer 2-qubit gates
    # have the lower infidelity? (pairwise concordance; isolates the gate count from circuit size and content)
    per_dev = {}
    for d in DEVICES:
        R = [r for r in raw["R"] if r["device"] == d]
        agree = total = 0
        for r in R:
            ks = ("REL", "C2", "A1", "L3")
            for i in range(4):
                for j in range(i + 1, 4):
                    a_, b_ = r[ks[i]], r[ks[j]]
                    if a_["two_q"] == b_["two_q"]:
                        continue
                    total += 1
                    fewer, more = (a_, b_) if a_["two_q"] < b_["two_q"] else (b_, a_)
                    agree += (1 - fewer["noisy"]) < (1 - more["noisy"])
        per_dev[d] = (agree / total if total else float("nan"), total)
    res["N3"] = V(all(v[0] >= 0.70 for v in per_dev.values()), any(v[0] < 0.55 for v in per_dev.values()))
    res["N4"] = V(ok4, bad4)
    res["N0"] = V(not ideal_bad, bool(ideal_bad))
    print(f"N0 every compiled output exact without noise (fidelity >= 1-1e-9): {len(ideal_bad)} failures -> {res['N0']}")
    print(f"N1 A1 lower infidelity than the released stack (mean, and >= 70 % of differing circuits) -> {res['N1']}")
    print(f"N2 A1 at L3 level (mean infidelity <= 1.05 x L3) -> {res['N2']}")
    print(f"N3 fewer 2-qubit gates -> lower infidelity, within-circuit arm pairs (>= 70 % per device): "
          f"{', '.join(f'{d} {v[0]:.1%} of {v[1]}' for d, v in per_dev.items())} -> {res['N3']}")
    print(f"N4 A1 not worse than C2 (mean infidelity) -> {res['N4']}")
    M = raw["MODEL"]
    if M:
        inf = {k: sum(1 - r[k]["noisy"] for r in M) / len(M) for k in ("REL", "C2", "A1", "L3")}
        print(f"   model circuits (FakeAuckland, unique, n={len(M)}; reported without prediction): mean infidelity "
              + " ".join(f"{k} {v:.4f}" for k, v in inf.items()))
    print("SUMMARY", json.dumps(res))
    print("DECISION:", "ALL CONFIRMED" if c0 and all(v == "CONFIRMED" for v in res.values()) else "NOT ALL CONFIRMED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "score"])
    ap.add_argument("--rel-compile")
    ap.add_argument("--rel-layout")
    ap.add_argument("--compile")
    ap.add_argument("--layout")
    ap.add_argument("--a1")
    ap.add_argument("--model-dir")
    ap.add_argument("--v10-dir", default=".")
    ap.add_argument("--out", default="nf_raw.json")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    run(a) if a.mode == "run" else score(a)


if __name__ == "__main__":
    main()
