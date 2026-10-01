"""rp_eval.py -- pre-registered replay of model-written circuits through the AI front end a2 (workplace, 2026-10-01).

Inputs: every circuit the models wrote in the 2026-09-30 pod runs (rounds.jsonl, field "spec") that is NOT one of the
best_circuit.json files (those were used in development), unique per task, with at most 8 qubits, converted exactly
as the e2e harness does (to_tape -> tape_to_qiskit). They include wrong answers: the comparison is between compilers
of the same logical circuit, so correctness against the task does not matter here.
Arms (psf_compile / psf_smart_layout 2026-10-01.c2, core 2026-09-29.1):
  C2   compile_for_hardware(cx, layout_search=True, seed 0)           (what v10 used, adopted version)
  A2   psf_ai_compile 2026-10-01.a2 with the device Target            (what v11 --compiler ai uses)
  L3   Qiskit level 3 with coupling_map/basis only
  L3T  Qiskit level 3 with target                                      (the harness's own "Q3" comparison)
Devices: FakeAuckland (the e2e device) and FakeTorino. Noise: NoiseModel.from_backend; fidelity on final-layout
qubits (density matrix).

  python rp_eval.py run --compile <c2> --layout <c2> --a2 <a2.py> --data <data/2026-09-30> --v10-dir <dir>
                        [--out rp_raw.json] [--dry]
  python rp_eval.py score [--out rp_raw.json]
"""
import argparse
import contextlib
import glob
import io
import json
import os
import statistics
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core_fix_c2_eval as H  # noqa: E402

DEVICES = ("FakeAuckland", "FakeTorino")
ARMS = ("C2", "A2", "L3", "L3T")


def collect(data_dir, E, sub="pod_outputs"):
    best = set()
    for f in glob.glob(os.path.join(data_dir, "**", "best_circuit.json"), recursive=True):
        task = f.split(os.sep)[-2]
        best.add(task + json.dumps(json.load(open(f)).get("gates"), sort_keys=True))
    items, seen = [], set()
    for f in sorted(glob.glob(os.path.join(data_dir, "**", sub, "**", "rounds.jsonl"), recursive=True)):
        task = f.split(os.sep)[-2]
        if task not in E.TASKS or E.TASKS[task][0] > 8:
            continue
        for line in open(f):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            spec = r.get("spec")
            if not isinstance(spec, dict) or "gates" not in spec:
                continue
            key = task + json.dumps(spec["gates"], sort_keys=True)
            if key in best or key in seen:
                continue
            seen.add(key)
            items.append((task, spec, os.path.relpath(f, data_dir)))
    return items


def run(args):
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime import fake_provider
    comp = H.load_module(args.compile, "psf_compile")
    lay = H.load_module(args.layout, "psf_smart_layout")
    a2 = H.load_module(args.a2, "psf_ai_compile_a2")
    sys.path.insert(0, args.v10_dir)
    import e2e_vllm_psf_v10 as E
    from psf_pennylane_gpu_prototype import tape_to_qiskit
    import psf_zero_core
    # the dry run uses the sandbox dry-run folders (mock/fake-model circuits), never the scored pod circuits
    items = collect(args.data, E, sub="sandbox_dryrun")[:6] if args.dry else collect(args.data, E)
    meta = dict(c2=comp.VERSION, layout=lay.LAYOUT_VERSION, a2=a2.AI_COMPILE_VERSION,
                core=getattr(psf_zero_core, "CORE_VERSION", None), dry=bool(args.dry), n_items=len(items),
                qiskit=__import__("qiskit").__version__, aer=__import__("qiskit_aer").__version__,
                sha={"c2": H.norm_sha(args.compile), "layout": H.norm_sha(args.layout), "a2": H.norm_sha(args.a2),
                     "script": H.norm_sha(os.path.abspath(__file__)), "helpers": H.norm_sha(H.__file__)})
    print("META", json.dumps(meta), flush=True)
    raw = {"meta": meta, "rows": [], "skipped": []}

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
        for task, spec, src in items:
            n = E.TASKS[task][0]
            try:
                tape, _, _, _ = E.to_tape(spec, n)
                qc, _ = tape_to_qiskit(tape, wire_order=list(range(n)))
            except Exception as exc:  # the harness would have rejected it too
                raw["skipped"].append(dict(device=bname, task=task, src=src, error=type(exc).__name__))
                continue
            outs, times = {}, {}
            t = time.perf_counter()
            outs["C2"] = H.cfh(comp, qc, cm, nat)
            times["C2"] = time.perf_counter() - t
            with contextlib.redirect_stdout(io.StringIO()):
                t = time.perf_counter()
                outs["A2"] = a2.compile_for_model_circuit(qc, cm, nat, target=tgt)
                times["A2"] = time.perf_counter() - t
            outs["L3"] = transpile(qc, coupling_map=cm, basis_gates=nat, optimization_level=3, seed_transpiler=0)
            outs["L3T"] = transpile(qc, target=tgt, optimization_level=3, seed_transpiler=0)
            row = dict(device=bname, task=task, src=src, n=n, logical_2q=H.two_q(qc))
            for k, o in outs.items():
                row[k] = dict(two_q=H.two_q(o), ideal=fidelity(qc, o, ideal), noisy=fidelity(qc, o, noisy),
                              s=round(times.get(k, 0.0), 4))
            raw["rows"].append(row)
            print("ROW", json.dumps(row), flush=True)
    with open(args.out, "w") as f:
        json.dump(raw, f, indent=1)
    print("wrote", args.out)


def score(args):
    raw = json.load(open(args.out))
    m = raw["meta"]
    c0 = m["c2"] == "2026-10-01.c2" and m["layout"].startswith("2026-10-01.c2") and m["a2"] == "2026-10-01.a2"
    print("C0 versions:", "OK" if c0 else "MISMATCH", m["c2"], m["layout"], m["a2"], "core", m["core"],
          "| items", m["n_items"], "| skipped", len(raw["skipped"]))
    V = H.verdict
    res = {}
    rows = raw["rows"]
    bad = [(r["device"], r["task"], k) for r in rows for k in ARMS if r[k]["ideal"] < 1 - 1e-9]
    res["E1"] = V(not bad, bool(bad))
    print(f"E1 all outputs exact without noise: {len(bad)} failures of {4 * len(rows)} -> {res['E1']}")
    ok = {k: True for k in ("E2", "E3", "E4")}
    badf = {k: False for k in ok}
    meds, mxs = {}, []
    for d in DEVICES:
        D = [r for r in rows if r["device"] == d]
        mi = {k: sum(1 - r[k]["noisy"] for r in D) / len(D) for k in ARMS}
        tq = {k: sum(r[k]["two_q"] for r in D) for k in ARMS}
        meds[d] = statistics.median(r["A2"]["s"] for r in D)
        mxs.append(max(r["A2"]["s"] for r in D))
        w = sum((1 - r["A2"]["noisy"]) < (1 - r["L3T"]["noisy"]) - 1e-12 for r in D)
        t_ = sum((1 - r["A2"]["noisy"]) > (1 - r["L3T"]["noisy"]) + 1e-12 for r in D)
        print(f"   {d} (n={len(D)}): mean infidelity " + " ".join(f"{k} {mi[k]:.4f}" for k in ARMS)
              + " | 2q " + " ".join(f"{k} {tq[k]}" for k in ARMS)
              + f" | A2/C2 {mi['A2'] / mi['C2']:.3f} A2/L3T {mi['A2'] / mi['L3T']:.3f} | A2 better/worse than L3T "
              f"{w}/{t_} | A2 median {meds[d] * 1000:.0f} ms")
        for task in sorted({r["task"] for r in D}):
            T = [r for r in D if r["task"] == task]
            print(f"      {task:9s} n={len(T):3d} mean infidelity " + " ".join(
                f"{k} {sum(1 - r[k]['noisy'] for r in T) / len(T):.4f}" for k in ARMS)
                + " | 2q " + " ".join(f"{k} {sum(r[k]['two_q'] for r in T)}" for k in ARMS))
        ok["E2"] &= mi["A2"] <= 0.90 * mi["C2"]
        badf["E2"] |= mi["A2"] >= mi["C2"]
        ok["E3"] &= mi["A2"] <= 1.05 * mi["L3T"]
        badf["E3"] |= mi["A2"] > 1.20 * mi["L3T"]
        ok["E4"] &= tq["A2"] <= tq["L3T"]
        badf["E4"] |= tq["A2"] > 1.05 * tq["L3T"]
    for k in ok:
        res[k] = V(ok[k], badf[k])
    print(f"E2 A2 beats C2 (the compiler v10 used) under noise: mean infidelity <= 0.90 x C2 -> {res['E2']}")
    print(f"E3 A2 at the harness's own Qiskit comparison (L3 with target): <= 1.05 x L3T -> {res['E3']}")
    print(f"E4 A2 2-qubit sum <= L3T's -> {res['E4']}")
    res["E5"] = V(all(v <= 0.6 for v in meds.values()) and max(mxs) <= 3.0, any(v > 1.2 for v in meds.values()))
    print(f"E5 A2 time per compile: medians {', '.join(f'{d} {v * 1000:.0f} ms' for d, v in meds.items())}, "
          f"max {max(mxs):.2f} s -> {res['E5']}")
    print("SUMMARY", json.dumps(res))
    print("DECISION:", "ALL CONFIRMED" if c0 and all(v == "CONFIRMED" for v in res.values()) else "NOT ALL CONFIRMED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "score"])
    ap.add_argument("--compile")
    ap.add_argument("--layout")
    ap.add_argument("--a2")
    ap.add_argument("--data")
    ap.add_argument("--v10-dir", default=".")
    ap.add_argument("--out", default="rp_raw.json")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    run(a) if a.mode == "run" else score(a)


if __name__ == "__main__":
    main()
