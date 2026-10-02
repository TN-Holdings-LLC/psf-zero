"""ai6_eval.py -- pre-registered home test (2026-10-02): the AI front end integrated with release psf_compile
2026-10-02.2. Does psf_ai_compile a6 (a5 with the release's exact re-placement inside every compile) keep or improve
a5, does the AI front end still add anything over the release itself, and is a6's fast mode good enough?

Circuit sets:
  GAP    the five GAP families of Addendum 300, via the locked `benchmarks/gap_eval.py` (`family()`), same seeds and
         sizes (2,079 circuits per device; all have at most 8 qubits);
  MODEL  every circuit the models wrote in the 2026-09-30 pod runs that is not a best_circuit.json, unique per task,
         with at most 8 qubits (the 153-circuit replay set of Addendum 285), converted as the e2e harness does
         (`e2e_vllm_psf_v10.to_tape` -> `psf_pennylane_gpu_prototype.tape_to_qiskit`). The smoke run uses the sandbox
         dry-run folders instead (mock circuits), never these.
Arms (release psf_compile 2026-10-02.2, core and layout unchanged):
  C5   compile_for_hardware(cx, layout_search=True, seed 0, target, placement_refine=True)   (the release alone)
  A5   psf_ai_compile 2026-10-01.a5 (benchmarks/psf_ai_compile.py) with the target
  A6   candidate psf_ai_compile 2026-10-02.a6 with the target
  A6F  the candidate with the target and state_aware_placement=False                          (fast mode)
  L3T  Qiskit transpile with the Target, optimization_level 3, approximation_degree 1.0
Devices: FakeAuckland, FakeTorino, FakeKingston. Metric: infidelity 1 - <psi|rho|psi> of the final-layout qubits
under NoiseModel.from_backend (density matrix), as in GAP. Recorded per circuit: two-qubit count, depth, compile time,
failed-edge and failed-qubit uses, and instructions not in the Target ("off-target", reported only).

  python ai6_eval.py prep  --repo <repo> --out <dir> [--smoke]
  python ai6_eval.py run   --repo <repo> --out <dir> --device <d> --arm <a> --set <F1..F5|MODEL> [--smoke]
  python ai6_eval.py score --out <dir>
"""
import argparse
import contextlib
import glob
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")
ARMS = ("C5", "A5", "A6", "A6F", "L3T")
DEVICES = ("FakeAuckland", "FakeTorino", "FakeKingston")
HERON = ("FakeTorino", "FakeKingston")
FAMILIES = ("F1", "F2", "F3", "F4", "F5")
SETS = FAMILIES + ("MODEL",)
CAND = os.path.join("patches", "psf_ai_compile_a6_2026-10-02", "psf_ai_compile.py")
MAX_ACTIVE = 11
MODEL_MAX_QUBITS = 8


def norm_sha(path):
    with open(path, "rb") as f:
        txt = f.read().decode("utf-8").replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in txt.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def collect(data_dir, tasks, sub):
    """The replay set of Addendum 285 (rp_eval.collect), restricted to tasks with at most MODEL_MAX_QUBITS qubits."""
    best = set()
    for f in glob.glob(os.path.join(data_dir, "**", "best_circuit.json"), recursive=True):
        task = f.split(os.sep)[-2]
        best.add(task + json.dumps(json.load(open(f)).get("gates"), sort_keys=True))
    items, seen = [], set()
    for f in sorted(glob.glob(os.path.join(data_dir, "**", sub, "**", "rounds.jsonl"), recursive=True)):
        task = f.split(os.sep)[-2]
        if task not in tasks or tasks[task][0] > MODEL_MAX_QUBITS:
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


def prep(args):
    """Converts the model-written circuits once (needs PennyLane) and stores them as QPY for the run jobs."""
    from qiskit import qpy
    sys.path.insert(0, os.path.join(args.repo, "benchmarks"))
    import e2e_vllm_psf_v10 as E
    from psf_pennylane_gpu_prototype import tape_to_qiskit
    items = collect(os.path.join(args.repo, "data", "2026-09-30"), E.TASKS,
                    "sandbox_dryrun" if args.smoke else "pod_outputs")
    if args.smoke:
        items = items[:6]
    circs, index, skipped = [], [], []
    for task, spec, src in items:
        n = E.TASKS[task][0]
        try:
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tape, _, _, _ = E.to_tape(spec, n)
                qc, _ = tape_to_qiskit(tape, wire_order=list(range(n)))
        except Exception as exc:  # the harness would have rejected it too
            skipped.append(dict(task=task, src=src, error=type(exc).__name__))
            continue
        qc = qc.remove_final_measurements(inplace=False)
        circs.append(qc)
        index.append(dict(task=task, src=src, n=n))
    tag = "_smoke" if args.smoke else ""
    with open(os.path.join(args.out, f"model_circuits{tag}.qpy"), "wb") as f:
        qpy.dump(circs, f)
    with open(os.path.join(args.out, f"model_index{tag}.json"), "w") as f:
        json.dump(dict(collected=len(items), converted=len(circs), skipped=skipped, index=index), f)
    print(f"prep: {len(items)} collected, {len(circs)} converted, {len(skipped)} skipped", flush=True)


def circuits(args, G):
    if args.set != "MODEL":
        yield from G.family(args.set, args.smoke)
        return
    from qiskit import qpy
    tag = "_smoke" if args.smoke else ""
    with open(os.path.join(args.out, f"model_circuits{tag}.qpy"), "rb") as f:
        circs = qpy.load(f)
    index = json.load(open(os.path.join(args.out, f"model_index{tag}.json")))["index"]
    for i, (meta, qc) in enumerate(zip(index, circs)):
        yield dict(meta, k=i), qc


def run(args):
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime import fake_provider
    sys.path.insert(0, os.path.join(args.repo, "benchmarks"))
    sys.path.insert(0, args.repo)
    import core_fix_c2_eval as H
    import gap_eval as G
    rel = H.load_module(os.path.join(args.repo, "psf_compile.py"), "psf_compile")
    lay = H.load_module(os.path.join(args.repo, "benchmarks", "psf_smart_layout.py"), "psl_c2")
    sys.modules["psf_smart_layout"] = lay
    a5 = H.load_module(os.path.join(args.repo, "benchmarks", "psf_ai_compile.py"), "psf_ai_compile")
    a6 = H.load_module(os.path.join(args.repo, CAND), "psf_ai_compile_a6")
    import psf_zero_core
    import qiskit
    backend = getattr(fake_provider, args.device)()
    tgt = backend.target
    cm = tgt.build_coupling_map()
    nat = [g for g in ("cx", "cz", "rz", "sx", "x") if g in tgt.operation_names]
    g2 = next(g for g in ("cz", "ecr", "cx") if g in tgt.operation_names)
    failed = {tuple(q) for q, p in tgt[g2].items() if p is not None and p.error is not None and p.error >= 0.5}
    failed_q = {q[0] for q, p in tgt["sx"].items() if p is not None and p.error is not None and p.error >= 0.5}
    opts = dict(method="density_matrix", max_parallel_threads=1, max_parallel_experiments=1)
    sims = {"noisy": AerSimulator(noise_model=NoiseModel.from_backend(backend), **opts), "ideal": AerSimulator(**opts)}
    meta = dict(release=rel.VERSION, a5=a5.AI_COMPILE_VERSION, a6=a6.AI_COMPILE_VERSION, layout=lay.LAYOUT_VERSION,
                core=getattr(psf_zero_core, "CORE_VERSION", None), qiskit=qiskit.__version__, python=sys.version.split()[0],
                git_head=subprocess.run(["git", "-C", args.repo, "rev-parse", "--short=7", "HEAD"], capture_output=True,
                                        text=True).stdout.strip(),
                sha=dict(script=norm_sha(os.path.abspath(__file__)), gap=norm_sha(G.__file__),
                         a6=norm_sha(os.path.join(args.repo, CAND)), release=norm_sha(os.path.join(args.repo, "psf_compile.py")),
                         a5=norm_sha(os.path.join(args.repo, "benchmarks", "psf_ai_compile.py"))),
                device=args.device, arm=args.arm, set=args.set, smoke=args.smoke, failed_edges=len(failed),
                failed_qubits=len(failed_q), started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print("META", json.dumps(meta), flush=True)

    def off_target(out):
        bad = 0
        for ins in out.data:
            name = ins.operation.name
            if name in ("barrier", "measure", "delay"):
                continue
            q = tuple(out.find_bit(b).index for b in ins.qubits)
            if name not in tgt.operation_names or q not in tgt[name]:
                bad += 1
        return bad

    t0 = time.time()
    rows, circs, keep = [], [], []
    for params, qc in circuits(args, G):
        n = qc.num_qubits
        t1 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if args.arm == "L3T":
                out = transpile(qc, target=tgt, optimization_level=3, seed_transpiler=0, approximation_degree=1.0)
            elif args.arm == "C5":
                out = rel.compile_for_hardware(qc, coupling_map=cm, basis_gates=nat, entangling_basis="cx",
                                               layout_search=True, seed_transpiler=0, target=tgt, placement_refine=True)
            elif args.arm == "A5":
                out = a5.compile_for_model_circuit(qc, cm, nat, target=tgt)
            else:
                out = a6.compile_for_model_circuit(qc, cm, nat, target=tgt, state_aware_placement=(args.arm == "A6"))
        tc = time.perf_counter() - t1
        fin = list(out.layout.final_index_layout(filter_ancillas=True)[:n])
        idx = [tuple(out.find_bit(q).index for q in ins.qubits) for ins in out.data]
        active = {i for t in idx for i in t}
        row = dict(params=params, n=n, compile_s=tc, two_q=sum(1 for t in idx if len(t) == 2), depth=out.depth(),
                   active=len(active), failed_uses=sum(1 for t in idx if len(t) == 2 and (t in failed or t[::-1] in failed)),
                   failed_q_uses=sum(1 for t in idx if any(i in failed_q for i in t)), off_target=off_target(out))
        rows.append(row)
        if len(active) > MAX_ACTIVE:
            row["too_wide"] = True
            continue
        c = out.copy()
        c.remove_final_measurements(inplace=True)
        c.save_density_matrix(qubits=fin)
        circs.append(c)
        keep.append((row, Statevector(qc).data))
    for which, key in (("noisy", "infid"), ("ideal", "infid_ideal")):
        if not circs:
            break
        res = sims[which].run(circs).result()
        for j, (row, psi) in enumerate(keep):
            rho = np.asarray(res.data(j)["density_matrix"])
            row[key] = float(1 - np.real(np.conj(psi) @ rho @ psi))
    name = f"ai6_{args.device}_{args.arm}_{args.set}{'_smoke' if args.smoke else ''}.json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(dict(meta=meta, rows=rows, wall_s=time.time() - t0), f)
    print(f"wrote {name}: {len(rows)} circuits, {len(circs)} simulated, {time.time() - t0:.0f} s", flush=True)


def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def score(args):
    D, smoke = {}, None
    for p in sorted(glob.glob(os.path.join(args.out, "ai6_*.json"))):
        r = json.load(open(p))
        m = r["meta"]
        smoke = m["smoke"]
        D[(m["device"], m["arm"], m["set"])] = r["rows"]
    rows = [x for v in D.values() for x in v]
    missing = [(d, a, s) for d in DEVICES for a in ARMS for s in SETS if (d, a, s) not in D]
    wide = sum(1 for x in rows if x.get("too_wide"))
    p0i = max((x["infid_ideal"] for x in rows if "infid_ideal" in x), default=1.0)
    idx = os.path.join(args.out, "model_index%s.json" % ("_smoke" if smoke else ""))
    nmodel = json.load(open(idx))["converted"] if os.path.exists(idx) else 0
    p0 = (not missing and p0i <= 1e-6 and wide <= 0.05 * max(len(rows), 1) and (smoke or nmodel >= 150))
    L = [f"# ai6 score{' (SMOKE -- not a result)' if smoke else ''}", "",
         f"P0: {'PASS' if p0 else 'FAIL'} -- files {len(D)} of 90 (missing {missing}); noiseless infidelity max "
         f"{p0i:.1e} (<= 1e-6); too wide {wide} of {len(rows)}; model circuits converted {nmodel} (>= 150)"]

    def pairs(d, sets, a, b):
        out = []
        for s in sets:
            for x, y in zip(D.get((d, a, s), []), D.get((d, b, s), [])):
                if "infid" in x and "infid" in y:
                    out.append((x, y))
        return out

    ratio = lambda pr: (np.mean([x["infid"] for x, _ in pr]) / max(np.mean([y["infid"] for _, y in pr]), 1e-12)
                        if pr else float("nan"))
    GROUPS = {"GAP": FAMILIES, "MODEL": ("MODEL",)}
    R = {(g, d, a, b): ratio(pairs(d, sets, a, b)) for g, sets in GROUPS.items() for d in DEVICES
         for a, b in (("A6", "A5"), ("A6", "C5"), ("A6F", "A6"), ("A6", "L3T"), ("A5", "L3T"), ("C5", "L3T"),
                      ("A5", "C5"))}
    L += ["", "| set | device | A6/A5 | A6/C5 | A6F/A6 | A6/L3T | A5/L3T | C5/L3T | mean infidelity C5 / A5 / A6 / A6F / L3T |",
          "|---|---|---|---|---|---|---|---|---|"]
    for g, sets in GROUPS.items():
        for d in DEVICES:
            mi = []
            for a in ARMS:
                v = [x["infid"] for s in sets for x in D.get((d, a, s), []) if "infid" in x]
                mi.append(np.mean(v) if v else float("nan"))
            L.append(f"| {g} | {d} | " + " | ".join(f"{R[(g, d, a, b)]:.3f}" for a, b in
                     (("A6", "A5"), ("A6", "C5"), ("A6F", "A6"), ("A6", "L3T"), ("A5", "L3T"), ("C5", "L3T"))) +
                     " | " + " / ".join(f"{v:.4f}" for v in mi) + " |")
    L += ["", "| set | device | A6/A5 | A6/C5 | A6/L3T | mean 2q C5 / A5 / A6 / A6F / L3T |", "|---|---|---|---|---|---|"]
    for s in SETS:
        for d in DEVICES:
            tq = [np.mean([x["two_q"] for x in D.get((d, a, s), [])]) if D.get((d, a, s)) else float("nan") for a in ARMS]
            L.append(f"| {s} | {d} | {ratio(pairs(d, [s], 'A6', 'A5')):.3f} | {ratio(pairs(d, [s], 'A6', 'C5')):.3f} | "
                     f"{ratio(pairs(d, [s], 'A6', 'L3T')):.3f} | " + " / ".join(f"{v:.1f}" for v in tq) + " |")
    h1 = verdict(all(R[(g, d, "A6", "A5")] <= 1.00 for g in GROUPS for d in DEVICES),
                 any(R[(g, d, "A6", "A5")] > 1.02 for g in GROUPS for d in DEVICES))
    h2 = verdict(all(R[("GAP", d, "A6", "A5")] <= 0.98 for d in HERON),
                 all(R[("GAP", d, "A6", "A5")] >= 1.00 for d in HERON))
    h3 = verdict(all(R[("MODEL", d, "A6", "C5")] <= 0.95 for d in DEVICES),
                 any(R[("MODEL", d, "A6", "C5")] >= 1.00 for d in DEVICES))
    h4 = verdict(all(R[("GAP", d, "A6", "C5")] <= 1.00 for d in DEVICES),
                 any(R[("GAP", d, "A6", "C5")] > 1.05 for d in DEVICES))
    med = {a: float(np.median([x["compile_s"] for (d, aa, s), v in D.items() if aa == a for x in v])) for a in ARMS}
    fq = {(g, d): R[(g, d, "A6F", "A6")] for g in GROUPS for d in HERON}
    h5 = verdict(all(v <= 1.05 for v in fq.values()) and med["A6F"] <= 0.5 * med["A6"],
                 any(v > 1.15 for v in fq.values()) or med["A6F"] > med["A6"])
    h6 = verdict(all(R[(g, d, "A6", "L3T")] <= 1.00 for g in GROUPS for d in DEVICES),
                 any(R[(g, d, "A6", "L3T")] > 1.05 for g in GROUPS for d in DEVICES))
    nf = {a: sum(x["failed_uses"] + x["failed_q_uses"] for (d, aa, s), v in D.items() if aa == a for x in v) for a in ARMS}
    h7 = verdict(nf["C5"] + nf["A6"] + nf["A6F"] == 0, nf["C5"] + nf["A6"] + nf["A6F"] > 0)
    h8 = verdict(med["A6"] <= 1.3 * med["A5"], med["A6"] > 2 * med["A5"])
    fmt = lambda keys: ", ".join(f"{g}/{d} {R[(g, d) + keys]:.3f}" for g in GROUPS for d in DEVICES)
    h2txt = ", ".join("%s %.3f" % (d, R[("GAP", d, "A6", "A5")]) for d in HERON)
    one = lambda g, a, b: ", ".join("%s %.3f" % (d, R[(g, d, a, b)]) for d in DEVICES)
    offt = {a: sum(x["off_target"] > 0 for (d, aa, s), v in D.items() if aa == a for x in v) for a in ARMS}
    L += ["", "## Predictions", "",
          f"- H1 (A6/A5 <= 1.00 on every device, GAP and MODEL): **{h1}** ({fmt(('A6', 'A5'))})",
          f"- H2 (GAP: A6/A5 <= 0.98 on both Heron devices): **{h2}** ({h2txt})",
          f"- H3 (MODEL: A6/C5 <= 0.95 on every device): **{h3}** ({one('MODEL', 'A6', 'C5')})",
          f"- H4 (GAP: A6/C5 <= 1.00 on every device): **{h4}** ({one('GAP', 'A6', 'C5')})",
          f"- H5 (fast mode: A6F/A6 <= 1.05 on the Heron devices, GAP and MODEL, and median time A6F <= 0.5 x A6): "
          f"**{h5}** ({', '.join(f'{g}/{d} {v:.3f}' for (g, d), v in fq.items())}; median A6F {med['A6F']:.3f} s, "
          f"A6 {med['A6']:.3f} s)",
          f"- H6 (A6/L3T <= 1.00 on every device, GAP and MODEL): **{h6}** ({fmt(('A6', 'L3T'))})",
          f"- H7 (C5, A6 and A6F never use a failed coupler or qubit): **{h7}** ({nf})",
          f"- H8 (median compile time A6 <= 1.3 x A5): **{h8}** (A5 {med['A5']:.3f} s, A6 {med['A6']:.3f} s)"]
    L.append("\nReported: median compile s " + ", ".join(f"{a} {v:.3f}" for a, v in med.items()))
    L.append("Reported: circuits with an off-target instruction, by arm: " + ", ".join(f"{a} {v}" for a, v in offt.items()))
    txt = "\n".join(L) + "\n"
    with open(os.path.join(args.out, "score.md"), "w") as f:
        f.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("prep", "run", "score"))
    ap.add_argument("--repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", choices=DEVICES)
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--set", choices=SETS)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    {"prep": prep, "run": run, "score": score}[args.part](args)


if __name__ == "__main__":
    main()
