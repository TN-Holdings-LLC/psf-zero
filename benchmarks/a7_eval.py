"""a7_eval.py -- pre-registered home test (2026-10-02): does psf_ai_compile a7 (a5 plus Qiskit level 3's own output as a
candidate, Addendum 314) remove a5's loss to Qiskit L3T on the F3 Heisenberg chains on FakeAuckland without costing
anything elsewhere?

Circuit sets (as in Addendum 312):
  GAP    the five GAP families via the locked `benchmarks/gap_eval.py` (`family()`): 693 circuits per device;
  MODEL  the 153-circuit replay set of model-written circuits (Addendum 285), converted as the e2e harness does. The
         smoke run uses the sandbox dry-run (mock) circuits instead.
Arms: A5 (benchmarks/psf_ai_compile.py, a5, with the target), A7 (the candidate with the target), L3T (Qiskit level 3
  with the Target, approximation_degree 1.0). Release psf_compile 2026-10-02.2 underneath, without its opt-in
  arguments. Devices and metric as in Addendum 312. Recorded for A7: which candidate it returned (PSF or L3T).

  python a7_eval.py prep  --repo <repo> --out <dir> [--smoke]
  python a7_eval.py run   --repo <repo> --out <dir> --device <d> --arm <a> --set <F1..F5|MODEL> [--smoke]
  python a7_eval.py score --out <dir>
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
ARMS = ("A5", "A7", "L3T")
DEVICES = ("FakeAuckland", "FakeTorino", "FakeKingston")
HERON = ("FakeTorino", "FakeKingston")
FAMILIES = ("F1", "F2", "F3", "F4", "F5")
SETS = FAMILIES + ("MODEL",)
CAND = os.path.join("patches", "psf_ai_compile_a7_2026-10-02", "psf_ai_compile.py")
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
    a7 = H.load_module(os.path.join(args.repo, CAND), "psf_ai_compile_a7")
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
    meta = dict(release=rel.VERSION, a5=a5.AI_COMPILE_VERSION, a7=a7.AI_COMPILE_VERSION, layout=lay.LAYOUT_VERSION,
                core=getattr(psf_zero_core, "CORE_VERSION", None), qiskit=qiskit.__version__, python=sys.version.split()[0],
                git_head=subprocess.run(["git", "-C", args.repo, "rev-parse", "--short=7", "HEAD"], capture_output=True,
                                        text=True).stdout.strip(),
                sha=dict(script=norm_sha(os.path.abspath(__file__)), gap=norm_sha(G.__file__),
                         a7=norm_sha(os.path.join(args.repo, CAND)), release=norm_sha(os.path.join(args.repo, "psf_compile.py")),
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
        chosen = None
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if args.arm == "L3T":
                out = transpile(qc, target=tgt, optimization_level=3, seed_transpiler=0, approximation_degree=1.0)
            elif args.arm == "A5":
                out = a5.compile_for_model_circuit(qc, cm, nat, target=tgt)
            else:
                out, info = a7.compile_for_model_circuit(qc, cm, nat, target=tgt, return_info=True)
                chosen = info.get("chosen", info.get("path"))
        tc = time.perf_counter() - t1
        fin = list(out.layout.final_index_layout(filter_ancillas=True)[:n])
        idx = [tuple(out.find_bit(q).index for q in ins.qubits) for ins in out.data]
        active = {i for t in idx for i in t}
        row = dict(params=params, n=n, compile_s=tc, two_q=sum(1 for t in idx if len(t) == 2), depth=out.depth(),
                   active=len(active), failed_uses=sum(1 for t in idx if len(t) == 2 and (t in failed or t[::-1] in failed)),
                   failed_q_uses=sum(1 for t in idx if any(i in failed_q for i in t)), off_target=off_target(out),
                   chosen=chosen)
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
    name = f"a7_{args.device}_{args.arm}_{args.set}{'_smoke' if args.smoke else ''}.json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(dict(meta=meta, rows=rows, wall_s=time.time() - t0), f)
    print(f"wrote {name}: {len(rows)} circuits, {len(circs)} simulated, {time.time() - t0:.0f} s", flush=True)


def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def score(args):
    D, smoke = {}, None
    for p in sorted(glob.glob(os.path.join(args.out, "a7_*.json"))):
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
    L = [f"# a7 score{' (SMOKE -- not a result)' if smoke else ''}", "",
         f"P0: {'PASS' if p0 else 'FAIL'} -- files {len(D)} of 54 (missing {missing}); noiseless infidelity max "
         f"{p0i:.1e} (<= 1e-6); too wide {wide} of {len(rows)}; model circuits converted {nmodel} (>= 150)"]

    def sel(d, a, s, bc=None):
        return [x for x in D.get((d, a, s), []) if bc is None or x["params"].get("bc") == bc]

    def ratio(d, sets, a, b, bc=None):
        xs = [x["infid"] for s in sets for x in sel(d, a, s, bc) if "infid" in x]
        ys = [y["infid"] for s in sets for y in sel(d, b, s, bc) if "infid" in y]
        return float(np.mean(xs) / max(np.mean(ys), 1e-12)) if xs and ys else float("nan")

    GROUPS = {"GAP": FAMILIES, "MODEL": ("MODEL",)}
    L += ["", "| set | device | A7/A5 | A7/L3T | A5/L3T | A7 chose L3T's output |", "|---|---|---|---|---|---|"]
    cells = [(f, None) for f in ("F1", "F2")] + [("F3", "o"), ("F3", "p")] + [(f, None) for f in ("F4", "F5", "MODEL")]
    for f, bc in cells:
        for d in DEVICES:
            n_l3 = sum(x.get("chosen") == "L3T" for x in sel(d, "A7", f, bc))
            L.append(f"| {f}{bc or ''} | {d} | {ratio(d, [f], 'A7', 'A5', bc):.3f} | {ratio(d, [f], 'A7', 'L3T', bc):.3f} | "
                     f"{ratio(d, [f], 'A5', 'L3T', bc):.3f} | {n_l3} of {len(sel(d, 'A7', f, bc))} |")
    f3 = {bc: ratio("FakeAuckland", ["F3"], "A7", "L3T", bc) for bc in ("o", "p")}
    h1 = verdict(all(v <= 1.02 for v in f3.values()), any(v >= 1.10 for v in f3.values()))
    r75 = {(g, d): ratio(d, s, "A7", "A5") for g, s in GROUPS.items() for d in DEVICES}
    h2 = verdict(all(v <= 1.00 for v in r75.values()), any(v > 1.02 for v in r75.values()))
    ra = ratio("FakeAuckland", FAMILIES, "A7", "L3T")
    h3 = verdict(ra <= 1.00, ra > 1.02)
    frac = {}
    for d in DEVICES:
        pr = [(x, y) for s in SETS for x, y in zip(sel(d, "A7", s), sel(d, "A5", s)) if "infid" in x and "infid" in y]
        frac[d] = sum(x["infid"] <= y["infid"] + 1e-12 for x, y in pr) / max(len(pr), 1)
    h4 = verdict(all(v >= 0.95 for v in frac.values()), any(v < 0.90 for v in frac.values()))
    med = {a: float(np.median([x["compile_s"] for (d, aa, s), v in D.items() if aa == a for x in v])) for a in ARMS}
    h5 = verdict(med["A7"] <= 1.1 * med["A5"], med["A7"] > 1.5 * med["A5"])
    nf = sum(x["failed_uses"] + x["failed_q_uses"] for (d, aa, s), v in D.items() if aa == "A7" for x in v)
    h6 = verdict(nf == 0, nf > 0)
    fmt = lambda dct: ", ".join("%s %.4f" % ("/".join(k) if isinstance(k, tuple) else k, v) for k, v in dct.items())
    offt = {a: sum(x["off_target"] > 0 for (d, aa, s), v in D.items() if aa == a for x in v) for a in ARMS}
    L += ["", "## Predictions", "",
          f"- H1 (FakeAuckland F3: A7/L3T <= 1.02 on open and periodic chains): **{h1}** ({fmt(f3)})",
          f"- H2 (A7/A5 <= 1.00 on every device, GAP and MODEL): **{h2}** ({fmt(r75)})",
          f"- H3 (FakeAuckland GAP: A7/L3T <= 1.00): **{h3}** ({ra:.4f})",
          f"- H4 (per circuit A7 <= A5 in >= 95% on every device): **{h4}** ({fmt(frac)})",
          f"- H5 (median compile time A7 <= 1.1 x A5): **{h5}** (A5 {med['A5']:.3f} s, A7 {med['A7']:.3f} s)",
          f"- H6 (A7 never uses a failed coupler or qubit): **{h6}** ({nf} uses)"]
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
