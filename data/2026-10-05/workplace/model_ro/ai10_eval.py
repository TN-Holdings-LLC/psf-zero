"""ai10_eval.py -- workplace test MODEL-RO (2026-10-05): does candidate psf_ai_compile 2026-10-05.a10 (item 15:
readout of measured qubits in the state-aware estimate) give model-written circuits, sampled in the computational
basis, a more faithful output distribution than a9?

Set: the 153 model-written circuits of Addendum 285 (data/2026-10-02/ai6/outputs/model_circuits.qpy, 3-6 qubits),
each with measure_all(). Dry run: synthetic GHZ/W circuits instead (no model circuit).
Arms: A9 (a9), A10 (a10) -- both with candidate psf_compile c13 underneath and the device Target -- and L3TM (Qiskit
level 3 with the Target, approximation_degree 1.0). Devices: FakeTorino, FakeKingston, FakeAuckland.
Metric per circuit: the classical infidelity 1 - (sum_x sqrt(p_x q_x))^2 between the ideal output distribution p of
the measured logical qubits and the noisy one q: Aer density matrix of the compiled circuit without its measurements,
noise model restricted to the touched qubits (DEPTH's Noisy), marginal on the measured physical qubits in clbit order,
then each qubit's readout assignment probabilities (Aer's, asymmetric). Also: summed Target measure error of the
measured qubits, two-qubit count, compile time; exactness (noiseless distribution vs ideal, total variation <= 1e-6)
and clbit j measuring the final position of logical j.

  python ai10_eval.py run --device D --out DIR [--dry]      python ai10_eval.py score --out DIR
"""
import argparse, contextlib, glob, importlib.util, io, json, os, sys, time, warnings
import numpy as np
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import psf_compile as pc  # noqa: E402  (candidate c13, next to this file)
DEVICES = ("FakeTorino", "FakeKingston", "FakeAuckland")
ARMS = ("A9", "A10", "L3TM")
QPY = os.environ.get("MODEL_QPY", os.path.join(HERE, "model_circuits.qpy"))


def norm_sha(path):
    import hashlib
    t = open(path, "rb").read().decode("utf-8").replace("\r\n", "\n")
    ls = [x.rstrip() for x in t.split("\n")]
    while ls and ls[-1] == "":
        ls.pop()
    return hashlib.sha256("\n".join(ls).encode()).hexdigest()


def load(p, n):
    s = importlib.util.spec_from_file_location(n, os.path.join(HERE, p))
    m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m


def circuits(dry):
    from qiskit import QuantumCircuit, qpy
    if dry:
        out = []
        rng = np.random.default_rng(77)
        for n in (3, 4, 5):
            for _ in range(2):
                qc = QuantumCircuit(n); qc.h(0)
                for i in range(n - 1):
                    qc.cx(i, i + 1)
                for i in range(n):
                    qc.ry(float(rng.uniform(-1, 1)), i)
                out.append(qc)
        return out
    with open(QPY, "rb") as f:
        return list(qpy.load(f))


def reduced_probs(out, sim, noisy):
    """Distribution over the measured physical qubits, in clbit order, before readout."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    meas = sorted(((out.find_bit(i.clbits[0]).index, out.find_bit(i.qubits[0]).index)
                   for i in out.data if i.operation.name == "measure"))
    mq = [q for _, q in meas]
    act = sorted({out.find_bit(b).index for i in out.data for b in i.qubits} | set(mq))
    idx = {p: k for k, p in enumerate(act)}
    red = QuantumCircuit(len(act))
    for i in out.data:
        if i.operation.name in ("measure", "barrier", "delay"):
            continue
        red.append(i.operation, [idx[out.find_bit(b).index] for b in i.qubits])
    red.save_probabilities(qubits=[idx[q] for q in mq])
    s = AerSimulator(method="density_matrix", noise_model=sim.reduced_model(act) if noisy else None)
    return np.asarray(s.run(red).result().data()["probabilities"]), mq, [c for c, _ in meas]


def apply_readout(p, mq, sim):
    k = len(mq)
    t = p.reshape([2] * k)   # axis 0 = last clbit (little-endian)
    for j, q in enumerate(mq):
        e01, e10 = sim.readout(q)
        A = np.array([[1 - e01, e10], [e01, 1 - e10]])    # A[measured, true]
        ax = k - 1 - j
        t = np.moveaxis(np.tensordot(A, t, axes=([1], [ax])), 0, ax)
    return t.reshape(-1)


def do_run(args):
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    from qiskit_ibm_runtime import fake_provider as fp
    import depth_eval as DE   # Noisy (restricted noise model), next to this file
    A9, A10 = load("psf_ai_compile_a9.py", "e_a9"), load("psf_ai_compile.py", "e_a10")
    assert A9.AI_COMPILE_VERSION == "2026-10-05.a9" and A10.AI_COMPILE_VERSION == "2026-10-05.a10" and pc.VERSION == "2026-10-05.c13"
    be = getattr(fp, args.device)(); t = be.target; cm = t.build_coupling_map()
    basis = [g for g in t.operation_names if g in ("cx", "cz", "rz", "sx", "x")]
    comp = {"A9": lambda qc: A9.compile_for_model_circuit(qc, cm, basis, target=t),
            "A10": lambda qc: A10.compile_for_model_circuit(qc, cm, basis, target=t),
            "L3TM": lambda qc: transpile(qc, target=t, optimization_level=3, seed_transpiler=0, approximation_degree=1.0)}
    sim = DE.Noisy(be)
    rows = []
    for k, qc0 in enumerate(circuits(args.dry)):
        qc = qc0.copy(); qc.measure_all()
        ideal = Statevector(qc0).probabilities()
        row = dict(k=k, n=qc0.num_qubits)
        for a in ARMS:
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                out = comp[a](qc)
            cs = time.perf_counter() - t0
            fin = list(out.layout.final_index_layout(filter_ancillas=True))
            p0, mq, cl = reduced_probs(out, sim, False)
            pn, _, _ = reduced_probs(out, sim, True)
            q = apply_readout(pn, mq, sim)
            row[a] = dict(tv_noiseless=float(0.5 * np.abs(p0 - ideal).sum()),
                          meas_ok=(cl == list(range(qc0.num_qubits)) and mq == fin[:qc0.num_qubits]),
                          infid=float(1 - np.sum(np.sqrt(np.clip(ideal, 0, None) * np.clip(q, 0, None))) ** 2),
                          meas_err=float(sum(t["measure"][(m,)].error for m in mq)),
                          n2q=sum(1 for g in out.data if len(g.qubits) == 2), compile_s=round(cs, 4))
        rows.append(row)
    meta = dict(script_sha=norm_sha(os.path.abspath(__file__)), a9_sha=norm_sha(os.path.join(HERE, "psf_ai_compile_a9.py")),
                a10_sha=norm_sha(os.path.join(HERE, "psf_ai_compile.py")), c13_sha=norm_sha(pc.__file__),
                device=args.device, dry=bool(args.dry))
    json.dump(dict(meta=meta, rows=rows), open(os.path.join(args.out, f"ai10_{args.device}.json"), "w"))
    print("done", args.device, len(rows))


def verdict(ok, bad):
    return "CONFIRMED" if ok else ("REFUTED" if bad else "AMBIGUOUS")


def do_score(args):
    R = {json.load(open(p))["meta"]["device"]: json.load(open(p)) for p in glob.glob(os.path.join(args.out, "ai10_*.json"))}
    dry = any(d["meta"]["dry"] for d in R.values())
    rows = [r for d in R.values() for r in d["rows"]]
    tv = max(r[a]["tv_noiseless"] for r in rows for a in ARMS)
    mok = all(r[a]["meas_ok"] for r in rows for a in ARMS)
    nexp = 6 if dry else 153
    p0 = len(R) == 3 and all(len(d["rows"]) == nexp for d in R.values()) and tv <= 1e-6 and mok
    L = [f"# MODEL-RO score{' (DRY RUN -- not a result)' if dry else ''}", "",
         f"P0: {'PASS' if p0 else 'FAIL'} -- devices {len(R)}/3, circuits per device {[len(d['rows']) for d in R.values()]} "
         f"(expected {nexp}); max noiseless total variation {tv:.2e} (<= 1e-6); clbit j measures logical j's final qubit: {mok}",
         "", "| device | arm | classical infidelity | summed measure error | 2q | median compile s |", "|---|---|---|---|---|---|"]
    S = {}
    for dev in DEVICES:
        rr = R[dev]["rows"]
        for a in ARMS:
            S[(dev, a)] = dict(inf=np.mean([r[a]["infid"] for r in rr]), me=np.mean([r[a]["meas_err"] for r in rr]),
                               n2q=np.mean([r[a]["n2q"] for r in rr]), cs=float(np.median([r[a]["compile_s"] for r in rr])))
            s = S[(dev, a)]
            L.append(f"| {dev} | {a} | {s['inf']:.5f} | {s['me']:.5f} | {s['n2q']:.2f} | {s['cs']:.3f} |")
    L.append("")
    H = ("FakeTorino", "FakeKingston")
    res = {}
    r1 = {d: S[(d, "A10")]["me"] / S[(d, "A9")]["me"] for d in DEVICES}
    res["M1"] = verdict(all(r1[d] <= 0.8 for d in H) and all(v <= 1 for v in r1.values()), any(v > 1 for v in r1.values()))
    L.append(f"- M1 (A10 puts the measured qubits on better readout: A10/A9 summed measure error <= 0.80 on both Heron "
             f"devices, <= 1 everywhere): **{res['M1']}** ({', '.join(f'{d} {v:.3f}' for d, v in r1.items())})")
    r2 = {d: S[(d, "A10")]["inf"] / S[(d, "A9")]["inf"] for d in DEVICES}
    res["M2"] = verdict(all(v <= 1.0 for v in r2.values()) and all(r2[d] <= 0.95 for d in H), any(v > 1.01 for v in r2.values()))
    L.append(f"- M2 (the sampled distribution is more faithful: A10/A9 classical infidelity <= 1.00 everywhere and <= 0.95 "
             f"on both Heron devices): **{res['M2']}** ({', '.join(f'{d} {v:.3f}' for d, v in r2.items())})")
    r3 = {d: S[(d, "A10")]["inf"] / S[(d, "L3TM")]["inf"] for d in DEVICES}
    res["M3"] = verdict(all(v <= 1.0 for v in r3.values()), any(v > 1.05 for v in r3.values()))
    L.append(f"- M3 (A10 at or ahead of L3TM on every device): **{res['M3']}** ({', '.join(f'{d} {v:.3f}' for d, v in r3.items())})")
    r4 = {d: float(np.mean([r["A10"]["infid"] <= r["A9"]["infid"] + 1e-12 for r in R[d]["rows"]])) for d in H}
    res["M4"] = verdict(all(v >= 0.8 for v in r4.values()), any(v < 0.6 for v in r4.values()))
    L.append(f"- M4 (per circuit A10 <= A9 in >= 80% of circuits on both Heron devices): **{res['M4']}** "
             f"({', '.join(f'{d} {v:.3f}' for d, v in r4.items())})")
    r5 = max(S[(d, "A10")]["cs"] / S[(d, "A9")]["cs"] for d in DEVICES)
    res["M5"] = verdict(r5 <= 1.2, r5 > 2.0)
    L.append(f"- M5 (median compile time A10 <= 1.2 x A9): **{res['M5']}** (max {r5:.3f})")
    L += ["", "SUMMARY " + json.dumps(res)]
    txt = "\n".join(L); open(os.path.join(args.out, "score.md"), "w").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "score"]); ap.add_argument("--device"); ap.add_argument("--out", required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    {"run": do_run, "score": do_score}[a.mode](a)
