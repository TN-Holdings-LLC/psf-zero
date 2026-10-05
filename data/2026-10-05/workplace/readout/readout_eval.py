"""readout_eval.py -- workplace test READOUT (2026-10-05): does compiling a quantum classifier WITH its final
measurement, and candidate psf_compile 2026-10-05.c13 (item 40: readout of measured qubits in hybrid_cost), put the
classifier's output on a qubit with lower readout error, and does that buy margin and few-shot accuracy?

Classifier, data and training: depth_eval.py (DEPTH stage 1), imported unchanged, split seed 3 / init seed 3
(new; DEPTH used 0 pilot, 1 scored, 2 dry). n in {4, 6}; L in {4, 12}; test points capped at 40 per dataset.
Arms (all with the device Target and the recommended call: placement_refine, final_resynthesis="select",
compare_level3, compare_floor, candidate_score="hybrid"):
  C12   candidate c12, circuit WITHOUT measurement (as in DEPTH)
  C13   candidate c13, circuit WITHOUT measurement (identity check: must equal C12; compared, not simulated)
  C12M  candidate c12, circuit WITH measure(0 -> c0)
  C13M  candidate c13, circuit WITH measure(0 -> c0)
  L3TM  Qiskit level 3 with the Target, approximation_degree 1.0, circuit WITH measurement
Devices: FakeTorino, FakeKingston (Heron, uneven readout), FakeAuckland (cx, control).
Per circuit: exactness as a state infidelity on the touched qubits (Addendum-340 probe's check, measurements
removed; <= 1e-6); the physical qubit that carries logical 0 at the end (for M arms it must be the measured one);
its Target measure error and Aer's assignment probabilities (e01, e10); noisy z (Aer density matrix, noise model
restricted to the touched qubits, as DEPTH); effective margin y * (z (1 - e01 - e10) - (e01 - e10)) -- the
measured expectation; shot accuracy at 32 and 4,000 shots (200 / 20 repetitions, same random numbers in every arm).

  python readout_eval.py train  --out DIR [--dry]
  python readout_eval.py run    --device D --out DIR --c12 P --c13 P [--dry]
  python readout_eval.py score  --out DIR
"""
import argparse, contextlib, glob, hashlib, importlib.util, io, json, os, sys, time, warnings, zlib
import numpy as np

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import depth_eval as DE  # noqa: E402  (DEPTH stage 1, unchanged)

DATASETS = ("BC", "D38"); NS = (4, 6); LS = (4, 12)
DEVICES = ("FakeTorino", "FakeKingston", "FakeAuckland")
ARMS = ("C12", "C13", "C12M", "C13M", "L3TM")
NPTS = 40
SPLIT = 3          # scored; the dry run uses 2 (DEPTH's dry split)


def split(args):
    return 2 if args.dry else SPLIT


def norm_sha(path):
    return DE.norm_sha(path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def measured(qc):
    from qiskit import QuantumCircuit
    m = QuantumCircuit(qc.num_qubits, 1)
    m.compose(qc, inplace=True)
    m.measure(0, 0)
    return m


def strip_measure(out):
    c = out.copy_empty_like()
    for ins in out.data:
        if ins.operation.name != "measure":
            c.append(ins.operation, ins.qubits, ins.clbits)
    c._layout = out._layout
    return c


def state_infid(qc, out):
    """The workplace probe's check (Addendum 340), measurements removed from `out`."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity
    ideal = Statevector(qc)
    fin = list(out.layout.final_index_layout(filter_ancillas=True))
    active = sorted({out.find_bit(b).index for ins in out.data for b in ins.qubits} | set(fin))
    idx = {p: i for i, p in enumerate(active)}
    red = QuantumCircuit(len(active))
    for ins in out.data:
        if ins.operation.name in ("barrier", "measure", "delay"):
            continue
        red.append(ins.operation, [idx[out.find_bit(b).index] for b in ins.qubits])
    sv = Statevector(red)
    keep = [idx[p] for p in fin]
    trace_out = [i for i in range(len(active)) if i not in keep]
    rho = DensityMatrix(partial_trace(sv, trace_out) if trace_out else sv)
    order = sorted(keep); perm = [order.index(k) for k in keep]; n = len(keep)
    t = rho.data.reshape([2] * (2 * n))
    t = np.transpose(t, [n - 1 - perm[v] for v in range(n)][::-1] + [2 * n - 1 - perm[v] for v in range(n)][::-1])
    return float(1 - state_fidelity(DensityMatrix(t.reshape(2 ** n, 2 ** n)), ideal))


def sig(c):
    return [(i.operation.name, [c.find_bit(q).index for q in i.qubits], [round(float(p), 12) for p in i.operation.params])
            for i in c.data]


def do_train(args):
    for ds in DATASETS:
        for n in NS:
            Xtr, ytr, Xte, yte = DE.data(ds, n, split(args))
            for L in LS:
                th = DE.train(Xtr, ytr, n, L, seed=split(args) + 1000 * L, steps=40 if args.dry else 300)
                np.save(os.path.join(args.out, f"theta_{ds}_n{n}_L{L}.npy"), th)
                z = DE.forward(th, Xte, n, L)
                print("TRAIN", ds, n, L, "ideal acc %.3f margin %.3f" % (np.mean(np.sign(z) == yte), np.mean(yte * z)),
                      flush=True)


def shots(zm_true_p0, key, nshots, reps):
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    k = rng.binomial(nshots, zm_true_p0, size=reps)
    return 2.0 * k / nshots - 1.0


def do_run(args):
    from qiskit import transpile
    from qiskit_ibm_runtime import fake_provider as fp
    be = getattr(fp, args.device)(); t = be.target
    P12, P13 = load(args.c12, "psf_c12"), load(args.c13, "psf_c13")
    assert P12.VERSION == "2026-10-05.c12" and P13.VERSION == "2026-10-05.c13", (P12.VERSION, P13.VERSION)
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x", "id")]
    full = dict(coupling_map=t.build_coupling_map(), basis_gates=basis, entangling_basis="cx", layout_search=True,
                seed_transpiler=0, target=t, placement_refine=True, final_resynthesis="select", compare_level3=True,
                compare_floor=True, candidate_score="hybrid")
    comp = {"C12": lambda qc: P12.compile_for_hardware(qc, **full),
            "C13": lambda qc: P13.compile_for_hardware(qc, **full),
            "C12M": lambda qc: P12.compile_for_hardware(measured(qc), **full),
            "C13M": lambda qc: P13.compile_for_hardware(measured(qc), **full),
            "L3TM": lambda qc: transpile(measured(qc), target=t, optimization_level=3, seed_transpiler=0,
                                         approximation_degree=1.0)}
    sim = DE.Noisy(be)
    rows = []
    for ds in DATASETS:
        for n in NS:
            Xtr, ytr, Xte, yte = DE.data(ds, n, split(args))
            npts = 6 if args.dry else NPTS
            Xte, yte = Xte[:npts], yte[:npts]
            for L in LS:
                th = np.load(os.path.join(args.out, f"theta_{ds}_n{n}_L{L}.npy"))
                zid = DE.forward(th, Xte, n, L)
                for i, (x, yv) in enumerate(zip(Xte, yte)):
                    qc = DE.circuit(x, th, n, L)
                    outs = {}
                    for a in ARMS:
                        t0 = time.perf_counter()
                        with contextlib.redirect_stdout(io.StringIO()):
                            outs[a] = comp[a](qc)
                        outs[a + "_s"] = time.perf_counter() - t0
                    row = dict(dataset=ds, n=n, L=L, i=i, y=float(yv), z_ideal=float(zid[i]),
                               c13_same_as_c12=sig(outs["C13"]) == sig(outs["C12"]))
                    key = f"{args.device}|{ds}|{n}|{L}|{i}"
                    for a in ("C12", "C12M", "C13M", "L3TM"):
                        out = outs[a]
                        fin0 = out.layout.final_index_layout(filter_ancillas=True)[0]
                        mq = [out.find_bit(g.qubits[0]).index for g in out.data if g.operation.name == "measure"]
                        so = strip_measure(out)
                        zn, p, touched = sim.z(so)
                        e01, e10 = sim.readout(fin0)
                        zm = zn * (1 - e01 - e10) - (e01 - e10)
                        p0m = (1 + zm) / 2
                        s32 = shots(p0m, key + "|32", 32, 200); s4k = shots(p0m, key + "|4000", 4000, 20)
                        row[a] = dict(infid=state_infid(qc, so), fin0=int(fin0), measured=mq,
                                      meas_err=float(t["measure"][(fin0,)].error or 0.0), e01=e01, e10=e10, z=zn,
                                      eff_margin=float(yv * zm), acc32=float(np.mean(np.sign(s32) == yv)),
                                      acc4k=float(np.mean(np.sign(s4k) == yv)),
                                      n2q=sum(1 for g in out.data if len(g.qubits) == 2), touched=touched,
                                      compile_s=round(outs[a + "_s"], 4))
                    row["C13_compile_s"] = round(outs["C13_s"], 4)
                    rows.append(row)
                print("RUN", ds, n, L, "done", flush=True)
    meta = dict(script_sha=norm_sha(os.path.abspath(__file__)), depth_eval_sha=norm_sha(DE.__file__),
                c12_sha=norm_sha(args.c12), c13_sha=norm_sha(args.c13), device=args.device, dry=bool(args.dry))
    json.dump(dict(meta=meta, rows=rows), open(os.path.join(args.out, f"readout_{args.device}.json"), "w"))


def verdict(ok, bad):
    return "CONFIRMED" if ok else ("REFUTED" if bad else "AMBIGUOUS")


def do_score(args):
    R = {}
    for p in glob.glob(os.path.join(args.out, "readout_*.json")):
        d = json.load(open(p)); R[d["meta"]["device"]] = d
    dry = any(d["meta"]["dry"] for d in R.values())
    L = [f"# READOUT score{' (DRY RUN -- not a result)' if dry else ''}", ""]
    M = ("C12", "C12M", "C13M", "L3TM")
    allrows = [r for d in R.values() for r in d["rows"]]
    infid = max(r[a]["infid"] for r in allrows for a in M)
    meas_ok = all(r[a]["measured"] == [r[a]["fin0"]] for r in allrows for a in ("C12M", "C13M", "L3TM"))
    p0 = len(R) == len(DEVICES) and infid <= 1e-6 and meas_ok
    L.append(f"P0: {'PASS' if p0 else 'FAIL'} -- devices {len(R)}/{len(DEVICES)}; max state infidelity {infid:.2e} "
             f"(<= 1e-6); measured qubit = final position of logical 0 in every measured arm: {meas_ok}")
    L += ["", "| device | arm | measure error of the output qubit | eff. margin | acc 32 shots | acc 4000 shots | 2q | median compile s |",
          "|---|---|---|---|---|---|---|---|"]
    S = {}
    for dev in DEVICES:
        rows = R[dev]["rows"]
        for a in M:
            S[(dev, a)] = dict(err=np.mean([r[a]["meas_err"] for r in rows]), em=np.mean([r[a]["eff_margin"] for r in rows]),
                               a32=np.mean([r[a]["acc32"] for r in rows]), a4k=np.mean([r[a]["acc4k"] for r in rows]),
                               n2q=np.mean([r[a]["n2q"] for r in rows]), cs=float(np.median([r[a]["compile_s"] for r in rows])))
            s = S[(dev, a)]
            L.append(f"| {dev} | {a} | {s['err']:.4f} | {s['em']:.4f} | {s['a32']:.4f} | {s['a4k']:.4f} | {s['n2q']:.1f} | {s['cs']:.3f} |")
    L.append("")
    res = {}
    T, K = "FakeTorino", "FakeKingston"
    r1 = S[(T, "C12M")]["err"] / S[(T, "C12")]["err"]
    res["R1"] = verdict(r1 <= 0.70, r1 >= 1.0)
    L.append(f"- R1 (compiling with the measurement moves the output to a better-readout qubit, FakeTorino: C12M/C12 "
             f"measure error <= 0.70): **{res['R1']}** ({r1:.3f})")
    d2 = {dev: S[(dev, "C13M")]["err"] - S[(dev, "C12M")]["err"] for dev in DEVICES}
    res["R2"] = verdict(all(v <= 1e-12 for v in d2.values()) and any(v < 0 for v in d2.values()),
                        any(v > 0.001 for v in d2.values()))
    L.append(f"- R2 (c13's readout term never worse, somewhere better: C13M - C12M measure error <= 0 on every device, "
             f"< 0 on one): **{res['R2']}** ({', '.join(f'{k} {v:+.5f}' for k, v in d2.items())})")
    d3 = S[(T, "C13M")]["em"] - S[(T, "C12")]["em"]
    res["R3"] = verdict(d3 >= 0.01, d3 < 0)
    L.append(f"- R3 (effective margin C13M - C12 >= +0.01 on FakeTorino): **{res['R3']}** ({d3:+.4f})")
    d4 = {dev: S[(dev, "C13M")]["em"] - S[(dev, "L3TM")]["em"] for dev in DEVICES}
    res["R4"] = verdict(all(v >= -0.01 for v in d4.values()), any(v < -0.02 for v in d4.values()))
    L.append(f"- R4 (C13M level with or above L3TM in effective margin, every device, >= -0.01): **{res['R4']}** "
             f"({', '.join(f'{k} {v:+.4f}' for k, v in d4.items())})")
    same = float(np.mean([r["c13_same_as_c12"] for r in allrows]))
    res["R5"] = verdict(same == 1.0, same < 0.999)
    L.append(f"- R5 (without measurements c13 equals c12, instruction by instruction): **{res['R5']}** ({same:.4f})")
    r6 = max(S[(dev, "C13M")]["cs"] / S[(dev, "C12M")]["cs"] for dev in DEVICES)
    res["R6"] = verdict(r6 <= 1.2, r6 > 2.0)
    L.append(f"- R6 (median compile time C13M <= 1.2 x C12M on every device): **{res['R6']}** (max ratio {r6:.3f})")
    d7 = S[(T, "C13M")]["a32"] - S[(T, "C12")]["a32"]
    res["R7"] = verdict(d7 >= 0.0, d7 < -0.01)
    L.append(f"- R7 (32-shot accuracy C13M >= C12 on FakeTorino): **{res['R7']}** ({d7:+.4f})")
    L += ["", "SUMMARY " + json.dumps(res)]
    txt = "\n".join(L); open(os.path.join(args.out, "score.md"), "w").write(txt + "\n"); print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "run", "score"])
    ap.add_argument("--device"); ap.add_argument("--c12"); ap.add_argument("--c13")
    ap.add_argument("--out", required=True); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    {"train": do_train, "run": do_run, "score": do_score}[a.mode](a)


if __name__ == "__main__":
    main()
