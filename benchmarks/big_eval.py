"""big_eval.py -- test BIG (2026-10-06): does candidate psf_ai_compile 2026-10-06.a13 (changelog item 18: the release's
whole recommended call, with compare_floor and candidate_score "hybrid", for circuits above SMALL_MAX_QUBITS) give
sampled 9-10-qubit circuits a lower classical infidelity than the adopted front end a12?

Circuits: HOLD6's wide families W1-W6 (hold6_eval.family_w, unchanged, with its module constant W_BASE set to
130,000,000: seeds between 131 and 137 million, none used before), HOLD6's sizes, each with measure_all(). Smoke:
hold6_eval's smoke sizes. Devices: HOLD6's nine.
Arms (all with measurements):
  A12   the adopted front end (benchmarks/psf_ai_compile.py)
  A13   the candidate
  L3TM  Qiskit level 3 with the Target, approximation_degree 1.0
Per arm: compile time; exactness (the workplace probe's state infidelity, measurements removed, <= 1e-6); whether
clbit j measures logical j's final qubit; the summed Target measure error of the measured qubits; and, if it touches
at most 11 qubits, MODEL-RO2's classical infidelity 1 - (sum_x sqrt(p_x q_x))^2 of the sampled distribution (Aer
density matrix on the touched qubits with the device's noise model restricted to them, then Aer's readout). A
circuit is in the means only if all three arms could be simulated. The helpers are kro_eval.py's (Addendum 363),
copied from the workplace harnesses, unchanged in what they compute.

    python benchmarks/big_eval.py run --device D --family W --out DIR [--smoke]
    python benchmarks/big_eval.py score --out DIR
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
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for _p in (HERE, REPO):
    sys.path.insert(0, _p)

DEVICES = ("FakeAuckland", "FakeTorino", "FakeKingston", "FakeHanoiV2", "FakeAlgiers", "FakeGeneva", "FakeFez",
           "FakeMarrakesh", "FakeAachen")
FAMILIES = ("W1", "W2", "W3", "W4", "W5", "W6")
A13_PATH = os.path.join(REPO, "patches", "psf_ai_compile_a13_2026-10-06", "psf_ai_compile.py")
W_BASE = 130_000_000
MAX_ACTIVE = 11
ARMS = ("A12", "A13", "L3TM")


def norm_sha(path):
    t = open(path, "rb").read().decode("utf-8").replace("\r\n", "\n")
    lines = [ln.rstrip() for ln in t.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def load():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    rel = H.load_module(os.path.join(REPO, "psf_compile.py"), "psf_compile")  # before the front ends import it
    a12 = H.load_module(os.path.join(REPO, "benchmarks", "psf_ai_compile.py"), "psf_ai_compile_a12_big")
    a13 = H.load_module(A13_PATH, "psf_ai_compile_a13_big")
    hold6 = H.load_module(os.path.join(REPO, "benchmarks", "hold6_eval.py"), "hold6_eval_big")
    if rel.VERSION != "2026-10-06.1" or a12.AI_COMPILE_VERSION != "2026-10-06.a12" or \
            a13.AI_COMPILE_VERSION != "2026-10-06.a13":
        raise SystemExit(f"STOP: versions {rel.VERSION}, {a12.AI_COMPILE_VERSION}, {a13.AI_COMPILE_VERSION}")
    hold6.W_BASE = W_BASE  # new seeds for HOLD6's wide generator
    return rel, a12, a13, hold6


def git_head():
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


class Noisy:
    """kro_eval.Noisy (depth_eval.Noisy), copied."""

    def __init__(self, be):
        from qiskit_aer.noise import NoiseModel
        self.nm = NoiseModel.from_backend(be)
        self.cache = {}

    def reduced_model(self, active):
        key = tuple(active)
        if key in self.cache:
            return self.cache[key]
        from qiskit_aer.noise import NoiseModel
        idx = {p: i for i, p in enumerate(active)}
        m = NoiseModel(basis_gates=self.nm.basis_gates)
        for gate, d in self.nm._local_quantum_errors.items():
            for qs, err in d.items():
                if all(q in idx for q in qs):
                    m.add_quantum_error(err, gate, [idx[q] for q in qs])
        if len(self.cache) > 2000:
            self.cache.clear()
        self.cache[key] = m
        return m

    def readout(self, p):
        e = self.nm._local_readout_errors.get((p,))
        if e is None:
            return 0.0, 0.0
        pr = np.asarray(e.probabilities)
        return float(pr[0][1]), float(pr[1][0])


def reduced_probs(out, sim):
    """kro_eval.reduced_probs (ai10_eval2.reduced_probs, noisy), copied."""
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
    s = AerSimulator(method="density_matrix", noise_model=sim.reduced_model(act), max_parallel_threads=1)
    return np.asarray(s.run(red).result().data()["probabilities"]), mq, [c for c, _ in meas]


def apply_readout(p, mq, sim):
    """kro_eval.apply_readout (ai10_eval2.apply_readout), copied."""
    k = len(mq)
    t = p.reshape([2] * k)
    for j, q in enumerate(mq):
        e01, e10 = sim.readout(q)
        A = np.array([[1 - e01, e10], [e01, 1 - e10]])
        ax = k - 1 - j
        t = np.moveaxis(np.tensordot(A, t, axes=([1], [ax])), 0, ax)
    return t.reshape(-1)


def state_infid(qc, out):
    """kro_eval.state_infid (readout_eval.state_infid), copied; `qc` without measurements."""
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
    order = sorted(keep)
    perm = [order.index(k) for k in keep]
    n = len(keep)
    t = rho.data.reshape([2] * (2 * n))
    t = np.transpose(t, [n - 1 - perm[v] for v in range(n)][::-1] + [2 * n - 1 - perm[v] for v in range(n)][::-1])
    return float(1 - state_fidelity(DensityMatrix(t.reshape(2 ** n, 2 ** n)), ideal))


def run(args):
    import qiskit
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    from qiskit_ibm_runtime import fake_provider
    rel, a12, a13, hold6 = load()
    be = getattr(fake_provider, args.device)()
    t = be.target
    cm = t.build_coupling_map()
    basis = [g for g in t.operation_names if g in ("cx", "cz", "ecr", "rz", "sx", "x")]
    comp = {"A12": lambda qc: a12.compile_for_model_circuit(qc, cm, basis, target=t),
            "A13": lambda qc: a13.compile_for_model_circuit(qc, cm, basis, target=t),
            "L3TM": lambda qc: transpile(qc, target=t, optimization_level=3, seed_transpiler=0, approximation_degree=1.0)}
    sim = Noisy(be)
    rows, t0 = [], time.time()
    for k, (params, qc0) in enumerate(hold6.family_w(args.family, args.smoke)):
        n = qc0.num_qubits
        qc = qc0.copy()
        qc.measure_all()
        ideal = Statevector(qc0).probabilities()
        row = dict(params=params, n=n)
        order = list(ARMS) if k % 2 == 0 else list(reversed(ARMS))
        for arm in order:
            t1 = time.perf_counter()
            try:
                with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = comp[arm](qc)
            except Exception as e:  # recorded; P0 requires none
                row[arm] = dict(error=f"{type(e).__name__}: {e}"[:300])
                continue
            cs = time.perf_counter() - t1
            meas = sorted(((out.find_bit(i.clbits[0]).index, out.find_bit(i.qubits[0]).index)
                           for i in out.data if i.operation.name == "measure"))
            fin = list(out.layout.final_index_layout(filter_ancillas=True))
            active = {out.find_bit(b).index for i in out.data for b in i.qubits if i.operation.name != "barrier"}
            r = dict(compile_s=round(cs, 4), state_infid=state_infid(qc0, out),
                     meas_ok=bool([c for c, _ in meas] == list(range(n)) and [q for _, q in meas] == fin[:n]),
                     meas_err=float(sum(t["measure"][(q,)].error or 0.0 for _, q in meas)), active=len(active),
                     n2q=sum(1 for g in out.data if len(g.qubits) == 2 and g.operation.name != "barrier"))
            if len(active) <= MAX_ACTIVE:
                p, mq, _ = reduced_probs(out, sim)
                q = apply_readout(p, mq, sim)
                r["infid"] = float(1 - np.sum(np.sqrt(np.clip(ideal, 0, None) * np.clip(q, 0, None))) ** 2)
            row[arm] = r
        rows.append(row)
    meta = dict(device=args.device, family=args.family, smoke=bool(args.smoke), git_head=git_head(),
                qiskit=qiskit.__version__,
                versions=dict(release=rel.VERSION, a12=a12.AI_COMPILE_VERSION, a13=a13.AI_COMPILE_VERSION),
                sha=dict(script=norm_sha(os.path.abspath(__file__)), release=norm_sha(rel.__file__),
                         a12=norm_sha(a12.__file__), a13=norm_sha(A13_PATH), hold6=norm_sha(hold6.__file__)),
                wall_s=round(time.time() - t0, 1))
    os.makedirs(args.out, exist_ok=True)
    name = f"big_{args.device}_{args.family}{'_smoke' if args.smoke else ''}.json"
    json.dump(dict(meta=meta, rows=rows), open(os.path.join(args.out, name), "w"))
    print(f"wrote {name}: {len(rows)} circuits, {meta['wall_s']} s", flush=True)


def verdict(ok, bad):
    return "REFUTED" if bad else ("CONFIRMED" if ok else "AMBIGUOUS")


def score(args):
    R = {}
    for p in sorted(glob.glob(os.path.join(args.out, "big_*.json"))):
        r = json.load(open(p))
        R[(r["meta"]["device"], r["meta"]["family"])] = r["rows"]
    smoke = any("_smoke" in p for p in glob.glob(os.path.join(args.out, "big_*.json")))
    rows = {d: [x for f in FAMILIES for x in R.get((d, f), [])] for d in DEVICES}
    allr = [x for d in DEVICES for x in rows[d]]
    errors = sum(1 for x in allr for a in ARMS if "error" in x.get(a, {"error": 1}))
    ok = [x for x in allr if all("error" not in x.get(a, {"error": 1}) for a in ARMS)]
    inexact = sum(1 for x in ok for a in ARMS if x[a]["state_infid"] > 1e-6)
    badm = sum(1 for x in ok for a in ARMS if not x[a]["meas_ok"])
    p0 = len(R) == len(DEVICES) * len(FAMILIES) and not errors and not inexact and not badm
    lines = [f"# BIG score{' (SMOKE)' if smoke else ''}", "",
             f"P0: {'PASS' if p0 else 'FAIL'} -- files {len(R)} of {len(DEVICES) * len(FAMILIES)}; errors {errors}; "
             f"inexact outputs {inexact}; wrong measurement mapping {badm}"]
    if not p0:
        lines.append("Nothing below is scored.")
        out = "\n".join(lines) + "\n"
        print(out)
        open(os.path.join(args.out, "score.md"), "w").write(out)
        return
    S = {}
    lines += ["", "| device | sim / all | A12 | A13 | L3TM | A13/A12 | A13/L3TM | meas err A12 / A13 | med time A12 / A13 s |",
              "|---|---|---|---|---|---|---|---|---|"]
    for d in DEVICES:
        rs = [x for x in rows[d] if all("infid" in x[a] for a in ARMS)]
        m = {a: float(np.mean([x[a]["infid"] for x in rs])) if rs else float("nan") for a in ARMS}
        e = {a: float(np.mean([x[a]["meas_err"] for x in rows[d]])) for a in ARMS}
        tm = {a: float(np.median([x[a]["compile_s"] for x in rows[d]])) for a in ARMS}
        S[d] = dict(m=m, e=e, tm=tm, n=len(rs))
        lines.append(f"| {d} | {len(rs)}/{len(rows[d])} | {m['A12']:.5f} | {m['A13']:.5f} | {m['L3TM']:.5f} | "
                     f"{m['A13'] / m['A12']:.4f} | {m['A13'] / m['L3TM']:.4f} | {e['A12']:.4f} / {e['A13']:.4f} | "
                     f"{tm['A12']:.3f} / {tm['A13']:.3f} |")
    b1 = {d: S[d]["m"]["A13"] / S[d]["m"]["A12"] for d in DEVICES}
    b2 = {d: S[d]["e"]["A13"] - S[d]["e"]["A12"] for d in DEVICES}
    b3 = {d: S[d]["m"]["A13"] / S[d]["m"]["L3TM"] for d in DEVICES}
    b4 = {d: S[d]["tm"]["A13"] / S[d]["tm"]["A12"] for d in DEVICES}
    V = [("B1", "a13 samples better than a12 on wide circuits (mean classical infidelity A13/A12 <= 1.000 on >= 8 of 9 "
          "devices; refuted > 1.005 on any)", verdict(sum(v <= 1.0 for v in b1.values()) >= 8,
                                                       any(v > 1.005 for v in b1.values())), b1),
         ("B2", "a13 measures on better readout (mean summed measure error A13 - A12 <= 0 on >= 8 of 9 devices; refuted "
          "> 0 on >= 3)", verdict(sum(v <= 0 for v in b2.values()) >= 8, sum(v > 0 for v in b2.values()) >= 3), b2),
         ("B3", "a13 is level with or ahead of Qiskit level 3 (A13/L3TM <= 1.00 on >= 7 of 9 devices; refuted > 1.02 on "
          "any)", verdict(sum(v <= 1.0 for v in b3.values()) >= 7, any(v > 1.02 for v in b3.values())), b3),
         ("B4", "the cost is moderate (median compile time A13/A12 <= 2.0 on every device; refuted > 3.0 on any)",
          verdict(all(v <= 2.0 for v in b4.values()), any(v > 3.0 for v in b4.values())), b4)]
    lines += ["", "## Predictions", ""]
    for qid, text, v, num in V:
        lines.append(f"- {qid} ({text}): **{v}** -- {json.dumps(num, default=lambda x: round(x, 5))}")
    lines += ["", "Reported: circuits too wide to simulate for at least one arm are excluded from the means: " +
              json.dumps({d: len(rows[d]) - S[d]["n"] for d in DEVICES})]
    out = "\n".join(lines) + "\n"
    print(out)
    open(os.path.join(args.out, "score.md"), "w").write(out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--device", required=True, choices=DEVICES)
    r.add_argument("--family", required=True, choices=FAMILIES)
    r.add_argument("--out", required=True)
    r.add_argument("--smoke", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else score(a)


if __name__ == "__main__":
    main()
