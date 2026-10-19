"""Check psfz r5 templates (prototype, 2026-10-19): fixed shape, exact binding, cost against r4.

    python rust/psfz/harness/template_check.py OUT_DIR        (from the repository root; psfz built with cargo)

For each device (FakeFez, FakeTorino: CZ) and each input family:
- S1 (fixed shape): templates built at two random value sets have the same gate list (names and qubits).
- Exactness: the template built at values A, bound to values C by `psfz bind`, gives the same distribution of the
  measured bits as the input circuit at values C (Qiskit state vector on the active qubits; largest difference of
  any outcome's probability). The final rz of a run before a measurement is dropped, so only the distribution is
  compared, not the state.
- Same angles: the bound circuit equals the template built directly at C (largest angle difference, mod 2 pi).
- Cost: the bound circuit's sx and cz counts against r4's `compile` at the same values C.
- Bind time: the `bind_s` psfz reports (inside the process, without start-up).
Writes OUT_DIR/template_check.json and prints a table.
"""
import contextlib
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.getcwd()
PSFZ = os.environ.get("PSFZ_BIN", os.path.join(REPO, "rust", "psfz", "target", "release", "psfz"))


def load_pc():
    sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
    spec = importlib.util.spec_from_file_location("psf_compile", os.path.join(REPO, "psf_compile.py"))
    pc = importlib.util.module_from_spec(spec)
    sys.modules["psf_compile"] = pc
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(pc)
    return pc


def ring_classifier(n, L, rng):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    for _ in range(L):
        for q in range(n):
            qc.ry(rng.uniform(-np.pi, np.pi), q)
        for q in range(n):
            qc.ry(rng.uniform(-np.pi, np.pi), q)
            qc.rz(rng.uniform(-np.pi, np.pi), q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    for q in range(n):
        qc.ry(rng.uniform(-np.pi, np.pi), q)
    qc.measure_all()
    return qc


def tfim_chain(n, steps, rng, fixed):
    """Trotter steps of a TFIM chain: rzz with fixed angles (not slots), rx and u slots."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.h(q)
    for s in range(steps):
        for q in range(0, n - 1, 2):
            qc.rzz(fixed[s], q, q + 1)
        for q in range(1, n - 1, 2):
            qc.rzz(fixed[s], q, q + 1)
        for q in range(n):
            qc.rx(rng.uniform(-np.pi, np.pi), q)
    for q in range(n):
        qc.u(*rng.uniform(-np.pi, np.pi, 3), q)
    qc.measure_all()
    return qc


def slot_values(qc):
    """One list of parameters per single-qubit gate with parameters, in input order (as psfz numbers slots)."""
    return [[float(p) for p in inst.operation.params] for inst in qc.data
            if inst.operation.num_qubits == 1 and inst.operation.params and inst.operation.name != "measure"]


def run(args, stdin=None):
    r = subprocess.run([PSFZ] + args, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def probs_of_input(qc):
    from qiskit.quantum_info import Statevector
    q = qc.remove_final_measurements(inplace=False)
    p = Statevector(q).probabilities()
    return p  # index bit k = qubit k = clbit k (measure_all)


def probs_of_output(path, nc):
    """Distribution of the clbits of psfz's output (OpenQASM 3 on device qubits), on its active qubits only."""
    from qiskit import QuantumCircuit, qasm3
    from qiskit.quantum_info import Statevector
    qc = qasm3.loads(open(path).read())
    act = sorted({qc.find_bit(q).index for inst in qc.data for q in inst.qubits})
    pos = {p: i for i, p in enumerate(act)}
    small = QuantumCircuit(len(act))
    meas = {}
    for inst in qc.data:
        qs = [pos[qc.find_bit(q).index] for q in inst.qubits]
        if inst.operation.name == "measure":
            meas[qs[0]] = qc.find_bit(inst.clbits[0]).index
        elif inst.operation.name != "barrier":
            small.append(inst.operation, qs)
    p = Statevector(small).probabilities()
    out = np.zeros(2 ** nc)
    for idx in np.nonzero(p > 1e-15)[0]:
        c = 0
        for qi, cb in meas.items():
            if (idx >> qi) & 1:
                c |= 1 << cb
        out[c] += p[idx]
    return out


def skeleton(t):
    return [(o[0], tuple(o[1])) for o in t["ops"]]


def counts(ops):
    return {g: sum(1 for o in ops if o[0] == g) for g in ("sx", "x", "cz", "rz")}


def qasm_ops(path):
    from qiskit import qasm3
    qc = qasm3.loads(open(path).read())
    return [(i.operation.name,) for i in qc.data]


def main(out_dir):
    from qiskit import qasm3
    from qiskit_ibm_runtime import fake_provider as fp
    pc = load_pc()
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(20261019)
    cases = [("ring_classifier", dict(n=12, L=L)) for L in (1, 2, 8, 16)]
    cases += [("tfim_chain", dict(n=n, steps=s)) for n, s in ((6, 3), (10, 4), (16, 3))]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for dev in ("FakeFez", "FakeTorino"):
            dfile = os.path.join(tmp, dev + ".json")
            open(dfile, "w").write(pc._psfz105_device(getattr(fp, dev)().target))
            for fam, kw in cases:
                fixed = rng.uniform(-2, 2, 8)
                make = (lambda r: ring_classifier(kw["n"], kw["L"], r)) if fam == "ring_classifier" else \
                    (lambda r: tfim_chain(kw["n"], kw["steps"], r, fixed))
                seeds = rng.integers(0, 2 ** 31, 3)
                qa, qb, qcc = (make(np.random.default_rng(int(s))) for s in seeds)
                paths = {}
                for tag, qc in (("A", qa), ("B", qb), ("C", qcc)):
                    f = os.path.join(tmp, f"{dev}-{fam}-{tag}.qasm")
                    open(f, "w").write(qasm3.dumps(qc))
                    paths[tag] = f
                tj = {}
                for tag in ("A", "B", "C"):
                    rc, _, err = run(["template", dfile, paths[tag], os.path.join(tmp, f"t{tag}.json")])
                    if rc != 0:
                        raise SystemExit(f"{dev} {fam}: template refused: {err}")
                    tj[tag] = json.load(open(os.path.join(tmp, f"t{tag}.json")))
                s1 = skeleton(tj["A"]) == skeleton(tj["B"]) == skeleton(tj["C"])
                vf = os.path.join(tmp, "values.json")
                json.dump(slot_values(qcc), open(vf, "w"))
                bound = os.path.join(tmp, "bound.qasm")
                rc, _, err = run(["bind", os.path.join(tmp, "tA.json"), vf, bound])
                if rc != 0:
                    raise SystemExit(f"{dev} {fam}: bind failed: {err}")
                bind_s = float([l for l in err.split("\n") if l.startswith("bind_s")][0].split()[1])
                nc = qcc.num_clbits
                dp = float(np.max(np.abs(probs_of_output(bound, nc) - probs_of_input(qcc))))
                # same angles as the template built directly at C
                from qiskit import qasm3 as q3
                bq = q3.loads(open(bound).read())
                angs_b = [float(i.operation.params[0]) for i in bq.data if i.operation.name == "rz"]
                angs_c = [o[2][0] for o in tj["C"]["ops"] if o[0] == "rz"]
                dang = max((abs(math.remainder(x - y, 2 * math.pi)) for x, y in zip(angs_b, angs_c)), default=0.0)
                r4 = os.path.join(tmp, "r4.qasm")
                rc, _, _ = run(["compile", dfile, paths["C"], r4])
                r4c = {g: sum(1 for i in q3.loads(open(r4).read()).data if i.operation.name == g) for g in ("sx", "x", "cz", "rz")} if rc == 0 else None
                tc = {g: sum(1 for o in tj["A"]["ops"] if o[0] == g) for g in ("sx", "x", "cz", "rz")}
                row = dict(device=dev, family=fam, **kw, slots=len(tj["A"]["slots"]), runs=len(tj["A"]["runs"]),
                           ops=len(tj["A"]["ops"]), s1_fixed_shape=s1, max_prob_diff=dp, max_angle_diff=dang,
                           template_counts=tc, r4_counts=r4c, bind_s=bind_s, build_s=tj["A"].get("build_s"))
                rows.append(row)
                print(f"{dev:10s} {fam:15s} {str(kw):22s} slots {row['slots']:4d} S1 {s1!s:5s} |dP| {dp:.1e} "
                      f"|dangle| {dang:.1e} sx/cz template {tc['sx']}/{tc['cz']} r4 "
                      f"{(r4c or {}).get('sx')}/{(r4c or {}).get('cz')} bind {bind_s * 1e3:.3f} ms", flush=True)
    json.dump(rows, open(os.path.join(out_dir, "template_check.json"), "w"), indent=1)
    ok = all(r["s1_fixed_shape"] and r["max_prob_diff"] < 1e-9 and r["max_angle_diff"] < 1e-9 for r in rows)
    print("ALL PASS" if ok else "SOME FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
