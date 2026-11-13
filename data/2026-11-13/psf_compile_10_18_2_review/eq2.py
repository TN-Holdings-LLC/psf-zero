"""For the outputs that differ (>12 active qubits): (1) the classical-output distribution from |0..0>, old vs new vs the
input; (2) if old and new act on the same qubits with the same initial layout, the states from 3 random product inputs
(fidelity, so a global phase does not count); (3) gate counts."""
import json, sys, os, runpy, contextlib, io, random
import numpy as np
sys.argv = [sys.argv[0]] + sys.argv[1:]
OLD, NEW_, CMP, OUT = sys.argv[1:5]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare.py")).read()
head = src.split("DEVICES = [")[0]
sys.argv = ["x", OLD, NEW_, "/dev/null"]
ns = {"__name__": "eq2"}
exec(compile(head, "compare_head", "exec"), ns)
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import fake_provider
DEV = {"sherbrooke": fake_provider.FakeSherbrooke(), "brisbane": fake_provider.FakeBrisbane()}
todo = [json.loads(l) for l in open(CMP)]
todo = [r for r in todo if not r["same"] and r.get("equivalent") is None]
def active(c):
    return sorted({c.find_bit(q).index for ci in c.data if ci.operation.name not in ("barrier", "measure") for q in ci.qubits} |
                  {c.find_bit(ci.qubits[0]).index for ci in c.data if ci.operation.name == "measure"})
def strip(c, act, prep=None):
    pos = {p: i for i, p in enumerate(act)}
    r = QuantumCircuit(len(act), global_phase=c.global_phase); meas = []
    if prep is not None:
        for i, u in enumerate(prep): r.u(*u, i)
    for ci in c.data:
        nm = ci.operation.name
        if nm == "barrier": continue
        if nm == "measure": meas.append((pos[c.find_bit(ci.qubits[0]).index], c.find_bit(ci.clbits[0]).index)); continue
        r.append(ci.operation, [pos[c.find_bit(q).index] for q in ci.qubits])
    return r, meas
def dist(c, nclb):
    act = active(c); r, meas = strip(c, act)
    p = Statevector(r).probabilities(); out = np.zeros(2 ** nclb)
    idx = np.arange(len(p))
    key = np.zeros(len(p), dtype=np.int64)
    for qi, cb in meas: key |= ((idx >> qi) & 1) << cb
    np.add.at(out, key, p); return out
def counts(c):
    d = {}
    for ci in c.data: d[ci.operation.name] = d.get(ci.operation.name, 0) + 1
    return d
cache = {}
with open(OUT, "w") as f:
    for r in todo:
        os.environ["PSFZ_BIN"] = os.path.abspath("rust/psfz/target/release/psfz") if r["mode"] == "psfz" else "/nonexistent/psfz"
        base = ns["build"](r["test"]); rng = random.Random(f"home-cmp|{r['device']}|{r['test']}")
        inputs = {"build": base}; inputs["same"] = ns["P"]["randomize"](base, rng, same=True); inputs["free"] = ns["P"]["randomize"](base, rng)
        qc = inputs[r["kind"]]; t = DEV[r["device"]].target
        oa, _, ea = ns["run"](ns["A"], qc, t); ob, _, eb = ns["run"](ns["B"], qc, t)
        rec = {k: r[k] for k in ("mode", "device", "test", "kind")}
        aa, ab = active(oa), active(ob)
        rec["active"] = (len(aa), len(ab)); rec["same_active"] = aa == ab
        rec["counts_old"] = counts(oa); rec["counts_new"] = counts(ob)
        if max(len(aa), len(ab)) <= 22:
            nclb = qc.num_clbits
            qin = qc.remove_final_measurements(inplace=False) if nclb else qc
            da, db = dist(oa, nclb), dist(ob, nclb)
            rec["tvd_old_new"] = float(0.5 * np.abs(da - db).sum())
            if qc.num_qubits <= 22 and nclb:
                # input distribution, measured bits as in qc
                from qiskit import transpile
                qd = transpile(qc, basis_gates=['u', 'cx', 'measure'], optimization_level=0)
                act = list(range(qd.num_qubits)); rr, meas = strip(qd, act)
                p = Statevector(rr).probabilities(); di = np.zeros(2 ** nclb); idx = np.arange(len(p)); key = np.zeros(len(p), dtype=np.int64)
                for qi, cb in meas: key |= ((idx >> qi) & 1) << cb
                np.add.at(di, key, p)
                rec["tvd_old_in"] = float(0.5 * np.abs(da - di).sum()); rec["tvd_new_in"] = float(0.5 * np.abs(db - di).sum())
            il_a = tuple(oa._layout.initial_index_layout(filter_ancillas=False)) if oa._layout else None
            il_b = tuple(ob._layout.initial_index_layout(filter_ancillas=False)) if ob._layout else None
            if aa == ab and il_a == il_b:
                fids = []
                g = np.random.default_rng(7)
                for _ in range(3):
                    prep = g.uniform(-np.pi, np.pi, (len(aa), 3))
                    sa = Statevector(strip(oa, aa, prep)[0]); sb = Statevector(strip(ob, aa, prep)[0])
                    fids.append(float(abs(np.vdot(sa.data, sb.data)) ** 2))
                rec["fid_random_inputs"] = fids
        f.write(json.dumps(rec) + "\n"); f.flush(); print(json.dumps(rec)[:300], flush=True)
