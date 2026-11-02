"""Before/after comparison of psfz (eb864ead against the workplace's three patches) on chain circuits whose pairs carry
general two-qubit blocks (Gen) and blocks with angles at pi. Exploratory, not declared. Per circuit: the output is
checked against the input (outcome distribution, as psfz's own harness: psfz may drop diagonal gates before measurement), and the counts of cz, sx, x and
the as-soon-as-possible duration (device durations) are compared.
usage: python ab.py BEFORE_BIN AFTER_BIN OUT.json"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime.fake_provider import FakeFez

BEFORE, AFTER, OUT = sys.argv[1:4]
target = FakeFez().target


def device_json(t):
    qs = []
    for q in range(t.num_qubits):
        sx, m = t["sx"].get((q,)), t["measure"].get((q,))
        qs.append(dict(sx_error=None if sx is None else sx.error, sx_duration=None if sx is None else sx.duration,
                       meas_error=None if m is None else m.error))
    cps = [[a, b, None if p is None else p.error, None if p is None else p.duration] for (a, b), p in t["cz"].items()]
    return json.dumps(dict(name="fake_fez", num_qubits=t.num_qubits, two_qubit_gate="cz", qubits=qs, couplers=cps))


def chain(n, seed, kind):
    """kind 'gen': per pair rxx(a) rz(b) ryy(c) (not commuting: a Gen block); 'pi': rzz(pi) rxx(a) ryy(c) (a Can
    block with one angle at pi); 'mix': both, alternating."""
    r = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(r.uniform(0, np.pi), q)
    for layer in range(2):
        for a in range(layer % 2, n - 1, 2):
            b = a + 1
            k = kind if kind != "mix" else ("gen" if (a // 2 + layer) % 2 == 0 else "pi")
            if k == "gen2":  # varied single-qubit parts inside the block, on either qubit
                qc.rxx(r.uniform(-3, 3), a, b)
                getattr(qc, ["rx", "ry", "rz"][int(r.integers(3))])(r.uniform(-3, 3), [a, b][int(r.integers(2))])
                qc.rzz(r.uniform(-3, 3), a, b)
                getattr(qc, ["rx", "ry", "rz"][int(r.integers(3))])(r.uniform(-3, 3), [a, b][int(r.integers(2))])
                qc.ryy(r.uniform(-3, 3), a, b)
            elif k == "gen":
                qc.rxx(r.uniform(-3, 3), a, b)
                qc.rz(r.uniform(-3, 3), b)
                qc.ryy(r.uniform(-3, 3), a, b)
                qc.rx(r.uniform(-3, 3), a)
            else:
                qc.rzz(np.pi, a, b)
                qc.rxx(r.uniform(-3, 3), a, b)
                qc.ryy(r.uniform(-3, 3), a, b)
        for q in range(n):
            qc.rz(r.uniform(-3, 3), q)
    return qc


def run(binary, dev, src):
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "dev.json"), "w").write(dev)
        open(os.path.join(d, "in.qasm"), "w").write(src)
        p = subprocess.run([binary, "compile", f"{d}/dev.json", f"{d}/in.qasm", f"{d}/out.qasm", f"{d}/rep.json"],
                           capture_output=True, text=True)
        rep = json.load(open(f"{d}/rep.json"))
    return rep


def to_circuit(rep, n):
    # the outcome order is the classical bits': physical qubit q is read into bit c
    pos = {int(q): int(c) for q, c in rep["measure"]}
    qc = QuantumCircuit(n)
    names = {}
    for name, qs, ps in rep["ops"]:
        names[name] = names.get(name, 0) + 1
        qs = [pos[int(q)] for q in qs]
        getattr(qc, name)(*ps, *qs)
    return qc, names


def asap(rep):
    t = {}
    for name, qs, ps in rep["ops"]:
        qs = [int(q) for q in qs]
        if name == "rz":
            continue
        dur = (target["cz"][tuple(qs)].duration if name == "cz" else target[name][(qs[0],)].duration) or 0
        s = max(t.get(q, 0) for q in qs) + dur
        for q in qs:
            t[q] = s
    return max(t.values()) * 1e9 if t else 0.0


dev = device_json(target)
rows = []
for kind, seeds in (("gen", 8), ("pi", 8), ("mix", 8), ("gen2", 40)):
    for seed in range(seeds):
        n = 8
        qc = chain(n, {"gen": 100, "pi": 200, "mix": 300, "gen2": 400}[kind] + seed, kind)
        qm = qc.copy()
        qm.measure_all()
        src = qasm3.dumps(transpile(qm, basis_gates=["rx", "ry", "rz", "h", "x", "y", "z", "s", "sdg", "sx", "p",
                                                     "cx", "cz", "rzz", "rxx", "ryy"], optimization_level=0))
        ideal = Statevector(qc)
        row = dict(kind=kind, seed=seed)
        for tag, b in (("before", BEFORE), ("after", AFTER)):
            rep = run(b, dev, src)
            if not rep.get("ok"):
                row[tag] = dict(ok=False, reason=rep.get("reason"))
                continue
            out, names = to_circuit(rep, n)
            pi_, po = ideal.probabilities(), Statevector(out).probabilities()
            fid = 1 - 0.5 * float(np.abs(pi_ - po).sum())  # 1 - total variation distance
            row[tag] = dict(ok=True, fid=round(float(fid), 12), cz=names.get("cz", 0), sx=names.get("sx", 0),
                            x=names.get("x", 0), ns=round(asap(rep), 1))
        rows.append(row)
        b, a = row["before"], row["after"]
        print(kind, seed, {k: (b.get(k), a.get(k)) for k in ("fid", "cz", "sx", "x", "ns")})
json.dump(rows, open(OUT, "w"), indent=1)
