"""Exploratory comparison (not pre-registered): psfz r1 (the Rust prototype) against psf_compile 2026-10-17.1's default
call, on C63-VAL's chain and ring tests.

    python rust/psfz/harness/compare.py <out.json> [--draws K]     (from the PSF-Zero root; psfz built in release mode)

Per device (FakeFez, FakeMarrakesh, FakeTorino: CZ; FakeSherbrooke, FakeBrisbane: ECR) and test, the build input and K
draws (C63-VAL's `randomize`, same-sign and free alternately, seed f"PSFZ|{device}|{test}"): the input written as
OpenQASM 3 at gate level (Qiskit's transpile to a gate basis, no layout, no routing), compiled by psfz; the release
compiles the input itself. Per output: native, exact (outcome distribution, up to 20 active qubits), two-qubit count,
duration (as soon as possible with the device's durations, measurements included), compile time (psfz: its own clock,
without process start; the release: the call). This harness is for checking only: psfz itself reads no Qiskit.
"""
import contextlib
import io
import json
import math
import os
import random
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

warnings.simplefilter("ignore")
OUT = sys.argv[1]
DRAWS = int(sys.argv[sys.argv.index("--draws") + 1]) if "--draws" in sys.argv else 4
if os.path.exists(OUT):
    raise SystemExit(f"STOP: {OUT} exists")
BIN = os.path.join("rust", "psfz", "target", "release", "psfz")
_tmp = tempfile.mkdtemp() + "/imp"
sys.argv = ["c55_val", _tmp]
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join("benchmarks", "c55_val.py"), run_name="psfz_import")
shutil.rmtree(_tmp, ignore_errors=True)
sys.path[:0] = [os.path.join("data", "2026-10-15", "workplace", "workplace_2026-10-15b_W1-W2", "w30"), "benchmarks"]
import c60_tests as TT  # noqa: E402
import test_release_2026_10_16_2 as T  # noqa: E402
from qiskit import QuantumCircuit, qasm3, transpile  # noqa: E402
from qiskit.circuit.library import ECRGate, PauliEvolutionGate  # noqa: E402
from qiskit.quantum_info import SparsePauliOp  # noqa: E402
from qiskit_ibm_runtime import fake_provider  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    REL = g["load"]("psf_compile.py", "psf_compile_release")
REL.WARN_WITHOUT_TARGET = False
CHAINS = ("tfim_n12", "tfim_n16", "tfimP_n12", "tfimP_n16", "ghz_n12", "chain_cz_n12", "chain_cz_n16", "cx_chain_n12",
          "cx_chain_n16")
RINGS = ("ring12_tfim", "heis_n12_r2", "tfim_n20_r3", "tfim_n40_r2", "ring_cz_n12")
FORMULA = {"tfim_n12", "tfim_n16", "tfimP_n12", "tfimP_n16", "ghz_n12", "ring12_tfim", "heis_n12_r2", "tfim_n20_r3",
           "tfim_n40_r2"}
DEVICES = ("FakeFez", "FakeMarrakesh", "FakeTorino", "FakeSherbrooke", "FakeBrisbane")
BASIS = ["rx", "ry", "rz", "h", "x", "y", "z", "s", "sdg", "sx", "p", "cx", "cz", "rzz", "rxx", "ryy"]


def build(name):
    return g["build"](name) if name in FORMULA else TT.build(name)


def randomize(qc, rng, same=False):  # c63_val.py's
    out = qc.copy_empty_like()
    for ins in qc.data:
        op = ins.operation
        if op.name == "PauliEvolution":
            H = op.operator
            if same:
                new = SparsePauliOp(H.paulis, [float(c.real) * rng.uniform(0.5, 1.5) for c in H.coeffs])
            else:
                new = SparsePauliOp(H.paulis, [rng.uniform(-1.5, 1.5) for _ in range(len(H))])
            op = PauliEvolutionGate(new, time=op.time, synthesis=op.synthesis)
        elif op.params and op.name not in ("barrier", "measure", "delay") and \
                all(isinstance(p, (int, float)) for p in op.params):
            op = op.__class__(*[(float(p) * rng.uniform(0.5, 1.5)) if same else rng.uniform(-math.pi, math.pi)
                                for p in op.params])
        out.append(op, ins.qubits, ins.clbits)
    return out


def device_json(t, name):
    g2 = "cz" if "cz" in t.operation_names else "ecr"
    qs = []
    for q in range(t.num_qubits):
        sx, m = t["sx"].get((q,)), t["measure"].get((q,))
        qs.append(dict(sx_error=None if sx is None else sx.error, meas_error=None if m is None else m.error))
    cps = [[a, b, None if p is None else p.error, None if p is None else p.duration] for (a, b), p in t[g2].items()]
    return dict(name=name, num_qubits=t.num_qubits, two_qubit_gate=g2, qubits=qs, couplers=cps)


def numbers(c, t):
    clock, n2 = {}, 0
    pos = {q: i for i, q in enumerate(c.qubits)}
    for ins in c.data:
        nm = ins.operation.name
        if nm in ("barrier", "delay"):
            continue
        qs = tuple(pos[q] for q in ins.qubits)
        p = t[nm].get(qs) if nm in t.operation_names else None
        d = (p.duration or 0.0) if p is not None else 0.0
        s = max(clock.get(q, 0.0) for q in qs)
        for q in qs:
            clock[q] = s + d
        n2 += len(qs) == 2
    return dict(q2=n2, dur_us=1e6 * max(clock.values()) if clock else 0.0, active=len(clock), native=T._native(c, t))


def from_report(rep, n):
    c = QuantumCircuit(n, rep["num_clbits"])
    for name, qs, ps in rep["ops"]:
        qs = [int(q) for q in qs]
        if name == "ecr":
            c.append(ECRGate(), qs)
        else:
            getattr(c, name)(*ps, *qs)
    for q, cb in rep["measure"]:
        c.measure(int(q), int(cb))
    return c


work = tempfile.mkdtemp()
rows = []
for d in DEVICES:
    t = getattr(fake_provider, d)().target
    dj = os.path.join(work, f"{d}.json")
    json.dump(device_json(t, d), open(dj, "w"))
    for name in CHAINS + RINGS:
        qc0 = build(name)
        rng = random.Random(f"PSFZ|{d}|{name}")
        for k in range(-1, DRAWS):
            qc = qc0 if k < 0 else randomize(qc0, rng, same=k % 2 == 0)
            row = dict(device=d, test=name, k=k)
            gate = transpile(qc, basis_gates=BASIS, optimization_level=0)
            src = os.path.join(work, "in.qasm")
            open(src, "w").write(qasm3.dumps(gate))
            rep_f, out_f = os.path.join(work, "rep.json"), os.path.join(work, "out.qasm")
            for f in (rep_f, out_f):
                if os.path.exists(f):
                    os.remove(f)
            r = subprocess.run([BIN, "compile", dj, src, out_f, rep_f], capture_output=True, text=True)
            rep = json.load(open(rep_f))
            row["psfz_s"] = rep["compile_s"]
            if rep["ok"]:
                out = from_report(rep, t.num_qubits)
                row["psfz"] = numbers(out, t)
                ref = qc.decompose() if any(i.operation.name == "PauliEvolution" for i in qc.data) else qc
                row["psfz"]["tvd"] = T._tvd(T._dist(ref), T._dist(out)) if row["psfz"]["active"] <= 20 else None
                row["layout"] = rep["layout"]
            else:
                row["psfz"] = dict(refused=rep["reason"])
            s = time.perf_counter()
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    o2 = REL.compile_for_hardware(qc, target=t)
                row["rel_s"] = time.perf_counter() - s
                row["rel"] = numbers(o2, t)
            except Exception as e:  # noqa: BLE001
                row["rel_s"] = time.perf_counter() - s
                row["rel"] = dict(refused=f"{type(e).__name__}: {str(e)[:100]}")
            rows.append(row)
            p, q = row["psfz"], row["rel"]
            print(d, name, k, "psfz", p.get("refused") or f"q2 {p['q2']} dur {p['dur_us']:.2f} native {p['native']} tvd {p['tvd']}",
                  "| rel", q.get("refused") or f"q2 {q['q2']} dur {q['dur_us']:.2f}", flush=True)
shutil.rmtree(work, ignore_errors=True)
json.dump(dict(rows=rows, release=REL.VERSION), open(OUT, "w"), indent=1)
print("done")
