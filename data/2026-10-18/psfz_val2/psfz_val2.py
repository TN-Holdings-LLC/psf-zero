"""PSFZ-VAL2: psfz r4 against release 2026-10-17.1's default call, against its pre-registration
(PSFZ-VAL2_preregistration.md next to this file). PSFZ-VAL's script (data/2026-10-17/psfz_val/psfz_val.py) with
three changes: new draws (random.Random(f"PSFZ-VAL2|{device}|{test}")); on the devices PSFZ-VAL used (`KNOWN`) the
draws only, not the build input (already seen); on FakeCusco the two tests that psfz and the release both refuse on
the build input (`LEFT_OUT`, found before the run) are left out.

    python data/2026-10-18/psfz_val2/psfz_val2.py <out.jsonl> <device> [<device> ...]   (from the PSF-Zero root)

Per input: psfz on the input written as gate-level OpenQASM 3 (Qiskit's transpile to a gate basis, no layout, no
routing), called twice (outputs compared); the release on the circuit itself. Per output: native, failed elements
(error >= 0.5), exactness (outcome distribution; every input up to 16 active qubits, the build input and the first two
draws from 17 to 20), two-qubit count, duration (as soon as possible with the device's durations, measurements
included), compile time (psfz: its own clock; the release: the call). One JSON line per input.
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
OUT, DEVS = sys.argv[1], sys.argv[2:]
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
SAME, FREE = 5, 5
BASIS = ["rx", "ry", "rz", "h", "x", "y", "z", "s", "sdg", "sx", "p", "cx", "cz", "rzz", "rxx", "ryy"]


KNOWN = ("FakeAachen", "FakeBoston", "FakeKingston", "FakePittsburgh", "FakeKyiv", "FakeOsaka", "FakeKawasaki",
         "FakeQuebec", "FakeFez", "FakeMarrakesh", "FakeTorino", "FakeSherbrooke", "FakeBrisbane")
LEFT_OUT = {("FakeCusco", "tfim_ring_n28"), ("FakeCusco", "tfim_n40_r2")}
NEW = ("tfim_chain_n24", "tfim_ring_n28", "heis_chain_n16", "cz_ring_n20", "cx_chain_n20", "ghz_n20")


def build_new(name):
    """The 6 tests of new sizes: Pauli evolutions as C55-VAL builds its own (`_evolution`: an RY layer, then a
    second-order Suzuki-Trotter evolution, every qubit measured), and gate circuits as C63-VAL's native tests."""
    import random as _r
    rng = _r.Random("PSFZ-VAL-build|" + name)
    lab, evo = g["_lab"], g["_evolution"]
    if name == "tfim_chain_n24":
        n = 24
        terms = [(lab(n, {q: "Z", q + 1: "Z"}), rng.uniform(-1.5, 1.5)) for q in range(n - 1)]
        terms += [(lab(n, {q: "X"}), 0.7) for q in range(n)]
        return evo(n, terms, 1.0, 3)
    if name == "tfim_ring_n28":
        n = 28
        terms = [(lab(n, {q: "Z", (q + 1) % n: "Z"}), rng.uniform(-1.5, 1.5)) for q in range(n)]
        terms += [(lab(n, {q: "X"}), 0.7) for q in range(n)]
        return evo(n, terms, 1.0, 3)
    if name == "heis_chain_n16":
        n = 16
        terms = [(lab(n, {q: p, q + 1: p}), rng.uniform(-1, 1)) for q in range(n - 1) for p in "XYZ"]
        terms += [(lab(n, {q: "Z"}), rng.uniform(-1, 1)) for q in range(n)]
        return evo(n, terms, 0.6, 2)
    n = int(name.split("_n")[1])
    qc = QuantumCircuit(n)
    if name.startswith("ghz"):
        qc.h(0)
        for q in range(n - 1):
            qc.cx(q, q + 1)
    else:
        for _ in range(3):
            for q in range(n):
                qc.ry(rng.uniform(-3, 3), q)
                qc.rz(rng.uniform(-3, 3), q)
            if name.startswith("cz_ring"):
                for q in range(n):
                    qc.cz(q, (q + 1) % n)
            else:
                for q in range(n - 1):
                    qc.cx(q, q + 1)
        for q in range(n):
            qc.ry(rng.uniform(-3, 3), q)
    qc.measure_all()
    return qc


def build(name):
    if name in NEW:
        return build_new(name)
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
        qs.append(dict(sx_error=None if sx is None else sx.error, sx_duration=None if sx is None else sx.duration,
                       meas_error=None if m is None else m.error))
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


def fp(c):
    import hashlib
    q = {b: i for i, b in enumerate(c.qubits)}
    k = {b: i for i, b in enumerate(c.clbits)}
    return hashlib.sha256(repr([(i.operation.name, tuple(q[x] for x in i.qubits), tuple(k[x] for x in i.clbits),
                                 tuple(float(p).hex() for p in i.operation.params)) for i in c.data]).encode()).hexdigest()


if os.path.exists(OUT):
    raise SystemExit(f"STOP: {OUT} exists")
work = tempfile.mkdtemp()
with open(OUT, "w", encoding="utf-8") as fh:
    for d in DEVS:
        t = getattr(fake_provider, d)().target
        fe, fq = REL._failed_elements(t, 0.5)
        dj = os.path.join(work, f"{d}.json")
        json.dump(device_json(t, d), open(dj, "w"))
        for name in CHAINS + RINGS + NEW:
            qc0 = build(name)
            if (d, name) in LEFT_OUT:
                continue
            rng = random.Random(f"PSFZ-VAL2|{d}|{name}")
            for k in range(-1, SAME + FREE):
                if k < 0 and d in KNOWN:
                    continue
                qc = qc0 if k < 0 else randomize(qc0, rng, same=k < SAME)
                row = dict(device=d, test=name, k=k)
                src = os.path.join(work, "in.qasm")
                open(src, "w").write(qasm3.dumps(transpile(qc, basis_gates=BASIS, optimization_level=0)))
                reps = []
                for rep_try in range(2):
                    rep_f, out_f = os.path.join(work, f"rep{rep_try}.json"), os.path.join(work, f"out{rep_try}.qasm")
                    for f in (rep_f, out_f):
                        if os.path.exists(f):
                            os.remove(f)
                    subprocess.run([BIN, "compile", dj, src, out_f, rep_f], capture_output=True, text=True)
                    reps.append(json.load(open(rep_f)) if os.path.exists(rep_f) else dict(ok=False, reason="no report"))
                rep = reps[0]
                row["psfz_s"] = rep.get("compile_s")
                if rep["ok"]:
                    out = from_report(rep, t.num_qubits)
                    row["psfz"] = numbers(out, t)
                    row["psfz"]["failed"] = bool(REL._uses_failed(out, fe, fq))
                    row["psfz"]["deterministic"] = reps[1]["ok"] and fp(from_report(reps[1], t.num_qubits)) == fp(out)
                    act = row["psfz"]["active"]
                    if act <= 16 or (act <= 20 and k < 2):
                        ref = qc.decompose() if any(i.operation.name == "PauliEvolution" for i in qc.data) else qc
                        row["psfz"]["tvd"] = T._tvd(T._dist(ref), T._dist(out))
                    else:
                        row["psfz"]["tvd"] = None
                    row["layout"] = rep["layout"]
                else:
                    row["psfz"] = dict(refused=rep.get("reason"))
                s = time.perf_counter()
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        o2 = REL.compile_for_hardware(qc, target=t)
                    row["rel_s"] = time.perf_counter() - s
                    row["rel"] = numbers(o2, t)
                except Exception as e:  # noqa: BLE001
                    row["rel_s"] = time.perf_counter() - s
                    row["rel"] = dict(refused=f"{type(e).__name__}: {str(e)[:100]}")
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            print(d, name, "done", flush=True)
shutil.rmtree(work, ignore_errors=True)
print("PSFZ-VAL2 DONE", REL.VERSION)
