"""w2_tfim.py -- W2 (2026-10-14): TFIM chains at scale, c48 against Qiskit level 3 given the PauliEvolutionGate and
Qiskit level 3 given a hand-written brickwork circuit. Measured: correctness, CZ count, circuit duration. No ESP, no
search, no seed variation (every compile uses seed_transpiler=0, once).

    python w2_tfim.py --out <dir> [--devices FakeFez,FakeMarrakesh] [--n 12,16,24,32,50,100] [--steps 2,4,8,10]

Run from the psf-zero repository root (loads patches/psf_compile_c48_2026-10-12 and benchmarks/psf_smart_layout.py).

Model (as HW1/HW2's tfim tests): H = J sum Z_i Z_{i+1} + h sum X_i on an open chain, J = h = 1, time 0.6, second-order
Suzuki-Trotter with `steps` repetitions, start in |0...0>, all qubits measured.

Arms
  PSF      c48, `compile_for_hardware(qc, backend=b, seed_transpiler=0)` as shipped (item 83 on). Its item 80 may
           return Qiskit level 3's circuit when that has the lower summed -ln(1 - e); which one came back is recorded.
  PSF-C    c48 with the estimate-based choices off: `level3_candidate=False, near_failed_error=None` (the constructive
           pipeline alone; placement as c48's default).
  QK3      `generate_preset_pass_manager(3, b, seed_transpiler=0)` on the PauliEvolutionGate circuit.
  QK3-BW   the same pass manager on a hand-written brickwork circuit with exactly the same unitary: per step the
           symmetric formula RZZ(dt/2) | RX(dt) | RZZ(dt/2), adjacent half layers merged (RZZ(dt) between RX layers),
           each RZZ layer written even edges then odd edges.

Correctness: n <= 24, total variation distance between the compiled and the logical outcome distributions (Aer
statevector of the active qubits); n > 24, the largest difference of <Z_i> and <Z_i Z_{i+1}> over the chain (Aer
matrix product state, same mapping through the measurements). Duration: gates scheduled as soon as possible on the
Target's durations, measurements left out (the same rule as HW2's estimates)."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import sys
import time
import warnings

import numpy as np

REPO = os.path.abspath(os.getcwd())
for p in (os.path.join(REPO, "benchmarks"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)
J, H, T = 1.0, 1.0, 0.6
C48_FILE = os.path.join(REPO, "patches", "psf_compile_c48_2026-10-12", "psf_compile.py")
LAYOUT_FILE = os.path.join(REPO, "benchmarks", "psf_smart_layout.py")
EXACT_MAX = 24
TOL = 1e-6


def stamp():
    import datetime as dt
    from zoneinfo import ZoneInfo
    return dt.datetime.now(ZoneInfo("Europe/Zurich")).strftime("%Y-%m-%d %H:%M:%S %Z")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _lab(n, d):
    s = ["I"] * n
    for q, p in d.items():
        s[n - 1 - q] = p
    return "".join(s)


def tfim_pauli(n, steps):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import SuzukiTrotter
    terms = [(_lab(n, {q: "Z", q + 1: "Z"}), J) for q in range(n - 1)] + [(_lab(n, {q: "X"}), H) for q in range(n)]
    qc = QuantumCircuit(n)
    qc.append(PauliEvolutionGate(SparsePauliOp.from_list(terms), time=T, synthesis=SuzukiTrotter(order=2, reps=steps)),
              range(n))
    qc.measure_all()
    return qc


def tfim_brickwork(n, steps):
    """exp(-i dt (J ZZ)) = RZZ(2 J dt), exp(-i dt h X) = RX(2 h dt); symmetric Trotter with merged half layers."""
    from qiskit import QuantumCircuit
    dt = T / steps
    qc = QuantumCircuit(n)

    def zz(theta):
        for start in (0, 1):
            for q in range(start, n - 1, 2):
                qc.rzz(theta, q, q + 1)
    zz(2 * J * dt / 2)
    for k in range(steps):
        for q in range(n):
            qc.rx(2 * H * dt, q)
        zz(2 * J * dt if k < steps - 1 else 2 * J * dt / 2)
    qc.measure_all()
    return qc


def unrolled(qc):
    from qiskit import transpile
    return transpile(qc, basis_gates=["cx", "rz", "sx", "x", "h", "rzz", "rx", "ry", "u", "measure"],
                     optimization_level=0)


def _reduce(circ):
    """(circuit on the active qubits without measurements, {clbit: local qubit})"""
    from qiskit import QuantumCircuit
    meas, body = {}, []
    for ins in circ.data:
        nm = ins.operation.name
        if nm == "measure":
            meas[circ.find_bit(ins.clbits[0]).index] = circ.find_bit(ins.qubits[0]).index
        elif nm in ("barrier", "delay"):
            continue
        else:
            body.append((ins.operation, [circ.find_bit(q).index for q in ins.qubits]))
    active = sorted({q for _, qs in body for q in qs} | set(meas.values()))
    pos = {q: i for i, q in enumerate(active)}
    c = QuantumCircuit(len(active))
    for op, qs in body:
        c.append(op, [pos[q] for q in qs])
    return c, {k: pos[q] for k, q in meas.items()}


def probs(circ):
    from qiskit_aer import AerSimulator
    c, m = _reduce(circ)
    ncl = circ.num_clbits
    c.save_probabilities([m[k] for k in range(ncl)])
    return np.asarray(AerSimulator(method="statevector").run(c).result().data()["probabilities"])


def tvd(p, q):
    return 0.5 * float(np.sum(np.abs(p - q)))


def corr(circ, n):
    """<Z_i> and <Z_i Z_{i+1}> for logical i (through the measurements), MPS."""
    from qiskit.quantum_info import Pauli
    from qiskit_aer import AerSimulator
    c, m = _reduce(circ)
    for i in range(n):
        c.save_expectation_value(Pauli("Z"), [m[i]], label=f"z{i}")
    for i in range(n - 1):
        c.save_expectation_value(Pauli("ZZ"), [m[i], m[i + 1]], label=f"zz{i}")
    d = AerSimulator(method="matrix_product_state").run(c).result().data()
    return np.array([d[f"z{i}"] for i in range(n)] + [d[f"zz{i}"] for i in range(n - 1)])


def duration_and_cz(out, target):
    pos = {q: i for i, q in enumerate(out.qubits)}
    clock, cz, two = {}, 0, None
    for ins in out.data:
        nm = ins.operation.name
        if nm in ("barrier", "measure", "delay"):
            continue
        qs = tuple(pos[q] for q in ins.qubits)
        pr = target[nm].get(qs) if nm in target.operation_names else None
        d = (pr.duration or 0.0) if pr is not None else 0.0
        s = max(clock.get(q, 0.0) for q in qs)
        for q in qs:
            clock[q] = s + d
        if len(qs) == 2:
            cz += 1
            two = nm
    return (max(clock.values()) if clock else 0.0) * 1e6, cz, two


def isa_ok(out, target):
    pos = {q: i for i, q in enumerate(out.qubits)}
    for ins in out.data:
        nm = ins.operation.name
        if nm in ("barrier",):
            continue
        qs = tuple(pos[q] for q in ins.qubits)
        if nm not in target.operation_names or qs not in target[nm]:
            return False
    return True


def main():
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--devices", default="FakeFez,FakeMarrakesh")
    ap.add_argument("--n", default="12,16,24,32,50,100")
    ap.add_argument("--steps", default="2,4,8,10")
    ap.add_argument("--arms", default="PSF,PSF-C,QK3,QK3-BW")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import qiskit
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import fake_provider
    load(LAYOUT_FILE, "psf_smart_layout")
    c48 = load(C48_FILE, "w2_c48")
    c48.WARN_WITHOUT_TARGET = False
    assert getattr(c48, "ST_SCHEDULE", True) is True
    print(f"{stamp()} | {platform.platform()} | python {platform.python_version()} | qiskit {qiskit.__version__}")
    print("SCRIPT", sha(os.path.abspath(__file__)), "| c48", sha(C48_FILE))
    path = os.path.join(a.out, "w2_rows.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["device"], r["n"], r["steps"], r["arm"]) for r in map(json.loads, open(path))}
    fh = open(path, "a")
    for dev in a.devices.split(","):
        b = getattr(fake_provider, dev)()
        tgt = b.target
        pm3 = generate_preset_pass_manager(3, b, seed_transpiler=0)
        for n in map(int, a.n.split(",")):
            for steps in map(int, a.steps.split(",")):
                logical = tfim_pauli(n, steps)
                bw = tfim_brickwork(n, steps)
                ref = probs(unrolled(logical)) if n <= EXACT_MAX else corr(unrolled(logical), n)
                if n <= 12:   # the brickwork circuit must have the same outcomes as the Pauli evolution
                    assert tvd(probs(unrolled(bw)), ref) < 1e-9
                for arm in a.arms.split(","):
                    if (dev, n, steps, arm) in done:
                        continue
                    t0 = time.perf_counter()
                    chosen = None
                    if arm in ("PSF", "PSF-C"):
                        kw = {} if arm == "PSF" else dict(level3_candidate=False, near_failed_error=None)
                        before = dict(c48.L3C_STATS)
                        with contextlib.redirect_stdout(io.StringIO()):
                            out = c48.compile_for_hardware(logical, backend=b, seed_transpiler=0, **kw)
                        if arm == "PSF":
                            d = {k: c48.L3C_STATS[k] - before.get(k, 0) for k in c48.L3C_STATS}
                            chosen = "level3" if d.get("kept_level3") else "psf"
                    elif arm == "QK3":
                        out = pm3.run(logical)
                    else:
                        out = pm3.run(bw)
                    secs = time.perf_counter() - t0
                    dur, cz, two = duration_and_cz(out, tgt)
                    if n <= EXACT_MAX:
                        err = tvd(probs(out), ref)
                        kind = "tvd"
                    else:
                        err = float(np.max(np.abs(corr(out, n) - ref)))
                        kind = "max|dZ,dZZ|"
                    r = dict(device=dev, n=n, steps=steps, arm=arm, cz=cz, twoq_gate=two, dur_us=round(dur, 4),
                             correct=bool(err <= TOL), err=err, err_kind=kind, isa=isa_ok(out, tgt),
                             compile_s=round(secs, 2), item80=chosen)
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    print(f"{dev} n={n:3d} steps={steps:2d} {arm:6s} CZ {cz:5d} dur {dur:7.2f} us err {err:.1e} "
                          f"isa {r['isa']} {secs:.1f}s{' item80=' + chosen if chosen else ''}", flush=True)
    fh.close()
    print(f"{stamp()} done")


if __name__ == "__main__":
    main()
