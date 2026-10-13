"""hw2_real.py -- HW2 (2026-10-13): (R) HW1 repeated on a different IBM device; (S) the same gates in a different order.

Pre-registered in HW2_preregistration.md (fixed before any real-device step; sha256 and CEST time kept on the owner's
PC). One job:

  part R  HW1's nine tests and the two tfimP tests, compiled by Qiskit level 3 (QK3) and by PSF-Zero's own pipeline
          (PSF: prototype c45 with level3_candidate=False), as in HW1 (Addenda 454-455), on another device. Asks whether HW1's findings
          repeat: ESP does not rank the pair; an estimate with idling does (form fixed before the run, from physics);
          PSF-Zero wins where its circuit is also shorter in time and not on the Barabasi-Albert QAOA circuits.
  part S  prototype c48 with item 83 off (S0, = c47's order) and on (S1): the same blocks of gates, only their order
          (and so the circuit's duration) differs. Asks whether a shorter schedule of the same gates gives a higher
          fidelity on the device. Tests: TFIM from a fixed random product state (tfimP; HW1's TFIM from |0...0> could
          not be scored by XEB at n = 16) and the two 3-regular QAOA tests; a pair is submitted only where S0 and S1
          differ.

Every circuit's ESP, duration (as soon as possible on the Target's gate durations) and idle-aware estimate
(ESP x exp(-sum over active qubits of (T - t_first) (1/(4 T1) + 1/(2 T2))), the Target's T1 and T2) are computed from
the device's Target when `prepare` runs and written into the manifest before submission.

Steps (only `submit` uses QPU time; `prepare`, `submit` and `fetch` need the IBM connection, the others do not):

  rehearse  the whole flow on a fake device (default FakeFez) with a noiseless local simulator. No IBM connection.
  prepare   device rule (of ibm_fez and ibm_marrakesh, the operational one with fewer pending jobs; neither: stop);
            the device's Target now; compiles every arm; checks every circuit (ISA, failed elements, exactness by
            noiseless simulation); writes the circuits (QPY) and the manifest BEFORE any submission. Refuses to run if
            the manifest exists.
  submit    submits ONE SamplerV2 job, 6,000 shots per circuit, dynamical decoupling and twirling off; records the job
            ID. Refuses if a job ID is already recorded.
  fetch     waits for the job (at most --wait-min minutes) and saves the raw counts and the job's metrics.
  score     offline: scores the predictions.

Usage (from the psf-zero repository root):
    python -u <kit>/hw2_real.py rehearse --out data/2026-10-13/hw2_rehearsal
    python -u <kit>/hw2_real.py prepare  --out data/2026-10-13/hw2
    python -u <kit>/hw2_real.py submit   --out data/2026-10-13/hw2
    python -u <kit>/hw2_real.py fetch    --out data/2026-10-13/hw2
    python -u <kit>/hw2_real.py score    --out data/2026-10-13/hw2

The IBM API key is never read from or written to any file by this script; it uses the account saved beforehand by
QiskitRuntimeService.save_account() in a terminal. This script runs no git and no pip.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import random
import sys
import time
import warnings

REPO = os.path.abspath(os.getcwd())
for p in (os.path.join(REPO, "benchmarks"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

SHOTS = 6000  # each circuit, per repeat
REPEATS = 2
USAGE_EST_CAP_S = 300.0  # prepare refuses a job whose rough QPU estimate (0.5 ms per shot + 1 s per circuit) exceeds this
DEVICES = ("ibm_fez", "ibm_marrakesh")  # the device rule: the operational one with fewer pending jobs (ties: this order)
FAILED = 0.5
EXACT_TOL = 1e-6
EXACT_MAX_ACTIVE = 24  # noiseless exactness check by statevector on at most this many active qubits
MAX_PUBS = 80
MISSING_T_S = 1e-6  # a qubit without a reported T1 or T2 is taken to have 1 us (as in HW1's exploratory estimate)
C45_FILE = os.path.join(REPO, "patches", "psf_compile_c45_2026-10-12", "psf_compile.py")
C48_FILE = os.path.join(REPO, "patches", "psf_compile_c48_2026-10-12", "psf_compile.py")
LAYOUT_FILE = os.path.join(REPO, "benchmarks", "psf_smart_layout.py")
TESTS = ("qaoa_n12_p2", "qaoa_n20_p2", "ba_n14", "ba_n16", "ba_n18", "ba_n20", "tfim_n12", "tfim_n16", "ghz_n12",
         "tfimP_n12", "tfimP_n16")
READING = ("qaoa_n12_p2", "qaoa_n20_p2", "ba_n14", "ba_n16", "ba_n18", "ba_n20", "tfim_n12")  # HW1's seven (Addendum 455)
BA = ("ba_n14", "ba_n16", "ba_n18", "ba_n20")
R4_TESTS = READING + ("tfimP_n12", "tfimP_n16")  # R4 also reads the two TFIM tests from a product state
TESTS_S = ("tfimP_n12", "tfimP_n16", "qaoa_n12_p2", "qaoa_n20_p2")
KIT = os.path.dirname(os.path.abspath(__file__))  # Benchpress's QAOA files (Apache-2.0) travel with the kit in qasm/
QAOA_GAMMA, QAOA_BETA = 0.55, 0.45  # linear ramp: gamma_k = QAOA_GAMMA (k + 1) / p, beta_k = QAOA_BETA (1 - k / p)
TFIM_J, TFIM_H, TFIM_T, TFIM_REPS = 1.0, 1.0, 0.6, 2


def now_utc():
    return dt.datetime.now(dt.timezone.utc)


def stamp():
    u = now_utc()
    try:
        from zoneinfo import ZoneInfo
        c = u.astimezone(ZoneInfo("Europe/Zurich"))
        return f"{u.isoformat(timespec='seconds')} ({c.strftime('%Y-%m-%d %H:%M:%S')} CEST)"
    except Exception:  # noqa: BLE001
        return u.isoformat(timespec="seconds")


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ circuits

def _label(n, d):
    lb = ["I"] * n
    for q, p in d.items():
        lb[n - 1 - q] = p
    return "".join(lb)


def regular3(n, rng):
    while True:
        stubs = [v for v in range(n) for _ in range(3)]
        rng.shuffle(stubs)
        edges, ok = set(), True
        for i in range(0, len(stubs), 2):
            a, b = stubs[i], stubs[i + 1]
            e = (min(a, b), max(a, b))
            if a == b or e in edges:
                ok = False
                break
            edges.add(e)
        if ok:
            return sorted(edges)


def build_test(name):
    """(measured logical circuit, observable spec) for an HW1 test. Spec: ("zz", [(i, j, w)]) for sum w <Z_i Z_j>,
    or ("ghz", n) for P(all 0) + P(all 1)."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate, QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import SuzukiTrotter
    kind = name.split("_")[0]
    n = int(name.split("_n")[1].split("_")[0])
    if kind == "qaoa":
        p = int(name.split("_p")[1])
        edges = regular3(n, random.Random("HW1|" + name))
        cost = SparsePauliOp.from_list([(_label(n, {a: "Z", b: "Z"}), 1.0) for a, b in edges])
        qa = QAOAAnsatz(cost, reps=p)
        vals = {}
        for prm in qa.parameters:
            k = int(prm.name.split("[")[1].rstrip("]"))
            vals[prm] = QAOA_GAMMA * (k + 1) / p if prm.name.startswith("γ") else QAOA_BETA * (1 - k / p)
        qc = qa.assign_parameters(vals)
        qc.measure_all()
        return qc, ("zz", [(a, b, 1.0) for a, b in edges])
    if kind == "ba":  # Benchpress qaoa_barabasi_albert_N{n}_3reps.qasm; the cost terms are its two-qubit pairs
        qc = QuantumCircuit.from_qasm_file(os.path.join(KIT, "qasm", f"qaoa_barabasi_albert_N{n}_3reps.qasm"))
        pairs = sorted({tuple(sorted(qc.find_bit(q).index for q in i.qubits)) for i in qc.data
                        if len(i.qubits) == 2 and i.operation.name != "barrier"})
        qc = qc.remove_final_measurements(inplace=False)
        qc.measure_all()
        return qc, ("zz", [(a, b, 1.0) for a, b in pairs])
    if kind == "tfimP":  # HW2: the TFIM of HW1 from a fixed random product state (ry layer), so that XEB can score it
        rng = random.Random("HW2|" + name)
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.ry(rng.uniform(0.3 * math.pi, 0.7 * math.pi), q)
        terms = [(_label(n, {q: "Z", q + 1: "Z"}), TFIM_J) for q in range(n - 1)]
        terms += [(_label(n, {q: "X"}), TFIM_H) for q in range(n)]
        qc.append(PauliEvolutionGate(SparsePauliOp.from_list(terms), time=TFIM_T,
                                     synthesis=SuzukiTrotter(order=2, reps=TFIM_REPS)), range(n))
        qc.measure_all()
        return qc, ("zz", [(q, q + 1, 1.0) for q in range(n - 1)])
    if kind == "tfim":
        terms = [(_label(n, {q: "Z", q + 1: "Z"}), TFIM_J) for q in range(n - 1)]
        terms += [(_label(n, {q: "X"}), TFIM_H) for q in range(n)]
        qc = QuantumCircuit(n)
        qc.append(PauliEvolutionGate(SparsePauliOp.from_list(terms), time=TFIM_T,
                                     synthesis=SuzukiTrotter(order=2, reps=TFIM_REPS)), range(n))
        qc.measure_all()
        return qc, ("zz", [(q, q + 1, 1.0) for q in range(n - 1)])
    if kind == "ghz":
        qc = QuantumCircuit(n)
        qc.h(0)
        for q in range(n - 1):
            qc.cx(q, q + 1)
        qc.measure_all()
        return qc, ("ghz", n)
    raise ValueError(name)


def _unrolled(qc):
    """The logical circuit with every composite gate (Pauli evolutions included) replaced by its definition, so that a
    simulation gives what the circuit's own synthesis does (a Trotterised evolution, not the exact exponential)."""
    from qiskit import transpile
    return transpile(qc, basis_gates=["cx", "rz", "sx", "x", "h", "rzz", "rx", "ry", "u", "measure"],
                     optimization_level=0)


def measured_distribution(circ):
    """Exact probabilities over the classical bits of a circuit whose measurements are all at the end: (probabilities
    as {bitstring: p} with clbit 0 rightmost, number of active qubits)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    meas = {}
    body = []
    for ins in circ.data:
        nm = ins.operation.name
        if nm == "measure":
            meas[circ.find_bit(ins.clbits[0]).index] = circ.find_bit(ins.qubits[0]).index
        elif nm in ("barrier", "delay"):
            continue
        else:
            body.append((ins.operation, [circ.find_bit(q).index for q in ins.qubits]))
    active = sorted({q for _, qs in body for q in qs} | set(meas.values()))
    if len(active) > EXACT_MAX_ACTIVE:
        return None, len(active)
    pos = {q: i for i, q in enumerate(active)}
    c = QuantumCircuit(len(active))
    for op, qs in body:
        c.append(op, [pos[q] for q in qs])
    probs = Statevector(c).probabilities()
    out = {}
    nclb = circ.num_clbits
    for idx in np.nonzero(probs > 1e-15)[0]:
        bits = ["0"] * nclb
        for cb, q in meas.items():
            if (int(idx) >> pos[q]) & 1:
                bits[nclb - 1 - cb] = "1"
        key = "".join(bits)
        out[key] = out.get(key, 0.0) + float(probs[idx])
    return out, len(active)


def observable(spec, dist_or_counts, shots=None):
    """(value, standard error) of the observable from probabilities ({bits: p}) or counts ({bits: int})."""
    tot = sum(dist_or_counts.values())
    vals, ws = [], []
    for key, c in dist_or_counts.items():
        b = key.replace(" ", "")
        bit = lambda q: int(b[len(b) - 1 - q])  # noqa: E731
        if spec[0] == "zz":
            v = sum(w * (1 - 2 * bit(i)) * (1 - 2 * bit(j)) for i, j, w in spec[1])
        else:
            n = spec[1]
            v = 1.0 if (b[-n:] == "0" * n or b[-n:] == "1" * n) else 0.0
        vals.append(v)
        ws.append(c / tot)
    vals, ws = np.array(vals, dtype=float), np.array(ws, dtype=float)
    mean = float(np.dot(vals, ws))
    se = float(math.sqrt(max(np.dot(ws, (vals - mean) ** 2), 0.0) / shots)) if shots else 0.0
    return mean, se


def xeb(dist, counts, n, shots):
    """Normalised linear cross-entropy fidelity of sampled counts against the ideal distribution: 1 for the ideal
    sampler, 0 for uniform noise; (value, standard error)."""
    s2 = sum(v * v for v in dist.values())
    den = (2 ** n) * s2 - 1.0
    tot = sum(counts.values())
    vals = np.array([(2 ** n) * dist.get(k.replace(" ", ""), 0.0) for k in counts], dtype=float)
    w = np.array([c / tot for c in counts.values()], dtype=float)
    m = float(np.dot(vals, w))
    sd = math.sqrt(max(float(np.dot(w, (vals - m) ** 2)), 0.0))
    return (m - 1.0) / den, sd / math.sqrt(shots) / den


def check_circuit(out, backend, mods):
    """(ISA ok, uses a failed element, -ln ESP on the device's Target now, two-qubit count)."""
    t = backend.target
    m = mods["C45"]
    isa = True
    for ins in out.data:
        nm = ins.operation.name
        if nm in ("barrier",):
            continue
        qs = tuple(out.find_bit(q).index for q in ins.qubits)
        if nm not in t.operation_names or qs not in t[nm]:
            isa = False
            break
    edges, bad = m._failed_elements(t, FAILED)
    return isa, bool(m._uses_failed(out, edges, bad)), float(m._st_cost(out, t)), \
        sum(1 for i in out.data if len(i.qubits) == 2 and i.operation.name != "barrier")


def sig(out):
    from c29_identity import value_sig
    return hashlib.sha256(repr(value_sig(out)).encode()).hexdigest()[:16]


def header():
    import qiskit
    print(f"{stamp()} | platform {platform.platform()} | python {platform.python_version()} | qiskit "
          f"{qiskit.__version__}")
    for m in ("qiskit_aer", "qiskit_ibm_runtime", "psf_zero_core"):
        try:
            mod = __import__(m)
            print(f"  {m} {getattr(mod, '__version__', getattr(mod, 'CORE_VERSION', '?'))}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {m} MISSING ({type(exc).__name__})")
    print("SCRIPT", os.path.abspath(__file__), sha256_file(os.path.abspath(__file__)))
    for f in (C45_FILE, C48_FILE, LAYOUT_FILE):
        print("FILE", os.path.relpath(f, REPO), sha256_file(f) if os.path.exists(f) else "MISSING")


# ------------------------------------------------------------------ compiling

def compilers():
    load(LAYOUT_FILE, "psf_smart_layout")
    mods = {"C45": load(C45_FILE, "hw2_c45"), "C48": load(C48_FILE, "hw2_c48")}
    for m in mods.values():
        m.WARN_WITHOUT_TARGET = False
    return mods


def compile_arm(arm, qc, backend, mods):
    """QK3: Qiskit level 3. PSF: c45, PSF-Zero's own pipeline (as HW1). S0 / S1: c48 with item 83 off / on."""
    from qiskit.transpiler import generate_preset_pass_manager
    t0 = time.perf_counter()
    if arm == "QK3":
        out = generate_preset_pass_manager(3, backend, seed_transpiler=0).run(qc)
    else:
        mod = mods["C45" if arm == "PSF" else "C48"]
        if arm in ("S0", "S1"):
            mod.ST_SCHEDULE = arm == "S1"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out = mod.compile_for_hardware(qc, backend=backend, entangling_basis="cx", layout_search=True,
                                               seed_transpiler=0, level3_candidate=False)
        finally:
            if arm in ("S0", "S1"):
                mod.ST_SCHEDULE = True
    return out, time.perf_counter() - t0


def estimates(out, target, mods):
    """(ESP, duration in us, idle-aware estimate) of a compiled circuit on the device's Target. The duration schedules
    every gate as soon as its qubits are free (measurements and barriers left out); the idle-aware estimate multiplies
    the ESP by exp(-sum over active qubits of (T - t_first) (1/(4 T1) + 1/(2 T2))): the Pauli-twirled relaxation and
    dephasing of each qubit from its first gate to the end (HW1's exploratory form, fixed here before the run)."""
    pos = {q: i for i, q in enumerate(out.qubits)}
    clock, first = {}, {}
    for ins in out.data:
        nm = ins.operation.name
        if nm in ("barrier", "measure", "delay"):
            continue
        qs = tuple(pos[q] for q in ins.qubits)
        pr = target[nm].get(qs) if nm in target.operation_names else None
        d = (pr.duration or 0.0) if pr is not None else 0.0
        s = max(clock.get(q, 0.0) for q in qs)
        for q in qs:
            first.setdefault(q, s)
            clock[q] = s + d
    T = max(clock.values()) if clock else 0.0
    ln_idle = 0.0
    for q in clock:
        p = target.qubit_properties[q] if target.qubit_properties is not None else None
        t1 = getattr(p, "t1", None) or MISSING_T_S
        t2 = getattr(p, "t2", None) or MISSING_T_S
        ln_idle += (T - first[q]) * (1 / (4 * t1) + 1 / (2 * t2))
    ln_esp = float(mods["C45"]._st_cost(out, target))
    return math.exp(-ln_esp), T * 1e6, math.exp(-ln_esp - ln_idle)


def build_all(backend, out_dir):
    """Compiles and checks every circuit; returns (manifest entries, pubs, ideal)."""
    from qiskit import qpy
    mods = compilers()
    entries, circs, ideal = [], [], {}
    print(f"{stamp()} compiling on {backend.name}", flush=True)
    plan = [("R", t, ("QK3", "PSF")) for t in TESTS] + [("S", t, ("S0", "S1")) for t in TESTS_S]
    for part, test, arms in plan:
        qc, spec = build_test(test)
        dist, _ = measured_distribution(_unrolled(qc))
        if test not in ideal:
            ideal[test] = dict(value=observable(spec, dist)[0], spec=spec, n=qc.num_qubits,
                               sum_p2=float(sum(v * v for v in dist.values())))
        outs = {arm: compile_arm(arm, qc, backend, mods) for arm in arms}
        if part == "S" and sig(outs["S0"][0]) == sig(outs["S1"][0]):
            print(f"  {test:12s} S0 and S1 identical (item 83 kept the mover's order): pair not submitted", flush=True)
            continue
        for arm, (out, secs) in outs.items():
            isa, failed, cost, n2 = check_circuit(out, backend, mods)
            esp, dur, idle = estimates(out, backend.target, mods)
            d, active = measured_distribution(out)
            exact = None
            if d is not None:
                exact = max(abs(d.get(k, 0.0) - dist.get(k, 0.0)) for k in set(d) | set(dist))
            ok = isa and not failed and (exact is None or exact <= EXACT_TOL)
            print(f"  {part} {test:12s} {arm:4s} 2q {n2:5d} ESP {esp:.4g} dur {dur:6.2f} us idle-est {idle:.4g} "
                  f"active {active:3d} exact {'n/a' if exact is None else f'{exact:.1e}'} isa {isa} failed {failed} "
                  f"({secs:.1f} s){'' if ok else '  -> NOT SUBMITTED'}", flush=True)
            entries.append(dict(part=part, test=test, arm=arm, twoq=n2, neg_ln_esp=cost, esp=esp, dur_us=dur,
                                idle_est=idle, active=active, exact_maxdiff=exact, isa=isa, uses_failed=failed,
                                compile_s=round(secs, 3), submitted=ok, sig=sig(out)))
            if ok:
                circs.append((len(entries) - 1, out))
        if part == "S":  # a pair is scored only if both circuits go
            last = [e for e in entries[-2:]]
            if not all(e["submitted"] for e in last):
                for e in last:
                    if e["submitted"]:
                        e["submitted"] = False
                        circs = [(k, c) for k, c in circs if entries[k] is not e]
                print(f"  {test}: pair incomplete, neither submitted", flush=True)
    # submission order: repeat 1 in one order, repeat 2 in the reverse (drift shows as a repeat difference)
    order = [(k, c, 1) for k, c in circs] + [(k, c, 2) for k, c in circs[::-1]]
    if len(order) > MAX_PUBS:
        raise SystemExit(f"{len(order)} circuits > {MAX_PUBS}: not prepared")
    pubs = [dict(entry=k, repeat=r, shots=SHOTS) for k, _, r in order]
    est = sum(0.0005 * p["shots"] + 1.0 for p in pubs)
    print(f"{len(pubs)} circuits, {sum(p['shots'] for p in pubs)} shots; rough QPU estimate {est:.0f} s")
    if est > USAGE_EST_CAP_S:
        raise SystemExit(f"rough QPU estimate {est:.0f} s > {USAGE_EST_CAP_S:.0f} s: not prepared")
    with open(os.path.join(out_dir, "hw2_circuits.qpy"), "wb") as f:
        qpy.dump([c for _, c, _ in order], f)
    return entries, pubs, ideal


# ------------------------------------------------------------------ steps

def manifest_path(out_dir):
    return os.path.join(out_dir, "hw2_manifest.json")


def write_manifest(out_dir, man):
    with open(manifest_path(out_dir), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, default=str)


def calibration_time(backend):
    try:
        return str(backend.properties().last_update_date)
    except Exception:  # noqa: BLE001
        return None


def _read(a):
    with open(manifest_path(a.out), encoding="utf-8") as f:
        man = json.load(f)
    if sha256_file(os.path.join(a.out, "hw2_circuits.qpy")) != man["qpy_sha256"]:
        raise SystemExit("hw2_circuits.qpy differs from the manifest's hash: stop.")
    return man


def submit(a):
    header()
    man = _read(a)
    if man.get("job_id"):
        raise SystemExit(f"job {man['job_id']} already recorded: not submitting again.")
    from qiskit import qpy
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    with open(os.path.join(a.out, "hw2_circuits.qpy"), "rb") as f:
        circs = qpy.load(f)
    backend = QiskitRuntimeService().backend(man["device"])
    sampler = SamplerV2(mode=backend)
    sampler.options.dynamical_decoupling.enable = False
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    t_sub = stamp()
    if len(circs) != len(man["pubs"]):
        raise SystemExit("the QPY file and the manifest disagree on the number of circuits: stop")
    job = sampler.run([(c, None, p["shots"]) for c, p in zip(circs, man["pubs"])])
    man["job_id"], man["submitted"] = job.job_id(), t_sub
    write_manifest(a.out, man)
    print(f"{stamp()} submitted job {job.job_id()} ({len(circs)} circuits, {sum(p['shots'] for p in man['pubs'])} "
          f"shots) to {man['device']}. Results are not read here.")


def _sample_local(a, man):
    """rehearse only: noiseless local sampling of the prepared circuits."""
    from qiskit import qpy
    from qiskit_aer import AerSimulator
    with open(os.path.join(a.out, "hw2_circuits.qpy"), "rb") as f:
        circs = qpy.load(f)
    sim = AerSimulator()
    rows = []
    for i, (c, p) in enumerate(zip(circs, man["pubs"])):
        counts = sim.run(c, shots=p["shots"], seed_simulator=1000 + i).result().get_counts()
        rows.append(dict(pub=i, counts=counts))
    return rows, dict(usage=dict(quantum_seconds=0), note="local noiseless rehearsal")


def fetch(a):
    header()
    man = _read(a)
    if not man.get("job_id"):
        raise SystemExit("no job recorded")
    from qiskit_ibm_runtime import QiskitRuntimeService
    job = QiskitRuntimeService().job(man["job_id"])
    t_end = time.time() + 60 * a.wait_min
    while True:
        st = str(job.status())
        print(f"{stamp()} job {man['job_id']} status {st}", flush=True)
        if any(x in st.upper() for x in ("DONE", "ERROR", "CANCEL")):
            break
        if time.time() > t_end:
            raise SystemExit("not finished within the wait limit; run fetch again later")
        time.sleep(30)
    if "DONE" not in st.upper():
        raise SystemExit(f"job ended with status {st}")
    res = job.result()
    rows = [dict(pub=i, counts=dict(r.join_data().get_counts())) for i, r in enumerate(res)]
    try:
        metrics = job.metrics()
    except Exception as exc:  # noqa: BLE001
        metrics = dict(error=type(exc).__name__)
    with open(os.path.join(a.out, "hw2_raw_counts.json"), "w", encoding="utf-8") as f:
        json.dump(dict(fetched=stamp(), job_id=man["job_id"], status=st, metrics=metrics, rows=rows), f,
                  default=str)
    print(f"{stamp()} saved hw2_raw_counts.json ({len(rows)} results); usage {metrics.get('usage')}")


def prepare(a, fake=None):
    header()
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(manifest_path(a.out)):
        raise SystemExit(f"{manifest_path(a.out)} exists: already prepared. Not preparing again.")
    if fake is None:
        from qiskit_ibm_runtime import QiskitRuntimeService
        try:
            svc = QiskitRuntimeService()
        except Exception as exc:  # noqa: BLE001
            raise SystemExit("No usable saved IBM account (" + type(exc).__name__ + "). Save it once in the terminal "
                             "with QiskitRuntimeService.save_account(...); never paste the key into a file or chat.")
        rule, cands = [], []
        for k, name in enumerate(DEVICES):
            try:
                b = svc.backend(name)
                st = b.status()
                rule.append(dict(name=name, operational=bool(st.operational), status=st.status_msg,
                                 pending=st.pending_jobs))
                if st.operational:
                    cands.append((st.pending_jobs, k, b))
            except Exception as exc:  # noqa: BLE001
                rule.append(dict(name=name, error=type(exc).__name__))
        print("device rule:", rule)
        if not cands:
            raise SystemExit("Neither ibm_fez nor ibm_marrakesh is operational now: nothing prepared.")
        backend = min(cands, key=lambda c: (c[0], c[1]))[2]
    else:
        from qiskit_ibm_runtime import fake_provider
        backend, rule = getattr(fake_provider, fake)(), [dict(name=fake, fake=True)]
    entries, pubs, ideal = build_all(backend, a.out)
    man = dict(written=stamp(), device=backend.name, device_rule=rule, calibration_time=calibration_time(backend),
               shots=SHOTS, repeats=REPEATS, script_sha256=sha256_file(os.path.abspath(__file__)),
               files={os.path.relpath(f, REPO): sha256_file(f) for f in (C45_FILE, C48_FILE, LAYOUT_FILE)},
               qpy_sha256=sha256_file(os.path.join(a.out, "hw2_circuits.qpy")), entries=entries, pubs=pubs,
               ideal=ideal, params=dict(QAOA_GAMMA=QAOA_GAMMA, QAOA_BETA=QAOA_BETA, TFIM=[TFIM_J, TFIM_H, TFIM_T,
                                                                                          TFIM_REPS]),
               job_id=None)
    write_manifest(a.out, man)
    print(f"{stamp()} manifest written before any submission: {len(pubs)} circuits; device {backend.name}; "
          f"calibration {man['calibration_time']}")
    return backend


def _ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0] * len(v)
    for k, i in enumerate(o):
        r[i] = k
    return r


def _rho(x, y):
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def score(a):
    man = _read(a)
    with open(os.path.join(a.out, "hw2_raw_counts.json"), encoding="utf-8") as f:
        raw = json.load(f)
    ent = man["entries"]
    by = {}
    for p, row in zip(man["pubs"], raw["rows"]):
        e = ent[p["entry"]]
        by.setdefault((e["part"], e["test"], e["arm"]), []).append((p["repeat"], row["counts"]))
    out = [f"# HW2 scores ({stamp()})", "",
           f"device {man['device']}; calibration {man['calibration_time']}; job {man.get('job_id')}; usage "
           f"{(raw.get('metrics') or {}).get('usage')}", "",
           "F: normalised linear XEB against the ideal distribution, mean of the two repeats.", "",
           "| part | test | arm | 2q | ESP | duration us | idle-est | F rep 1 | F rep 2 | F mean (SE) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    dists, R = {}, {}
    for e in ent:
        if not e["submitted"]:
            continue
        t = e["test"]
        if t not in dists:
            qc, _ = build_test(t)
            d, _ = measured_distribution(_unrolled(qc))
            if abs(sum(v * v for v in d.values()) - man["ideal"][t]["sum_p2"]) > 1e-9:
                raise SystemExit(f"ideal distribution of {t} differs from the manifest's: stop")
            dists[t] = d
        n = man["ideal"][t]["n"]
        reps = sorted(by.get((e["part"], t, e["arm"]), []))
        fs = [xeb(dists[t], c, n, sum(c.values())) for _, c in reps]
        fm = float(np.mean([f for f, _ in fs])) if fs else float("nan")
        fse = math.sqrt(sum(s_ ** 2 for _, s_ in fs)) / max(len(fs), 1)
        R[(e["part"], t, e["arm"])] = dict(f=fm, se=fse, reps=[f for f, _ in fs], esp=e["esp"], dur=e["dur_us"],
                                           idle=e["idle_est"], twoq=e["twoq"], sig=e["sig"])
        out.append(f"| {e['part']} | {t} | {e['arm']} | {e['twoq']} | {e['esp']:.4g} | {e['dur_us']:.2f} | "
                   f"{e['idle_est']:.4g} | " + " | ".join(f"{x:.4f}" for x in ([f for f, _ in fs] + [float('nan')] * 2)[:2])
                   + f" | {fm:.4f} ({fse:.4f}) |")

    def pair(part, t, a1, a0):
        x, y = R.get((part, t, a1)), R.get((part, t, a0))
        if x is None or y is None:
            return None
        fr = x["f"] / y["f"] if x["f"] > 0 and y["f"] > 0 else float("nan")
        return dict(test=t, esp=x["esp"] / y["esp"], idle=x["idle"] / y["idle"], dur=x["dur"] / y["dur"], f=fr,
                    f1=x["f"], f0=y["f"], drift=max(abs(x["reps"][0] - x["reps"][-1]), abs(y["reps"][0] - y["reps"][-1])),
                    same=x["sig"] == y["sig"])

    # ---------------------------------------------------------------- part R
    rp = {t: pair("R", t, "PSF", "QK3") for t in TESTS}
    out += ["", "## Part R: PSF / QK3", "", "| test | ESP ratio | idle-est ratio | duration ratio | F ratio | F PSF | F QK3 | "
            "largest repeat difference |", "|---|---|---|---|---|---|---|---|"]
    for t, x in rp.items():
        if x is not None:
            out.append(f"| {t} | {x['esp']:.3f} | {x['idle']:.3f} | {x['dur']:.3f} | {x['f']:.3f} | {x['f1']:.4f} | "
                       f"{x['f0']:.4f} | {x['drift']:.4f} |")
    rd = [x for t, x in rp.items() if t in READING and x is not None and not math.isnan(x["f"])]
    verdict = {}
    big = [x for x in rd if x["esp"] >= 1.2]
    wins = sum(1 for x in big if x["f1"] >= x["f0"])
    verdict["R1"] = "NOT TESTABLE" if not big else ("CONFIRMED" if 2 * wins <= len(big) else "REFUTED")
    rho_e = _rho([math.log(x["esp"]) for x in rd], [math.log(x["f"]) for x in rd])
    rho_i = _rho([math.log(x["idle"]) for x in rd], [math.log(x["f"]) for x in rd])
    verdict["R2"] = "CONFIRMED" if rho_e <= 0.3 else "REFUTED"
    agree_e = sum(1 for x in rd if (x["esp"] > 1) == (x["f"] > 1))
    agree_i = sum(1 for x in rd if (x["idle"] > 1) == (x["f"] > 1))
    need = math.ceil(5 * len(rd) / 7)
    verdict["R3"] = "CONFIRMED" if (agree_i >= need and rho_i > rho_e) else "REFUTED"
    r4 = [x for t, x in rp.items() if t in R4_TESTS and x is not None and not math.isnan(x["f"])]
    short = [x for x in r4 if x["dur"] <= 0.8]
    verdict["R4"] = "NOT TESTABLE" if len(short) < 2 else ("CONFIRMED" if all(x["f1"] >= x["f0"] for x in short)
                                                          else "REFUTED")
    ba = [x for t, x in rp.items() if t in BA and x is not None and not math.isnan(x["f"])]
    ba_ok = sum(1 for x in ba if x["f1"] <= 1.05 * x["f0"])
    verdict["R5"] = "CONFIRMED" if ba_ok >= 3 else "REFUTED"
    g = rp.get("ghz_n12")
    verdict["R6"] = "n/a" if g is None or math.isnan(g["f"]) else (
        ("CONFIRMED" if abs(g["f"] - 1.0) <= 0.10 else "REFUTED") + ("" if g["same"] else " (circuits differ)"))
    replicated = all(verdict[k] == "CONFIRMED" for k in ("R1", "R3", "R4"))
    out += ["", "### Predictions, part R", "",
            f"- R1 (ESP >= 1.2 x QK3 does not give the higher F on more than half): {len(big)} tests, PSF's F >= QK3's on "
            f"{wins} -> {verdict['R1']}",
            f"- R2 (ESP does not rank the pair: rank correlation <= 0.3): {rho_e:+.3f} over {len(rd)} -> {verdict['R2']}",
            f"- R3 (the idle-aware estimate: same winner on >= {need} of {len(rd)} and rank correlation above ESP's): "
            f"same winner {agree_i} (ESP {agree_e}); rank correlation {rho_i:+.3f} (ESP {rho_e:+.3f}) -> {verdict['R3']}",
            f"- R4 (where PSF's duration <= 0.8 x QK3's, PSF's F >= QK3's): {len(short)} tests "
            f"({', '.join(x['test'] for x in short)}); -> {verdict['R4']}",
            f"- R5 (Barabasi-Albert: PSF's F <= 1.05 x QK3's on >= 3 of 4): {ba_ok} of {len(ba)} -> {verdict['R5']}",
            f"- R6 (GHZ control: |F ratio - 1| <= 0.10): {verdict['R6']}",
            f"- **Replication (R1, R3 and R4 all confirmed): {'REPLICATED' if replicated else 'NOT REPLICATED'}**"]
    if "tfim_n16" in rp and rp["tfim_n16"] is not None:
        out.append(f"- tfim_n16 (reported, not read: XEB failed there in HW1): F PSF {rp['tfim_n16']['f1']:.4f}, "
                   f"QK3 {rp['tfim_n16']['f0']:.4f}")
    # ---------------------------------------------------------------- part S
    sp = {t: pair("S", t, "S1", "S0") for t in TESTS_S}
    out += ["", "## Part S: S1 (item 83 on) / S0 (off), the same gate blocks", "",
            "| test | 2q S1 / S0 | ESP ratio | idle-est ratio | duration ratio | F ratio | F S1 | F S0 | largest repeat "
            "difference |", "|---|---|---|---|---|---|---|---|---|"]
    for t, x in sp.items():
        if x is not None:
            out.append(f"| {t} | {R[('S', t, 'S1')]['twoq']} / {R[('S', t, 'S0')]['twoq']} | {x['esp']:.3f} | "
                       f"{x['idle']:.3f} | {x['dur']:.3f} | {x['f']:.3f} | {x['f1']:.4f} | {x['f0']:.4f} | "
                       f"{x['drift']:.4f} |")
    tp = [x for t, x in sp.items() if t.startswith("tfimP") and x is not None and not math.isnan(x["f"])]
    up = sum(1 for x in tp if x["f1"] > x["f0"])
    verdict["S1"] = "NOT TESTABLE" if not tp else ("CONFIRMED" if up == len(tp) else ("REFUTED" if up == 0 else "AMBIGUOUS"))
    gm = math.exp(float(np.mean([math.log(x["f"]) for x in tp]))) if tp else float("nan")
    verdict["S2"] = "NOT TESTABLE" if not tp else ("CONFIRMED" if gm >= 1.05 else "REFUTED")
    out += ["", "### Predictions, part S", "",
            f"- S1 (TFIM from a product state: S1's F > S0's on every pair): {up} of {len(tp)} -> {verdict['S1']}",
            f"- S2 (TFIM: geometric mean F ratio S1 / S0 >= 1.05): {gm:.3f} -> {verdict['S2']}",
            "- S3 (reported): QAOA pairs: " + ("; ".join(f"{x['test']} duration {x['dur']:.3f}, F ratio {x['f']:.3f}"
                                                         for t, x in sp.items() if x is not None and t.startswith("qaoa"))
                                              or "none submitted"),
            "- S4 (reported): which estimate picked the device's winner on each pair: " + "; ".join(
                f"{x['test']}: ESP {'yes' if (x['esp'] > 1) == (x['f'] > 1) else 'no'}, idle-est "
                f"{'yes' if (x['idle'] > 1) == (x['f'] > 1) else 'no'}" for x in sp.values()
                if x is not None and not math.isnan(x["f"]))]
    fe = {}
    for (part, t, arm), v in R.items():
        if v["f"] > 0 and t != "tfim_n16":
            fe.setdefault(arm, []).append(math.log(v["f"] / v["esp"]))
    out.append("- reported: F / ESP per arm (geometric mean): " + ", ".join(f"{k} {math.exp(np.mean(v)):.3f}"
                                                                         for k, v in fe.items()))
    txt = "\n".join(out) + "\n"
    with open(os.path.join(a.out, "hw2_score.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    with open(os.path.join(a.out, "hw2_score.json"), "w", encoding="utf-8") as f:
        json.dump(dict(F={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in R.items()}, part_r=rp, part_s=sp, verdict=verdict,
                       rho_esp=rho_e, rho_idle=rho_i, replicated=replicated), f, indent=1)
    print(txt)


def rehearse(a):
    warnings.simplefilter("ignore")
    prepare(a, fake=a.fake)
    man = _read(a)
    rows, metrics = _sample_local(a, man)
    with open(os.path.join(a.out, "hw2_raw_counts.json"), "w", encoding="utf-8") as f:
        json.dump(dict(fetched=stamp(), job_id="rehearsal", status="DONE (local)", metrics=metrics, rows=rows), f)
    man["job_id"] = "rehearsal"
    write_manifest(a.out, man)
    score(a)
    print("REHEARSAL DONE (noiseless: every F should be 1 within shot noise; the verdicts mean nothing here)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("rehearse", "prepare", "submit", "fetch", "score"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--fake", default="FakeFez")
    ap.add_argument("--wait-min", type=float, default=40.0)
    a = ap.parse_args()
    if a.step == "prepare":
        prepare(a)
    else:
        dict(rehearse=rehearse, submit=submit, fetch=fetch, score=score)[a.step](a)


if __name__ == "__main__":
    main()
