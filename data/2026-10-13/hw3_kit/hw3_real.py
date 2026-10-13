"""hw3_real.py -- HW3: the same compiled circuits with and without dynamical decoupling, to test why the device was more
sensitive to time than its T1/T2 explain (Addenda 455, 462).

Pre-registered in HW3_preregistration.md (fixed before any real-device step). One job. Three pairs of circuits, each a
shorter and a longer circuit of the same test:

  tfimP_n12    c48 with item 83 on (S1) and off (S0): the same gate blocks, about 5 times apart in duration
  tfim_n12     PSF-Zero (c45, level3_candidate=False) and Qiskit level 3: HW1's 1.20 win of the shorter circuit
  qaoa_n20_p2  the same two arms: HW1's other 1.20 win of the shorter circuit

Every compiled circuit is submitted in three variants:

  plain  as compiled (the device schedules it itself; HW1 and HW2 ran so)
  pad    scheduled as late as possible (ALAP) on the device's Target, idle windows filled with delays
  dd     the same schedule, idle windows of the circuit's active qubits filled with X-X dynamical decoupling

'pad' and 'dd' differ only by the decoupling pulses; 'plain' and 'pad' only by who schedules (reported).

Steps (only `submit` uses QPU time; `prepare`, `submit` and `fetch` need the IBM connection, the others do not):

  rehearse  the whole flow on a fake device (default FakeFez) with a noiseless local simulator. No IBM connection.
  prepare   device rule (of ibm_fez and ibm_marrakesh, the operational one with fewer pending jobs; neither: stop);
            compiles, makes the variants, checks every circuit (ISA, failed elements, exactness by noiseless
            simulation), writes the circuits and the manifest BEFORE any submission. Refuses if the manifest exists.
  submit    ONE SamplerV2 job, 6,000 shots per circuit, the runtime's own decoupling and twirling off.
  fetch     waits for the job and saves the raw counts.
  score     offline: scores the predictions.

Usage (from the psf-zero repository root):
    python -u <kit>/hw3_real.py rehearse --out data/<date>/hw3_rehearsal
    python -u <kit>/hw3_real.py prepare  --out data/<date>/hw3
    python -u <kit>/hw3_real.py submit   --out data/<date>/hw3
    python -u <kit>/hw3_real.py fetch    --out data/<date>/hw3
    python -u <kit>/hw3_real.py score    --out data/<date>/hw3

The IBM API key is never read from or written to any file by this script. This script runs no git and no pip.
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
USAGE_EST_CAP_S = 200.0  # prepare refuses a job whose rough QPU estimate (0.5 ms per shot + 1 s per circuit) exceeds this
DEVICES = ("ibm_fez", "ibm_marrakesh")  # the device rule: the operational one with fewer pending jobs (ties: this order)
FAILED = 0.5
EXACT_TOL = 1e-6
EXACT_MAX_ACTIVE = 24
MAX_PUBS = 60
MISSING_T_S = 1e-6
C45_FILE = os.path.join(REPO, "patches", "psf_compile_c45_2026-10-12", "psf_compile.py")
C48_FILE = os.path.join(REPO, "patches", "psf_compile_c48_2026-10-12", "psf_compile.py")
LAYOUT_FILE = os.path.join(REPO, "benchmarks", "psf_smart_layout.py")
PAIRS = (("tfimP_n12", "S0", "S1"), ("tfim_n12", "QK3", "PSF"), ("qaoa_n20_p2", "QK3", "PSF"))
VARIANTS = ("plain", "pad", "dd")
KIT = os.path.dirname(os.path.abspath(__file__))  # Benchpress's QAOA files travel with the kit in qasm/ (unused here)
QAOA_GAMMA, QAOA_BETA = 0.55, 0.45
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
    mods = {"C45": load(C45_FILE, "hw3_c45"), "C48": load(C48_FILE, "hw3_c48")}
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


def variant(out, target, kind):
    """The 'plain' circuit itself, or its ALAP schedule padded with delays ('pad') or with X-X decoupling on the
    circuit's active qubits ('dd'); the decoupling pulses and delays act as the identity."""
    if kind == "plain":
        return out
    from qiskit.circuit.library import XGate
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay, PadDynamicalDecoupling
    if kind == "pad":
        return PassManager([ALAPScheduleAnalysis(target=target), PadDelay(target=target)]).run(out)
    active = sorted({out.find_bit(q).index for i in out.data if i.operation.name not in ("barrier",)
                     for q in i.qubits})
    return PassManager([ALAPScheduleAnalysis(target=target),
                        PadDynamicalDecoupling(target=target, dd_sequence=[XGate(), XGate()], qubits=active)]).run(out)


def n_pulses(c):
    return sum(1 for i in c.data if i.operation.name == "x")


def build_all(backend, out_dir):
    """Compiles, makes the variants and checks every circuit; returns (manifest entries, pubs, ideal)."""
    from qiskit import qpy
    mods = compilers()
    entries, circs, ideal = [], [], {}
    t = backend.target
    print(f"{stamp()} compiling on {backend.name}", flush=True)
    for test, long_arm, short_arm in PAIRS:
        qc, spec = build_test(test)
        dist, _ = measured_distribution(_unrolled(qc))
        ideal[test] = dict(value=observable(spec, dist)[0], spec=spec, n=qc.num_qubits,
                           sum_p2=float(sum(v * v for v in dist.values())))
        group = []
        for arm in (long_arm, short_arm):
            out, secs = compile_arm(arm, qc, backend, mods)
            esp, dur, idle = estimates(out, t, mods)
            for kind in VARIANTS:
                c = variant(out, t, kind)
                isa, failed, cost, n2 = check_circuit(c, backend, mods)
                d, active = measured_distribution(c)
                exact = None if d is None else max(abs(d.get(k, 0.0) - dist.get(k, 0.0)) for k in set(d) | set(dist))
                ok = isa and not failed and (exact is None or exact <= EXACT_TOL)
                print(f"  {test:12s} {arm:4s} {kind:5s} 2q {n2:4d} x-pulses {n_pulses(c):4d} ESP {esp:.4g} dur "
                      f"{dur:6.2f} us idle-est {idle:.4g} exact {'n/a' if exact is None else f'{exact:.1e}'} isa {isa} "
                      f"failed {failed}{'' if ok else '  -> NOT SUBMITTED'}", flush=True)
                entries.append(dict(test=test, arm=arm, variant=kind, twoq=n2, pulses=n_pulses(c), esp=esp,
                                    dur_us=dur, idle_est=idle, active=active, exact_maxdiff=exact, isa=isa,
                                    uses_failed=failed, compile_s=round(secs, 3), submitted=ok, sig=sig(c)))
                group.append((len(entries) - 1, c))
        if all(entries[k]["submitted"] for k, _ in group):
            circs += group
        else:
            for k, _ in group:
                entries[k]["submitted"] = False
            print(f"  {test}: a circuit failed a check; the whole pair is not submitted", flush=True)
    order = [(k, c, 1) for k, c in circs] + [(k, c, 2) for k, c in circs[::-1]]
    if len(order) > MAX_PUBS:
        raise SystemExit(f"{len(order)} circuits > {MAX_PUBS}: not prepared")
    pubs = [dict(entry=k, repeat=r, shots=SHOTS) for k, _, r in order]
    est = sum(0.0005 * p["shots"] + 1.0 for p in pubs)
    print(f"{len(pubs)} circuits, {sum(p['shots'] for p in pubs)} shots; rough QPU estimate {est:.0f} s")
    if est > USAGE_EST_CAP_S:
        raise SystemExit(f"rough QPU estimate {est:.0f} s > {USAGE_EST_CAP_S:.0f} s: not prepared")
    with open(os.path.join(out_dir, "hw3_circuits.qpy"), "wb") as f:
        qpy.dump([c for _, c, _ in order], f)
    return entries, pubs, ideal


# ------------------------------------------------------------------ steps

def manifest_path(out_dir):
    return os.path.join(out_dir, "hw3_manifest.json")


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
    if sha256_file(os.path.join(a.out, "hw3_circuits.qpy")) != man["qpy_sha256"]:
        raise SystemExit("hw3_circuits.qpy differs from the manifest's hash: stop.")
    return man


def submit(a):
    header()
    man = _read(a)
    if man.get("job_id"):
        raise SystemExit(f"job {man['job_id']} already recorded: not submitting again.")
    from qiskit import qpy
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    with open(os.path.join(a.out, "hw3_circuits.qpy"), "rb") as f:
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
    with open(os.path.join(a.out, "hw3_circuits.qpy"), "rb") as f:
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
    with open(os.path.join(a.out, "hw3_raw_counts.json"), "w", encoding="utf-8") as f:
        json.dump(dict(fetched=stamp(), job_id=man["job_id"], status=st, metrics=metrics, rows=rows), f,
                  default=str)
    print(f"{stamp()} saved hw3_raw_counts.json ({len(rows)} results); usage {metrics.get('usage')}")


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
               qpy_sha256=sha256_file(os.path.join(a.out, "hw3_circuits.qpy")), entries=entries, pubs=pubs,
               ideal=ideal, params=dict(QAOA_GAMMA=QAOA_GAMMA, QAOA_BETA=QAOA_BETA, TFIM=[TFIM_J, TFIM_H, TFIM_T,
                                                                                          TFIM_REPS]),
               job_id=None)
    write_manifest(a.out, man)
    print(f"{stamp()} manifest written before any submission: {len(pubs)} circuits; device {backend.name}; "
          f"calibration {man['calibration_time']}")
    return backend


def score(a):
    man = _read(a)
    with open(os.path.join(a.out, "hw3_raw_counts.json"), encoding="utf-8") as f:
        raw = json.load(f)
    ent = man["entries"]
    by = {}
    for p, row in zip(man["pubs"], raw["rows"]):
        e = ent[p["entry"]]
        by.setdefault((e["test"], e["arm"], e["variant"]), []).append((p["repeat"], row["counts"]))
    out = [f"# HW3 scores ({stamp()})", "",
           f"device {man['device']}; calibration {man['calibration_time']}; job {man.get('job_id')}; usage "
           f"{(raw.get('metrics') or {}).get('usage')}", "",
           "F: normalised linear XEB against the ideal distribution, mean of the two repeats.", "",
           "| test | arm | variant | 2q | X pulses | duration us (as compiled) | F rep 1 | F rep 2 | F mean (SE) |",
           "|---|---|---|---|---|---|---|---|---|"]
    F, dists = {}, {}
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
        reps = sorted(by.get((t, e["arm"], e["variant"]), []))
        fs = [xeb(dists[t], c, n, sum(c.values())) for _, c in reps]
        fm = float(np.mean([f for f, _ in fs])) if fs else float("nan")
        fse = math.sqrt(sum(s_ ** 2 for _, s_ in fs)) / max(len(fs), 1)
        F[(t, e["arm"], e["variant"])] = dict(f=fm, se=fse, reps=[f for f, _ in fs], dur=e["dur_us"])
        out.append(f"| {t} | {e['arm']} | {e['variant']} | {e['twoq']} | {e['pulses']} | {e['dur_us']:.2f} | "
                   + " | ".join(f"{x:.4f}" for x in ([f for f, _ in fs] + [float('nan')] * 2)[:2])
                   + f" | {fm:.4f} ({fse:.4f}) |")
    rows = []
    for test, a_long, a_short in PAIRS:
        if any(F.get((test, arm, k)) is None for arm in (a_long, a_short) for k in VARIANTS):
            continue
        dl, ds = F[(test, a_long, "plain")]["dur"], F[(test, a_short, "plain")]["dur"]
        if ds > dl:  # 'short' is the shorter as compiled on this device
            a_long, a_short = a_short, a_long
        f = lambda arm, k: F[(test, arm, k)]["f"]  # noqa: E731
        if min(f(a_long, k) for k in VARIANTS) <= 0 or min(f(a_short, k) for k in VARIANTS) <= 0:
            rows.append(dict(test=test, skip=True))
            continue
        r = {k: f(a_short, k) / f(a_long, k) for k in VARIANTS}
        rows.append(dict(test=test, short=a_short, long=a_long, r=r,
                         gain_long=f(a_long, "dd") / f(a_long, "pad"), gain_short=f(a_short, "dd") / f(a_short, "pad"),
                         drift=max(abs(F[(test, arm, k)]["reps"][0] - F[(test, arm, k)]["reps"][-1])
                                   for arm in (a_long, a_short) for k in VARIANTS)))
    ok = [x for x in rows if not x.get("skip")]
    out += ["", "| test | shorter / longer | ratio plain | ratio pad | ratio dd | DD gain longer | DD gain shorter | largest repeat difference |",
            "|---|---|---|---|---|---|---|---|"]
    for x in ok:
        out.append(f"| {x['test']} | {x['short']} / {x['long']} | {x['r']['plain']:.3f} | {x['r']['pad']:.3f} | "
                   f"{x['r']['dd']:.3f} | {x['gain_long']:.3f} | {x['gain_short']:.3f} | {x['drift']:.4f} |")
    # Version 2 (Addendum 469): D1, D2 select on repeat 1 and are judged on repeat 2 only, so that the noise that lets
    # a pair in is not the noise that is then compared (the workplace's review, section 3). A pair is selected if,
    # under pad in repeat 1, the shorter circuit's F exceeds the longer's by more than the pair's largest repeat
    # difference of the plain variant (separate circuits, independent of pad).
    def rep(test, arm, k, i):
        rs = F[(test, arm, k)]["reps"]
        return rs[i] if len(rs) > i else float("nan")
    for x in ok:
        t_, s_, l_ = x["test"], x["short"], x["long"]
        x["drift_plain"] = max(abs(rep(t_, arm, "plain", 0) - rep(t_, arm, "plain", 1)) for arm in (s_, l_))
        x["sel"] = rep(t_, s_, "pad", 0) - rep(t_, l_, "pad", 0) > x["drift_plain"]
        v2 = {(arm, k): rep(t_, arm, k, 1) for arm in (s_, l_) for k in ("pad", "dd")}
        x["rep2_ok"] = all(val > 0 for val in v2.values())
        if x["rep2_ok"]:
            x["r2"] = {k: v2[(s_, k)] / v2[(l_, k)] for k in ("pad", "dd")}
            x["gain2_long"], x["gain2_short"] = v2[(l_, "dd")] / v2[(l_, "pad")], v2[(s_, "dd")] / v2[(s_, "pad")]
    eff = [x for x in ok if x["sel"] and x["rep2_ok"]]
    d1 = sum(1 for x in eff if math.log(x["r2"]["dd"]) < math.log(x["r2"]["pad"]))
    d1abs = sum(1 for x in eff if abs(math.log(x["r2"]["dd"])) < abs(math.log(x["r2"]["pad"])))
    d2 = sum(1 for x in eff if x["gain2_long"] > x["gain2_short"])
    d3 = sum(1 for x in ok if x["gain_long"] > 1.0)
    need = len(eff) if len(eff) < 3 else 2
    v = dict(D1="NOT TESTABLE" if len(eff) < 2 else ("CONFIRMED" if d1 >= need else "REFUTED"),
             D2="NOT TESTABLE" if len(eff) < 2 else ("CONFIRMED" if d2 >= need else "REFUTED"),
             D3="NOT TESTABLE" if not ok else ("CONFIRMED" if d3 == len(ok) else "REFUTED"))
    out += ["", "### Predictions (version 2: selected on repeat 1, judged on repeat 2)", "",
            f"- pairs where time mattered without decoupling (pad, repeat 1: shorter above longer by more than the "
            f"plain variant's repeat difference): {len(eff)} of {len(ok)} "
            f"({', '.join(x['test'] for x in eff) or 'none'})",
            "- repeat 2 on those pairs: " + ("; ".join(
                f"{x['test']} ratio pad {x['r2']['pad']:.3f}, dd {x['r2']['dd']:.3f}, DD gain longer "
                f"{x['gain2_long']:.3f}, shorter {x['gain2_short']:.3f}" for x in eff) or "none"),
            f"- D1 (on those pairs, decoupling shrinks the shorter circuit's advantage, ln ratio dd < ln ratio pad, "
            f"repeat 2, on >= {need} of {len(eff)}): {d1} -> {v['D1']}",
            f"- D1, reported (the first version's form, |ln ratio dd| < |ln ratio pad|, repeat 2): {d1abs} of {len(eff)}",
            f"- D2 (on those pairs, decoupling helps the longer circuit more than the shorter, repeat 2, on >= {need} "
            f"of {len(eff)}): {d2} -> {v['D2']}",
            f"- D3 (decoupling raises the longer circuit's F, every pair): {d3} of {len(ok)} -> {v['D3']}",
            "- D4 (reported): F plain / F pad per circuit (the device's own scheduling against ALAP): " + "; ".join(
                f"{t} {arm} {F[(t, arm, 'plain')]['f'] / F[(t, arm, 'pad')]['f']:.3f}" for (t, arm, k) in F
                if k == "plain" and F[(t, arm, "pad")]["f"] > 0),
            "- D5 (reported): does the shorter circuit still win with decoupling (ratio dd > 1): "
            + "; ".join(f"{x['test']} {x['r']['dd']:.3f}" for x in ok)]
    txt = "\n".join(out) + "\n"
    with open(os.path.join(a.out, "hw3_score.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    with open(os.path.join(a.out, "hw3_score.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(F={f"{k[0]}|{k[1]}|{k[2]}": x for k, x in F.items()}, pairs=rows, verdict=v), fh, indent=1)
    print(txt)


def rehearse(a):
    warnings.simplefilter("ignore")
    prepare(a, fake=a.fake)
    man = _read(a)
    rows, metrics = _sample_local(a, man)
    with open(os.path.join(a.out, "hw3_raw_counts.json"), "w", encoding="utf-8") as f:
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
