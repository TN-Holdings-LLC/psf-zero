"""hw1_real.py -- HW1 (2026-10-12): does the ESP advantage PSF-Zero predicts on fake devices hold on a real IBM device?
And PSF-XOR: the PennyLane-trained XOR classifier (Stage 2's circuits) compiled by PSF-Zero on the real device.

Pre-registered in HW1_preregistration.md (fixed before any real-device step; sha256 and CEST time kept on the owner's
PC). One job: every HW1 circuit compiled by Qiskit level 3 (QK3) and by PSF-Zero's own pipeline (PSF: prototype c45
with level3_candidate=False), each submitted twice (two repeats, interleaved); prototype c47 (item 82) as a third arm
only where its circuit differs from PSF's; and the four XOR circuits compiled by release 2026-10-11.4 (R4) and c45.

Steps (only `submit` uses QPU time; `prepare`, `submit` and `fetch` need the IBM connection, the others do not):

  rehearse  the whole flow on a fake device (default FakeKingston) with a noiseless local simulator: compiles every
            arm, checks every circuit, samples, scores. No IBM connection, no QPU.
  prepare   device rule (ibm_kingston if operational, else ibm_fez, else stop); the device's Target now; compiles every
            arm; checks every circuit (ISA, failed elements, exactness by noiseless simulation); writes the circuits
            (QPY) and the manifest (device, calibration time, predicted ESP of every circuit, ideal values) BEFORE any
            submission. Refuses to run if the manifest exists.
  submit    submits ONE SamplerV2 job with the prepared circuits, 4,000 shots each, dynamical decoupling and twirling
            off; records the job ID. Does not read results. Refuses if a job ID is already recorded.
  fetch     waits for the job (at most --wait-min minutes) and saves the raw counts and the job's metrics. Does not
            score.
  score     offline: reads the manifest and the raw counts, scores HW1's and PSF-XOR's predictions.

Usage (from the psf-zero repository root):
    python -u <kit>/hw1_real.py rehearse --out data/2026-10-12/hw1_rehearsal
    python -u <kit>/hw1_real.py prepare  --out data/2026-10-12/hw1
    python -u <kit>/hw1_real.py submit   --out data/2026-10-12/hw1
    python -u <kit>/hw1_real.py fetch    --out data/2026-10-12/hw1
    python -u <kit>/hw1_real.py score    --out data/2026-10-12/hw1

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

SHOTS = 6000  # each HW1 circuit, per repeat
XOR_SHOTS = 4000  # each XOR circuit, as Stage 2
REPEATS = 2
USAGE_EST_CAP_S = 240.0  # prepare refuses a job whose rough QPU estimate (0.5 ms per shot + 1 s per circuit) exceeds this
DEVICE_ORDER = ("ibm_kingston", "ibm_fez")  # the device rule: the first operational one
FAILED = 0.5
EXACT_TOL = 1e-6
EXACT_MAX_ACTIVE = 24  # noiseless exactness check by statevector on at most this many active qubits
MAX_PUBS = 80
R4_FILE = os.path.join(REPO, "psf_compile.py")
C45_FILE = os.path.join(REPO, "patches", "psf_compile_c45_2026-10-12", "psf_compile.py")
C47_FILE = os.path.join(REPO, "patches", "psf_compile_c47_2026-10-12", "psf_compile.py")
LAYOUT_FILE = os.path.join(REPO, "benchmarks", "psf_smart_layout.py")
TESTS = ("qaoa_n12_p2", "qaoa_n20_p2", "ba_n14", "ba_n16", "ba_n18", "ba_n20", "tfim_n12", "tfim_n16", "ghz_n12")
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


def header():
    import qiskit
    print(f"{stamp()} | platform {platform.platform()} | python {platform.python_version()} | qiskit "
          f"{qiskit.__version__}")
    for m in ("qiskit_aer", "qiskit_ibm_runtime", "pennylane", "psf_zero_core"):
        try:
            mod = __import__(m)
            print(f"  {m} {getattr(mod, '__version__', getattr(mod, 'CORE_VERSION', '?'))}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {m} MISSING ({type(exc).__name__})")
    print("SCRIPT", os.path.abspath(__file__), sha256_file(os.path.abspath(__file__)))
    for f in (R4_FILE, C45_FILE, C47_FILE, LAYOUT_FILE):
        print("FILE", os.path.relpath(f, REPO), sha256_file(f) if os.path.exists(f) else "MISSING")


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


# ------------------------------------------------------------------ compiling

def compilers():
    load(LAYOUT_FILE, "psf_smart_layout")
    mods = {"R4": load(R4_FILE, "hw1_r4"), "C45": load(C45_FILE, "hw1_c45"), "C47": load(C47_FILE, "hw1_c47")}
    for m in mods.values():
        m.WARN_WITHOUT_TARGET = False
    return mods


def compile_arm(arm, qc, backend, mods):
    from qiskit.transpiler import generate_preset_pass_manager
    t0 = time.perf_counter()
    if arm == "QK3":
        out = generate_preset_pass_manager(3, backend, seed_transpiler=0).run(qc)
    else:
        mod = mods["C45" if arm == "PSF" else arm]
        kw = dict(level3_candidate=False) if arm in ("PSF", "C47") else {}
        with contextlib.redirect_stdout(io.StringIO()):
            out = mod.compile_for_hardware(qc, backend=backend, entangling_basis="cx", layout_search=True,
                                           seed_transpiler=0, **kw)
    return out, time.perf_counter() - t0


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


def build_all(backend, out_dir):
    """Compiles and checks every circuit; returns (pubs, manifest entries)."""
    from qiskit import qpy
    mods = compilers()
    entries, circs = [], []
    ideal = {}
    print(f"{stamp()} compiling on {backend.name}", flush=True)
    for test in TESTS:
        qc, spec = build_test(test)
        dist, _ = measured_distribution(_unrolled(qc))
        ival = observable(spec, dist)[0]
        ideal[test] = dict(value=ival, spec=spec, n=qc.num_qubits, sum_p2=float(sum(v * v for v in dist.values())))
        outs = {}
        for arm in ("QK3", "PSF", "C47"):
            out, secs = compile_arm(arm, qc, backend, mods)
            if arm == "C47" and sig(out) == sig(outs["PSF"][0]):
                print(f"  {test:12s} C47 identical to PSF: not submitted", flush=True)
                continue
            outs[arm] = (out, secs)
        for arm, (out, secs) in outs.items():
            isa, failed, cost, n2 = check_circuit(out, backend, mods)
            d, active = measured_distribution(out)
            exact = None
            if d is not None:
                keys = set(d) | set(dist)
                exact = max(abs(d.get(k, 0.0) - dist.get(k, 0.0)) for k in keys)
            ok = isa and not failed and (exact is None or exact <= EXACT_TOL)
            print(f"  {test:12s} {arm:4s} 2q {n2:5d} ESP {math.exp(-cost):.4g} active {active:3d} exact "
                  f"{'n/a' if exact is None else f'{exact:.1e}'} isa {isa} failed {failed} ({secs:.1f} s)"
                  f"{'' if ok else '  -> NOT SUBMITTED'}", flush=True)
            entries.append(dict(part="HW1", test=test, arm=arm, twoq=n2, neg_ln_esp=cost, esp=math.exp(-cost),
                                active=active, exact_maxdiff=exact, isa=isa, uses_failed=failed,
                                compile_s=round(secs, 3), submitted=ok, sig=sig(out)))
            if ok:
                circs.append((len(entries) - 1, out))
    # PSF-XOR: Stage 2's PennyLane-trained circuits, compiled by R4 and C45
    import xor_prereg_stage1_sweep as S1
    import xor_prereg_stage1b_sweep as S1b
    xc = S1.build_logical_circuits()
    S1.validate_circuits(xc)
    labels = [int(x) for x in S1.R.XOR_LABELS]
    inputs = ["".join(str(x) for x in b) for b in S1.R.XOR_INPUTS]
    try:
        from qiskit_aer import AerSimulator
        sim = AerSimulator.from_backend(backend)
    except Exception as exc:  # noqa: BLE001
        sim = None
        print("Aer noise model unavailable:", type(exc).__name__)
    for arm in ("R4", "C45"):
        for i, lq in enumerate(xc):
            mqc = S1b.with_z0_measure(lq)
            out, secs = compile_arm(arm, mqc, backend, mods)
            isa, failed, cost, n2 = check_circuit(out, backend, mods)
            d, active = measured_distribution(out)
            z_exact = (d.get("0", 0.0) - d.get("1", 0.0)) if d is not None else None
            z_pred = S1b.z_from_sim(sim, out, 20000, 1000) if sim is not None else None
            ok = isa and not failed and z_exact is not None and abs(z_exact - labels[i] * 0.99776) <= 6e-5
            print(f"  xor_{inputs[i]}     {arm:4s} 2q {n2:5d} ESP {math.exp(-cost):.4g} z_exact {z_exact} z_pred "
                  f"{z_pred} isa {isa} failed {failed}{'' if ok else '  -> NOT SUBMITTED'}", flush=True)
            entries.append(dict(part="XOR", test="xor_" + inputs[i], arm=arm, label=labels[i], twoq=n2,
                                neg_ln_esp=cost, esp=math.exp(-cost), active=active, z_exact=z_exact, z_pred=z_pred,
                                isa=isa, uses_failed=failed, compile_s=round(secs, 3), submitted=ok, sig=sig(out)))
            if ok:
                circs.append((len(entries) - 1, out))
    # the submission order: HW1 circuits twice (repeat 1 in one order, repeat 2 in the reverse), the XOR circuits once
    hw = [(k, c) for k, c in circs if entries[k]["part"] == "HW1"]
    xo = [(k, c) for k, c in circs if entries[k]["part"] == "XOR"]
    order = [(k, c, 1) for k, c in hw] + [(k, c, 1) for k, c in xo] + [(k, c, 2) for k, c in hw[::-1]]
    if len(order) > MAX_PUBS:
        raise SystemExit(f"{len(order)} circuits > {MAX_PUBS}: not prepared")
    pubs = [dict(entry=k, repeat=r, shots=SHOTS if entries[k]["part"] == "HW1" else XOR_SHOTS) for k, _, r in order]
    est = sum(0.0005 * p["shots"] + 1.0 for p in pubs)
    print(f"{len(pubs)} circuits, {sum(p['shots'] for p in pubs)} shots; rough QPU estimate {est:.0f} s")
    if est > USAGE_EST_CAP_S:
        raise SystemExit(f"rough QPU estimate {est:.0f} s > {USAGE_EST_CAP_S:.0f} s: not prepared")
    with open(os.path.join(out_dir, "hw1_circuits.qpy"), "wb") as f:
        qpy.dump([c for _, c, _ in order], f)
    return entries, pubs, ideal


# ------------------------------------------------------------------ steps

def manifest_path(out_dir):
    return os.path.join(out_dir, "hw1_manifest.json")


def write_manifest(out_dir, man):
    with open(manifest_path(out_dir), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, default=str)


def calibration_time(backend):
    try:
        return str(backend.properties().last_update_date)
    except Exception:  # noqa: BLE001
        return None


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
        backend, rule = None, []
        for name in DEVICE_ORDER:
            try:
                b = svc.backend(name)
                st = b.status()
                rule.append(dict(name=name, operational=bool(st.operational), status=st.status_msg,
                                 pending=st.pending_jobs))
                if st.operational and backend is None:
                    backend = b
            except Exception as exc:  # noqa: BLE001
                rule.append(dict(name=name, error=type(exc).__name__))
        print("device rule:", rule)
        if backend is None:
            raise SystemExit("Neither ibm_kingston nor ibm_fez is operational now: nothing prepared.")
    else:
        from qiskit_ibm_runtime import fake_provider
        backend, rule = getattr(fake_provider, fake)(), [dict(name=fake, fake=True)]
    entries, pubs, ideal = build_all(backend, a.out)
    man = dict(written=stamp(), device=backend.name, device_rule=rule, calibration_time=calibration_time(backend),
               shots=SHOTS, xor_shots=XOR_SHOTS, repeats=REPEATS, script_sha256=sha256_file(os.path.abspath(__file__)),
               files={os.path.relpath(f, REPO): sha256_file(f) for f in (R4_FILE, C45_FILE, C47_FILE, LAYOUT_FILE)},
               qpy_sha256=sha256_file(os.path.join(a.out, "hw1_circuits.qpy")), entries=entries, pubs=pubs,
               ideal=ideal, params=dict(QAOA_GAMMA=QAOA_GAMMA, QAOA_BETA=QAOA_BETA, TFIM=[TFIM_J, TFIM_H, TFIM_T,
                                                                                          TFIM_REPS]),
               job_id=None)
    write_manifest(a.out, man)
    print(f"{stamp()} manifest written before any submission: {len(pubs)} circuits; device {backend.name}; "
          f"calibration {man['calibration_time']}")
    return backend


def _read(a):
    with open(manifest_path(a.out), encoding="utf-8") as f:
        man = json.load(f)
    if sha256_file(os.path.join(a.out, "hw1_circuits.qpy")) != man["qpy_sha256"]:
        raise SystemExit("hw1_circuits.qpy differs from the manifest's hash: stop.")
    return man


def submit(a):
    header()
    man = _read(a)
    if man.get("job_id"):
        raise SystemExit(f"job {man['job_id']} already recorded: not submitting again.")
    from qiskit import qpy
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    with open(os.path.join(a.out, "hw1_circuits.qpy"), "rb") as f:
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
    with open(os.path.join(a.out, "hw1_circuits.qpy"), "rb") as f:
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
    with open(os.path.join(a.out, "hw1_raw_counts.json"), "w", encoding="utf-8") as f:
        json.dump(dict(fetched=stamp(), job_id=man["job_id"], status=st, metrics=metrics, rows=rows), f,
                  default=str)
    print(f"{stamp()} saved hw1_raw_counts.json ({len(rows)} results); usage {metrics.get('usage')}")


def score(a):
    man = _read(a)
    with open(os.path.join(a.out, "hw1_raw_counts.json"), encoding="utf-8") as f:
        raw = json.load(f)
    ent = man["entries"]
    by = {}
    for p, row in zip(man["pubs"], raw["rows"]):
        e = ent[p["entry"]]
        by.setdefault((e["part"], e["test"], e["arm"]), []).append((p["repeat"], row["counts"]))
    out = [f"# HW1 / PSF-XOR scores ({stamp()})", "",
           f"device {man['device']}; calibration {man['calibration_time']}; job {man.get('job_id')}; usage "
           f"{(raw.get('metrics') or {}).get('usage')}", ""]
    # HW1: F = normalised linear XEB against the ideal distribution (primary); R = observable / ideal (secondary)
    out += ["## HW1", "", "F: normalised linear XEB (primary). R: observable / ideal (secondary; n/a where the ideal "
            "is below 0.1 per term).", "", "| test | arm | 2q | ESP | F rep 1 | F rep 2 | F mean (SE) | R mean |",
            "|---|---|---|---|---|---|---|---|"]
    R = {}
    dists = {}
    for e in ent:
        if e["part"] != "HW1" or not e["submitted"]:
            continue
        t = e["test"]
        if t not in dists:
            qc, spec = build_test(t)
            d, _ = measured_distribution(_unrolled(qc))
            if abs(sum(v * v for v in d.values()) - man["ideal"][t]["sum_p2"]) > 1e-9:
                raise SystemExit(f"ideal distribution of {t} differs from the manifest's: stop")
            dists[t] = d
        spec, iv, n = man["ideal"][t]["spec"], man["ideal"][t]["value"], man["ideal"][t]["n"]
        reps = sorted(by.get(("HW1", t, e["arm"]), []))
        fs = [xeb(dists[t], c, n, sum(c.values())) for _, c in reps]
        terms = len(spec[1]) if spec[0] == "zz" else 1
        rs = [observable(spec, c, SHOTS)[0] / iv for _, c in reps] if abs(iv) / terms >= 0.1 else []
        fm = float(np.mean([f for f, _ in fs])) if fs else float("nan")
        fse = math.sqrt(sum(s_ ** 2 for _, s_ in fs)) / max(len(fs), 1)
        R[(t, e["arm"])] = dict(f=fm, se=fse, reps=[f for f, _ in fs], r=float(np.mean(rs)) if rs else float("nan"),
                                esp=e["esp"])
        out.append(f"| {t} | {e['arm']} | {e['twoq']} | {e['esp']:.4g} | "
                   + " | ".join(f"{x:.4f}" for x in ([f for f, _ in fs] + [float('nan')] * 2)[:2])
                   + f" | {fm:.4f} ({fse:.4f}) | {R[(t, e['arm'])]['r']:.4f} |")
    pairs = []
    for t in TESTS:
        if (t, "PSF") in R and (t, "QK3") in R:
            p_, q_ = R[(t, "PSF")], R[(t, "QK3")]
            pairs.append(dict(test=t, esp_ratio=p_["esp"] / q_["esp"],
                              f_ratio=(p_["f"] / q_["f"]) if q_["f"] > 0 and p_["f"] > 0 else float("nan"),
                              f_psf=p_["f"], f_qk3=q_["f"], se_psf=p_["se"], se_qk3=q_["se"],
                              drift=max(abs(p_["reps"][0] - p_["reps"][-1]), abs(q_["reps"][0] - q_["reps"][-1]))))
    out += ["", "| test | ESP ratio PSF/QK3 | F ratio PSF/QK3 | F PSF (SE) | F QK3 (SE) | largest repeat difference |",
            "|---|---|---|---|---|---|"]
    for x in pairs:
        out.append(f"| {x['test']} | {x['esp_ratio']:.3f} | {x['f_ratio']:.3f} | {x['f_psf']:.4f} ({x['se_psf']:.4f}) | "
                   f"{x['f_qk3']:.4f} ({x['se_qk3']:.4f}) | {x['drift']:.4f} |")
    main = [x for x in pairs if not x["test"].startswith("ghz")]
    big = [x for x in main if x["esp_ratio"] >= 1.2]
    wins = sum(1 for x in big if x["f_psf"] >= x["f_qk3"])

    def ranks(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r
    rho = float("nan")
    ok = [x for x in main if not math.isnan(x["f_ratio"])]
    if len(ok) >= 3:
        a1, b1 = ranks([math.log(x["esp_ratio"]) for x in ok]), ranks([math.log(x["f_ratio"]) for x in ok])
        rho = float(np.corrcoef(a1, b1)[0, 1])
    ghz = [x for x in pairs if x["test"].startswith("ghz")]
    gm = lambda arm: math.exp(np.mean([math.log(v["f"] / v["esp"]) for (t, ar), v in R.items()  # noqa: E731
                                       if ar == arm and v["f"] > 0]))
    out += ["", "### Predictions (F)", "",
            f"- H1: tests with ESP ratio >= 1.2: {len(big)}; PSF's F >= QK3's on {wins} -> "
            + ("CONFIRMED" if big and wins * 2 > len(big) else ("REFUTED" if big and wins * 2 < len(big)
                                                                  else "AMBIGUOUS")),
            f"- H2: rank correlation of ESP ratio and F ratio over {len(ok)} tests: {rho:.3f} -> "
            + ("CONFIRMED" if rho > 0 else ("REFUTED" if rho < 0 else "AMBIGUOUS")),
            "- H3: GHZ control: " + ("; ".join(f"ESP ratio {x['esp_ratio']:.3f}, F ratio {x['f_ratio']:.3f}"
                                               for x in ghz) or "not submitted")
            + ((" -> " + ("CONFIRMED" if abs(ghz[0]["f_ratio"] - ghz[0]["esp_ratio"]) <= 0.10 else "NOT CONFIRMED"))
               if ghz else ""),
            "- H4 (reported): F / ESP per arm (geometric mean over HW1 tests): "
            + ", ".join(f"{arm} {gm(arm):.3f}" for arm in ("QK3", "PSF", "C47")
                        if any(ar == arm and v["f"] > 0 for (t, ar), v in R.items()))]
    # PSF-XOR
    out += ["", "## PSF-XOR", "", "| arm | input | label | z observed | z predicted (Aer) |", "|---|---|---|---|---|"]
    xm = {}
    for arm in ("R4", "C45"):
        zs, zp, lb = [], [], []
        for e in ent:
            if e["part"] != "XOR" or e["arm"] != arm or not e["submitted"]:
                continue
            c = by[("XOR", e["test"], arm)][0][1]
            n0, n1 = c.get("0", 0), c.get("1", 0)
            z = (n0 - n1) / max(n0 + n1, 1)
            zs.append(z)
            zp.append(e["z_pred"])
            lb.append(e["label"])
            zp_s = "n/a" if e["z_pred"] is None else f"{e['z_pred']:+.4f}"
            out.append(f"| {arm} | {e['test'][4:]} | {e['label']:+d} | {z:+.4f} | {zp_s} |")
        if zs:
            m_obs = float(np.mean([l * z for l, z in zip(lb, zs)]))
            m_pred = float(np.mean([l * z for l, z in zip(lb, zp)])) if None not in zp else float("nan")
            xm[arm] = dict(m_obs=m_obs, m_pred=m_pred, correct=sum(1 for l, z in zip(lb, zs) if l * z > 0), n=len(zs))
    for arm, v in xm.items():
        out.append(f"- X1 {arm}: correct {v['correct']}/{v['n']} -> {'CONFIRMED' if v['correct'] == v['n'] == 4 else 'NOT CONFIRMED'}")
        out.append(f"- X2 {arm}: observed M {v['m_obs']:.4f} >= predicted M {v['m_pred']:.4f} - 0.15 -> "
                   f"{'CONFIRMED' if v['m_obs'] >= v['m_pred'] - 0.15 else 'NOT CONFIRMED'}")
    out.append("- X3 (against Stage 2's chosen arm, another job): scored in the Addendum from both scores")
    txt = "\n".join(out) + "\n"
    with open(os.path.join(a.out, "hw1_score.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    with open(os.path.join(a.out, "hw1_score.json"), "w", encoding="utf-8") as f:
        json.dump(dict(F={f"{k[0]}|{k[1]}": v for k, v in R.items()}, pairs=pairs, xor=xm, rho=rho), f, indent=1)
    print(txt)


def rehearse(a):
    warnings.simplefilter("ignore")
    prepare(a, fake=a.fake)
    man = _read(a)
    rows, metrics = _sample_local(a, man)
    with open(os.path.join(a.out, "hw1_raw_counts.json"), "w", encoding="utf-8") as f:
        json.dump(dict(fetched=stamp(), job_id="rehearsal", status="DONE (local)", metrics=metrics, rows=rows), f)
    man["job_id"] = "rehearsal"
    write_manifest(a.out, man)
    score(a)
    print("REHEARSAL DONE (noiseless: every F and R should be 1 within shot noise; X1 4/4)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("rehearse", "prepare", "submit", "fetch", "score"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--fake", default="FakeKingston")
    ap.add_argument("--wait-min", type=float, default=40.0)
    a = ap.parse_args()
    if a.step == "prepare":
        prepare(a)
    else:
        dict(rehearse=rehearse, submit=submit, fetch=fetch, score=score)[a.step](a)


if __name__ == "__main__":
    main()
