"""time1_real.py -- TIME1 (2026-10-14): does a compiled circuit's duration pick the winner on a third IBM device?

Pre-registered in TIME1_preregistration.md (fixed before any real-device step; sha256 and CEST time kept with it).
One job, on a Heron device other than ibm_kingston (HW1) and ibm_marrakesh (HW2):

  T1 (order only)    TFIM chains from a fixed random product state, n = 12, 16, 20, 4 and 8 second-order Trotter
                     steps (dt 0.3): prototype c49 (PSF-Zero's own pipeline, level3_candidate=False) with item 83 on
                     (S1) and off (S0). The same gate blocks; a pair is submitted only where the two differ.
  T2/T3 (HW2's seven) HW2's seven reading tests compiled by Qiskit level 3 (QK3), by c49's default call (C49: choices
                     by duration) and by c48's default call (C48: choices by ESP). Pairs C49 / QK3 and C49 / C48 (a
                     C49 / C48 pair is scored only where the circuits differ).
  control            ghz_n12, QK3 and C49.

Every circuit's duration (as soon as possible on the Target's gate durations), two-qubit count and ESP are computed
from the device's Target when `prepare` runs and written into the manifest before submission. ESP is computed only
to be scored as a competing predictor; it decides nothing.

Steps (only `submit` uses QPU time; `prepare`, `submit` and `fetch` need the IBM connection):
  rehearse  the whole flow on a fake device (default FakeFez) with a noiseless local simulator. No IBM connection.
  prepare   device rule; the device's Target now; compiles; checks (ISA, failed elements, exactness by noiseless
            simulation up to 24 active qubits); writes circuits (QPY) and manifest BEFORE any submission.
  submit    ONE SamplerV2 job, 6,000 shots per circuit and repeat, dynamical decoupling and twirling off.
  fetch     waits for the job and saves the raw counts and the job's metrics.
  score     offline: scores the predictions.

    python -u <kit>/time1_real.py rehearse --out data/2026-10-14/time1_rehearsal
    python -u <kit>/time1_real.py prepare  --out data/2026-10-14/time1_run
    python -u <kit>/time1_real.py submit   --out data/2026-10-14/time1_run
    python -u <kit>/time1_real.py fetch    --out data/2026-10-14/time1_run
    python -u <kit>/time1_real.py score    --out data/2026-10-14/time1_run

The IBM API key is never read from or written to any file by this script (the account saved beforehand by
QiskitRuntimeService.save_account() is used). This script runs no git and no pip.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import os
import random
import sys
import time
import warnings

REPO = os.path.abspath(os.getcwd())
for p in (os.path.join(REPO, "benchmarks"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

KIT = os.path.dirname(os.path.abspath(__file__))
HW2_FILE = os.path.join(REPO, "data", "2026-10-13", "hw2_kit", "hw2_real.py")
HW2_SHA = "80f356c4a6bb3061f5c315ca2312d8a71bc09ac2dd5987ed04e7b15ecc7e368d"
C48_FILE = os.path.join(REPO, "patches", "psf_compile_c48_2026-10-12", "psf_compile.py")
C49_FILE = os.path.join(REPO, "patches", "psf_compile_c49_2026-10-14", "psf_compile.py")
LAYOUT_FILE = os.path.join(REPO, "benchmarks", "psf_smart_layout.py")

SHOTS = 6000  # each circuit, per repeat (as HW2)
REPEATS = 2
USAGE_EST_CAP_S = 300.0
MAX_PUBS = 80
FAILED = 0.5
EXACT_TOL = 1e-6
EXCLUDED = ("ibm_kingston", "ibm_marrakesh")  # HW1's and HW2's devices
FIRST_CHOICE = "ibm_fez"
T1_TESTS = tuple(f"tfimP_n{n}_r{r}" for n in (12, 16, 20) for r in (4, 8))
READING = ("qaoa_n12_p2", "qaoa_n20_p2", "ba_n14", "ba_n16", "ba_n18", "ba_n20", "tfim_n12")
CONTROL = ("ghz_n12",)
DT = 0.3  # Trotter time step of T1's chains (HW2's tfimP: 2 steps over t = 0.6)
TIE = 1.10  # TB: a pair counts when its duration ratio is <= 1/TIE or >= TIE


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


H2 = _load(HW2_FILE, "time1_hw2")
H2.KIT = KIT  # the Barabasi-Albert QASM files travel with this kit in qasm/
stamp, sha256_file = H2.stamp, H2.sha256_file


def build_test(name):
    """(measured logical circuit) for a TIME1 input."""
    if name.startswith("tfimP_n") and "_r" in name:
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.quantum_info import SparsePauliOp
        from qiskit.synthesis import SuzukiTrotter
        n, r = int(name.split("_n")[1].split("_")[0]), int(name.split("_r")[1])
        rng = random.Random("TIME1|" + name)
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.ry(rng.uniform(0.3 * math.pi, 0.7 * math.pi), q)
        terms = [(H2._label(n, {q: "Z", q + 1: "Z"}), H2.TFIM_J) for q in range(n - 1)]
        terms += [(H2._label(n, {q: "X"}), H2.TFIM_H) for q in range(n)]
        qc.append(PauliEvolutionGate(SparsePauliOp.from_list(terms), time=DT * r,
                                     synthesis=SuzukiTrotter(order=2, reps=r)), range(n))
        qc.measure_all()
        return qc
    return H2.build_test(name)[0]


def header():
    import qiskit
    print(f"{stamp()} | qiskit {qiskit.__version__}")
    for m in ("qiskit_aer", "qiskit_ibm_runtime", "psf_zero_core"):
        try:
            mod = __import__(m)
            print(f"  {m} {getattr(mod, '__version__', getattr(mod, 'CORE_VERSION', '?'))}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {m} MISSING ({type(exc).__name__})")
    print("SCRIPT", os.path.abspath(__file__), sha256_file(os.path.abspath(__file__)))
    for f in (HW2_FILE, C48_FILE, C49_FILE, LAYOUT_FILE):
        print("FILE", os.path.relpath(f, REPO), sha256_file(f) if os.path.exists(f) else "MISSING")
    if sha256_file(HW2_FILE) != HW2_SHA:
        raise SystemExit("hw2_real.py differs from the hash fixed in this kit: stop")


def compilers():
    _load(LAYOUT_FILE, "psf_smart_layout")
    mods = {"C48": _load(C48_FILE, "time1_c48"), "C49": _load(C49_FILE, "time1_c49")}
    for m in mods.values():
        m.WARN_WITHOUT_TARGET = False
    return mods


def compile_arm(arm, qc, backend, mods):
    """QK3: Qiskit level 3. C49 / C48: the default call (with the device). S1 / S0: c49's own pipeline
    (level3_candidate=False) with item 83 on / off."""
    from qiskit.transpiler import generate_preset_pass_manager
    t0 = time.perf_counter()
    kw = dict(backend=backend, entangling_basis="cx", layout_search=True, seed_transpiler=0)
    if arm == "QK3":
        out = generate_preset_pass_manager(3, backend, seed_transpiler=0).run(qc)
    elif arm in ("C49", "C48"):
        with contextlib.redirect_stdout(io.StringIO()):
            out = mods[arm].compile_for_hardware(qc, **kw)
    else:
        m = mods["C49"]
        m.ST_SCHEDULE = arm == "S1"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out = m.compile_for_hardware(qc, level3_candidate=False, **kw)
        finally:
            m.ST_SCHEDULE = True
    return out, time.perf_counter() - t0


def metrics(out, target, mods):
    """(two-qubit count, ESP, duration in us without measurements, ISA ok, uses a failed element)."""
    m = mods["C49"]
    pos = {q: i for i, q in enumerate(out.qubits)}
    clock, isa, n2 = {}, True, 0
    for ins in out.data:
        nm = ins.operation.name
        if nm == "barrier":
            continue
        qs = tuple(pos[q] for q in ins.qubits)
        if nm not in target.operation_names or qs not in target[nm]:
            isa = False
        if nm in ("measure", "delay"):
            continue
        pr = target[nm].get(qs) if nm in target.operation_names else None
        d = (pr.duration or 0.0) if pr is not None else 0.0
        s = max(clock.get(q, 0.0) for q in qs)
        for q in qs:
            clock[q] = s + d
        n2 += len(qs) == 2
    edges, bad = m._failed_elements(target, FAILED)
    return n2, math.exp(-m._st_cost(out, target)), (max(clock.values()) if clock else 0.0) * 1e6, isa, \
        bool(m._uses_failed(out, edges, bad))


def build_all(backend, out_dir):
    from qiskit import qpy
    mods = compilers()
    entries, circs, ideal = [], [], {}
    print(f"{stamp()} compiling on {backend.name}", flush=True)
    plan = [("T1", t, ("S0", "S1")) for t in T1_TESTS] + [("R", t, ("QK3", "C49", "C48")) for t in READING] + \
        [("CTL", t, ("QK3", "C49")) for t in CONTROL]
    for part, test, arms in plan:
        qc = build_test(test)
        dist, _ = H2.measured_distribution(H2._unrolled(qc))
        ideal[test] = dict(n=qc.num_qubits, sum_p2=float(sum(v * v for v in dist.values())))
        outs = {arm: compile_arm(arm, qc, backend, mods) for arm in arms}
        sigs = {arm: H2.sig(o) for arm, (o, _) in outs.items()}
        if part == "T1" and sigs["S0"] == sigs["S1"]:
            print(f"  {test:16s} S0 and S1 identical: pair not submitted", flush=True)
            continue
        start = len(entries)
        for arm, (out, secs) in outs.items():
            n2, esp, dur, isa, failed = metrics(out, backend.target, mods)
            d, active = H2.measured_distribution(out)
            exact = None if d is None else max(abs(d.get(k, 0.0) - dist.get(k, 0.0)) for k in set(d) | set(dist))
            ok = isa and not failed and (exact is None or exact <= EXACT_TOL)
            print(f"  {part:3s} {test:16s} {arm:4s} 2q {n2:5d} dur {dur:7.2f} us ESP {esp:.4g} active {active:3d} "
                  f"exact {'n/a' if exact is None else f'{exact:.1e}'} isa {isa} failed {failed} ({secs:.1f} s)"
                  f"{'' if ok else '  -> NOT SUBMITTED'}", flush=True)
            entries.append(dict(part=part, test=test, arm=arm, twoq=n2, esp=esp, dur_us=dur, active=active,
                                exact_maxdiff=exact, isa=isa, uses_failed=failed, compile_s=round(secs, 3),
                                submitted=ok, sig=sigs[arm]))
            if ok:
                circs.append((len(entries) - 1, out))
        if part == "T1" and not all(e["submitted"] for e in entries[start:]):
            for e in entries[start:]:
                e["submitted"] = False
            circs = [(k, c) for k, c in circs if k < start]
            print(f"  {test}: pair incomplete, neither submitted", flush=True)
    order = [(k, c, 1) for k, c in circs] + [(k, c, 2) for k, c in circs[::-1]]
    if len(order) > MAX_PUBS:
        raise SystemExit(f"{len(order)} circuits > {MAX_PUBS}: not prepared")
    pubs = [dict(entry=k, repeat=r, shots=SHOTS) for k, _, r in order]
    est = sum(0.0005 * p["shots"] + 1.0 for p in pubs)
    print(f"{len(pubs)} circuits, {sum(p['shots'] for p in pubs)} shots; rough QPU estimate {est:.0f} s")
    if est > USAGE_EST_CAP_S:
        raise SystemExit(f"rough QPU estimate {est:.0f} s > {USAGE_EST_CAP_S:.0f} s: not prepared")
    with open(os.path.join(out_dir, "time1_circuits.qpy"), "wb") as f:
        qpy.dump([c for _, c, _ in order], f)
    return entries, pubs, ideal


def manifest_path(out_dir):
    return os.path.join(out_dir, "time1_manifest.json")


def write_manifest(out_dir, man):
    with open(manifest_path(out_dir), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, default=str)


def _read(a):
    with open(manifest_path(a.out), encoding="utf-8") as f:
        man = json.load(f)
    if sha256_file(os.path.join(a.out, "time1_circuits.qpy")) != man["qpy_sha256"]:
        raise SystemExit("time1_circuits.qpy differs from the manifest's hash: stop.")
    return man


def device_rule(svc):
    """ibm_fez if operational; otherwise the operational non-simulator device with at least 127 qubits, other than
    ibm_kingston and ibm_marrakesh, with the fewest pending jobs (ties: name)."""
    rule = []
    try:
        b = svc.backend(FIRST_CHOICE)
        st = b.status()
        rule.append(dict(name=FIRST_CHOICE, operational=bool(st.operational), pending=st.pending_jobs))
        if st.operational:
            return b, rule
    except Exception as exc:  # noqa: BLE001
        rule.append(dict(name=FIRST_CHOICE, error=type(exc).__name__))
    cands = []
    for b in svc.backends(simulator=False, operational=True, min_num_qubits=127):
        if b.name in EXCLUDED or b.name == FIRST_CHOICE:
            continue
        st = b.status()
        rule.append(dict(name=b.name, operational=bool(st.operational), pending=st.pending_jobs))
        if st.operational:
            cands.append((st.pending_jobs, b.name, b))
    if not cands:
        raise SystemExit("no eligible device is operational now: nothing prepared")
    return min(cands, key=lambda c: (c[0], c[1]))[2], rule


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
        backend, rule = device_rule(svc)
    else:
        from qiskit_ibm_runtime import fake_provider
        backend, rule = getattr(fake_provider, fake)(), [dict(name=fake, fake=True)]
    print("device rule:", rule)
    entries, pubs, ideal = build_all(backend, a.out)
    man = dict(written=stamp(), device=backend.name, device_rule=rule, calibration_time=H2.calibration_time(backend),
               shots=SHOTS, repeats=REPEATS, script_sha256=sha256_file(os.path.abspath(__file__)),
               files={os.path.relpath(f, REPO): sha256_file(f) for f in (HW2_FILE, C48_FILE, C49_FILE, LAYOUT_FILE)},
               qpy_sha256=sha256_file(os.path.join(a.out, "time1_circuits.qpy")), entries=entries, pubs=pubs,
               ideal=ideal, params=dict(DT=DT, TIE=TIE), job_id=None)
    write_manifest(a.out, man)
    print(f"{stamp()} manifest written before any submission: {len(pubs)} circuits; device {backend.name}; "
          f"calibration {man['calibration_time']}")
    return backend


def submit(a):
    header()
    man = _read(a)
    if man.get("job_id"):
        raise SystemExit(f"job {man['job_id']} already recorded: not submitting again.")
    from qiskit import qpy
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    with open(os.path.join(a.out, "time1_circuits.qpy"), "rb") as f:
        circs = qpy.load(f)
    if len(circs) != len(man["pubs"]):
        raise SystemExit("the QPY file and the manifest disagree on the number of circuits: stop")
    backend = QiskitRuntimeService().backend(man["device"])
    sampler = SamplerV2(mode=backend)
    sampler.options.dynamical_decoupling.enable = False
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    t_sub = stamp()
    job = sampler.run([(c, None, p["shots"]) for c, p in zip(circs, man["pubs"])])
    man["job_id"], man["submitted"] = job.job_id(), t_sub
    write_manifest(a.out, man)
    print(f"{stamp()} submitted job {job.job_id()} ({len(circs)} circuits, {sum(p['shots'] for p in man['pubs'])} "
          f"shots) to {man['device']}. Results are not read here.")


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
        metrics_ = job.metrics()
    except Exception as exc:  # noqa: BLE001
        metrics_ = dict(error=type(exc).__name__)
    with open(os.path.join(a.out, "time1_raw_counts.json"), "w", encoding="utf-8") as f:
        json.dump(dict(fetched=stamp(), job_id=man["job_id"], status=st, metrics=metrics_, rows=rows), f, default=str)
    print(f"{stamp()} saved time1_raw_counts.json ({len(rows)} results); usage {metrics_.get('usage')}")


def _rho(x, y):
    return H2._rho(x, y)


def score(a):
    man = _read(a)
    with open(os.path.join(a.out, "time1_raw_counts.json"), encoding="utf-8") as f:
        raw = json.load(f)
    ent = man["entries"]
    by = {}
    for p, row in zip(man["pubs"], raw["rows"]):
        e = ent[p["entry"]]
        by.setdefault((e["part"], e["test"], e["arm"]), []).append((p["repeat"], row["counts"]))
    out = [f"# TIME1 scores ({stamp()})", "",
           f"device {man['device']}; calibration {man['calibration_time']}; job {man.get('job_id')}; usage "
           f"{(raw.get('metrics') or {}).get('usage')}", "",
           "F: normalised linear XEB against the ideal distribution, mean of the two repeats.", "",
           "| part | test | arm | 2q | duration us | ESP | F rep 1 | F rep 2 | F mean (SE) |", "|---|---|---|---|---|---|---|---|---|"]
    dists, R = {}, {}
    for e in ent:
        if not e["submitted"]:
            continue
        t = e["test"]
        if t not in dists:
            d, _ = H2.measured_distribution(H2._unrolled(build_test(t)))
            if abs(sum(v * v for v in d.values()) - man["ideal"][t]["sum_p2"]) > 1e-9:
                raise SystemExit(f"ideal distribution of {t} differs from the manifest's: stop")
            dists[t] = d
        n = man["ideal"][t]["n"]
        reps = sorted(by.get((e["part"], t, e["arm"]), []))
        fs = [H2.xeb(dists[t], c, n, sum(c.values())) for _, c in reps]
        fm = float(np.mean([f for f, _ in fs])) if fs else float("nan")
        fse = math.sqrt(sum(s_ ** 2 for _, s_ in fs)) / max(len(fs), 1)
        R[(e["part"], t, e["arm"])] = dict(f=fm, se=fse, reps=[f for f, _ in fs], esp=e["esp"], dur=e["dur_us"],
                                           twoq=e["twoq"], sig=e["sig"])
        out.append(f"| {e['part']} | {t} | {e['arm']} | {e['twoq']} | {e['dur_us']:.2f} | {e['esp']:.4g} | "
                   + " | ".join(f"{x:.4f}" for x in ([f for f, _ in fs] + [float('nan')] * 2)[:2])
                   + f" | {fm:.4f} ({fse:.4f}) |")

    def pair(part, t, a1, a0, kind):
        x, y = R.get((part, t, a1)), R.get((part, t, a0))
        if x is None or y is None or x["sig"] == y["sig"]:
            return None
        if not (x["f"] > 0 and y["f"] > 0):
            return dict(kind=kind, test=t, a=a1, b=a0, valid=False)
        return dict(kind=kind, test=t, a=a1, b=a0, valid=True, dur=x["dur"] / y["dur"], twoq=x["twoq"] / y["twoq"],
                    esp=x["esp"] / y["esp"], f=x["f"] / y["f"], fa=x["f"], fb=y["f"])

    pairs = [pair("T1", t, "S1", "S0", "T1") for t in T1_TESTS]
    pairs += [pair("R", t, "C49", "C48", "T2") for t in READING]
    pairs += [pair("R", t, "C49", "QK3", "T3") for t in READING]
    pairs = [p for p in pairs if p is not None]
    valid = [p for p in pairs if p["valid"]]
    out += ["", "## Pairs (ratio = first arm / second arm)", "",
            "| class | test | pair | duration ratio | 2q ratio | ESP ratio | F ratio | F first | F second |",
            "|---|---|---|---|---|---|---|---|---|"]
    for p in pairs:
        if p["valid"]:
            out.append(f"| {p['kind']} | {p['test']} | {p['a']}/{p['b']} | {p['dur']:.3f} | {p['twoq']:.3f} | "
                       f"{p['esp']:.3f} | {p['f']:.3f} | {p['fa']:.4f} | {p['fb']:.4f} |")
        else:
            out.append(f"| {p['kind']} | {p['test']} | {p['a']}/{p['b']} | - | - | - | F <= 0: not scored | | |")
    verdict = {}
    t1 = [p for p in valid if p["kind"] == "T1"]
    up = sum(1 for p in t1 if p["f"] > 1)
    need_a = math.ceil(5 * len(t1) / 6)
    verdict["TA"] = "NOT TESTABLE" if len(t1) < 3 else ("CONFIRMED" if up >= need_a else "REFUTED")
    clear = [p for p in valid if p["dur"] <= 1 / TIE or p["dur"] >= TIE]
    short_wins = sum(1 for p in clear if (p["dur"] < 1) == (p["f"] > 1))
    verdict["TB"] = "NOT TESTABLE" if len(clear) < 4 else ("CONFIRMED" if short_wins >= 0.75 * len(clear) else "REFUTED")
    lf = [math.log(p["f"]) for p in valid]
    rho_d = _rho([-math.log(p["dur"]) for p in valid], lf)
    rho_g = _rho([-math.log(p["twoq"]) for p in valid], lf)
    rho_e = _rho([math.log(p["esp"]) for p in valid], lf)
    verdict["TC"] = "NOT TESTABLE" if len(valid) < 5 else (
        "CONFIRMED" if (rho_d >= 0.5 and rho_d > rho_g and rho_d > rho_e) else "REFUTED")
    t3 = [p for p in pairs if p["kind"] == "T3"]
    same3 = [t for t in READING if R.get(("R", t, "C49")) and R.get(("R", t, "QK3"))
             and R[("R", t, "C49")]["sig"] == R[("R", t, "QK3")]["sig"]]
    ok_d = sum(1 for p in t3 if p["valid"] and p["f"] >= 0.95) + len(same3)
    verdict["TD"] = "CONFIRMED" if ok_d >= 6 else "REFUTED"
    g = (R.get(("CTL", "ghz_n12", "C49")), R.get(("CTL", "ghz_n12", "QK3")))
    verdict["TE"] = "n/a" if None in g or g[1]["f"] <= 0 else (
        "CONFIRMED" if abs(g[0]["f"] / g[1]["f"] - 1) <= 0.10 else "REFUTED")
    confirmed = all(verdict[k] == "CONFIRMED" for k in ("TA", "TB", "TC"))
    out += ["", "## Predictions", "",
            f"- TA (T1: item 83 on gives the higher F on >= {need_a} of {len(t1)}): {up} -> {verdict['TA']}",
            f"- TB (pairs with duration ratio outside 1/{TIE}-{TIE}: the shorter circuit has the higher F on >= 75%): "
            f"{short_wins} of {len(clear)} -> {verdict['TB']}",
            f"- TC (rank correlation over {len(valid)} pairs: 1/duration {rho_d:+.3f}, 1/two-qubit count {rho_g:+.3f}, "
            f"ESP {rho_e:+.3f}; duration >= +0.5 and above both) -> {verdict['TC']}",
            f"- TD (C49's F >= 0.95 x QK3's on >= 6 of 7; identical circuits count): {ok_d} -> {verdict['TD']}",
            f"- TE (control ghz_n12: |F ratio - 1| <= 0.10): {verdict['TE']}",
            f"- **Duration as the choice rule (TA, TB, TC all confirmed): {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}**"]
    drift = [abs(v["reps"][0] - v["reps"][-1]) for v in R.values() if len(v["reps"]) == 2]
    out.append(f"- reported: largest repeat difference {max(drift) if drift else float('nan'):.4f}; median "
               f"{float(np.median(drift)) if drift else float('nan'):.4f}")
    txt = "\n".join(out) + "\n"
    with open(os.path.join(a.out, "time1_score.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    with open(os.path.join(a.out, "time1_score.json"), "w", encoding="utf-8") as f:
        json.dump(dict(F={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in R.items()}, pairs=pairs, verdict=verdict,
                       rho=dict(duration=rho_d, twoq=rho_g, esp=rho_e), confirmed=confirmed), f, indent=1)
    print(txt)


def rehearse(a):
    warnings.simplefilter("ignore")
    prepare(a, fake=a.fake)
    man = _read(a)
    from qiskit import qpy
    from qiskit_aer import AerSimulator
    with open(os.path.join(a.out, "time1_circuits.qpy"), "rb") as f:
        circs = qpy.load(f)
    sim = AerSimulator()
    rows = [dict(pub=i, counts=sim.run(c, shots=p["shots"], seed_simulator=1000 + i).result().get_counts())
            for i, (c, p) in enumerate(zip(circs, man["pubs"]))]
    with open(os.path.join(a.out, "time1_raw_counts.json"), "w", encoding="utf-8") as f:
        json.dump(dict(fetched=stamp(), job_id="rehearsal", status="DONE (local)",
                       metrics=dict(usage=dict(quantum_seconds=0), note="local noiseless rehearsal"), rows=rows), f)
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
