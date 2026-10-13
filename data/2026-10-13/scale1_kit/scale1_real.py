"""scale1_real.py -- SCALE1: do compiled circuits keep a measurable known-answer signal at 12-144 qubits on a real
Heron device? Clifford mirror circuits (scale1_circuits.py, the workplace's file, unchanged); no claim of quantum
advantage: a laptop simulates every circuit here instantly.

Pre-registered in SCALE1_preregistration.md (fixed before any real-device step). Three arms on the device's Target:

  PSF   PSF-Zero's release (psf_compile.py at the repository root): compile_for_hardware(qc, backend=backend,
        seed_transpiler=0)
  QK3   Qiskit level 3 as a typical user calls it: transpile(qc, target=backend.target, optimization_level=3,
        seed_transpiler=0)
  QK3F  the same, given a copy of the Target without the elements whose reported error is >= 0.5 (what a careful user
        can do in a few lines; it separates "avoids failed elements" from everything else a compiler does)

Steps (only `submit` uses QPU time; `prepare`, `submit` and `fetch` need the IBM connection, the others do not):

  rehearse  the whole flow on a fake device (default FakeFez), sampled noiselessly by a matrix-product-state simulator.
  prepare   device rule (of ibm_fez and ibm_marrakesh, the operational one with fewer pending jobs; neither: stop);
            compiles, checks every circuit (ISA; the ideal bitstring by noiseless MPS simulation of the compiled
            circuit, at every width; two-qubit count at least half the logical circuit's: cancelling the
            mirror across the barrier would remove most of them, while CZ pairs cancelling inside one half are legitimate), writes circuits and manifest
            BEFORE any submission. Circuits that use an element with error >= 0.5 go to a separate job (part F).
  submit    the main job, then the part-F job (if any): SamplerV2, the runtime's decoupling and twirling off.
  fetch     waits for the job(s) and saves the raw counts.
  score     offline: S (share of shots equal to the ideal bitstring) per circuit and the predictions C1-C5.

Usage (from the psf-zero repository root):
    python -u <kit>/scale1_real.py rehearse --out data/<date>/scale1_rehearsal
    python -u <kit>/scale1_real.py prepare  --out data/<date>/scale1
    python -u <kit>/scale1_real.py submit   --out data/<date>/scale1
    python -u <kit>/scale1_real.py fetch    --out data/<date>/scale1
    python -u <kit>/scale1_real.py score    --out data/<date>/scale1

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
import sys
import time
import warnings

REPO = os.path.abspath(os.getcwd())
KIT = os.path.dirname(os.path.abspath(__file__))
for p in (KIT, os.path.join(REPO, "benchmarks"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

DEVICES = ("ibm_fez", "ibm_marrakesh")  # the device rule: the operational one with fewer pending jobs (ties: this order)
FAILED = 0.5
ARMS = ("PSF", "QK3", "QK3F")
CELLS = (("L", 12, 0), ("L", 24, 0), ("L", 48, 0),
         ("R", 24, 2), ("R", 48, 2), ("R", 96, 2), ("R", 144, 2),
         ("R", 24, 4), ("R", 48, 4))
INSTANCES = (0, 1, 2)
FAMILIES = (("L", 0), ("R", 2), ("R", 4))  # signal width is read per family and depth
MAX_PUBS = 90
USAGE_EST_CAP_S = 220.0  # rough QPU estimate (0.5 ms per shot + 1 s per circuit), both jobs together
MPS_SHOTS = 32
RESOLVED_SE = 5.0
MISSING_T_S = 1e-6
PSF_FILE = os.path.join(REPO, "psf_compile.py")
CIRC_FILE = os.path.join(KIT, "scale1_circuits.py")


def shots_for(n):
    return 4000 if n >= 96 else 2000


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
    for m in ("qiskit_aer", "qiskit_ibm_runtime", "psf_zero_core"):
        try:
            mod = __import__(m)
            print(f"  {m} {getattr(mod, '__version__', getattr(mod, 'CORE_VERSION', '?'))}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {m} MISSING ({type(exc).__name__})")
    print("SCRIPT", os.path.abspath(__file__), sha256_file(os.path.abspath(__file__)))
    for f in (PSF_FILE, CIRC_FILE):
        print("FILE", os.path.relpath(f, REPO), sha256_file(f) if os.path.exists(f) else "MISSING")


# ------------------------------------------------------------------ circuits and checks

def logical(fam, n, d, inst):
    """(logical circuit with measure_all, ideal bitstring, logical two-qubit count)."""
    sc = load(CIRC_FILE, "scale1_circuits")
    qc = sc.family_L(n, inst) if fam == "L" else sc.family_R(n, d, inst)
    ideal = sc.ideal_bitstring(qc)
    n2 = sum(1 for i in qc.data if len(i.qubits) == 2 and i.operation.name != "barrier")
    qc = qc.copy()
    qc.measure_all()
    return qc, ideal, n2


def cell_name(fam, n, d):
    return f"L_n{n}" if fam == "L" else f"R{d}_n{n}"


def failed_elements(target):
    """{(name, qubits)} with reported error >= FAILED."""
    bad = set()
    for nm in target.operation_names:
        props = target[nm]
        if props is None:
            continue
        for qs, pr in props.items():
            if qs and pr is not None and pr.error is not None and pr.error >= FAILED:
                bad.add((nm, tuple(qs)))
    return bad


def filtered_target(target):
    """A copy of the Target without the elements in failed_elements() (QK3F's Target)."""
    from qiskit.transpiler import Target
    t = Target(num_qubits=target.num_qubits, dt=target.dt, qubit_properties=target.qubit_properties,
               granularity=target.granularity, min_length=target.min_length, pulse_alignment=target.pulse_alignment,
               acquire_alignment=target.acquire_alignment)
    for nm in target.operation_names:
        op = target.operation_from_name(nm)
        props = target[nm]
        if isinstance(op, type) or props is None or None in props:
            t.add_instruction(op, name=nm) if isinstance(op, type) else t.add_instruction(op, props, name=nm)
            continue
        t.add_instruction(op, {qs: pr for qs, pr in props.items()
                               if not (pr is not None and pr.error is not None and pr.error >= FAILED)}, name=nm)
    return t


def compile_arm(arm, qc, backend, psf, tf):
    from qiskit import transpile
    t0 = time.perf_counter()
    if arm == "PSF":
        with contextlib.redirect_stdout(io.StringIO()):
            out = psf.compile_for_hardware(qc, backend=backend, seed_transpiler=0)
        out = out[0] if isinstance(out, tuple) else out
    elif arm == "QK3":
        out = transpile(qc, target=backend.target, optimization_level=3, seed_transpiler=0)
    else:
        out = transpile(qc, target=tf, optimization_level=3, seed_transpiler=0)
    return out, time.perf_counter() - t0


def active_circuit(c):
    """The compiled circuit on its used qubits only (for the MPS check); clbits kept."""
    from qiskit import QuantumCircuit
    used = sorted({c.find_bit(q).index for i in c.data if i.operation.name != "barrier" for q in i.qubits})
    pos = {q: k for k, q in enumerate(used)}
    out = QuantumCircuit(len(used), c.num_clbits)
    for i in c.data:
        if i.operation.name == "barrier":
            continue
        out.append(i.operation, [pos[c.find_bit(q).index] for q in i.qubits], [c.find_bit(x).index for x in i.clbits])
    return out


def mps_outcomes(c, seed):
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="matrix_product_state")
    return sim.run(active_circuit(c), shots=MPS_SHOTS, seed_simulator=seed).result().get_counts()


def check(out, backend, bad):
    """(ISA ok, failed elements used, two-qubit count)."""
    t = backend.target
    isa, used_bad, n2 = True, 0, 0
    for ins in out.data:
        nm = ins.operation.name
        if nm == "barrier":
            continue
        qs = tuple(out.find_bit(q).index for q in ins.qubits)
        if nm not in t.operation_names or qs not in t[nm]:
            isa = False
        if (nm, qs) in bad:
            used_bad += 1
        if len(qs) == 2:
            n2 += 1
    return isa, used_bad, n2


def estimate(out, target):
    """(ESP with readout, idle-weighted estimate, readout-only product, duration in us). ESP: product of (1 - error)
    over every operation on the Target, measurements included. Idle term (the T2-placement draft's physics form, no
    free weight): for each used qubit, its idle time I = (end - first gate) - own gate time, as soon as possible on the
    Target's durations, charged I (1/(6 T1) + 1/(3 T2)), T2 capped at 2 T1."""
    pos = {q: i for i, q in enumerate(out.qubits)}
    ln_esp, ln_ro = 0.0, 0.0
    clock, first, busy = {}, {}, {}
    for ins in out.data:
        nm = ins.operation.name
        if nm in ("barrier", "delay"):
            continue
        qs = tuple(pos[q] for q in ins.qubits)
        pr = target[nm].get(qs) if nm in target.operation_names else None
        e = (pr.error or 0.0) if pr is not None else 0.0
        ln_esp += math.log(max(1e-300, 1.0 - min(e, 1.0)))
        if nm == "measure":
            ln_ro += math.log(max(1e-300, 1.0 - min(e, 1.0)))
            continue
        d = (pr.duration or 0.0) if pr is not None else 0.0
        s = max(clock.get(q, 0.0) for q in qs)
        for q in qs:
            first.setdefault(q, s)
            clock[q] = s + d
            busy[q] = busy.get(q, 0.0) + d
    T = max(clock.values()) if clock else 0.0
    ln_idle = 0.0
    for q in clock:
        p = target.qubit_properties[q] if target.qubit_properties is not None else None
        t1 = getattr(p, "t1", None) or MISSING_T_S
        t2 = min(getattr(p, "t2", None) or MISSING_T_S, 2 * t1)
        ln_idle += max(0.0, T - first[q] - busy[q]) * (1 / (6 * t1) + 1 / (3 * t2))
    return math.exp(ln_esp), math.exp(ln_esp - ln_idle), math.exp(ln_ro), T * 1e6


def build_all(backend, out_dir, psf):
    from qiskit import qpy
    t = backend.target
    bad = failed_elements(t)
    tf = filtered_target(t)
    print(f"{stamp()} compiling on {backend.name}; elements with error >= {FAILED}: {len(bad)}", flush=True)
    entries, main, part_f = [], [], []
    for fam, n, d in CELLS:
        for inst in INSTANCES:
            qc, ideal, n2_log = logical(fam, n, d, inst)
            for arm in ARMS:
                out, secs = compile_arm(arm, qc, backend, psf, tf)
                isa, used_bad, n2 = check(out, backend, bad)
                esp, pred, ro, dur = estimate(out, t)
                got = mps_outcomes(out, 4670 + len(entries))
                exact = list(got) == [ideal]
                ok = isa and exact and 2 * n2 >= n2_log  # a collapse across the barrier would remove most two-qubit gates
                part = "F" if used_bad else "main"
                cell = cell_name(fam, n, d)
                e = dict(cell=cell, fam=fam, n=n, d=d, inst=inst, arm=arm, ideal=ideal, twoq=n2, twoq_logical=n2_log,
                         failed_used=used_bad, part=part, esp=esp, pred=pred, readout_only=ro, dur_us=dur,
                         active=len({out.find_bit(q).index for i in out.data if i.operation.name != "barrier"
                                     for q in i.qubits}),
                         isa=isa, exact_mps=exact, compile_s=round(secs, 3), shots=shots_for(n), submitted=ok)
                entries.append(e)
                if ok:
                    (part_f if part == "F" else main).append((len(entries) - 1, out))
                print(f"  {cell:8s} i{inst} {arm:4s} 2q {n2:4d} (logical {n2_log:4d}) failed {used_bad:2d} ESP {esp:.3g} "
                      f"pred {pred:.3g} readout-only {ro:.3g} dur {dur:5.2f} us exact {exact} isa {isa}"
                      f"{'' if ok else '  -> NOT SUBMITTED'}", flush=True)
    # arms interleaved: the build order is already cell, instance, arm
    jobs = {"main": main, "F": part_f}
    pubs = {}
    for name, items in jobs.items():
        pubs[name] = [dict(entry=k, shots=entries[k]["shots"]) for k, _ in items]
        if items:
            with open(os.path.join(out_dir, f"scale1_circuits_{name}.qpy"), "wb") as f:
                qpy.dump([c for _, c in items], f)
    npub = sum(len(v) for v in pubs.values())
    if npub > MAX_PUBS:
        raise SystemExit(f"{npub} circuits > {MAX_PUBS}: not prepared")
    est = sum(0.0005 * p["shots"] + 1.0 for v in pubs.values() for p in v)
    print(f"{len(pubs['main'])} circuits in the main job, {len(pubs['F'])} in part F; "
          f"{sum(p['shots'] for v in pubs.values() for p in v)} shots; rough QPU estimate {est:.0f} s")
    if est > USAGE_EST_CAP_S:
        raise SystemExit(f"rough QPU estimate {est:.0f} s > {USAGE_EST_CAP_S:.0f} s: not prepared")
    return entries, pubs, len(bad)


# ------------------------------------------------------------------ steps

def manifest_path(out_dir):
    return os.path.join(out_dir, "scale1_manifest.json")


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
    for name, h in man["qpy_sha256"].items():
        if sha256_file(os.path.join(a.out, f"scale1_circuits_{name}.qpy")) != h:
            raise SystemExit(f"scale1_circuits_{name}.qpy differs from the manifest's hash: stop.")
    return man


def _circuits(a, name):
    from qiskit import qpy
    with open(os.path.join(a.out, f"scale1_circuits_{name}.qpy"), "rb") as f:
        return qpy.load(f)


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
    psf = load(PSF_FILE, "scale1_psf")
    psf.WARN_WITHOUT_TARGET = False
    entries, pubs, nbad = build_all(backend, a.out, psf)
    man = dict(written=stamp(), device=backend.name, device_rule=rule, calibration_time=calibration_time(backend),
               failed_elements=nbad, script_sha256=sha256_file(os.path.abspath(__file__)),
               files={os.path.relpath(f, REPO): sha256_file(f) for f in (PSF_FILE, CIRC_FILE)},
               psf_version=getattr(psf, "__version__", None),
               qpy_sha256={name: sha256_file(os.path.join(a.out, f"scale1_circuits_{name}.qpy"))
                           for name in pubs if pubs[name]},
               entries=entries, pubs=pubs, job_ids={})
    write_manifest(a.out, man)
    print(f"{stamp()} manifest written before any submission: {sum(len(v) for v in pubs.values())} circuits; "
          f"device {backend.name}; calibration {man['calibration_time']}")
    return backend


def submit(a):
    header()
    man = _read(a)
    if man.get("job_ids"):
        raise SystemExit(f"jobs {man['job_ids']} already recorded: not submitting again.")
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    backend = QiskitRuntimeService().backend(man["device"])
    for name in ("main", "F"):
        if not man["pubs"].get(name):
            continue
        circs = _circuits(a, name)
        if len(circs) != len(man["pubs"][name]):
            raise SystemExit("the QPY file and the manifest disagree on the number of circuits: stop")
        sampler = SamplerV2(mode=backend)
        sampler.options.dynamical_decoupling.enable = False
        sampler.options.twirling.enable_gates = False
        sampler.options.twirling.enable_measure = False
        t_sub = stamp()
        try:
            job = sampler.run([(c, None, p["shots"]) for c, p in zip(circs, man["pubs"][name])])
        except Exception as exc:  # noqa: BLE001
            print(f"{stamp()} job {name} NOT submitted: {type(exc).__name__}: {str(exc)[:300]}")
            man.setdefault("submit_errors", {})[name] = f"{type(exc).__name__}: {str(exc)[:300]}"
            write_manifest(a.out, man)
            continue
        man["job_ids"][name] = job.job_id()
        man.setdefault("submitted", {})[name] = t_sub
        write_manifest(a.out, man)
        print(f"{stamp()} submitted job {name} {job.job_id()} ({len(circs)} circuits) to {man['device']}.")
    if "main" not in man["job_ids"]:
        raise SystemExit("the main job was not submitted")
    print(f"{stamp()} submitted jobs {man['job_ids']}. Results are not read here.")


def _sample_local(circs, pubs, base):
    rows = []
    for i, (c, p) in enumerate(zip(circs, pubs)):
        from qiskit_aer import AerSimulator
        counts = AerSimulator(method="matrix_product_state").run(active_circuit(c), shots=p["shots"],
                                                                 seed_simulator=base + i).result().get_counts()
        rows.append(dict(pub=i, counts=counts))
    return rows


def fetch(a):
    header()
    man = _read(a)
    if not man.get("job_ids"):
        raise SystemExit("no job recorded")
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService()
    for name, jid in man["job_ids"].items():
        path = os.path.join(a.out, f"scale1_raw_counts_{name}.json")
        if os.path.exists(path):
            print(f"{name}: already saved")
            continue
        job = svc.job(jid)
        t_end = time.time() + 60 * a.wait_min
        while True:
            st = str(job.status())
            print(f"{stamp()} job {name} {jid} status {st}", flush=True)
            if any(x in st.upper() for x in ("DONE", "ERROR", "CANCEL")):
                break
            if time.time() > t_end:
                raise SystemExit("not finished within the wait limit; run fetch again later")
            time.sleep(30)
        if "DONE" not in st.upper():
            print(f"job {name} ended with status {st}; recorded, nothing to score for it")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dict(fetched=stamp(), job_id=jid, status=st, rows=[]), f)
            continue
        res = job.result()
        rows = [dict(pub=i, counts=dict(r.join_data().get_counts())) for i, r in enumerate(res)]
        try:
            metrics = job.metrics()
        except Exception as exc:  # noqa: BLE001
            metrics = dict(error=type(exc).__name__)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(fetched=stamp(), job_id=jid, status=st, metrics=metrics, rows=rows), f, default=str)
        print(f"{stamp()} saved {os.path.basename(path)} ({len(rows)} results); usage {metrics.get('usage')}")
    print(f"{stamp()} saved all scale1 results")


# ------------------------------------------------------------------ scoring

def _s(counts, ideal, shots):
    hit = counts.get(ideal, 0)
    s = hit / shots
    se = math.sqrt(max(s * (1 - s), 0.0) / shots)
    ham = {}
    for k, v in counts.items():
        h = sum(1 for x, y in zip(k.replace(" ", ""), ideal) if x != y)
        ham[h] = ham.get(h, 0) + v
    return s, se, hit, ham


def score(a):
    man = _read(a)
    E = man["entries"]
    for name, plist in man["pubs"].items():
        path = os.path.join(a.out, f"scale1_raw_counts_{name}.json")
        if not plist or not os.path.exists(path):
            continue
        raw = json.load(open(path, encoding="utf-8"))
        for p, row in zip(plist, raw["rows"]):
            e = E[p["entry"]]
            s, se, hit, ham = _s(row["counts"], e["ideal"], sum(row["counts"].values()))
            e.update(S=s, S_se=se, hits=hit, hamming=dict(sorted(ham.items())), resolved=hit > 0 and s >= RESOLVED_SE * se)
    key = {(e["cell"], e["inst"], e["arm"]): e for e in E}
    out = [f"# SCALE1 scores ({stamp()})", "",
           f"device {man['device']}; calibration {man['calibration_time']}; jobs {man.get('job_ids')}; elements with "
           f"error >= {FAILED}: {man['failed_elements']}", "",
           "Clifford mirror circuits with a known answer; no claim of quantum advantage. S: share of shots equal to the "
           "ideal bitstring; resolved: S >= 5 standard errors.", "",
           "| cell | inst | arm | part | 2q | failed used | dur us | ESP | predicted | readout only | S (SE) | resolved |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in E:
        s = e.get("S")
        out.append(f"| {e['cell']} | {e['inst']} | {e['arm']} | {e['part']} | {e['twoq']} | {e['failed_used']} | "
                   f"{e['dur_us']:.2f} | {e['esp']:.3g} | {e['pred']:.3g} | {e['readout_only']:.3g} | "
                   + (f"{s:.4f} ({e['S_se']:.4f})" if s is not None else "not run") + " | "
                   + ("yes" if e.get("resolved") else "no") + " |")

    def width(arm, fam, d):
        best = 0
        for f2, n, d2 in CELLS:
            if (f2, d2) != (fam, d):
                continue
            if all(key.get((cell_name(f2, n, d2), i, arm), {}).get("resolved") for i in INSTANCES):
                best = max(best, n)
        return best
    W = {arm: {cell_name(f, 0, d).split("_")[0]: width(arm, f, d) for f, d in FAMILIES} for arm in ARMS}
    fams = list(W["PSF"])
    c1 = [W["PSF"][k] >= W["QK3"][k] for k in fams]
    v = {}
    v["C1"] = "CONFIRMED" if all(c1) else "REFUTED"
    # C2: pairs where QK3 used a failed element and PSF is resolved
    c2 = [(k, key[k[:2] + ("PSF",)]["S"], key[k]["S"]) for k in key
          if k[2] == "QK3" and key[k]["failed_used"] and key[k].get("S") is not None
          and key.get(k[:2] + ("PSF",), {}).get("resolved")]
    if len(c2) < 2:
        v["C2"] = "NOT TESTABLE"
    elif all(sq <= 0.2 * sp for _, sp, sq in c2):
        v["C2"] = "CONFIRMED"
    elif any(sq >= sp for _, sp, sq in c2):
        v["C2"] = "REFUTED"
    else:
        v["C2"] = "AMBIGUOUS"

    def versus(other, only_clean):
        prs = []
        for (cell, inst, arm), e in key.items():
            if arm != "PSF":
                continue
            o = key.get((cell, inst, other))
            if o is None or e.get("S") is None or o.get("S") is None:
                continue
            if only_clean and o["failed_used"]:
                continue
            if not (e.get("resolved") or o.get("resolved")):
                continue
            prs.append((cell, inst, e["S"], o["S"]))
        if not prs:
            return "NOT TESTABLE", prs
        ge = sum(1 for _, _, sp, so in prs if sp >= so)
        lo = sum(1 for _, _, sp, so in prs if sp < 0.8 * so)
        if ge * 2 >= len(prs):
            return "CONFIRMED", prs
        if lo * 2 > len(prs):
            return "REFUTED", prs
        return "AMBIGUOUS", prs
    v["C3a"], p3a = versus("QK3", True)
    v["C3b"], p3b = versus("QK3F", False)
    # C4: the fixed predictor picks the device's winner where it predicts a clear difference
    c4 = []
    for other in ("QK3", "QK3F"):
        for (cell, inst, arm), e in key.items():
            o = key.get((cell, inst, other))
            if arm != "PSF" or o is None or e.get("S") is None or o.get("S") is None:
                continue
            if not (e.get("resolved") or o.get("resolved")) or min(e["pred"], o["pred"]) <= 0:
                continue
            r = e["pred"] / o["pred"]
            if 0.9 <= r <= 1.1 or e["S"] == o["S"]:
                continue
            c4.append((cell, inst, other, r, (r > 1) == (e["S"] > o["S"])))
    hits = sum(1 for x in c4 if x[4])
    if len(c4) < 3:
        v["C4"] = "NOT TESTABLE"
    elif 3 * hits >= 2 * len(c4):
        v["C4"] = "CONFIRMED"
    elif 2 * hits < len(c4):
        v["C4"] = "REFUTED"
    else:
        v["C4"] = "AMBIGUOUS"
    largest = {}
    for arm in ARMS:
        r = [e for e in E if e["arm"] == arm and e.get("resolved")]
        if r:
            b = max(r, key=lambda e: (e["n"], e["twoq"]))
            largest[arm] = dict(cell=b["cell"], inst=b["inst"], n=b["n"], twoq=b["twoq"], dur_us=b["dur_us"], S=b["S"])
    out += ["", "### Signal width (largest n whose three instances are all resolved)", "",
            "| family | " + " | ".join(ARMS) + " |", "|---|" + "---|" * len(ARMS)]
    out += [f"| {k} | " + " | ".join(str(W[arm][k]) for arm in ARMS) + " |" for k in fams]
    out += ["", "### Predictions", "",
            f"- C1 (signal width PSF >= QK3 in each of {', '.join(fams)}): {c1} -> {v['C1']}",
            f"- C2 (QK3 on a failed element, PSF resolved: S_QK3 <= 0.2 S_PSF on every pair, >= 2 pairs): "
            + "; ".join(f"{k[0]} i{k[1]} PSF {sp:.4f} QK3 {sq:.4f}" for k, sp, sq in c2) + f" -> {v['C2']}",
            f"- C3a (PSF >= QK3 where QK3 uses no failed element, on at least half): {len(p3a)} pairs, PSF >= QK3 on "
            f"{sum(1 for x in p3a if x[2] >= x[3])} -> {v['C3a']}",
            f"- C3b (PSF >= QK3F, on at least half): {len(p3b)} pairs, PSF >= QK3F on "
            f"{sum(1 for x in p3b if x[2] >= x[3])} -> {v['C3b']}",
            f"- C4 (ESP x idle term picks the winner on >= 2 of 3 pairs it calls outside 0.9-1.1, >= 3 pairs): "
            f"{hits} of {len(c4)} -> {v['C4']}",
            "- C5 (reported): largest resolved circuit per arm: " + "; ".join(
                f"{arm} {x['cell']} i{x['inst']} ({x['n']} qubits, {x['twoq']} two-qubit gates, {x['dur_us']:.2f} us, "
                f"S {x['S']:.4f})" for arm, x in largest.items()),
            "- C5 (reported): S / predicted per circuit and Hamming profiles are in scale1_score.json."]
    txt = "\n".join(out) + "\n"
    with open(os.path.join(a.out, "scale1_score.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    with open(os.path.join(a.out, "scale1_score.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(entries=E, width=W, verdict=v, c2=c2, c3a=p3a, c3b=p3b, c4=c4, largest=largest), fh, indent=1,
                  default=str)
    print(txt)


def rehearse(a):
    warnings.simplefilter("ignore")
    prepare(a, fake=a.fake)
    man = _read(a)
    for k, name in enumerate(("main", "F")):
        if man["pubs"].get(name):
            rows = _sample_local(_circuits(a, name), man["pubs"][name], 10_000 * (k + 1))
            with open(os.path.join(a.out, f"scale1_raw_counts_{name}.json"), "w", encoding="utf-8") as f:
                json.dump(dict(fetched=stamp(), job_id="rehearsal", status="DONE (local)", rows=rows), f)
            man["job_ids"][name] = "rehearsal"
    write_manifest(a.out, man)
    score(a)
    print("REHEARSAL DONE (noiseless: every S should be 1; the verdicts mean nothing here)")


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
