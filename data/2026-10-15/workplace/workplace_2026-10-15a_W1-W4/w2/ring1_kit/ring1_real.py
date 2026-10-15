"""ring1_real.py -- RING1 (draft, 2026-10-15): does PSF-Zero's AI front end (a18, release 2026-10-14.1: the ring
formula) make a data re-uploading classifier classify better on a real device than Qiskit level 3 does?

To be pre-registered in RING1_preregistration.md (fixed before any real-device step). One job:

  model   DEPTH1's / W3's classifier: per layer RY(pi/2 x_q), RY(a) RZ(b) on every qubit, CZ ring (0,1)...(n-1,0);
          then RY(c); prediction sign(<Z_0>). Breast-cancer data, PCA to n. n = 4 and 6, L = 4 and 8 layers. Weights
          trained noiselessly (DEPTH1's recipe) and frozen with 24 test points per configuration (12 per class) in
          ring1_model.json, so that no training or scikit-learn runs here.
  arms    A18: the AI front end of release 2026-10-14.1 (benchmarks/psf_ai_compile.py, a18, on the release's
          psf_compile.py) given the device's Target (no fractional gates);
          QK3: Qiskit level 3 on the same Target, seed 0.
  control ghz_n6 compiled once by Qiskit level 3, submitted twice (CTL-A, CTL-B): its F ratio checks the job.

All circuits go in one SamplerV2 job, 800 shots per circuit and repeat, two repeats (the second in reverse order),
dynamical decoupling and twirling off.

Steps: rehearse (fake device, local sampler; --noisy uses the fake device's noise model, reported only), prepare,
submit, fetch, score.
    python -u <kit>/ring1_real.py rehearse --out data/2026-10-15/ring1_rehearsal [--noisy]
    python -u <kit>/ring1_real.py prepare  --out data/2026-10-15/ring1_run
    python -u <kit>/ring1_real.py submit   --out data/2026-10-15/ring1_run
    python -u <kit>/ring1_real.py fetch    --out data/2026-10-15/ring1_run
    python -u <kit>/ring1_real.py score    --out data/2026-10-15/ring1_run
The IBM API key is never read from or written to any file by this script. It runs no git and no pip. No
calibration value chooses anything.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
import warnings

REPO = os.path.abspath(os.getcwd())
for p in (os.path.join(REPO, "benchmarks"), REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

KIT = os.path.dirname(os.path.abspath(__file__))
T1_FILE = os.path.join(REPO, "data", "2026-10-14", "time1_kit", "time1_real.py")
T1_SHA = "171f8922446443bf8a27e1552f1ff604a4d243346759bc8f1c1fdad7b1025505"
PSF_FILE = os.path.join(REPO, "psf_compile.py")  # release 2026-10-14.1 (prototype c57)
PSF_SHA = "23d4aa4d1cdea7729d323c82d63386f43ccb302fc28ffe50f9ae58363459d0f8"
AI_FILE = os.path.join(REPO, "benchmarks", "psf_ai_compile.py")  # release 2026-10-14.1 (a18)
AI_SHA = "94bd54eeed0d2c2941d3dda574737acf029a71041ee47f572710e24ce545e5c1"
A15_FILE = os.path.join(REPO, "patches", "psf_ai_compile_a15_2026-10-14", "psf_ai_compile.py")  # loaded by a18
A15_SHA = "2c9771a3bed8cda90f3be6fa61ceeec605e1e7879eea61aa9a920fd0deac5bb5"
A14_FILE = os.path.join(REPO, "patches", "psf_ai_compile_a14_2026-10-13", "psf_ai_compile.py")  # loaded by a15
A14_SHA = "55a502c372b773d9667e0daf042ee448383d777da5d889d68c456462926791b5"
MODEL_FILE = os.path.join(KIT, "ring1_model.json")

SHOTS = 800  # per circuit and repeat
REPEATS = 2
USAGE_EST_CAP_S = 150.0  # the owner's remaining QPU time is about 2.5 min
EST_S_PER_SHOT, EST_S_PER_CIRCUIT = 0.0003, 0.10  # TIME1 / FRAC1 on ibm_fez fit 0.27 ms per shot + 0.04 s per circuit
FAILED = 0.5
EXACT_TOL = 1e-6
MIN_QK3_MARGIN = 0.05  # RA is not testable below this pooled QK3 margin


def _load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


T1 = _load(T1_FILE, "ring1_time1")
H2 = T1.H2
stamp, sha256_file = T1.stamp, T1.sha256_file


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
    fixed = ((T1_FILE, T1_SHA), (PSF_FILE, PSF_SHA), (AI_FILE, AI_SHA), (A15_FILE, A15_SHA), (A14_FILE, A14_SHA),
             (T1.HW2_FILE, T1.HW2_SHA))
    for f, h in fixed:
        got = sha256_file(f) if os.path.exists(f) else "MISSING"
        print("FILE", os.path.relpath(f, REPO), got)
        if got != h:
            raise SystemExit(f"{os.path.relpath(f, REPO)} differs from the hash fixed in this kit: stop")
    print("MODEL", os.path.relpath(MODEL_FILE, REPO), sha256_file(MODEL_FILE))


def compilers():
    """The release's psf_compile.py registered as `psf_compile` (a18 imports it under that name), then a18."""
    T1._load(T1.LAYOUT_FILE, "psf_smart_layout")
    pc = _load(PSF_FILE, "psf_compile")
    pc.WARN_WITHOUT_TARGET = False
    ai = _load(AI_FILE, "ring1_a18")
    return pc, ai


def classifier(x, theta, n, L):
    """DEPTH1's circuit (depth_eval.circuit), measured on every qubit; clbit q = logical qubit q."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    k = 0
    for _ in range(L):
        for q in range(n):
            qc.ry(np.pi / 2 * x[q], q)
        for q in range(n):
            qc.ry(theta[k], q)
            k += 1
            qc.rz(theta[k], q)
            k += 1
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    for q in range(n):
        qc.ry(theta[k], q)
        k += 1
    qc.measure_all()
    return qc


def ghz(n):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    qc.h(0)
    for q in range(n - 1):
        qc.cx(q, q + 1)
    qc.measure_all()
    return qc


def z0(dist_or_counts):
    """<Z> of clbit 0 (rightmost) from a distribution or counts: P(0) - P(1)."""
    tot = sum(dist_or_counts.values())
    p1 = sum(v for k, v in dist_or_counts.items() if k.replace(" ", "")[-1] == "1")
    return (tot - 2 * p1) / tot


def metrics(out, target, pc):
    """(two-qubit count, duration in us without measurements, ISA ok, uses a failed element)."""
    pos = {q: i for i, q in enumerate(out.qubits)}
    clock, isa, n2 = {}, True, 0
    for ins in out.data:
        nm = ins.operation.name
        if nm == "barrier":
            continue
        qs = tuple(pos[q] for q in ins.qubits)
        if nm not in target.operation_names or (qs not in target[nm] and None not in target[nm]):
            isa = False
        if nm in ("measure", "delay"):
            continue
        pr = target[nm].get(qs) if nm in target.operation_names else None
        d = (pr.duration or 0.0) if pr is not None else 0.0
        s = max(clock.get(q, 0.0) for q in qs)
        for q in qs:
            clock[q] = s + d
        n2 += len(qs) == 2
    edges, bad = pc._failed_elements(target, FAILED)
    return n2, (max(clock.values()) if clock else 0.0) * 1e6, isa, bool(pc._uses_failed(out, edges, bad))


def readout_qubit(out):
    """The physical qubit whose measurement is written to clbit 0 (logical qubit 0): reported, chooses nothing."""
    for ins in out.data:
        if ins.operation.name == "measure" and out.find_bit(ins.clbits[0]).index == 0:
            return out.find_bit(ins.qubits[0]).index
    return None


def compile_arm(arm, qc, target, pc, ai):
    t0 = time.perf_counter()
    if arm == "QK3":
        from qiskit.transpiler import generate_preset_pass_manager
        out = generate_preset_pass_manager(3, target=target, seed_transpiler=0).run(qc)
        chosen = "QK3"
    else:
        basis = [g for g in target.operation_names if g in ("cz", "rz", "sx", "x")]
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, info = ai.compile_for_model_circuit(qc, target.build_coupling_map(), basis, target=target,
                                                     return_info=True)
        chosen = info.get("chosen")
    return out, time.perf_counter() - t0, chosen


def build_all(name, target, out_dir):
    from qiskit import qpy
    pc, ai = compilers()
    model = json.load(open(MODEL_FILE, encoding="utf-8"))
    entries, circs, ideal = [], [], {}
    print(f"{stamp()} compiling for {name}", flush=True)
    for cfg in model["configs"]:
        n, L, th = cfg["n"], cfg["L"], cfg["theta"]
        for i, (x, y, zi) in enumerate(zip(cfg["X"], cfg["y"], cfg["z_ideal"])):
            test = f"clf_n{n}_L{L}_p{i:02d}"
            qc = classifier(x, th, n, L)
            dist, _ = H2.measured_distribution(qc)
            if abs(z0(dist) - zi) > 1e-9:
                raise SystemExit(f"{test}: the ideal <Z_0> differs from ring1_model.json: stop")
            ideal[test] = dict(n=n, L=L, y=y, z=zi, sum_p2=float(sum(v * v for v in dist.values())))
            start = len(entries)
            for arm in ("A18", "QK3"):
                out, secs, chosen = compile_arm(arm, qc, target, pc, ai)
                n2, dur, isa, failed = metrics(out, target, pc)
                d, active = H2.measured_distribution(out)
                exact = None if d is None else max(abs(d.get(k, 0.0) - dist.get(k, 0.0)) for k in set(d) | set(dist))
                ok = isa and not failed and exact is not None and exact <= EXACT_TOL
                entries.append(dict(part="CLF", test=test, n=n, L=L, point=i, arm=arm, chosen=chosen, twoq=n2,
                                    dur_us=dur, active=active, exact_maxdiff=exact, isa=isa, uses_failed=failed,
                                    compile_s=round(secs, 3), q0_phys=readout_qubit(out), submitted=ok,
                                    sig=H2.sig(out)))
                if ok:
                    circs.append((len(entries) - 1, out))
                else:
                    print(f"  {test} {arm}: NOT SUBMITTED (isa {isa}, failed {failed}, exact {exact})", flush=True)
            if not all(e["submitted"] for e in entries[start:]):
                for e in entries[start:]:
                    e["submitted"] = False
                circs = [(k, c) for k, c in circs if k < start]
                print(f"  {test}: pair incomplete, neither submitted", flush=True)
        sub = [e for e in entries if e["n"] == n and e["L"] == L and e["submitted"]]
        for arm in ("A18", "QK3"):
            es = [e for e in sub if e["arm"] == arm]
            if es:
                print(f"  n={n} L={L} {arm}: {len(es)} circuits; 2q {np.mean([e['twoq'] for e in es]):.1f}; duration "
                      f"{np.mean([e['dur_us'] for e in es]):.2f} us (mean); chosen {sorted({e['chosen'] for e in es})}",
                      flush=True)
    from qiskit.transpiler import generate_preset_pass_manager
    g = ghz(6)
    gd, _ = H2.measured_distribution(g)
    ideal["ghz_n6"] = dict(n=6, sum_p2=float(sum(v * v for v in gd.values())))
    go = generate_preset_pass_manager(3, target=target, seed_transpiler=0).run(g)
    n2, dur, isa, failed = metrics(go, target, pc)
    for arm in ("CTL-A", "CTL-B"):
        entries.append(dict(part="CTL", test="ghz_n6", arm=arm, twoq=n2, dur_us=dur, isa=isa, uses_failed=failed,
                            submitted=isa and not failed, sig=H2.sig(go)))
        if entries[-1]["submitted"]:
            circs.append((len(entries) - 1, go))
    order = [(k, c, 1) for k, c in circs] + [(k, c, 2) for k, c in circs[::-1]]
    pubs = [dict(entry=k, repeat=r, shots=SHOTS) for k, _, r in order]
    est = sum(EST_S_PER_SHOT * p["shots"] + EST_S_PER_CIRCUIT for p in pubs)
    print(f"{len(pubs)} circuits, {sum(p['shots'] for p in pubs)} shots; rough QPU estimate {est:.0f} s")
    if est > USAGE_EST_CAP_S:
        raise SystemExit(f"rough QPU estimate {est:.0f} s > {USAGE_EST_CAP_S:.0f} s: not prepared")
    with open(os.path.join(out_dir, "ring1_circuits.qpy"), "wb") as f:
        qpy.dump([c for _, c, _ in order], f)
    return entries, pubs, ideal


def manifest_path(out_dir):
    return os.path.join(out_dir, "ring1_manifest.json")


def write_manifest(out_dir, man):
    with open(manifest_path(out_dir), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, default=str)


def _read(a):
    with open(manifest_path(a.out), encoding="utf-8") as f:
        man = json.load(f)
    if sha256_file(os.path.join(a.out, "ring1_circuits.qpy")) != man["qpy_sha256"]:
        raise SystemExit("ring1_circuits.qpy differs from the manifest's hash: stop.")
    return man


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
            raise SystemExit("No usable saved IBM account (" + type(exc).__name__ + "). Save it once in the terminal; "
                             "never paste the key into a file or chat.")
        b, rule = T1.device_rule(svc)  # ibm_fez if operational, otherwise TIME1's rule
        name, target, cal = b.name, b.target, H2.calibration_time(b)
    else:
        from qiskit_ibm_runtime import fake_provider
        b = getattr(fake_provider, fake)()
        name, rule, target, cal = b.name, [dict(name=fake, fake=True)], b.target, None
    print("device rule:", rule)
    entries, pubs, ideal = build_all(name, target, a.out)
    man = dict(written=stamp(), device=name, device_rule=rule, calibration_time=cal, shots=SHOTS, repeats=REPEATS,
               script_sha256=sha256_file(os.path.abspath(__file__)), model_sha256=sha256_file(MODEL_FILE),
               files={os.path.relpath(f, REPO): sha256_file(f) for f in (T1_FILE, T1.HW2_FILE, PSF_FILE, AI_FILE,
                                                                        A15_FILE, A14_FILE, T1.LAYOUT_FILE)},
               qpy_sha256=sha256_file(os.path.join(a.out, "ring1_circuits.qpy")), entries=entries, pubs=pubs,
               ideal=ideal, job_id=None)
    write_manifest(a.out, man)
    print(f"{stamp()} manifest written before any submission: {len(pubs)} circuits; device {name}; calibration {cal}")


def submit(a):
    header()
    man = _read(a)
    if man.get("job_id"):
        raise SystemExit(f"job {man['job_id']} already recorded: not submitting again.")
    from qiskit import qpy
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    with open(os.path.join(a.out, "ring1_circuits.qpy"), "rb") as f:
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
    with open(os.path.join(a.out, "ring1_raw_counts.json"), "w", encoding="utf-8") as f:
        json.dump(dict(fetched=stamp(), job_id=man["job_id"], status=st, metrics=metrics_, rows=rows), f, default=str)
    print(f"{stamp()} saved ring1_raw_counts.json ({len(rows)} results); usage {metrics_.get('usage')}")


def _gm(v):
    v = [x for x in v if x > 0 and math.isfinite(x)]
    return math.exp(float(np.mean([math.log(x) for x in v]))) if v else float("nan")


def score(a):
    man = _read(a)
    with open(os.path.join(a.out, "ring1_raw_counts.json"), encoding="utf-8") as f:
        raw = json.load(f)
    ent = man["entries"]
    pooled = {}
    for p, row in zip(man["pubs"], raw["rows"]):
        e = ent[p["entry"]]
        c = pooled.setdefault((e["test"], e["arm"]), {})
        for k, v in row["counts"].items():
            c[k] = c.get(k, 0) + v
    model = json.load(open(MODEL_FILE, encoding="utf-8"))
    if sha256_file(MODEL_FILE) != man["model_sha256"]:
        raise SystemExit("ring1_model.json differs from the manifest's: stop")
    dist_cache = {}

    def ideal_dist(test):
        if test not in dist_cache:
            if test == "ghz_n6":
                qc = ghz(6)
            else:
                n, L, i = man["ideal"][test]["n"], man["ideal"][test]["L"], int(test.split("_p")[1])
                cfg = next(c for c in model["configs"] if c["n"] == n and c["L"] == L)
                qc = classifier(cfg["X"][i], cfg["theta"], n, L)
            d, _ = H2.measured_distribution(qc)
            if abs(sum(v * v for v in d.values()) - man["ideal"][test]["sum_p2"]) > 1e-9:
                raise SystemExit(f"ideal distribution of {test} differs from the manifest's: stop")
            dist_cache[test] = d
        return dist_cache[test]

    rows = {}
    for e in ent:
        if not e["submitted"] or e["part"] != "CLF":
            continue
        c = pooled.get((e["test"], e["arm"]))
        if not c:
            continue
        idl = man["ideal"][e["test"]]
        z = z0(c)
        f, _ = H2.xeb(ideal_dist(e["test"]), c, idl["n"], sum(c.values()))
        rows[(e["test"], e["arm"])] = dict(n=idl["n"], L=idl["L"], y=idl["y"], z_ideal=idl["z"], z=z, f=f,
                                           dur=e["dur_us"], twoq=e["twoq"], q0=e.get("q0_phys"))
    tests = sorted({t for t, arm in rows if (t, "A18") in rows and (t, "QK3") in rows})
    out = [f"# RING1 scores ({stamp()})", "",
           f"device {man['device']}; calibration {man['calibration_time']}; job {man.get('job_id')}; usage "
           f"{(raw.get('metrics') or {}).get('usage')}", "",
           "Per point: z = P(0) - P(1) of logical qubit 0 over both repeats' pooled counts; correct if sign(z) = y "
           "(z = 0 counts half); margin = y z (the correct class's probability minus the other's); F = normalised "
           "linear XEB against the exact distribution.", "",
           "| n | L | arm | points | accuracy | margin (SE) | ideal margin | F (mean) | 2q | duration us | read on |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    cfgs = sorted({(rows[(t, 'A18')]["n"], rows[(t, 'A18')]["L"]) for t in tests})
    per = {}
    for n, L in cfgs:
        ts = [t for t in tests if rows[(t, "A18")]["n"] == n and rows[(t, "A18")]["L"] == L]
        for arm in ("A18", "QK3"):
            R = [rows[(t, arm)] for t in ts]
            acc = float(np.mean([1.0 if r["y"] * r["z"] > 0 else (0.5 if r["z"] == 0 else 0.0) for r in R]))
            mg = np.array([r["y"] * r["z"] for r in R])
            per[(n, L, arm)] = dict(acc=acc, margin=float(mg.mean()), se=float(mg.std(ddof=1) / math.sqrt(len(mg))),
                                    f=float(np.mean([r["f"] for r in R])), dur=float(np.mean([r["dur"] for r in R])),
                                    twoq=float(np.mean([r["twoq"] for r in R])), k=len(R),
                                    ideal=float(np.mean([r["y"] * r["z_ideal"] for r in R])),
                                    q0=sorted({r["q0"] for r in R}, key=str))
            q = per[(n, L, arm)]
            out.append(f"| {n} | {L} | {arm} | {q['k']} | {q['acc']:.3f} | {q['margin']:.4f} ({q['se']:.4f}) | "
                       f"{q['ideal']:.4f} | {q['f']:.4f} | {q['twoq']:.1f} | {q['dur']:.2f} | {', '.join(map(str, q['q0']))} |")
    allR = {arm: [rows[(t, arm)] for t in tests] for arm in ("A18", "QK3")}
    mA = float(np.mean([r["y"] * r["z"] for r in allR["A18"]])) if tests else float("nan")
    mQ = float(np.mean([r["y"] * r["z"] for r in allR["QK3"]])) if tests else float("nan")
    d = np.array([rA["y"] * (rA["z"] - rQ["z"]) for rA, rQ in zip(allR["A18"], allR["QK3"])])
    dse = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
    accA = float(np.mean([1.0 if r["y"] * r["z"] > 0 else (0.5 if r["z"] == 0 else 0.0) for r in allR["A18"]])) if tests else float("nan")
    accQ = float(np.mean([1.0 if r["y"] * r["z"] > 0 else (0.5 if r["z"] == 0 else 0.0) for r in allR["QK3"]])) if tests else float("nan")
    v = {}
    up = sum(1 for n, L in cfgs if per[(n, L, "A18")]["margin"] > per[(n, L, "QK3")]["margin"])
    ratio = mA / mQ if mQ > 0 else float("nan")
    v["RA"] = "NOT TESTABLE" if not (mQ > MIN_QK3_MARGIN and len(cfgs) >= 3) else (
        "CONFIRMED" if ratio >= 1.05 else "REFUTED")
    v["RB"] = "NOT TESTABLE" if not tests else ("CONFIRMED" if accA >= accQ else "REFUTED")
    fr = [per[(n, L, "A18")]["f"] / per[(n, L, "QK3")]["f"] for n, L in cfgs
          if per[(n, L, "A18")]["f"] > 0 and per[(n, L, "QK3")]["f"] > 0]
    g = [None, None]
    for e in ent:
        if e["part"] == "CTL" and e["submitted"]:
            c = pooled.get((e["test"], e["arm"]))
            if c:
                g[0 if e["arm"] == "CTL-A" else 1] = H2.xeb(ideal_dist("ghz_n6"), c, 6, sum(c.values()))[0]
    v["RE"] = "n/a" if None in g or g[1] <= 0 else ("CONFIRMED" if abs(g[0] / g[1] - 1) <= 0.10 else "REFUTED")
    out += ["", "## Predictions", "",
            f"- RA (pooled margin ratio A18 / QK3 >= 1.05): {mA:.4f} / {mQ:.4f} = {ratio:.3f} -> {v['RA']}",
            f"- RA2 (reported): A18's margin higher in {up} of {len(cfgs)} configurations (not independent: the "
            "configurations of one n share each arm's placement)",
            f"- RB (pooled accuracy A18 >= QK3 over {len(tests)} points each): {accA:.3f} vs {accQ:.3f} -> {v['RB']}",
            f"- RC (reported): F ratio A18 / QK3 per configuration: {', '.join(f'{x:.3f}' for x in fr)}; geometric "
            f"mean {_gm(fr):.3f}",
            f"- RD (reported): paired margin difference A18 - QK3 over {len(d)} points: {float(d.mean()) if len(d) else float('nan'):+.4f} "
            f"(SE {dse:.4f})",
            f"- RE (control ghz_n6, the same circuit twice: |F ratio - 1| <= 0.10): "
            f"{'n/a' if None in g else f'{g[0]:.4f} / {g[1]:.4f}'} -> {v['RE']}",
            f"- **RING1 (RA and RB confirmed): {'CONFIRMED' if v['RA'] == 'CONFIRMED' and v['RB'] == 'CONFIRMED' else 'NOT CONFIRMED'}**"]
    txt = "\n".join(out) + "\n"
    with open(os.path.join(a.out, "ring1_score.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    with open(os.path.join(a.out, "ring1_score.json"), "w", encoding="utf-8") as f:
        json.dump(dict(per={f"{k[0]}|{k[1]}|{k[2]}": x for k, x in per.items()}, configs_higher=up,
                       rows={f"{k[0]}|{k[1]}": x for k, x in rows.items()}, verdict=v, margin_ratio=ratio,
                       acc=dict(A18=accA, QK3=accQ), ghz_f=g), f, indent=1)
    print(txt)


def rehearse(a):
    warnings.simplefilter("ignore")
    prepare(a, fake=a.fake)
    man = _read(a)
    from qiskit import qpy
    from qiskit_aer import AerSimulator
    with open(os.path.join(a.out, "ring1_circuits.qpy"), "rb") as f:
        circs = qpy.load(f)
    if a.noisy:
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay
        from qiskit_ibm_runtime import fake_provider
        fb = getattr(fake_provider, a.fake)()
        sim = AerSimulator.from_backend(fb)
        # idle qubits relax only on explicit delays in Aer's device model, so the simulated copies (not the
        # submitted circuits) are scheduled ALAP with delays first; the device does this physically
        sched = PassManager([ALAPScheduleAnalysis(target=fb.target), PadDelay(target=fb.target)])
        circs = [sched.run(c) for c in circs]
        note = (f"local rehearsal with {a.fake}'s noise model, idle relaxation included (reported only; nothing is "
                "chosen by it)")
    else:
        sim = AerSimulator()
        note = "local noiseless rehearsal"
    if len({p["shots"] for p in man["pubs"]}) != 1:
        raise SystemExit("the rehearsal expects one shot count for every circuit")
    res = sim.run(circs, shots=man["pubs"][0]["shots"], seed_simulator=3000).result()  # one call: the noise model's
    rows = [dict(pub=i, counts=res.get_counts(i)) for i in range(len(circs))]  # set-up is paid once
    with open(os.path.join(a.out, "ring1_raw_counts.json"), "w", encoding="utf-8") as f:
        json.dump(dict(fetched=stamp(), job_id="rehearsal", status="DONE (local)",
                       metrics=dict(usage=dict(quantum_seconds=0), note=note), rows=rows), f)
    man["job_id"] = "rehearsal"
    write_manifest(a.out, man)
    score(a)
    print("REHEARSAL DONE (" + note + "; the verdicts mean nothing here)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("rehearse", "prepare", "submit", "fetch", "score"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--fake", default="FakeFez")
    ap.add_argument("--noisy", action="store_true")
    ap.add_argument("--wait-min", type=float, default=40.0)
    a = ap.parse_args()
    if a.step == "prepare":
        prepare(a)
    else:
        dict(rehearse=rehearse, submit=submit, fetch=fetch, score=score)[a.step](a)


if __name__ == "__main__":
    main()
