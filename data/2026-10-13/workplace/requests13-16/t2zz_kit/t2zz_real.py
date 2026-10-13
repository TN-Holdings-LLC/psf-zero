"""t2zz_real.py -- T2*/ZZ v2: a small hardware measurement of unechoed dephasing (Ramsey against Hahn echo) and of
residual ZZ to one neighbour, on three (target, spectator) pairs in parallel. Design: t2star-zz-design-v2-2026-10-13.md.

Steps (only `submit` uses QPU time; `prepare`, `submit` and `fetch` need the IBM connection; the others do not):

  rehearse  the whole flow on a fake device (default FakeFez) with Aer and the fake device's noise model, plus a
            KNOWN injected residual ZZ and quasi-static detuning per pair, so that `score` can be checked against
            the truth. No IBM connection.
  prepare   device rule (of ibm_fez and ibm_marrakesh, the operational one with fewer pending jobs; neither: stop),
            qubit rule, circuits with explicit delays, checks, circuits and manifest written BEFORE any submission.
            Refuses if the manifest exists.
  submit    ONE SamplerV2 job; decoupling and twirling off; the executed schedule requested
            (experimental scheduler_timing, no extra circuits or shots).
  fetch     waits for the job, saves raw counts and the returned schedule.
  run       submit, then fetch.
  score     offline: coherences, ZZ rates, predictions E1-E6, and the schedule check.

Usage:
    python -u t2zz_real.py rehearse --out <dir>/t2zz_rehearsal
    python -u t2zz_real.py prepare  --out <dir>/t2zz
    python -u t2zz_real.py submit   --out <dir>/t2zz
    python -u t2zz_real.py fetch    --out <dir>/t2zz     (or: run = submit + fetch)
    python -u t2zz_real.py score    --out <dir>/t2zz

The IBM API key is never read from or written to any file by this script. It runs no git and no pip. It does not
need the psf-zero repository."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import time
import warnings

import numpy as np

TAUS_US = (0, 5, 10, 20, 30, 40)          # v2: 30 us added
FIT_MAX_US = 40                            # v2: ZZ fit over 0 < tau <= 40 us, weighted (v1: <= 20 us, unweighted)
SEQS = ("RX", "RY", "EX", "EY")            # Ramsey X/Y, Hahn echo X/Y
SPECS = (0, 1)
SHOTS = 2000                               # v2: 2,000 shots per circuit (v1: 1,000)
DEVICES = ("ibm_fez", "ibm_marrakesh")
FAILED = 0.5
MIN_T2_A = 25e-6                           # v2: rule (a) needs T2 >= 25 us (v1: > 10 us)
MIN_PAIR_DIST = 3                          # shortest-path distance between qubits of different pairs
USAGE_EST_CAP_S = 110.0                    # rough rule: 0.5 ms per shot + 1 s per circuit
BOOT = 0                                   # analytic standard errors (delta method); see zz_fit
# rehearsal truth (injected in the local simulation only): per pair (a, b, c)
REH_ZETA_HZ = (4000.0, 1500.0, 0.0)        # residual ZZ
REH_SIGMA_HZ = (6000.0, 3000.0, 0.0)       # quasi-static detuning spread (echo removes it)
REH_BATCHES = 10


# ------------------------------------------------------------------ small helpers

def stamp():
    u = dt.datetime.now(dt.timezone.utc)
    try:
        from zoneinfo import ZoneInfo
        c = u.astimezone(ZoneInfo("Europe/Zurich"))
        return f"{u.isoformat(timespec='seconds')} ({c.strftime('%Y-%m-%d %H:%M:%S %Z')})"
    except Exception:  # noqa: BLE001
        return u.isoformat(timespec="seconds")


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def header():
    import qiskit
    print(f"{stamp()} | platform {platform.platform()} | python {platform.python_version()} | qiskit "
          f"{qiskit.__version__}")
    for m in ("qiskit_aer", "qiskit_ibm_runtime"):
        try:
            mod = __import__(m)
            print(f"  {m} {getattr(mod, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {m} MISSING ({type(exc).__name__})")
    print("SCRIPT", os.path.basename(__file__), sha256_file(os.path.abspath(__file__)))


def manifest_path(out):
    return os.path.join(out, "t2zz_manifest.json")


def write_manifest(out, man):
    with open(manifest_path(out), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, default=str)


def _read(a):
    with open(manifest_path(a.out), encoding="utf-8") as f:
        man = json.load(f)
    if sha256_file(os.path.join(a.out, "t2zz_circuits.qpy")) != man["qpy_sha256"]:
        raise SystemExit("t2zz_circuits.qpy differs from the manifest's hash: stop.")
    return man


# ------------------------------------------------------------------ qubit rule

def _err(target, name, qargs):
    try:
        p = target[name][qargs]
        return None if p is None else p.error
    except KeyError:
        return None


def device_qubits(backend):
    """per qubit: T1, T2 (s), readout and sx error, operational flag; undirected couplers of usable CZ."""
    t = backend.target
    info = {}
    for q in range(t.num_qubits):
        qp = t.qubit_properties[q] if t.qubit_properties else None
        t1 = getattr(qp, "t1", None)
        t2 = getattr(qp, "t2", None)
        em, es = _err(t, "measure", (q,)), _err(t, "sx", (q,))
        ok = (t1 is not None and t2 is not None and em is not None and es is not None and em < FAILED
              and es < FAILED)
        info[q] = dict(t1=t1, t2=t2, ro=em, sx=es, ok=bool(ok))
    adj = {q: set() for q in info}
    twoq = "cz" if "cz" in t.operation_names else ("ecr" if "ecr" in t.operation_names else None)
    if twoq is None:
        raise SystemExit("no cz/ecr in the Target: stop")
    for qargs, p in t[twoq].items():
        if qargs is None or len(qargs) != 2:
            continue
        a, b = qargs
        if p is not None and p.error is not None and p.error < FAILED and info[a]["ok"] and info[b]["ok"]:
            adj[a].add(b)
            adj[b].add(a)
    return info, adj


def _dist_from(adj, src):
    d = {src: 0}
    fr = [src]
    while fr:
        nx = []
        for u in fr:
            for v in adj[u]:
                if v not in d:
                    d[v] = d[u] + 1
                    nx.append(v)
        fr = nx
    return d


def select_pairs(info, adj):
    """The rule fixed in the design (section 2). Returns [(label, target, spectator, why)]."""
    ok = [q for q in info if info[q]["ok"] and adj[q]]
    t2 = {q: info[q]["t2"] for q in ok}
    med = float(np.median(list(t2.values())))
    orders = {
        "a": sorted([q for q in ok if t2[q] >= MIN_T2_A], key=lambda q: (t2[q], q)),
        "b": sorted(ok, key=lambda q: (abs(t2[q] - med), q)),
        "c": sorted(ok, key=lambda q: (-t2[q], q)),
    }
    chosen, used = [], []
    for lab in ("a", "b", "c"):
        for q in orders[lab]:
            if q in used:
                continue
            s = max(adj[q], key=lambda v: (info[v]["t1"], -v))
            if s in used:
                continue
            far = True
            for u in used:
                du = _dist_from(adj, u)
                if du.get(q, 99) < MIN_PAIR_DIST or du.get(s, 99) < MIN_PAIR_DIST:
                    far = False
                    break
            if not far:
                continue
            chosen.append(dict(label=lab, target=q, spectator=s, t1_t=info[q]["t1"], t2_t=t2[q],
                               t1_s=info[s]["t1"], ro_t=info[q]["ro"], ro_s=info[s]["ro"], t2_median=med))
            used += [q, s]
            break
        else:
            raise SystemExit(f"no pair found for rule ({lab}): stop")
    return chosen


# ------------------------------------------------------------------ circuits

def _h(c, q):
    c.rz(math.pi / 2, q)
    c.sx(q)
    c.rz(math.pi / 2, q)


def _delay_dt(seconds, target):
    """seconds -> a delay length in dt, rounded to the Target's granularity and at least its minimum length
    (0 stays 0)."""
    if seconds <= 0:
        return 0
    g = max(1, int(getattr(target, "granularity", 1) or 1))
    n = int(round(seconds / target.dt / g)) * g
    return max(n, int(getattr(target, "min_length", 1) or 1))


def build_circuit(n_qubits, pairs, seq, tau_s, spec, target):
    """One circuit, all pairs in parallel: [prep] | [wait] | [final rotation] | measure (barriers between the
    layers, so that the spectator's X precedes the wait whatever the device's timing).
    Returns the circuit and the actually programmed free-evolution time in seconds."""
    from qiskit import QuantumCircuit
    m = 2 * len(pairs)
    c = QuantumCircuit(n_qubits, m, name=f"{seq}_{int(round(tau_s * 1e6))}us_s{spec}")
    allq = [p["target"] for p in pairs] + [p["spectator"] for p in pairs]
    for p in pairs:
        _h(c, p["target"])
        if spec:
            c.x(p["spectator"])
    c.barrier(allq)
    if seq in ("RX", "RY"):
        n = _delay_dt(tau_s, target)
        for p in pairs:
            if n:
                c.delay(n, p["target"], unit="dt")
        prog = n * target.dt
    else:
        n = _delay_dt(tau_s / 2, target)
        for p in pairs:
            if n:
                c.delay(n, p["target"], unit="dt")
            c.x(p["target"])
            if n:
                c.delay(n, p["target"], unit="dt")
        prog = 2 * n * target.dt
    c.barrier(allq)
    for p in pairs:
        if seq in ("RY", "EY"):
            c.rz(-math.pi / 2, p["target"])
        _h(c, p["target"])
    c.barrier(allq)
    for k, p in enumerate(pairs):
        c.measure(p["target"], 2 * k)
        c.measure(p["spectator"], 2 * k + 1)
    return c, prog


def build_reference(n_qubits, pairs, ones):
    from qiskit import QuantumCircuit
    c = QuantumCircuit(n_qubits, 2 * len(pairs), name=f"ref{int(ones)}")
    allq = [p["target"] for p in pairs] + [p["spectator"] for p in pairs]
    if ones:
        for q in allq:
            c.x(q)
    c.barrier(allq)
    for k, p in enumerate(pairs):
        c.measure(p["target"], 2 * k)
        c.measure(p["spectator"], 2 * k + 1)
    return c


def build_all(backend, pairs):
    t = backend.target
    circs, pubs = [], []
    for seq in SEQS:
        for tau in TAUS_US:
            for s in SPECS:
                c, prog = build_circuit(t.num_qubits, pairs, seq, tau * 1e-6, s, t)
                circs.append(c)
                pubs.append(dict(kind="seq", seq=seq, tau_us=tau, tau_prog_s=prog, spec=s, shots=SHOTS))
    for ones in (0, 1):
        circs.append(build_reference(t.num_qubits, pairs, ones))
        pubs.append(dict(kind="ref", ones=ones, shots=SHOTS))
    return circs, pubs


def check_circuits(circs, backend):
    t = backend.target
    for c in circs:
        for ins in c.data:
            name = ins.operation.name
            if name == "barrier":
                continue
            qargs = tuple(c.find_bit(q).index for q in ins.qubits)
            if name == "delay":
                if ins.operation.unit != "dt":
                    raise SystemExit(f"{c.name}: delay not in dt")
                continue
            if not t.instruction_supported(name, qargs):
                raise SystemExit(f"{c.name}: {name} on {qargs} not supported by the Target: stop")


# ------------------------------------------------------------------ steps

def _connect():
    from qiskit_ibm_runtime import QiskitRuntimeService
    try:
        return QiskitRuntimeService()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("No usable saved IBM account (" + type(exc).__name__ + "). Save it once in the terminal "
                         "with QiskitRuntimeService.save_account(...); never paste the key into a file or chat.")


def prepare(a, fake=None):
    header()
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(manifest_path(a.out)):
        raise SystemExit(f"{manifest_path(a.out)} exists: already prepared. Not preparing again.")
    if fake is None:
        svc = _connect()
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
    info, adj = device_qubits(backend)
    pairs = select_pairs(info, adj)
    for p in pairs:
        print(f"qubit rule ({p['label']}): target {p['target']} (T2 {p['t2_t']*1e6:.0f} us, T1 "
              f"{p['t1_t']*1e6:.0f} us), spectator {p['spectator']} (T1 {p['t1_s']*1e6:.0f} us)")
    circs, pubs = build_all(backend, pairs)
    check_circuits(circs, backend)
    shots = sum(p["shots"] for p in pubs)
    est = 0.5e-3 * shots + 1.0 * len(pubs)
    print(f"{len(pubs)} circuits, {shots} shots; rough QPU estimate {est:.0f} s (HW1's ratio 0.40: about "
          f"{0.4 * est:.0f} s)")
    if est > USAGE_EST_CAP_S:
        raise SystemExit(f"rough QPU estimate {est:.0f} s > {USAGE_EST_CAP_S:.0f} s: not prepared")
    from qiskit import qpy
    with open(os.path.join(a.out, "t2zz_circuits.qpy"), "wb") as f:
        qpy.dump(circs, f)
    try:
        cal = str(backend.properties().last_update_date)
    except Exception:  # noqa: BLE001
        cal = None
    man = dict(written=stamp(), device=backend.name, device_rule=rule, calibration_time=cal, dt=backend.target.dt,
               pairs=pairs, pubs=pubs, shots=SHOTS, taus_us=TAUS_US, fit_max_us=FIT_MAX_US,
               script_sha256=sha256_file(os.path.abspath(__file__)),
               qpy_sha256=sha256_file(os.path.join(a.out, "t2zz_circuits.qpy")),
               qubits={str(q): info[q] for p in pairs for q in (p["target"], p["spectator"])},
               rough_qpu_s=est)
    write_manifest(a.out, man)
    print(f"{stamp()} manifest written before any submission: {manifest_path(a.out)}")
    return man


def submit(a):
    header()
    man = _read(a)
    if man.get("job_id"):
        raise SystemExit(f"job {man['job_id']} already recorded: not submitting again.")
    from qiskit import qpy
    from qiskit_ibm_runtime import SamplerV2
    with open(os.path.join(a.out, "t2zz_circuits.qpy"), "rb") as f:
        circs = qpy.load(f)
    if len(circs) != len(man["pubs"]):
        raise SystemExit("the QPY file and the manifest disagree on the number of circuits: stop")
    backend = _connect().backend(man["device"])
    sampler = SamplerV2(mode=backend)
    sampler.options.dynamical_decoupling.enable = False
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    sampler.options.experimental = {"execution": {"scheduler_timing": True}}
    t_sub = stamp()
    job = sampler.run([(c, None, p["shots"]) for c, p in zip(circs, man["pubs"])])
    man["job_id"], man["submitted"] = job.job_id(), t_sub
    write_manifest(a.out, man)
    print(f"{stamp()} submitted job {job.job_id()} ({len(circs)} circuits, "
          f"{sum(p['shots'] for p in man['pubs'])} shots) to {man['device']}. Results are not read here.")


def fetch(a):
    header()
    man = _read(a)
    if not man.get("job_id"):
        raise SystemExit("no job recorded")
    job = _connect().job(man["job_id"])
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
    rows = []
    for i, r in enumerate(res):
        timing = None
        try:
            timing = r.metadata["compilation"]["scheduler_timing"]["timing"]
        except Exception:  # noqa: BLE001
            pass
        rows.append(dict(pub=i, counts=dict(r.join_data().get_counts()), timing=timing))
    try:
        metrics = job.metrics()
    except Exception as exc:  # noqa: BLE001
        metrics = dict(error=type(exc).__name__)
    with open(os.path.join(a.out, "t2zz_raw_counts.json"), "w", encoding="utf-8") as f:
        json.dump(dict(fetched=stamp(), job_id=man["job_id"], status=st, metrics=metrics, rows=rows), f,
                  default=str)
    print(f"{stamp()} saved t2zz_raw_counts.json ({len(rows)} results, schedule returned for "
          f"{sum(r['timing'] is not None for r in rows)}); usage {metrics.get('usage')}")


# ------------------------------------------------------------------ analysis (fixed with the design)

def _wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def _phase_var(x, y, nx, ny):
    vx = max(1 - x * x, 1e-9) / max(nx, 1)
    vy = max(1 - y * y, 1e-9) / max(ny, 1)
    r2 = max(x * x + y * y, 1e-9)
    return (y * y * vx + x * x * vy) / (r2 * r2)


def zz_fit(rows, fit_max_s, mode="origin"):
    """rows: dicts spec (0/1), tau (s), x, y, nx, ny (one per spectator state and delay, tau = 0 included).
    d(tau) = phase(spectator 1) - phase(spectator 0), wrapped, then unwrapped along tau starting at tau = 0.
    mode "origin" (the fixed analysis): zeta = weighted least-squares slope of d against tau THROUGH THE ORIGIN over
    0 < tau <= fit_max, divided by 2 pi; weights 1 / var(d(tau)) from shot noise (delta method).
    mode "intercept" (reported as a sensitivity check): the same with a free intercept, tau = 0 included.
    Returns (zeta in Hz, standard error in Hz, extra) with extra = dict(points=[(tau, d)], d0, d0_se)."""
    by = {}
    for r in rows:
        by.setdefault(r["tau"], {})[r["spec"]] = r
    taus = sorted(t for t in by if 0 in by[t] and 1 in by[t])
    nan = float("nan")
    if not taus or taus[0] != 0:
        return nan, nan, {}
    d, v = [], []
    for t in taus:
        r0, r1 = by[t][0], by[t][1]
        p0, p1 = math.atan2(r0["y"], r0["x"]), math.atan2(r1["y"], r1["x"])
        d.append(_wrap(p1 - p0))
        v.append(_phase_var(r0["x"], r0["y"], r0["nx"], r0["ny"]) + _phase_var(r1["x"], r1["y"], r1["nx"], r1["ny"]))
    d = np.unwrap(np.array(d))
    v = np.array(v)
    tt = np.array(taus)
    m = tt <= fit_max_s * (1 + 1e-6)
    extra = dict(points=[(float(a), float(b)) for a, b in zip(tt, d)], d0=float(d[0]), d0_se=float(math.sqrt(v[0])))
    if mode == "origin":
        k = m & (tt > 0)
        if not k.any():
            return nan, nan, extra
        w = 1 / v[k]
        S = float(np.sum(w * tt[k] ** 2))
        slope, se = float(np.sum(w * tt[k] * d[k]) / S), math.sqrt(1 / S)
    else:
        w = 1 / v[m]
        X = np.vstack([np.ones(int(m.sum())), tt[m]]).T
        cov = np.linalg.inv(X.T @ (w[:, None] * X))
        beta = cov @ (X.T @ (w * d[m]))
        slope, se = float(beta[1]), math.sqrt(cov[1, 1])
    return slope / (2 * math.pi), se / (2 * math.pi), extra


def target_expectations(man, raw):
    """{(pair index, seq, tau_us, spec): (value of <Z> on the target after post-selection, kept shots, all shots)}"""
    out = {}
    pubs = man["pubs"]
    for row in raw["rows"]:
        p = pubs[row["pub"]]
        if p["kind"] != "seq":
            continue
        for k in range(len(man["pairs"])):
            n0 = n1 = tot = 0
            for b, c in row["counts"].items():
                bits = b.replace(" ", "")[::-1]     # bits[i] = clbit i
                tot += c
                if int(bits[2 * k + 1]) != p["spec"]:
                    continue
                if bits[2 * k] == "0":
                    n0 += c
                else:
                    n1 += c
            n = n0 + n1
            out[(k, p["seq"], p["tau_us"], p["spec"])] = ((n0 - n1) / n if n else float("nan"), n, tot)
    return out


def coherence_tables(man, ex):
    """per pair, kind (R/E) and spectator state: tau_us -> dict(x, y, n, C_raw, phase)."""
    T = {}
    for k in range(len(man["pairs"])):
        for kind in ("R", "E"):
            for s in SPECS:
                for tau in TAUS_US:
                    x, nx, _ = ex[(k, kind + "X", tau, s)]
                    y, ny, _ = ex[(k, kind + "Y", tau, s)]
                    T.setdefault((k, kind, s), {})[tau] = dict(x=x, y=y, nx=nx, ny=ny, C_raw=math.hypot(x, y),
                                                             phase=math.atan2(y, x))
    for key, tab in T.items():
        c0 = tab[0]["C_raw"]
        for tau in tab:
            tab[tau]["C"] = tab[tau]["C_raw"] / c0 if c0 > 0 else float("nan")
    return T


def _tau_s(man, kind, tau_us):
    for p in man["pubs"]:
        if p["kind"] == "seq" and p["seq"] == kind + "X" and p["tau_us"] == tau_us:
            return p["tau_prog_s"]
    return tau_us * 1e-6


def timing_check(man, raw):
    """From the returned schedule (if any): for each sequence circuit and pair, does the spectator's X start before
    the target's wait, and do all six measurements start together? Returns a summary dict."""
    out = dict(returned=0, spectator_x_before_wait=0, spec1_circuits=0, measures_aligned=0)
    for row in raw["rows"]:
        tim = row.get("timing")
        if not tim:
            continue
        out["returned"] += 1
        lines = tim.split("\n") if isinstance(tim, str) else list(tim)
        recs = []
        for ln in lines:
            f = [x.strip() for x in str(ln).split(",")]
            if len(f) >= 6:
                try:
                    recs.append((f[1], f[2], int(float(f[3])), int(float(f[4]))))
                except ValueError:
                    pass
        meas = sorted({t0 for ins, ch, t0, du in recs if ins.startswith("measure") and ch.startswith("Qubit")})
        if len(meas) <= 1:
            out["measures_aligned"] += 1
        p = man["pubs"][row["pub"]]
        if p["kind"] == "seq" and p["spec"] == 1 and p["tau_us"] > 0:
            out["spec1_circuits"] += 1
            good = True
            for pr in man["pairs"]:
                xs = [t0 for ins, ch, t0, du in recs if ins.startswith("x") and ch == f"Qubit {pr['spectator']}"]
                ds = [t0 for ins, ch, t0, du in recs if ins.startswith("delay") and ch == f"Qubit {pr['target']}"]
                if not xs or (ds and min(xs) > min(ds)):
                    good = False
            out["spectator_x_before_wait"] += int(good)
    return out


def score(a):
    man = _read(a)
    with open(os.path.join(a.out, "t2zz_raw_counts.json"), encoding="utf-8") as f:
        raw = json.load(f)
    ex = target_expectations(man, raw)
    T = coherence_tables(man, ex)
    L = [f"# T2*/ZZ v2 results ({man['device']}, job {raw['job_id']}, scored {stamp()})", ""]
    res = dict(pairs=[])
    # readout references
    refs = {man["pubs"][r["pub"]]["ones"]: r["counts"] for r in raw["rows"] if man["pubs"][r["pub"]]["kind"] == "ref"}
    L += ["| pair | target / spectator | T1, T2 reported (us) | C_echo(20), exp(-20/T2) | C_Ramsey(20) / C_echo(20) |"
          " zeta Ramsey (kHz) +- SE | d(0) (rad) +- SE | zeta, free intercept (kHz) | echo phase diff at 20 us (rad) | T_idle from Ramsey (us) | shots dropped by"
          " post-selection (spec 0 / 1) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, pr in enumerate(man["pairs"]):
        R0, E0 = T[(k, "R", 0)], T[(k, "E", 0)]
        c_e20, c_r20 = E0[20]["C"], R0[20]["C"]
        pred = math.exp(-_tau_s(man, "E", 20) / pr["t2_t"])
        ratio = c_r20 / c_e20 if c_e20 > 0 else float("nan")
        rows = []
        for s in SPECS:
            for tau in TAUS_US:
                t = T[(k, "R", s)][tau]
                rows.append(dict(spec=s, tau=_tau_s(man, "R", tau), x=t["x"], y=t["y"], nx=t["nx"], ny=t["ny"]))
        zeta, se, zx = zz_fit(rows, FIT_MAX_US * 1e-6)
        zeta_i, se_i, _ = zz_fit(rows, FIT_MAX_US * 1e-6, mode="intercept")
        e_diff = _wrap((T[(k, "E", 1)][20]["phase"] - T[(k, "E", 0)][20]["phase"])
                       - (T[(k, "E", 1)][0]["phase"] - T[(k, "E", 0)][0]["phase"]))
        # E6: -ln C_Ramsey against tau through the origin over 10-40 us
        tt = np.array([_tau_s(man, "R", t) for t in (10, 20, 30, 40)])
        yy = np.array([-math.log(max(R0[t]["C"], 1e-6)) for t in (10, 20, 30, 40)])
        rate = float(np.sum(tt * yy) / np.sum(tt * tt))
        t_idle = 1 / rate if rate > 0 else float("inf")
        drop = []
        for s in SPECS:
            kept = sum(ex[(k, q, tau, s)][1] for q in SEQS for tau in TAUS_US)
            tot = sum(ex[(k, q, tau, s)][2] for q in SEQS for tau in TAUS_US)
            drop.append(1 - kept / tot)
        d = dict(label=pr["label"], target=pr["target"], spectator=pr["spectator"], t1=pr["t1_t"], t2=pr["t2_t"],
                 C_echo20=c_e20, pred_echo20=pred, ratio20=ratio, zeta_hz=zeta, zeta_se_hz=se, zz_points=zx.get("points"), d0=zx.get("d0"),
                 d0_se=zx.get("d0_se"), zeta_intercept_hz=zeta_i, zeta_intercept_se_hz=se_i,
                 echo_phase_diff20=e_diff, t_idle_s=t_idle, dropped=drop,
                 C={f"{kind}{s}": {t: T[(k, kind, s)][t]["C"] for t in TAUS_US} for kind in "RE" for s in SPECS})
        res["pairs"].append(d)
        L.append(f"| {pr['label']} | {pr['target']} / {pr['spectator']} | {pr['t1_t']*1e6:.0f}, {pr['t2_t']*1e6:.0f}"
                 f" | {c_e20:.3f}, {pred:.3f} | {ratio:.3f} | {zeta/1e3:+.2f} +- {se/1e3:.2f} | {zx['d0']:+.3f} +- "
                 f"{zx['d0_se']:.3f} | {zeta_i/1e3:+.2f} +- {se_i/1e3:.2f} | {e_diff:+.3f} | "
                 f"{t_idle*1e6:.0f} | {drop[0]:.3f} / {drop[1]:.3f} |")
    P = res["pairs"]
    v = {}
    n1 = sum(abs(p["C_echo20"] - p["pred_echo20"]) <= 0.10 for p in P)
    v["E1"] = ("CONFIRMED" if n1 >= 2 else "REFUTED" if all(abs(p["C_echo20"] - p["pred_echo20"]) > 0.20 for p in P)
               else "NOT DECIDED")
    n2 = sum(p["ratio20"] <= 0.90 for p in P)
    v["E2"] = "CONFIRMED" if n2 >= 2 else "REFUTED" if all(p["ratio20"] >= 0.98 for p in P) else "NOT DECIDED"
    for p in P:   # fixed rule: a phase offset at tau = 0 of >= 3 SE -> the free-intercept estimate is used for E3
        off = p["d0"] is not None and p["d0_se"] and abs(p["d0"]) >= 3 * p["d0_se"]
        p["zeta_used"] = "intercept" if off else "origin"
        p["z_e3"], p["se_e3"] = ((p["zeta_intercept_hz"], p["zeta_intercept_se_hz"]) if off
                                 else (p["zeta_hz"], p["zeta_se_hz"]))
    e3 = [p for p in P if abs(p["z_e3"]) >= 1e3 and abs(p["z_e3"]) >= 3 * p["se_e3"]]
    v["E3"] = ("CONFIRMED" if e3 else "REFUTED" if all(abs(p["z_e3"]) < 500 for p in P) else "NOT DECIDED")
    v["E3"] += " (estimate used: " + ", ".join(f"{p['label']} {p['zeta_used']}" for p in P) + ")"
    if not e3:
        v["E4"] = "NOT TESTABLE (E3 holds on no pair)"
    elif all(abs(p["echo_phase_diff20"]) <= 0.10 for p in e3):
        v["E4"] = "CONFIRMED"
    elif any(abs(p["echo_phase_diff20"]) > 0.30 for p in e3):
        v["E4"] = "REFUTED"
    else:
        v["E4"] = "NOT DECIDED"
    gaps = {p["label"]: 1 - p["ratio20"] for p in P}
    v["E5"] = "YES" if max(gaps, key=gaps.get) == "a" else "NO"
    v["E6"] = "; ".join(f"{p['label']}: T_idle {p['t_idle_s']*1e6:.0f} us vs T2 {p['t2']*1e6:.0f}, T2/2 "
                        f"{p['t2']*5e5:.0f}" for p in P)
    res["verdict"] = v
    res["timing"] = timing_check(man, raw)
    res["readout_refs"] = {str(k): v2 for k, v2 in refs.items()}
    L += ["", "| ID | verdict |", "|---|---|"] + [f"| {k} | {x} |" for k, x in v.items()]
    tm = res["timing"]
    L += ["", f"Schedule returned for {tm['returned']} of {len(raw['rows'])} circuits; spectator X before the wait in "
              f"{tm['spectator_x_before_wait']} of {tm['spec1_circuits']} spectator-|1> circuits; measurements "
              f"aligned in {tm['measures_aligned']}."]
    txt = "\n".join(L) + "\n"
    with open(os.path.join(a.out, "t2zz_score.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    with open(os.path.join(a.out, "t2zz_score.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    print(txt)
    return res


# ------------------------------------------------------------------ rehearsal

def _inject(c, man, delta_hz):
    """rehearsal only: after every delay on a pair's target, a Z rotation 2 pi delta t (quasi-static detuning of this
    batch) and an RZZ(-pi zeta t) on (target, spectator), i.e. a static ZZ of zeta Hz (phase difference
    phase(spectator 1) - phase(spectator 0) = +2 pi zeta t)."""
    from qiskit import QuantumCircuit
    tq = {p["target"]: k for k, p in enumerate(man["pairs"])}
    out = QuantumCircuit(*c.qregs, *c.cregs, name=c.name)
    for ins in c.data:
        out.append(ins.operation, ins.qubits, ins.clbits)
        if ins.operation.name == "delay":
            q = c.find_bit(ins.qubits[0]).index
            if q in tq:
                k = tq[q]
                t = ins.operation.duration * man["dt"]
                out.rz(2 * math.pi * delta_hz[k] * t, q)
                out.rzz(-math.pi * REH_ZETA_HZ[k] * t, q, man["pairs"][k]["spectator"])
    return out


def rehearse(a):
    warnings.simplefilter("ignore")
    man = prepare(a, fake=a.fake)
    from qiskit import qpy
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime import fake_provider
    with open(os.path.join(a.out, "t2zz_circuits.qpy"), "rb") as f:
        circs = qpy.load(f)
    sim = AerSimulator(noise_model=NoiseModel.from_backend(getattr(fake_provider, a.fake)()))
    rng = np.random.default_rng(int(os.environ.get("T2ZZ_REH_SEED", "4691013")))
    deltas = rng.normal(0, 1, size=(REH_BATCHES, len(man["pairs"]))) * np.array(REH_SIGMA_HZ)
    counts = [dict() for _ in circs]
    shots = {p["shots"] for p in man["pubs"]}
    assert len(shots) == 1
    for b in range(REH_BATCHES):                  # one simulator call per detuning batch (all circuits)
        res = sim.run([_inject(c, man, deltas[b]) for c in circs], shots=SHOTS // REH_BATCHES,
                      seed_simulator=int(os.environ.get("T2ZZ_REH_SEED", "4691013")) % 100000 + b).result()
        for i in range(len(circs)):
            for k2, n in res.get_counts(i).items():
                counts[i][k2] = counts[i].get(k2, 0) + n
    rows = [dict(pub=i, counts=counts[i], timing=None) for i in range(len(circs))]
    with open(os.path.join(a.out, "t2zz_raw_counts.json"), "w", encoding="utf-8") as f:
        json.dump(dict(fetched=stamp(), job_id="rehearsal", status="DONE (local)",
                       metrics=dict(usage=dict(quantum_seconds=0), note="local Aer rehearsal, injected truth"),
                       truth=dict(zeta_hz=REH_ZETA_HZ, sigma_hz=REH_SIGMA_HZ, batch_deltas_hz=deltas.tolist()),
                       rows=rows), f)
    man["job_id"] = "rehearsal"
    write_manifest(a.out, man)
    res = score(a)
    print("REHEARSAL truth check (injected -> recovered):")
    ok = True
    for k, p in enumerate(res["pairs"]):
        z, se = p["zeta_hz"], p["zeta_se_hz"]
        dev = (z - REH_ZETA_HZ[k]) / se if se > 0 else float("inf")
        # expected extra Ramsey decay from the batch detunings at 20 us, relative to the echo
        t20 = _tau_s(man, "R", 20)
        exp_ratio = abs(np.mean(np.exp(2j * math.pi * deltas[:, k] * t20)))
        print(f"  pair {p['label']}: zeta injected {REH_ZETA_HZ[k]/1e3:.2f} kHz, recovered {z/1e3:+.2f} +- "
              f"{se/1e3:.2f} kHz ({dev:+.1f} SE); C_R/C_E at 20 us {p['ratio20']:.3f}, expected from the injected "
              f"detunings about {exp_ratio:.3f}")
        ok &= abs(dev) < 4
    print("REHEARSAL DONE:", "truth recovered within 4 SE on every pair" if ok else "TRUTH NOT RECOVERED: check")
    return res


def run(a):
    """submit, then fetch (the job is submitted once; fetch can be repeated on its own)."""
    submit(a)
    fetch(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("rehearse", "prepare", "submit", "fetch", "run", "score"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--fake", default="FakeFez")
    ap.add_argument("--wait-min", type=float, default=40.0)
    a = ap.parse_args()
    if a.step == "prepare":
        prepare(a)
    else:
        dict(rehearse=rehearse, submit=submit, fetch=fetch, run=run, score=score)[a.step](a)


if __name__ == "__main__":
    main()
