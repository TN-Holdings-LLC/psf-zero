"""xor_stage2_real.py -- Stage 2 (2026-09-28): the XOR classifier on a real IBM device.

Pre-registered in spare-qubit-cliff-addendum-211-preregistration-2026-09-27.md.
Reuses the hash-checked Stage-1 and Stage-1b scripts for circuits, arms and
helpers (xor_prereg_stage1_sweep.py, xor_prereg_stage1b_sweep.py).

Three steps, run in this order; only `submit` uses QPU time:

  preflight  versions; saved IBM account; the four logical circuits and their
             exact values (H0); the local environment reproducing Stage 1 and
             1b on fake backends (H1, H2); the list of real devices available
             now. No job is sent.
  submit     applies the fixed device rule and arm rule using Aer noise models
             built from each real device's calibration now; writes the
             manifest (device, calibration of the qubits used, predictions,
             circuits' layouts, timings) to disk BEFORE submitting; submits
             ONE job of 8 circuits (chosen arm and control arm, 4 inputs each,
             4,000 shots); records the job ID. Does not read results.
  score      reads the job's results and metrics, scores S1-S4, records
             timings and QPU usage.
  layout     (no QPU) error-weighted PSF-Zero layout against Qiskit L3 on the
             chosen device's real Target (W1, W2).

Usage (from the repository root, with the Stage-1/1b scripts findable):
    python -u benchmarks/xor_stage2_real.py preflight 2>&1 | tee stage2_preflight.txt
    python -u benchmarks/xor_stage2_real.py submit    2>&1 | tee stage2_submit.txt
    python -u benchmarks/xor_stage2_real.py score     2>&1 | tee stage2_score.txt
    python -u benchmarks/xor_stage2_real.py layout    2>&1 | tee stage2_layout.txt

The IBM API key is never read from or written to any file by this script; it
uses the account already saved by QiskitRuntimeService.save_account().
"""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import io
import json
import math
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "benchmarks"), os.path.dirname(HERE), os.getcwd(),
          os.path.join(os.getcwd(), "benchmarks"), os.path.expanduser("~/pennylane_gpu_mock_test")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

MANIFEST = "stage2_manifest_2026-09-28.json"
SHOTS = 4000
PRED_SHOTS = 20000
PRED_SIM_SEED = 1000
MIN_QUBITS = 100
TIE_M = 0.005
ARMS = ("A", "B", "M1", "M4")
CONTROL = "A"
CONTROL_IF_CHOSEN_IS_A = "M4"
FLOOR = 0.15          # S2: observed M below prediction by more than this = worse than calibration explains
SEL_MARGIN = 0.02     # S4
USAGE_CAP_S = 60.0    # P0
STAGE1_BRISBANE_A = [-0.9000, 0.9105, 0.9065, -0.9095]   # Addendum 168, seed 0
STAGE1B_KINGSTON_M1 = 0.9815                              # Addendum 170
W_LOGICAL = 80
W_SEEDS = (0, 1, 2)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def normalized_sha256(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f.read().splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def header():
    import qiskit
    print(f"platform {platform.platform()} | python {platform.python_version()} | qiskit {qiskit.__version__}")
    for m in ("qiskit_aer", "qiskit_ibm_runtime", "pennylane"):
        try:
            mod = __import__(m)
            print(f"  {m} {getattr(mod, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001 -- reported
            print(f"  {m} MISSING ({type(exc).__name__})")
    print("SCRIPT", os.path.abspath(__file__), normalized_sha256(os.path.abspath(__file__)))


def _readable(path):
    try:
        os.stat(path)
    except PermissionError:
        return False
    except OSError:
        return True
    return True


def load_stage_modules():
    import xor_prereg_stage1_sweep as S1
    import xor_prereg_stage1b_sweep as S1b
    # The Stage-1 script (locked, not edited) puts its pod path /root/psf-zero/benchmarks
    # on sys.path. Qiskit's plugin loader stats every sys.path entry, and a path the
    # current user may not read raises PermissionError; such entries are dropped here.
    dropped = [p for p in sys.path if not _readable(p)]
    sys.path[:] = [p for p in sys.path if _readable(p)]
    if dropped:
        print("sys.path entries dropped (not readable):", dropped)
    for mod in (S1, S1b):
        print("LOADED", mod.__file__, normalized_sha256(mod.__file__))
    return S1, S1b


def service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    try:
        return QiskitRuntimeService()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("No usable saved IBM account (" + type(exc).__name__ + "). Save it once in the "
                         "terminal with QiskitRuntimeService.save_account(...); never paste the key into a file "
                         "or chat.") from exc


def real_devices(svc):
    out = []
    for b in svc.backends(simulator=False, operational=True):
        try:
            st = b.status()
            pending = st.pending_jobs
        except Exception:  # noqa: BLE001
            pending = None
        out.append((b, b.num_qubits, pending))
    return out


def mean_margin(zs, labels):
    return float(np.mean([l * z for l, z in zip(labels, zs)]))


def se_margin(zs, shots):
    return math.sqrt(sum(max(1.0 - z * z, 0.0) / shots for z in zs)) / len(zs)


def predict_arms(backend, circuits, S1b):
    """Aer model from this backend's calibration now: predicted z per arm and input."""
    from qiskit_aer import AerSimulator
    sim = AerSimulator.from_backend(backend)
    res = {}
    for arm in ARMS:
        t0 = time.perf_counter()
        built = S1b.build_arm(arm, circuits, backend)
        t_compile = time.perf_counter() - t0
        zs = [S1b.z_from_sim(sim, mqc, PRED_SHOTS, PRED_SIM_SEED) for mqc, _, _ in built]
        res[arm] = dict(built=built, z_pred=zs, compile_s=t_compile)
    return res


def qubit_calibration(backend, circuits_built):
    """Errors of every operation the submitted circuits use, from the Target now."""
    tgt = backend.target
    snap = {}
    for mqc, _, _ in circuits_built:
        for inst in mqc.data:
            name = inst.operation.name
            if name in ("barrier", "delay"):
                continue
            q = tuple(mqc.find_bit(x).index for x in inst.qubits)
            key = f"{name}{list(q)}"
            if key in snap:
                continue
            try:
                props = tgt[name][q]
                snap[key] = None if props is None else props.error
            except KeyError:
                snap[key] = None
    return snap


# ------------------------------------------------------------ preflight

def preflight():
    header()
    S1, S1b = load_stage_modules()
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import fake_provider as fp
    circuits = S1.build_logical_circuits()
    exacts = S1.validate_circuits(circuits)
    print("H0 exact <Z0>:", [round(z, 5) for z in exacts], "-> PASS")

    labels = S1.R.XOR_LABELS
    bris = fp.FakeBrisbane()
    sim = AerSimulator.from_backend(bris)
    zs = [S1b.z_from_sim(sim, mqc, SHOTS, 0) for mqc, _, _ in S1b.build_arm("A", circuits, bris)]
    tol = [3 * math.sqrt(2) * math.sqrt(max(1 - r * r, 0) / SHOTS) for r in STAGE1_BRISBANE_A]
    ok1 = all(abs(z - r) <= t for z, r, t in zip(zs, STAGE1_BRISBANE_A, tol))
    print("H1 FakeBrisbane arm A seed 0:", [round(z, 4) for z in zs], "vs Stage 1", STAGE1_BRISBANE_A,
          "->", "PASS" if ok1 else "FAIL")
    king = fp.FakeKingston()
    simk = AerSimulator.from_backend(king)
    zk = [S1b.z_from_sim(simk, mqc, SHOTS, 0) for mqc, _, _ in S1b.build_arm("M1", circuits, king)]
    mk = mean_margin(zk, labels)
    ok2 = abs(mk - STAGE1B_KINGSTON_M1) <= 0.03
    print(f"H2 FakeKingston arm M1 M = {mk:.4f} vs Stage 1b {STAGE1B_KINGSTON_M1} -> {'PASS' if ok2 else 'FAIL'}")

    svc = service()
    print("IBM account: loaded (channel/instance not printed)")
    devs = real_devices(svc)
    print(f"Real devices available now: {len(devs)}")
    for b, n, pend in devs:
        print(f"  {b.name:20s} {n:4d} qubits  pending jobs {pend}")
    print("Nighthawk-class device available:", any("nighthawk" in b.name.lower() for b, _, _ in devs))
    print("\nPREFLIGHT", "OK" if (ok1 and ok2) else "FAILED -- do not submit; send this log")


# ------------------------------------------------------------ submit

def submit():
    header()
    if os.path.exists(MANIFEST):
        raise SystemExit(f"{MANIFEST} already exists: a job was already prepared or submitted. Not submitting again.")
    S1, S1b = load_stage_modules()
    circuits = S1.build_logical_circuits()
    S1.validate_circuits(circuits)
    labels = S1.R.XOR_LABELS
    inputs = ["".join(str(x) for x in b) for b in S1.R.XOR_INPUTS]
    svc = service()
    devs = [(b, n, p) for b, n, p in real_devices(svc) if n >= MIN_QUBITS]
    if not devs:
        raise SystemExit("No real device with >= 100 qubits available now.")

    # Device rule: highest predicted M of the best arm; ties within TIE_M -> fewest pending jobs.
    cand = []
    for b, n, pend in devs:
        t0 = time.perf_counter()
        pr = predict_arms(b, circuits, S1b)
        best = max(ARMS, key=lambda a: mean_margin(pr[a]["z_pred"], labels))
        m = mean_margin(pr[best]["z_pred"], labels)
        cand.append(dict(backend=b, name=b.name, pending=pend, best_arm=best, m_pred=m, pred=pr,
                         seconds=time.perf_counter() - t0))
        print(f"  {b.name:20s} pending {pend}; predicted M: "
              + ", ".join(f"{a} {mean_margin(pr[a]['z_pred'], labels):.4f}" for a in ARMS)
              + f"  -> best {best}", flush=True)
    top = max(c["m_pred"] for c in cand)
    tied = [c for c in cand if top - c["m_pred"] <= TIE_M]
    chosen = min(tied, key=lambda c: (c["pending"] if c["pending"] is not None else 10 ** 9))
    backend = chosen["backend"]
    arm = chosen["best_arm"]
    control = CONTROL if arm != CONTROL else CONTROL_IF_CHOSEN_IS_A
    pr = chosen["pred"]
    print(f"\nChosen device {chosen['name']} (predicted M {chosen['m_pred']:.4f}); chosen arm {arm}; "
          f"control arm {control}")

    subs = []
    for which, a in (("chosen", arm), ("control", control)):
        for i, (mqc, q0, routed) in enumerate(pr[a]["built"]):
            ok, reason = S1.is_isa_compliant(routed, backend)
            if not ok:
                raise SystemExit(f"ISA check failed ({a}, input {inputs[i]}): {reason}")
            subs.append(dict(role=which, arm=a, input=inputs[i], label=int(labels[i]), q0_physical=int(q0),
                             z_pred=pr[a]["z_pred"][i],
                             layout=list(routed.layout.initial_index_layout(filter_ancillas=True))
                             if routed.layout is not None else None,
                             twoq=sum(1 for x in mqc.data if len(x.qubits) == 2), circuit=mqc))
    try:
        cal_time = str(backend.properties().last_update_date)
    except Exception:  # noqa: BLE001
        cal_time = None
    manifest = dict(
        written_utc=now(), script_sha256=normalized_sha256(os.path.abspath(__file__)),
        device=chosen["name"], device_pending_at_choice=chosen["pending"], calibration_time=cal_time,
        chosen_arm=arm, control_arm=control, shots=SHOTS, pred_shots=PRED_SHOTS, pred_seed=PRED_SIM_SEED,
        m_pred={a: mean_margin(pr[a]["z_pred"], labels) for a in ARMS},
        compile_s={a: pr[a]["compile_s"] for a in ARMS},
        candidates=[dict(name=c["name"], pending=c["pending"], best_arm=c["best_arm"], m_pred=c["m_pred"],
                         seconds=c["seconds"]) for c in cand],
        circuits=[{k: v for k, v in s.items() if k != "circuit"} for s in subs],
        calibration=qubit_calibration(backend, [(s["circuit"], None, None) for s in subs]),
        job_id=None,
    )
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, default=str)
    print(f"Manifest written ({MANIFEST}) before submission: predictions and calibration are fixed.")

    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(mode=backend)
    t_sub = now()
    job = sampler.run([(s["circuit"],) for s in subs], shots=SHOTS)
    manifest["job_id"] = job.job_id()
    manifest["submitted_utc"] = t_sub
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, default=str)
    print(f"Submitted job {job.job_id()} ({len(subs)} circuits x {SHOTS} shots). Results are NOT read here.")
    print("Next: python -u benchmarks/xor_stage2_real.py score  (after the job finishes)")


# ------------------------------------------------------------ score

def score():
    header()
    with open(MANIFEST, encoding="utf-8") as f:
        man = json.load(f)
    svc = service()
    job = svc.job(man["job_id"])
    status = str(job.status())
    print("job", man["job_id"], "status", status)
    if "DONE" not in status.upper():
        raise SystemExit("Job not finished; run score again later.")
    res = job.result()
    rows = []
    for i, c in enumerate(man["circuits"]):
        counts = res[i].data.z0.get_counts()
        n0, n1 = counts.get("0", 0), counts.get("1", 0)
        z = (n0 - n1) / (n0 + n1)
        rows.append(dict(c, z_obs=z, shots_obs=n0 + n1))
        print(f"  {c['role']:7s} {c['arm']:2s} input {c['input']} label {int(c['label']):+d} qubit {c['q0_physical']:4d}: "
              f"z_obs {z:+.4f}  z_pred {c['z_pred']:+.4f}")
    by = {r: [x for x in rows if x["role"] == r] for r in ("chosen", "control")}
    m_obs = {r: mean_margin([x["z_obs"] for x in xs], [x["label"] for x in xs]) for r, xs in by.items()}
    m_pred = {r: mean_margin([x["z_pred"] for x in xs], [x["label"] for x in xs]) for r, xs in by.items()}
    se = {r: se_margin([x["z_obs"] for x in xs], SHOTS) for r, xs in by.items()}

    metrics = {}
    try:
        metrics = job.metrics()
    except Exception as exc:  # noqa: BLE001
        print("metrics unavailable:", type(exc).__name__)
    usage = (metrics.get("usage") or {}).get("quantum_seconds") if metrics else None
    ts = metrics.get("timestamps", {}) if metrics else {}
    print("\nmetrics:", json.dumps({"usage": metrics.get("usage") if metrics else None, "timestamps": ts}, default=str))

    def v(ok):
        return "CONFIRMED" if ok else "NOT CONFIRMED"

    print("\n=== predictions ===")
    flips = [x for x in by["chosen"] if x["label"] * x["z_obs"] <= 0]
    print(f"S1 chosen arm ({man['chosen_arm']}): all four inputs correct ({4 - len(flips)}/4) -> {v(not flips)}")
    print(f"S2 chosen arm: observed M >= predicted M - {FLOOR} ({m_obs['chosen']:.4f} vs {m_pred['chosen']:.4f}) "
          f"-> {v(m_obs['chosen'] >= m_pred['chosen'] - FLOOR)}")
    print(f"S3 chosen arm: observed M < predicted M + 3 SE ({m_obs['chosen']:.4f} vs "
          f"{m_pred['chosen'] + 3 * se['chosen']:.4f}) -> {v(m_obs['chosen'] < m_pred['chosen'] + 3 * se['chosen'])}")
    print(f"S4 chosen arm observed M >= control ({man['control_arm']}) observed M - {SEL_MARGIN} "
          f"({m_obs['chosen']:.4f} vs {m_obs['control']:.4f}) -> {v(m_obs['chosen'] >= m_obs['control'] - SEL_MARGIN)}")
    print(f"P0 pipeline complete; QPU usage {usage} s <= {USAGE_CAP_S} s -> "
          f"{v(usage is not None and usage <= USAGE_CAP_S)}")
    print(f"(control arm, not a prediction: observed M {m_obs['control']:.4f}, predicted {m_pred['control']:.4f}; "
          f"sign flips {sum(1 for x in by['control'] if x['label'] * x['z_obs'] <= 0)}/4)")
    man["scored_utc"] = now()
    man["results"] = [{k: v2 for k, v2 in r.items()} for r in rows]
    man["m_obs"], man["m_pred_roles"], man["se"] = m_obs, m_pred, se
    man["metrics"] = metrics
    with open(MANIFEST.replace(".json", "_scored.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, default=str)
    print("Wrote", MANIFEST.replace(".json", "_scored.json"))


# ------------------------------------------------------------ layout (no QPU)

def layout():
    header()
    from qiskit import QuantumCircuit, transpile
    import loop_endurance as le
    import psf_compile as pc
    print("LOADED", pc.__file__, pc.VERSION)
    name = None
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as f:
            name = json.load(f)["device"]
    svc = service()
    backend = svc.backend(name) if name else max((b for b, n, _ in real_devices(svc)), key=lambda b: b.num_qubits)
    tgt = backend.target
    native = [g for g in backend.operation_names if g in ("cz", "ecr", "cx", "rz", "sx", "x", "id")]
    edge_err = pc.edge_errors_from_target(tgt)
    qubit_err = pc.qubit_errors_from_target(tgt, "sx")
    print(f"device {backend.name}, {backend.num_qubits} qubits; {W_LOGICAL} logical qubits in pair24 blocks")

    def esp_log10(qc):
        total = 0.0
        for inst in qc.data:
            nm = inst.operation.name
            if nm in ("barrier", "delay"):
                continue
            q = tuple(qc.find_bit(x).index for x in inst.qubits)
            try:
                props = tgt[nm][q]
            except KeyError:
                props = None
            if props is not None and props.error:
                total += math.log10(max(1e-300, 1.0 - props.error))
        return total

    wins, exact_ok = 0, True
    for seed in W_SEEDS:
        rng = np.random.default_rng(seed)
        qc = QuantumCircuit(W_LOGICAL)
        th = rng.uniform(-np.pi, np.pi, (W_LOGICAL // 2, 24))
        for k in range(W_LOGICAL // 2):
            le.add_pair24(qc, 2 * k, 2 * k + 1, th[k])
        pc._CX_CORE_CACHE.clear()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            w = pc.compile_for_hardware(qc, coupling_map=backend.coupling_map, basis_gates=native,
                                        entangling_basis="cx", layout_search=True, on_unsupported="raise",
                                        seed_transpiler=0, layout_edge_errors=edge_err,
                                        layout_qubit_errors=qubit_err)
        tw = time.perf_counter() - t0
        t0 = time.perf_counter()
        q3 = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=0)
        tq = time.perf_counter() - t0
        okw, ww = le.pair_check(qc, w, W_LOGICAL)
        okq, wq = le.pair_check(qc, q3, W_LOGICAL)
        exact_ok &= bool(okw and ww <= 1e-12)
        ew, eq = esp_log10(w), esp_log10(q3)
        wins += int(ew >= eq)
        print(f"  seed {seed}: log10 ESP weighted {ew:.4f} ({tw * 1000:.0f} ms, 2q "
              f"{sum(1 for x in w.data if len(x.qubits) == 2)}) | Qiskit L3 {eq:.4f} ({tq * 1000:.0f} ms, 2q "
              f"{sum(1 for x in q3.data if len(x.qubits) == 2)}); weighted per-pair check {ww}", flush=True)
    print(f"W1 weighted ESP >= Qiskit L3 ESP on the real Target: {wins}/{len(W_SEEDS)} -> "
          f"{'CONFIRMED' if wins == len(W_SEEDS) else 'NOT CONFIRMED'}")
    print(f"W2 weighted output exact (per-pair <= 1e-12) on every seed -> {'CONFIRMED' if exact_ok else 'NOT CONFIRMED'}")


if __name__ == "__main__":
    cmds = dict(preflight=preflight, submit=submit, score=score, layout=layout)
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        raise SystemExit("usage: xor_stage2_real.py {preflight|submit|score|layout}")
    cmds[sys.argv[1]]()
