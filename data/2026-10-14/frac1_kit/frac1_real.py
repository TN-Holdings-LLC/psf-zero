"""frac1_real.py -- FRAC1 (2026-10-14): do IBM Heron's native fractional gates (rzz, rx) raise the fidelity of the
same compiled circuits on the device, through their shorter duration?

Pre-registered in FRAC1_preregistration.md (fixed before any real-device step). One job, on TIME1's device (read from
TIME1's manifest; if TIME1 has not prepared, by TIME1's device rule):

  Ising  TIME1's T1 inputs (TFIM chains from a product state, n = 12, 16, 20, 4 and 8 Trotter steps)
  HW2    HW2's seven reading tests
  arms   C49: prototype c49's default call on the device's Target WITHOUT fractional gates (as TIME1's C49);
         C51: prototype c51's default call (item 86 as revised in c51) on the device's Target WITH fractional gates (use_fractional_gates=True).
         A pair is submitted only where the two circuits differ.
  control ghz_n12, C49 and C51.

All circuits go in one SamplerV2 job on the backend loaded with use_fractional_gates=True (circuits without rzz run
there as well), 4,000 shots per circuit and repeat, two repeats, dynamical decoupling and twirling off (Pauli
twirling is not supported with fractional gates).

Steps: rehearse (fake device, fractional copy of its Target, noiseless local sampler), prepare, submit, fetch, score.
    python -u <kit>/frac1_real.py rehearse --out data/2026-10-14/frac1_rehearsal
    python -u <kit>/frac1_real.py prepare  --out data/2026-10-14/frac1_run
    python -u <kit>/frac1_real.py submit   --out data/2026-10-14/frac1_run
    python -u <kit>/frac1_real.py fetch    --out data/2026-10-14/frac1_run
    python -u <kit>/frac1_real.py score    --out data/2026-10-14/frac1_run
The IBM API key is never read from or written to any file by this script. It runs no git and no pip.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
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
C51_FILE = os.path.join(REPO, "patches", "psf_compile_c51_2026-10-14", "psf_compile.py")
TIME1_MANIFEST = os.path.join(REPO, "data", "2026-10-14", "time1_run", "time1_manifest.json")

SHOTS = 4000  # per circuit and repeat (TIME1 runs first on the same evening; the QPU left is about 5.5 min)
REPEATS = 2
USAGE_EST_CAP_S = 300.0
MAX_PUBS = 70


def _load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


T1 = _load(T1_FILE, "frac1_time1")
H2 = T1.H2
stamp, sha256_file = T1.stamp, T1.sha256_file
ISING = T1.T1_TESTS
READING = T1.READING
CONTROL = T1.CONTROL


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
    for f in (T1_FILE, T1.HW2_FILE, T1.C49_FILE, C51_FILE, T1.LAYOUT_FILE):
        print("FILE", os.path.relpath(f, REPO), sha256_file(f) if os.path.exists(f) else "MISSING")
    if sha256_file(T1_FILE) != T1_SHA:
        raise SystemExit("time1_real.py differs from the hash fixed in this kit: stop")
    if sha256_file(T1.HW2_FILE) != T1.HW2_SHA:
        raise SystemExit("hw2_real.py differs from the hash fixed in TIME1: stop")


def fractional_copy(target):
    """Rehearsal only: a fake device's Target with rzz (a CZ's duration) and rx (an sx's) added."""
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RXGate, RZZGate
    from qiskit.transpiler import InstructionProperties
    t = copy.deepcopy(target)
    t.add_instruction(RZZGate(Parameter("t")), {qa: InstructionProperties(duration=p.duration, error=p.error)
                                                 for qa, p in target["cz"].items()})
    t.add_instruction(RXGate(Parameter("t")), {qa: InstructionProperties(duration=p.duration, error=p.error)
                                               for qa, p in target["sx"].items()})
    return t


def compile_arm(arm, qc, t_plain, t_frac, mods):
    t0 = time.perf_counter()
    mod, tg = (mods["C49"], t_plain) if arm == "C49" else (mods["C51"], t_frac)
    with contextlib.redirect_stdout(io.StringIO()):
        out = mod.compile_for_hardware(qc, target=tg, entangling_basis="cx", layout_search=True, seed_transpiler=0)
    return out, time.perf_counter() - t0


def build_all(name, t_plain, t_frac, out_dir):
    from qiskit import qpy
    T1._load(T1.LAYOUT_FILE, "psf_smart_layout")
    mods = {"C49": T1._load(T1.C49_FILE, "frac1_c49"), "C51": T1._load(C51_FILE, "frac1_c51")}
    for m in mods.values():
        m.WARN_WITHOUT_TARGET = False
    entries, circs, ideal = [], [], {}
    print(f"{stamp()} compiling for {name}", flush=True)
    plan = [("ISING", t) for t in ISING] + [("R", t) for t in READING] + [("CTL", t) for t in CONTROL]
    for part, test in plan:
        qc = T1.build_test(test)
        dist, _ = H2.measured_distribution(H2._unrolled(qc))
        ideal[test] = dict(n=qc.num_qubits, sum_p2=float(sum(v * v for v in dist.values())))
        outs = {arm: compile_arm(arm, qc, t_plain, t_frac, mods) for arm in ("C49", "C51")}
        sigs = {arm: H2.sig(o) for arm, (o, _) in outs.items()}
        if part != "CTL" and sigs["C49"] == sigs["C51"]:
            print(f"  {test:16s} C49 and C51 identical: pair not submitted", flush=True)
            continue
        start = len(entries)
        for arm, (out, secs) in outs.items():
            tg = t_plain if arm == "C49" else t_frac
            n2, esp, dur, isa, failed = T1.metrics(out, tg, mods)
            ang = [float(i.operation.params[0]) for i in out.data if i.operation.name == "rzz"]
            in_range = all(0 < a <= math.pi / 2 + 1e-12 for a in ang)
            d, active = H2.measured_distribution(out)
            exact = None if d is None else max(abs(d.get(k, 0.0) - dist.get(k, 0.0)) for k in set(d) | set(dist))
            ok = isa and in_range and not failed and (exact is None or exact <= T1.EXACT_TOL)
            print(f"  {part:5s} {test:16s} {arm:4s} 2q {n2:5d} rzz {len(ang):4d} dur {dur:7.2f} us active {active:3d} "
                  f"exact {'n/a' if exact is None else f'{exact:.1e}'} isa {isa} rzz-range {in_range} failed {failed} "
                  f"({secs:.1f} s){'' if ok else '  -> NOT SUBMITTED'}", flush=True)
            entries.append(dict(part=part, test=test, arm=arm, twoq=n2, rzz=len(ang), esp=esp, dur_us=dur,
                                active=active, exact_maxdiff=exact, isa=isa, rzz_in_range=in_range,
                                uses_failed=failed, compile_s=round(secs, 3), submitted=ok, sig=sigs[arm]))
            if ok:
                circs.append((len(entries) - 1, out))
        if not all(e["submitted"] for e in entries[start:]):
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
    with open(os.path.join(out_dir, "frac1_circuits.qpy"), "wb") as f:
        qpy.dump([c for _, c, _ in order], f)
    return entries, pubs, ideal


def manifest_path(out_dir):
    return os.path.join(out_dir, "frac1_manifest.json")


def write_manifest(out_dir, man):
    with open(manifest_path(out_dir), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, default=str)


def _read(a):
    with open(manifest_path(a.out), encoding="utf-8") as f:
        man = json.load(f)
    if sha256_file(os.path.join(a.out, "frac1_circuits.qpy")) != man["qpy_sha256"]:
        raise SystemExit("frac1_circuits.qpy differs from the manifest's hash: stop.")
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
        if os.path.exists(TIME1_MANIFEST):
            name = json.load(open(TIME1_MANIFEST, encoding="utf-8"))["device"]
            rule = [dict(rule="TIME1's device", name=name)]
        else:
            b, rule = T1.device_rule(svc)
            name = b.name
        plain = svc.backend(name)
        frac = svc.backend(name, use_fractional_gates=True)
        t_plain, t_frac = plain.target, frac.target
        if "rzz" not in t_frac.operation_names:
            raise SystemExit(f"{name} offers no fractional gates (rzz): nothing prepared")
        cal = H2.calibration_time(plain)
    else:
        from qiskit_ibm_runtime import fake_provider
        b = getattr(fake_provider, fake)()
        name, rule, t_plain, cal = b.name, [dict(name=fake, fake=True)], b.target, None
        t_frac = fractional_copy(t_plain)
    print("device rule:", rule)
    entries, pubs, ideal = build_all(name, t_plain, t_frac, a.out)
    man = dict(written=stamp(), device=name, device_rule=rule, calibration_time=cal, shots=SHOTS, repeats=REPEATS,
               script_sha256=sha256_file(os.path.abspath(__file__)),
               files={os.path.relpath(f, REPO): sha256_file(f) for f in (T1_FILE, T1.HW2_FILE, T1.C49_FILE, C51_FILE,
                                                                        T1.LAYOUT_FILE)},
               qpy_sha256=sha256_file(os.path.join(a.out, "frac1_circuits.qpy")), entries=entries, pubs=pubs,
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
    with open(os.path.join(a.out, "frac1_circuits.qpy"), "rb") as f:
        circs = qpy.load(f)
    if len(circs) != len(man["pubs"]):
        raise SystemExit("the QPY file and the manifest disagree on the number of circuits: stop")
    backend = QiskitRuntimeService().backend(man["device"], use_fractional_gates=True)
    sampler = SamplerV2(mode=backend)
    sampler.options.dynamical_decoupling.enable = False
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    t_sub = stamp()
    job = sampler.run([(c, None, p["shots"]) for c, p in zip(circs, man["pubs"])])
    man["job_id"], man["submitted"] = job.job_id(), t_sub
    write_manifest(a.out, man)
    print(f"{stamp()} submitted job {job.job_id()} ({len(circs)} circuits, {sum(p['shots'] for p in man['pubs'])} "
          f"shots) to {man['device']} (fractional gates on). Results are not read here.")


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
    with open(os.path.join(a.out, "frac1_raw_counts.json"), "w", encoding="utf-8") as f:
        json.dump(dict(fetched=stamp(), job_id=man["job_id"], status=st, metrics=metrics_, rows=rows), f, default=str)
    print(f"{stamp()} saved frac1_raw_counts.json ({len(rows)} results); usage {metrics_.get('usage')}")


def score(a):
    man = _read(a)
    with open(os.path.join(a.out, "frac1_raw_counts.json"), encoding="utf-8") as f:
        raw = json.load(f)
    ent = man["entries"]
    by = {}
    for p, row in zip(man["pubs"], raw["rows"]):
        e = ent[p["entry"]]
        by.setdefault((e["part"], e["test"], e["arm"]), []).append((p["repeat"], row["counts"]))
    out = [f"# FRAC1 scores ({stamp()})", "",
           f"device {man['device']}; calibration {man['calibration_time']}; job {man.get('job_id')}; usage "
           f"{(raw.get('metrics') or {}).get('usage')}", "",
           "F: normalised linear XEB against the ideal distribution, mean of the two repeats.", "",
           "| part | test | arm | 2q | rzz | duration us | F rep 1 | F rep 2 | F mean (SE) |", "|---|---|---|---|---|---|---|---|---|"]
    dists, R = {}, {}
    for e in ent:
        if not e["submitted"]:
            continue
        t = e["test"]
        if t not in dists:
            d, _ = H2.measured_distribution(H2._unrolled(T1.build_test(t)))
            if abs(sum(v * v for v in d.values()) - man["ideal"][t]["sum_p2"]) > 1e-9:
                raise SystemExit(f"ideal distribution of {t} differs from the manifest's: stop")
            dists[t] = d
        n = man["ideal"][t]["n"]
        reps = sorted(by.get((e["part"], t, e["arm"]), []))
        fs = [H2.xeb(dists[t], c, n, sum(c.values())) for _, c in reps]
        fm = float(np.mean([f for f, _ in fs])) if fs else float("nan")
        fse = math.sqrt(sum(s_ ** 2 for _, s_ in fs)) / max(len(fs), 1)
        R[(e["part"], t, e["arm"])] = dict(f=fm, se=fse, reps=[f for f, _ in fs], dur=e["dur_us"], twoq=e["twoq"],
                                           rzz=e["rzz"], sig=e["sig"])
        out.append(f"| {e['part']} | {t} | {e['arm']} | {e['twoq']} | {e['rzz']} | {e['dur_us']:.2f} | "
                   + " | ".join(f"{x:.4f}" for x in ([f for f, _ in fs] + [float('nan')] * 2)[:2])
                   + f" | {fm:.4f} ({fse:.4f}) |")
    pairs = []
    for part, tests in (("ISING", ISING), ("R", READING)):
        for t in tests:
            x, y = R.get((part, t, "C51")), R.get((part, t, "C49"))
            if x is None or y is None:
                continue
            ok = x["f"] > 0 and y["f"] > 0
            pairs.append(dict(part=part, test=t, valid=ok, dur=x["dur"] / y["dur"],
                              f=(x["f"] / y["f"]) if ok else float("nan"), f51=x["f"], f49=y["f"]))
    out += ["", "## Pairs C51 / C49", "", "| part | test | duration ratio | F ratio | F C51 | F C49 |", "|---|---|---|---|---|---|"]
    for p in pairs:
        out.append(f"| {p['part']} | {p['test']} | {p['dur']:.3f} | {p['f']:.3f} | {p['f51']:.4f} | {p['f49']:.4f} |")
    v = {}
    ising = [p for p in pairs if p["part"] == "ISING" and p["valid"]]
    up = sum(1 for p in ising if p["f"] > 1)
    need = math.ceil(5 * len(ising) / 6)
    v["FA"] = "NOT TESTABLE" if len(ising) < 3 else ("CONFIRMED" if up >= need else "REFUTED")
    gm = math.exp(float(np.mean([math.log(p["f"]) for p in ising]))) if ising else float("nan")
    v["FB"] = "NOT TESTABLE" if len(ising) < 3 else ("CONFIRMED" if gm >= 1.10 else "REFUTED")
    short = [p for p in pairs if p["valid"] and p["dur"] <= 0.9]
    ok_c = sum(1 for p in short if p["f"] >= 1)
    v["FC"] = "NOT TESTABLE" if len(short) < 4 else ("CONFIRMED" if ok_c >= 0.75 * len(short) else "REFUTED")
    val = [p for p in pairs if p["valid"]]
    rho = H2._rho([-math.log(p["dur"]) for p in val], [math.log(p["f"]) for p in val]) if len(val) >= 3 else float("nan")
    g = (R.get(("CTL", "ghz_n12", "C51")), R.get(("CTL", "ghz_n12", "C49")))
    v["FE"] = "n/a" if None in g or g[1]["f"] <= 0 else ("CONFIRMED" if abs(g[0]["f"] / g[1]["f"] - 1) <= 0.10 else "REFUTED")
    adopted = v["FA"] == "CONFIRMED" and v["FB"] == "CONFIRMED"
    out += ["", "## Predictions", "",
            f"- FA (Ising: C51's F > C49's on >= {need} of {len(ising)}): {up} -> {v['FA']}",
            f"- FB (Ising: geometric mean F ratio C51 / C49 >= 1.10): {gm:.3f} -> {v['FB']}",
            f"- FC (every pair with duration ratio <= 0.9: C51's F >= C49's on >= 75%): {ok_c} of {len(short)} -> {v['FC']}",
            f"- FD (reported): rank correlation of 1 / duration ratio with F ratio over {len(val)} pairs: {rho:+.3f}",
            f"- FE (control ghz_n12: |F ratio - 1| <= 0.10): {v['FE']}",
            f"- **Item 86 on the device (FA and FB confirmed): {'CONFIRMED' if adopted else 'NOT CONFIRMED'}**"]
    txt = "\n".join(out) + "\n"
    with open(os.path.join(a.out, "frac1_score.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    with open(os.path.join(a.out, "frac1_score.json"), "w", encoding="utf-8") as f:
        json.dump(dict(F={f"{k[0]}|{k[1]}|{k[2]}": x for k, x in R.items()}, pairs=pairs, verdict=v, rho=rho,
                       adopted=adopted), f, indent=1)
    print(txt)


def rehearse(a):
    warnings.simplefilter("ignore")
    prepare(a, fake=a.fake)
    man = _read(a)
    from qiskit import qpy
    from qiskit_aer import AerSimulator
    with open(os.path.join(a.out, "frac1_circuits.qpy"), "rb") as f:
        circs = qpy.load(f)
    sim = AerSimulator()
    rows = [dict(pub=i, counts=sim.run(c, shots=p["shots"], seed_simulator=2000 + i).result().get_counts())
            for i, (c, p) in enumerate(zip(circs, man["pubs"]))]
    with open(os.path.join(a.out, "frac1_raw_counts.json"), "w", encoding="utf-8") as f:
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
