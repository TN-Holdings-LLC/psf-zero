"""C53-VAL: prototype c53 (c52 + the workplace's items 87-88: chain Hamiltonians built by formula as a brickwork, with
rzz / rx in a Pauli frame on a fractional Target) against c52. Offline; no real device. Built from C52-VAL's script.

Devices: FakeFez, FakeMarrakesh, FakeKingston, FakeTorino and ibm_marrakesh's HW2 snapshot (2026-10-13 11:37 CEST).
The fake Targets have no fractional gates; a fractional copy adds rzz on every cz coupler with a CZ's duration and rx
on every qubit with an sx's, as ibm_marrakesh reported them (rzz 68 ns, rx 36 ns). Arms, all with seed 0:
  QK3   Qiskit level 3 on the Target without fractional gates
  C52   prototype c52's default call on the Target without fractional gates
  C53   prototype c53's default call on the Target without fractional gates
  C52F  prototype c52's default call on the fractional Target
  C53F  prototype c53's default call on the fractional Target
Tests: C49-VAL's 18 and six layered classifiers with an odd number of CZ rings (W3's model; n = 4, 6; L = 3, 5, 9).
Durations as soon as possible on each arm's own Target, measurements left out.

    python benchmarks/c53_val.py <out dir>   (refuses an existing out dir)"""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import statistics
import sys
import time
import warnings

warnings.simplefilter("ignore")
sys.path[:0] = ["benchmarks", ".", os.path.join("data", "2026-10-12", "c48_dev")]
OUT = sys.argv[1]
if os.path.exists(OUT):
    raise SystemExit(f"STOP: {OUT} exists (a run never overwrites another)")
os.makedirs(OUT)

DEVICES = ("FakeFez", "FakeMarrakesh", "FakeKingston", "FakeTorino", "snapshot_marrakesh")
SNAPSHOT = os.path.join("data", "2026-10-13", "hw2", "ns_snapshot_ibm_marrakesh.json")
TESTS = ("qaoa_n12_p2", "qaoa_n20_p2", "ba_n14", "ba_n16", "ba_n18", "ba_n20", "tfim_n12", "tfim_n16", "ghz_n12",
         "tfimP_n12", "tfimP_n16",
         "tfim_n20_r3", "heis_n12_r2", "heis2d_n16_r1", "chem_n12_r1", "qaoa3reg_n20_r3", "qaoa3reg_n40_r3",
         "tfim_n40_r2", "clf_n4_L3", "clf_n4_L5", "clf_n4_L9", "clf_n6_L3", "clf_n6_L5", "clf_n6_L9")
ODD = ("clf_n4_L3", "clf_n4_L5", "clf_n4_L9", "clf_n6_L3", "clf_n6_L5", "clf_n6_L9")
ISING = ("tfim_n12", "tfim_n16", "tfimP_n12", "tfimP_n16", "tfim_n20_r3", "tfim_n40_r2")
REPEAT = ("qaoa_n20_p2", "tfim_n16", "ba_n20")
ARMS = ("QK3", "C52", "C53", "C52F", "C53F")
EXACT_MAX_ACTIVE = 20


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


C49_FILE = os.path.join("patches", "psf_compile_c49_2026-10-14", "psf_compile.py")
C52_FILE = os.path.join("patches", "psf_compile_c52_2026-10-14", "psf_compile.py")
C53_FILE = os.path.join("patches", "psf_compile_c53_2026-10-14", "psf_compile.py")
load("benchmarks/psf_smart_layout.py", "psf_smart_layout")
M49 = load(C49_FILE, "psf_c49")
M52 = load(C52_FILE, "psf_c52")
M53 = load(C53_FILE, "psf_c53")
M49.WARN_WITHOUT_TARGET = M52.WARN_WITHOUT_TARGET = M53.WARN_WITHOUT_TARGET = False
H = load("benchmarks/hw1_real.py", "hw1")
H.KIT = os.path.join("data", "2026-10-12", "hw_kit")
H2 = load(os.path.join("data", "2026-10-13", "hw2_kit", "hw2_real.py"), "hw2")
H2.KIT = H.KIT
import gens  # noqa: E402
import snapshot_target as ST  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402
from qiskit.circuit.library import RXGate, RZZGate  # noqa: E402
from qiskit.transpiler import InstructionProperties, generate_preset_pass_manager  # noqa: E402
from qiskit_ibm_runtime import fake_provider  # noqa: E402


def fractional(target):
    t = copy.deepcopy(target)
    t.add_instruction(RZZGate(Parameter("t")), {qa: InstructionProperties(duration=p.duration, error=p.error)
                                                 for qa, p in target["cz"].items()})
    t.add_instruction(RXGate(Parameter("t")), {qa: InstructionProperties(duration=p.duration, error=p.error)
                                               for qa, p in target["sx"].items()})
    return t


def device(d):
    t = ST.target_from_snapshot(SNAPSHOT) if d == "snapshot_marrakesh" else getattr(fake_provider, d)().target
    return t, fractional(t)


def metrics(c, t):
    pos = {q: i for i, q in enumerate(c.qubits)}
    clock, n2, isa = {}, 0, True
    for ins in c.data:
        nm = ins.operation.name
        if nm == "barrier":
            continue
        qs = tuple(pos[q] for q in ins.qubits)
        if nm not in t.operation_names or (qs not in t[nm] and None not in t[nm]):
            isa = False
        if nm in ("measure", "delay") or nm not in t.operation_names:
            continue
        pr = t[nm].get(qs)
        d = (pr.duration or 0.0) if pr is not None else 0.0
        s = max(clock.get(q, 0.0) for q in qs)
        for q in qs:
            clock[q] = s + d
        n2 += len(qs) == 2
    ang = [float(i.operation.params[0]) for i in c.data if i.operation.name == "rzz"]
    return dict(twoq=n2, rzz=len(ang), rzz_in_range=all(0 < a <= math.pi / 2 + 1e-12 for a in ang), isa=isa,
                dur_us=(max(clock.values()) if clock else 0.0) * 1e6, active=len(clock))


def classifier(n, layers, seed):
    import random
    from qiskit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(rng.uniform(0, math.pi), q)
    for _ in range(layers):
        for q in range(n):
            qc.ry(rng.uniform(-3, 3), q)
            qc.rz(rng.uniform(-3, 3), q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    qc.measure_all()
    return qc


def build(name):
    if name.startswith("clf_"):
        n, L = int(name.split("_n")[1].split("_")[0]), int(name.split("_L")[1])
        return classifier(n, L, 52_000 + 100 * n + L)
    if name.startswith("tfimP"):
        return H2.build_test(name)[0]
    return H.build_test(name)[0] if name in H.TESTS else gens.build(name)


def compile_arm(arm, qc, t, ft):
    if arm == "QK3":
        return generate_preset_pass_manager(3, target=t, seed_transpiler=0).run(qc)
    mod, tg = {"C52": (M52, t), "C53": (M53, t), "C52F": (M52, ft), "C53F": (M53, ft)}[arm]
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.compile_for_hardware(qc, target=tg, entangling_basis="cx", layout_search=True, seed_transpiler=0)


def exactness(qc, c):
    act = len({q for i in c.data if i.operation.name != "barrier" for q in i.qubits})
    if act > EXACT_MAX_ACTIVE:
        return None
    p0, _ = H.measured_distribution(H._unrolled(qc))
    p1, _ = H.measured_distribution(c)
    return max(abs(p0.get(k, 0) - p1.get(k, 0)) for k in set(p0) | set(p1))


def sig(c):
    return hashlib.sha256(repr([(i.operation.name, [c.find_bit(q).index for q in i.qubits],
                                 [round(float(p), 12) for p in i.operation.params]) for i in c.data]).encode()).hexdigest()


if __name__ == "__main__":
    head = dict(started=time.strftime("%Y-%m-%d %H:%M:%S %Z"), c52_sha256=sha(C52_FILE), c53_sha256=sha(C53_FILE),
                script_sha256=sha(os.path.abspath(__file__)), devices=DEVICES, tests=TESTS)
    json.dump(head, open(os.path.join(OUT, "c53_val_head.json"), "w"), indent=1)
    rows = []
    f = open(os.path.join(OUT, "c53_val_rows.jsonl"), "w")
    for d in DEVICES:
        t, ft = device(d)
        for name in TESTS:
            qc = build(name)
            for arm in ARMS:
                t0 = time.time()
                err = None
                try:
                    c = compile_arm(arm, qc, t, ft)
                except Exception as e:  # noqa: BLE001 - recorded
                    c, err = None, f"{type(e).__name__}: {str(e)[:200]}"
                secs = time.time() - t0
                r = dict(device=d, test=name, arm=arm, secs=round(secs, 2), error=err)
                if c is not None:
                    r.update(sig=sig(c), **metrics(c, ft if arm in ("C52F", "C53F") else t))
                    if arm in ("C53", "C53F"):
                        r["exact"] = exactness(qc, c)
                    if arm == "C53F":
                        if name in REPEAT + ("clf_n6_L5",):
                            r["sig_repeat"] = sig(compile_arm(arm, qc, t, ft))
                rows.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"{d[-10:]:10s} {name:16s} {arm:4s} " + (f"2q {r['twoq']:5d} rzz {r['rzz']:4d} dur {r['dur_us']:7.2f} "
                      f"exact {r.get('exact')} " if c is not None else f"ERROR {err} ") + f"{secs:.1f}s", flush=True)
    f.close()
    by = {(r["device"], r["test"], r["arm"]): r for r in rows}
    cells = [(d, t) for d in DEVICES for t in TESTS]

    def dur(d, t, a):
        r = by[(d, t, a)]
        return r["dur_us"] if r.get("error") is None else math.inf

    lines = []
    gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))  # noqa: E731
    bad1 = [(d, t, a, by[(d, t, a)].get("exact", by[(d, t, a)].get("error"))) for d, t in cells for a in ("C53", "C53F")
            if by[(d, t, a)].get("error") is not None or (by[(d, t, a)].get("exact") or 0) > 1e-6]
    n1 = sum(1 for c in cells for a in ("C53", "C53F") if by[(*c, a)].get("exact") is not None)
    lines.append(f"V1 exact: every C53 and C53F output with <= {EXACT_MAX_ACTIVE} active qubits within 1e-6, no refusal: "
                 f"{'CONFIRMED' if not bad1 else 'REFUTED'} ({n1} checked; failures {bad1})")
    for new_, old_, tag in (("C53", "C52", "V2"), ("C53F", "C52F", "V3")):
        ok = [c for c in cells if dur(*c, new_) <= 1.02 * dur(*c, old_)]
        lines.append(f"{tag} {new_} <= 1.02 x {old_} duration on >= 95% of cells: "
                     f"{'CONFIRMED' if len(ok) >= 0.95 * len(cells) else 'REFUTED'} ({len(ok)}/{len(cells)}; above "
                     f"{[(d, t, round(dur(d, t, new_) / dur(d, t, old_), 3)) for d, t in cells if (d, t) not in ok]})")
    isc = [c for c in cells if c[1] in ISING]
    ri = [dur(*c, "C53F") / dur(*c, "C52F") for c in isc]
    lines.append(f"V4 Ising chains on the fractional Target: C53F / C52F geometric mean <= 0.85: "
                 f"{'CONFIRMED' if gm(ri) <= 0.85 else 'REFUTED'} ({gm(ri):.3f}; {min(ri):.3f}-{max(ri):.3f})")
    rep = [r for r in rows if "sig_repeat" in r]
    lines.append(f"V5 same output on a second call (C53F): {'CONFIRMED' if rep and all(r['sig'] == r['sig_repeat'] for r in rep) else 'REFUTED'} "
                 f"({sum(r['sig'] == r['sig_repeat'] for r in rep)}/{len(rep)})")
    bad6 = [(d, t) for d, t in cells if not (by[(d, t, "C53F")].get("isa") and by[(d, t, "C53F")].get("rzz_in_range"))]
    lines.append(f"V6 every C53F output native to the fractional Target, every rzz angle in (0, pi/2]: "
                 f"{'CONFIRMED' if not bad6 else 'REFUTED'} ({len(cells) - len(bad6)}/{len(cells)}; failures {bad6})")
    ok7 = [c for c in cells if by[(*c, "C53F")]["secs"] <= 1.5 * by[(*c, "C52F")]["secs"] + 2.0]
    lines.append(f"V7 compile time C53F <= 1.5 x C52F + 2 s on >= 90% of cells: "
                 f"{'CONFIRMED' if len(ok7) >= 0.9 * len(cells) else 'REFUTED'} ({len(ok7)}/{len(cells)})")
    fams = (("Ising", ISING), ("Barabasi-Albert", ("ba_n14", "ba_n16", "ba_n18", "ba_n20")),
            ("3-regular QAOA", ("qaoa_n12_p2", "qaoa_n20_p2", "qaoa3reg_n20_r3", "qaoa3reg_n40_r3")),
            ("other", ("heis_n12_r2", "heis2d_n16_r1", "chem_n12_r1", "ghz_n12")), ("odd classifiers", ODD))
    for fam, ts in fams:
        cc = [c for c in cells if c[1] in ts]
        a = [dur(*c, "C53") / dur(*c, "C52") for c in cc]
        b = [dur(*c, "C53F") / dur(*c, "C52F") for c in cc]
        q = [dur(*c, "C53F") / dur(*c, "QK3") for c in cc]
        lines.append(f"V8 reported, {fam}: C53/C52 geo {gm(a):.3f}; C53F/C52F geo {gm(b):.3f} ({min(b):.3f}-{max(b):.3f}); "
                     f"C53F/QK3 geo {gm(q):.3f}")
    lines.append(f"V9 reported: CHAIN_STATS c53 {getattr(M53, 'CHAIN_STATS', None)}")
    lines.append(f"{'device':10s} {'test':16s} {'dur us QK3 / C52 / C53 / C52F / C53F':>44s} {'C53F/C52F':>9s}")
    for d, t in cells:
        def g(a, k="dur_us"):
            r = by[(d, t, a)]
            return r.get(k, float("nan")) if r.get("error") is None else float("nan")
        lines.append(f"{d[-10:]:10s} {t:16s} {g('QK3'):8.2f} /{g('C52'):8.2f} /{g('C53'):8.2f} /{g('C52F'):8.2f} /"
                     f"{g('C53F'):8.2f} {g('C53F') / g('C52F'):9.3f}")
    lines.append(f"\nfinished {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    open(os.path.join(OUT, "c53_val_report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
