"""C51-VAL: prototype c51 (item 86 as revised: IBM Heron's native fractional gates rzz and rx, every rzz folded) against
c49 and Qiskit level 3. The same script as C50-VAL (Addendum 480) with c51 in place of c50.
Offline; no real device.

Devices: FakeFez, FakeMarrakesh, FakeKingston, FakeTorino and ibm_marrakesh's HW2 snapshot (2026-10-13 11:37 CEST).
The fake Targets have no fractional gates; a fractional copy adds rzz on every cz coupler with a CZ's duration and rx
on every qubit with an sx's, as ibm_marrakesh reported them (rzz 68 ns, rx 36 ns). Arms, all with seed 0:
  QK3   Qiskit level 3 on the Target without fractional gates
  QK3F  Qiskit level 3 on the fractional Target
  C49   prototype c49's default call on the Target without fractional gates (what runs today)
  C50   prototype c50's default call on the fractional Target
Durations as soon as possible on each arm's own Target, measurements left out.

    python benchmarks/c51_val.py <out dir>   (refuses an existing out dir)"""
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
         "tfim_n40_r2")
ISING = ("tfim_n12", "tfim_n16", "tfimP_n12", "tfimP_n16", "tfim_n20_r3", "tfim_n40_r2")
REPEAT = ("qaoa_n20_p2", "tfim_n16", "ba_n20")
ARMS = ("QK3", "QK3F", "C49", "C50")
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
C50_FILE = os.path.join("patches", "psf_compile_c51_2026-10-14", "psf_compile.py")  # the arm keeps the name C50
load("benchmarks/psf_smart_layout.py", "psf_smart_layout")
M49 = load(C49_FILE, "psf_c49")
M50 = load(C50_FILE, "psf_c51")
M49.WARN_WITHOUT_TARGET = M50.WARN_WITHOUT_TARGET = False
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


def build(name):
    if name.startswith("tfimP"):
        return H2.build_test(name)[0]
    return H.build_test(name)[0] if name in H.TESTS else gens.build(name)


def compile_arm(arm, qc, t, ft):
    if arm in ("QK3", "QK3F"):
        return generate_preset_pass_manager(3, target=t if arm == "QK3" else ft, seed_transpiler=0).run(qc)
    mod, tg = (M49, t) if arm == "C49" else (M50, ft)
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
    head = dict(started=time.strftime("%Y-%m-%d %H:%M:%S %Z"), c49_sha256=sha(C49_FILE), c50_sha256=sha(C50_FILE),
                script_sha256=sha(os.path.abspath(__file__)), devices=DEVICES, tests=TESTS)
    json.dump(head, open(os.path.join(OUT, "c50_val_head.json"), "w"), indent=1)
    rows = []
    f = open(os.path.join(OUT, "c50_val_rows.jsonl"), "w")
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
                    r.update(sig=sig(c), **metrics(c, ft if arm in ("QK3F", "C50") else t))
                    if arm == "C50":
                        r["exact"] = exactness(qc, c)
                        if name in REPEAT:
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
    bad1 = [(d, t, by[(d, t, "C50")].get("exact", by[(d, t, "C50")].get("error"))) for d, t in cells
            if by[(d, t, "C50")].get("error") is not None or (by[(d, t, "C50")].get("exact") or 0) > 1e-6]
    n1 = sum(1 for c in cells if by[(*c, "C50")].get("exact") is not None)
    lines.append(f"V1 exact: every C50 output with <= {EXACT_MAX_ACTIVE} active qubits within 1e-6, no refusal: "
                 f"{'CONFIRMED' if not bad1 else 'REFUTED'} ({n1} checked; failures {bad1})")
    ok2 = [c for c in cells if dur(*c, "C50") <= 1.02 * dur(*c, "C49")]
    lines.append(f"V2 C50 (fractional) <= 1.02 x C49 (today's) duration on >= 95% of cells: "
                 f"{'CONFIRMED' if len(ok2) >= 0.95 * len(cells) else 'REFUTED'} ({len(ok2)}/{len(cells)}; above "
                 f"{[(d, t, round(dur(d, t, 'C50') / dur(d, t, 'C49'), 3)) for d, t in cells if (d, t) not in ok2]})")
    isc = [c for c in cells if c[1] in ISING]
    ri = [dur(*c, "C50") / dur(*c, "C49") for c in isc]
    gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))  # noqa: E731
    bad3 = [(d, t, round(dur(d, t, "C50") / dur(d, t, "C49"), 3)) for d, t in isc if dur(d, t, "C50") > 0.95 * dur(d, t, "C49")]
    lines.append(f"V3 Ising: C50 / C49 duration geometric mean <= 0.70 and every Ising cell <= 0.95: "
                 f"{'CONFIRMED' if gm(ri) <= 0.70 and not bad3 else 'REFUTED'} (geometric mean {gm(ri):.3f}, "
                 f"{min(ri):.3f}-{max(ri):.3f}; above 0.95 {bad3})")
    ok4 = [c for c in cells if dur(*c, "C50") <= 1.02 * dur(*c, "QK3F")]
    lines.append(f"V4 C50 <= 1.02 x QK3F (Qiskit level 3 with the same fractional Target) on >= 90% of cells: "
                 f"{'CONFIRMED' if len(ok4) >= 0.9 * len(cells) else 'REFUTED'} ({len(ok4)}/{len(cells)}; above "
                 f"{[(d, t, round(dur(d, t, 'C50') / dur(d, t, 'QK3F'), 3)) for d, t in cells if (d, t) not in ok4]})")
    rep = [r for r in rows if "sig_repeat" in r]
    lines.append(f"V5 same output on a second call (C50): {'CONFIRMED' if rep and all(r['sig'] == r['sig_repeat'] for r in rep) else 'REFUTED'} "
                 f"({sum(r['sig'] == r['sig_repeat'] for r in rep)}/{len(rep)})")
    bad6 = [(d, t) for d, t in cells if not (by[(d, t, "C50")].get("isa") and by[(d, t, "C50")].get("rzz_in_range"))]
    lines.append(f"V6 every C50 output native to the fractional Target, every rzz angle in (0, pi/2]: "
                 f"{'CONFIRMED' if not bad6 else 'REFUTED'} ({len(cells) - len(bad6)}/{len(cells)}; failures {bad6})")
    ok7 = [c for c in cells if by[(*c, "C50")]["secs"] <= 1.5 * by[(*c, "C49")]["secs"] + 2.0]
    lines.append(f"V7 compile time C50 <= 1.5 x C49 + 2 s on >= 90% of cells: "
                 f"{'CONFIRMED' if len(ok7) >= 0.9 * len(cells) else 'REFUTED'} ({len(ok7)}/{len(cells)})")
    fams = (("Ising", ISING), ("Barabasi-Albert", ("ba_n14", "ba_n16", "ba_n18", "ba_n20")),
            ("3-regular QAOA", ("qaoa_n12_p2", "qaoa_n20_p2", "qaoa3reg_n20_r3", "qaoa3reg_n40_r3")),
            ("other", ("heis_n12_r2", "heis2d_n16_r1", "chem_n12_r1", "ghz_n12")))
    for fam, ts in fams:
        cc = [c for c in cells if c[1] in ts]
        a = [dur(*c, "C50") / dur(*c, "C49") for c in cc]
        b = [dur(*c, "C50") / dur(*c, "QK3") for c in cc]
        q = [dur(*c, "C50") / dur(*c, "QK3F") for c in cc]
        lines.append(f"V8 reported, {fam}: C50/C49 geo {gm(a):.3f} ({min(a):.3f}-{max(a):.3f}); C50/QK3 geo {gm(b):.3f}; "
                     f"C50/QK3F geo {gm(q):.3f}")
    lines.append(f"V9 reported: FRAC_STATS {M50.FRAC_STATS}; SELECT_STATS c50 {M50.SELECT_STATS}")
    lines.append("")
    lines.append(f"{'device':10s} {'test':16s} {'dur us QK3 / QK3F / C49 / C50':>38s} {'C50/C49':>7s} {'C50/QK3F':>8s} "
                 f"{'2q C49/C50 (rzz)':>18s}")
    for d, t in cells:
        def g(a, k="dur_us"):
            r = by[(d, t, a)]
            return r.get(k, float("nan")) if r.get("error") is None else float("nan")
        lines.append(f"{d[-10:]:10s} {t:16s} {g('QK3'):8.2f} /{g('QK3F'):8.2f} /{g('C49'):8.2f} /{g('C50'):8.2f} "
                     f"{g('C50') / g('C49'):7.3f} {g('C50') / g('QK3F'):8.3f} {g('C49', 'twoq'):6.0f}/{g('C50', 'twoq'):5.0f} "
                     f"({g('C50', 'rzz'):4.0f})")
    lines.append(f"\nfinished {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    open(os.path.join(OUT, "c50_val_report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
