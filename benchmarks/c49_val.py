"""C49-VAL: prototype c49 (items 84 and 85) against c48 and Qiskit level 3. Offline; no real device.

Devices: four fake IBM devices, and ibm_marrakesh's calibration snapshot of HW2 (2026-10-13 11:37 CEST, 4 failed
qubits, so item 85 is active there). Arms, all with seed 0:
  QK3   Qiskit level 3 (generate_preset_pass_manager(3, ...))
  C48   prototype c48, the default call (entangling_basis="cx", layout_search=True; level3_candidate "auto")
  C49   prototype c49, the same call
  C48o  c48 with level3_candidate=False (PSF-Zero's own pipeline, as HW2's arms)
  C49o  c49 with level3_candidate=False
Part F (item 85): SCALE1's R2 n=144 circuits (instances 0-2) on the snapshot with 6 more qubits reported failed
(sx error 1), three seeds; c48 and c49; c49's outputs sampled by MPS (20 shots) against the known answer.

    python benchmarks/c49_val.py <out dir> [--part F|R|all]   (refuses an existing out dir)
Needs qiskit-aer for part F."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import statistics
import sys
import time
import warnings

warnings.simplefilter("ignore")
sys.path[:0] = ["benchmarks", ".", os.path.join("data", "2026-10-12", "c48_dev")]
OUT = sys.argv[1]
PART = sys.argv[sys.argv.index("--part") + 1] if "--part" in sys.argv else "all"
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
READING = ("qaoa_n12_p2", "qaoa_n20_p2", "ba_n14", "ba_n16", "ba_n18", "ba_n20", "tfim_n12")  # HW2's seven pairs
REPEAT = ("qaoa_n20_p2", "tfim_n16", "ba_n20")
ARMS = ("QK3", "C48", "C49", "C48o", "C49o")
EXACT_MAX_ACTIVE = 20
F_SEEDS, F_EXTRA, F_INSTS, F_SHOTS = (1, 2, 3), 6, (0, 1, 2), 20


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


C48_FILE = os.path.join("patches", "psf_compile_c48_2026-10-12", "psf_compile.py")
C49_FILE = os.path.join("patches", "psf_compile_c49_2026-10-14", "psf_compile.py")
load("benchmarks/psf_smart_layout.py", "psf_smart_layout")
M48 = load(C48_FILE, "psf_c48")
M49 = load(C49_FILE, "psf_c49")
M48.WARN_WITHOUT_TARGET = M49.WARN_WITHOUT_TARGET = False
H = load("benchmarks/hw1_real.py", "hw1")
H.KIT = os.path.join("data", "2026-10-12", "hw_kit")
H2 = load(os.path.join("data", "2026-10-13", "hw2_kit", "hw2_real.py"), "hw2")
H2.KIT = H.KIT
import gens  # noqa: E402
import snapshot_target as ST  # noqa: E402
from qiskit.transpiler import InstructionProperties, generate_preset_pass_manager  # noqa: E402
from qiskit_ibm_runtime import fake_provider  # noqa: E402


def device(d):
    """(backend or None, Target)."""
    if d == "snapshot_marrakesh":
        return None, ST.target_from_snapshot(SNAPSHOT)
    b = getattr(fake_provider, d)()
    return b, b.target


def metrics(c, t):
    """Two-qubit count, ESP (reported only), duration without the measurements (as C48-VAL), active qubits."""
    pos = {q: i for i, q in enumerate(c.qubits)}
    clock = {}
    n2 = 0
    for ins in c.data:
        nm = ins.operation.name
        if nm in ("barrier", "measure"):
            continue
        qs = tuple(pos[q] for q in ins.qubits)
        pr = t[nm].get(qs) if nm in t.operation_names else None
        d = (pr.duration or 0.0) if pr is not None else 0.0
        s = max(clock.get(q, 0.0) for q in qs)
        for q in qs:
            clock[q] = s + d
        n2 += len(qs) == 2
    T = max(clock.values()) if clock else 0.0
    return dict(twoq=n2, esp=math.exp(-M48._st_cost(c, t)), dur_us=T * 1e6, active=len(clock))


def build(name):
    if name.startswith("tfimP"):
        return H2.build_test(name)[0]
    return H.build_test(name)[0] if name in H.TESTS else gens.build(name)


def compile_arm(arm, qc, b, t):
    if arm == "QK3":
        if b is not None:
            return generate_preset_pass_manager(3, b, seed_transpiler=0).run(qc)
        return generate_preset_pass_manager(3, target=t, seed_transpiler=0).run(qc)
    mod = M48 if arm.startswith("C48") else M49
    kw = dict(entangling_basis="cx", layout_search=True, seed_transpiler=0)
    if arm.endswith("o"):
        kw["level3_candidate"] = False
    kw.update(backend=b) if b is not None else kw.update(target=t)
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.compile_for_hardware(qc, **kw)


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


def part_r(lines):
    rows = []
    f = open(os.path.join(OUT, "c49_val_rows.jsonl"), "w")
    for d in DEVICES:
        b, t = device(d)
        for name in TESTS:
            qc = build(name)
            for arm in ARMS:
                t0 = time.time()
                err = None
                try:
                    c = compile_arm(arm, qc, b, t)
                except Exception as e:  # noqa: BLE001 - recorded: a refusal counts against the arm
                    c, err = None, f"{type(e).__name__}: {str(e)[:200]}"
                secs = time.time() - t0
                r = dict(device=d, test=name, arm=arm, secs=round(secs, 2), error=err)
                if c is not None:
                    r.update(sig=sig(c), **metrics(c, t))
                    if arm != "QK3":
                        r["exact"] = exactness(qc, c)
                    if arm == "C49" and name in REPEAT:
                        r["sig_repeat"] = sig(compile_arm(arm, qc, b, t))
                rows.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"{d[-10:]:10s} {name:16s} {arm:4s} " + (f"2q {r['twoq']:5d} dur {r['dur_us']:7.2f} exact "
                      f"{r.get('exact')} " if c is not None else f"ERROR {err} ") + f"{secs:.1f}s", flush=True)
    f.close()
    by = {(r["device"], r["test"], r["arm"]): r for r in rows}
    cells = [(d, t) for d in DEVICES for t in TESTS]

    def dur(d, t, a):
        r = by[(d, t, a)]
        return r["dur_us"] if r.get("error") is None else math.inf

    bad1 = [(d, t, a, by[(d, t, a)].get("exact", by[(d, t, a)].get("error"))) for d, t in cells for a in ("C49", "C49o")
            if by[(d, t, a)].get("error") is not None or (by[(d, t, a)].get("exact") or 0) > 1e-6]
    n1 = sum(1 for d, t in cells for a in ("C49", "C49o") if by[(d, t, a)].get("exact") is not None)
    lines.append(f"V1 exact: every C49 and C49o output with <= {EXACT_MAX_ACTIVE} active qubits within 1e-6, no refusal: "
                 f"{'CONFIRMED' if not bad1 else 'REFUTED'} ({n1} checked; failures {bad1})")
    for new, old in (("C49", "C48"), ("C49o", "C48o")):
        ok = [c for c in cells if dur(*c, new) <= 1.02 * dur(*c, old)]
        out = [(d, t, round(dur(d, t, new) / dur(d, t, old), 3)) for d, t in cells if (d, t) not in ok]
        lines.append(f"V2 {new} <= 1.02 x {old} duration on >= 95% of cells: "
                     f"{'CONFIRMED' if len(ok) >= 0.95 * len(cells) else 'REFUTED'} ({len(ok)}/{len(cells)}; outside {out})")
    ok3 = [c for c in cells if dur(*c, "C49") <= 1.02 * dur(*c, "QK3")]
    ok3_48 = [c for c in cells if dur(*c, "C48") <= 1.02 * dur(*c, "QK3")]
    lines.append(f"V3 C49 <= 1.02 x QK3 duration on >= 90% of cells: "
                 f"{'CONFIRMED' if len(ok3) >= 0.9 * len(cells) else 'REFUTED'} ({len(ok3)}/{len(cells)}; C48 "
                 f"{len(ok3_48)}/{len(cells)}; C49 above: "
                 f"{[(d, t, round(dur(d, t, 'C49') / dur(d, t, 'QK3'), 3)) for d, t in cells if (d, t) not in ok3]})")
    is4 = [(d, t) for d, t in cells if t in ISING]
    bad4 = [(d, t, round(dur(d, t, "C49") / dur(d, t, "QK3"), 3)) for d, t in is4 if dur(d, t, "C49") > 0.5 * dur(d, t, "QK3")]
    lines.append(f"V4 Ising: C49 <= 0.5 x QK3 duration on every Ising cell: {'CONFIRMED' if not bad4 else 'REFUTED'} "
                 f"({len(is4) - len(bad4)}/{len(is4)}; above {bad4})")
    rd = [("snapshot_marrakesh", t) for t in READING]
    bad5 = [(t, round(dur(d, t, "C49") / dur(d, t, "QK3"), 3)) for d, t in rd if dur(d, t, "C49") > 1.02 * dur(d, t, "QK3")]
    lines.append(f"V5 HW2's seven reading tests on the marrakesh snapshot: C49 <= 1.02 x QK3 on 7 of 7: "
                 f"{'CONFIRMED' if not bad5 else 'REFUTED'} ({7 - len(bad5)}/7; above {bad5})")
    rep = [r for r in rows if "sig_repeat" in r]
    lines.append(f"V6 same output on a second call (C49): {'CONFIRMED' if rep and all(r['sig'] == r['sig_repeat'] for r in rep) else 'REFUTED'} "
                 f"({sum(r['sig'] == r['sig_repeat'] for r in rep)}/{len(rep)})")
    ok7 = [c for c in cells if by[(*c, "C49")]["secs"] <= 1.5 * by[(*c, "C48")]["secs"] + 2.0]
    lines.append(f"V7 compile time C49 <= 1.5 x C48 + 2 s on >= 90% of cells: "
                 f"{'CONFIRMED' if len(ok7) >= 0.9 * len(cells) else 'REFUTED'} ({len(ok7)}/{len(cells)})")
    fin = [c for c in cells if math.isfinite(dur(*c, "C49")) and math.isfinite(dur(*c, "C48"))]
    rat = [dur(*c, "C49") / dur(*c, "C48") for c in fin]
    ratq = [dur(*c, "C49") / dur(*c, "QK3") for c in fin]
    esp = [by[(*c, "C49")]["esp"] / by[(*c, "C48")]["esp"] for c in fin if by[(*c, "C48")]["esp"] > 0]
    gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))  # noqa: E731
    lines.append(f"V9 reported: duration C49/C48 median {statistics.median(rat):.3f} geo {gm(rat):.3f} min {min(rat):.3f} "
                 f"max {max(rat):.3f}; C49/QK3 median {statistics.median(ratq):.3f} geo {gm(ratq):.3f}; "
                 f"ESP C49/C48 geo {gm([x for x in esp if x > 0]):.3f} (reported only)")
    lines.append(f"V10 reported: SELECT_STATS {M49.SELECT_STATS}; L3C_STATS c49 {getattr(M49, 'L3C_STATS', None)}; "
                 f"CUT_STATS {M49.CUT_STATS}")
    lines.append("")
    lines.append(f"{'device':10s} {'test':16s} {'dur us QK3 / C48 / C49 / C48o / C49o':>44s} {'C49/QK3':>7s} {'C49/C48':>7s} "
                 f"{'2q QK3/C48/C49':>16s}")
    for d, t in cells:
        def g(a, k="dur_us"):
            r = by[(d, t, a)]
            return r.get(k, float("nan")) if r.get("error") is None else float("nan")
        lines.append(f"{d[-10:]:10s} {t:16s} {g('QK3'):8.2f} /{g('C48'):8.2f} /{g('C49'):8.2f} /{g('C48o'):8.2f} /"
                     f"{g('C49o'):8.2f} {g('C49') / g('QK3'):7.3f} {g('C49') / g('C48'):7.3f} "
                     f"{g('QK3', 'twoq'):5.0f}/{g('C48', 'twoq'):5.0f}/{g('C49', 'twoq'):5.0f}")


def part_f(lines):
    from qiskit_aer import AerSimulator
    sc = load(os.path.join("data", "2026-10-13", "scale1_kit", "scale1_circuits.py"), "scale1_circuits")
    base = ST.target_from_snapshot(SNAPSHOT)
    _, bq0 = M48._failed_elements(base, 0.5)
    rows, f = [], open(os.path.join(OUT, "c49_val_partF.jsonl"), "w")
    for seed in F_SEEDS:
        t = copy.deepcopy(base)
        add = random.Random(seed).sample([q for q in range(t.num_qubits) if q not in bq0], F_EXTRA)
        for q in add:
            p = t["sx"][(q,)]
            t.update_instruction_properties("sx", (q,), InstructionProperties(duration=p.duration, error=1.0))
        _, bq = M48._failed_elements(t, 0.5)
        for inst in F_INSTS:
            qc = sc.family_R(144, 2, inst)
            ideal = sc.ideal_bitstring(qc)
            qc.measure_all()
            for arm, mod in (("C48", M48), ("C49", M49)):
                t0 = time.time()
                r = dict(seed=seed, failed=sorted(bq), inst=inst, arm=arm)
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        o = mod.compile_for_hardware(qc, target=t, seed_transpiler=0)
                    pos = {q: i for i, q in enumerate(o.qubits)}
                    used = {pos[q] for i in o.data for q in i.qubits if i.operation.name != "barrier"}
                    r.update(compiled=True, uses_failed=bool(used & bq), **metrics(o, t))
                    if arm == "C49":
                        res = AerSimulator(method="matrix_product_state", seed_simulator=1).run(
                            o, shots=F_SHOTS).result().get_counts()
                        r["mps_all_ideal"] = set(res) == {ideal}
                except Exception as e:  # noqa: BLE001
                    r.update(compiled=False, error=f"{type(e).__name__}: {str(e)[:160]}")
                r["secs"] = round(time.time() - t0, 1)
                rows.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(json.dumps(r)[:300], flush=True)
    f.close()
    c49 = [r for r in rows if r["arm"] == "C49"]
    good = [r for r in c49 if r.get("compiled") and not r.get("uses_failed") and r.get("mps_all_ideal")]
    lines.append(f"V8 item 85: C49 compiles SCALE1's R2 n=144 on the snapshot with {F_EXTRA} more failed qubits "
                 f"({len(F_SEEDS)} seeds x {len(F_INSTS)} instances), no failed qubit, MPS {F_SHOTS} shots all ideal: "
                 f"{'CONFIRMED' if len(good) == len(c49) else 'REFUTED'} ({len(good)}/{len(c49)}); C48 refused "
                 f"{sum(1 for r in rows if r['arm'] == 'C48' and not r.get('compiled'))}/{len(c49)} (reported)")


if __name__ == "__main__":
    head = dict(started=time.strftime("%Y-%m-%d %H:%M:%S %Z"), c48_sha256=sha(C48_FILE), c49_sha256=sha(C49_FILE),
                script_sha256=sha(os.path.abspath(__file__)), devices=DEVICES, tests=TESTS, part=PART)
    json.dump(head, open(os.path.join(OUT, "c49_val_head.json"), "w"), indent=1)
    lines = []
    if PART in ("R", "all"):
        part_r(lines)
    if PART in ("F", "all"):
        part_f(lines)
    lines.append(f"\nfinished {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    open(os.path.join(OUT, f"c49_val_report_{PART}.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
