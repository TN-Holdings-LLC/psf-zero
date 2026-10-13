"""C48-VAL: prototype c48 (item 83, time-aware block order) against itself with item 83 off (= c47) and Qiskit level 3,
on four fake IBM devices. Offline; no real device. Writes one JSON line per (device, test, arm) and a report.
    python c48_val.py <out dir>   (refuses an existing out dir: a run never overwrites another)"""
# Repository form of a development script: workspace paths replaced by repository-relative ones (run from the
# repository root; Benchpress's QASM files under $BENCHPRESS_QASM).

import sys, os, io, json, math, time, contextlib, importlib.util, warnings, hashlib, statistics
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = ["benchmarks", ".", os.path.join("data", "2026-10-12", "c48_dev")]
OUT = sys.argv[1]
if os.path.exists(OUT):
    raise SystemExit(f"STOP: {OUT} exists (a run never overwrites another)")
os.makedirs(OUT)

DEVICES = ("FakeFez", "FakeMarrakesh", "FakeKingston", "FakeTorino")
TESTS = ("qaoa_n12_p2", "qaoa_n20_p2", "ba_n14", "ba_n16", "ba_n18", "ba_n20", "tfim_n12", "tfim_n16", "ghz_n12",
         "tfim_n20_r3", "heis_n12_r2", "heis2d_n16_r1", "chem_n12_r1", "qaoa3reg_n20_r3", "qaoa3reg_n40_r3",
         "tfim_n40_r2")
TFIM = ("tfim_n12", "tfim_n16", "tfim_n20_r3", "tfim_n40_r2")
EXACT_MAX_ACTIVE = 20
COMPILE_CAP_S = 600.0  # per compile; a longer one is recorded and counted against V7


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


C48_FILE = os.path.join("patches", "psf_compile_c48_2026-10-12", "psf_compile.py")
load("benchmarks/psf_smart_layout.py", "psf_smart_layout")
M = load(C48_FILE, "psf_c48")
M.WARN_WITHOUT_TARGET = False
H = load("benchmarks/hw1_real.py", "hw1")
H.KIT = os.path.join("data", "2026-10-12", "hw_kit")  # its qasm/ (the kit as committed with Addendum 455)
import gens  # noqa: E402
from qiskit.transpiler import generate_preset_pass_manager  # noqa: E402
from qiskit_ibm_runtime import fake_provider  # noqa: E402


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def metrics(c, t):
    pos = {q: i for i, q in enumerate(c.qubits)}
    clock, first = {}, {}
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
            first.setdefault(q, s)
            clock[q] = s + d
        n2 += len(qs) == 2
    T = max(clock.values()) if clock else 0.0
    ln_idle = sum((T - first[q]) * (1 / (4 * t.qubit_properties[q].t1) + 1 / (2 * t.qubit_properties[q].t2))
                  for q in clock)
    ln_esp = M._st_cost(c, t)
    return dict(twoq=n2, esp=math.exp(-ln_esp), dur_us=T * 1e6, idle_est=math.exp(-ln_esp - ln_idle),
                active=len(clock))


def build(name):
    return H.build_test(name)[0] if name in H.TESTS else gens.build(name)


def compile_arm(arm, qc, b):
    if arm == "QK3":
        return generate_preset_pass_manager(3, b, seed_transpiler=0).run(qc)
    M.ST_SCHEDULE = arm == "C48"
    with contextlib.redirect_stdout(io.StringIO()):
        return M.compile_for_hardware(qc, backend=b, entangling_basis="cx", layout_search=True, seed_transpiler=0,
                                      level3_candidate=False)


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
    head = dict(started=time.strftime("%Y-%m-%d %H:%M:%S %Z"), c48_sha256=sha(C48_FILE),
                script_sha256=sha(os.path.abspath(__file__)), devices=DEVICES, tests=TESTS)
    json.dump(head, open(os.path.join(OUT, "c48_val_head.json"), "w"), indent=1)
    rows = []
    f = open(os.path.join(OUT, "c48_val_rows.jsonl"), "w")
    for d in DEVICES:
        b = getattr(fake_provider, d)()
        for name in TESTS:
            qc = build(name)
            for arm in ("QK3", "C47", "C48"):
                t0 = time.time()
                c = compile_arm(arm, qc, b)
                secs = time.time() - t0
                r = dict(device=d, test=name, arm=arm, secs=round(secs, 2), sig=sig(c), **metrics(c, b.target))
                if arm != "QK3":
                    r["exact"] = exactness(qc, c)
                if arm == "C48" and name in ("qaoa_n20_p2", "tfim_n16"):
                    r["sig_repeat"] = sig(compile_arm(arm, qc, b))
                rows.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"{d[4:]:10s} {name:16s} {arm:4s} 2q {r['twoq']:5d} ESP {r['esp']:.3f} dur {r['dur_us']:7.2f} "
                      f"exact {r.get('exact')} {secs:.1f}s", flush=True)
    f.close()
    # ---------------------------------------------------------------- scoring (as fixed in the prestatement)
    by = {(r["device"], r["test"], r["arm"]): r for r in rows}
    pairs = [(by[(d, t, "C47")], by[(d, t, "C48")], by[(d, t, "QK3")]) for d in DEVICES for t in TESTS]
    lines = []
    v1 = [(o["test"], o["device"]) for o, n, _ in pairs
          if not (abs(n["twoq"] - o["twoq"]) <= 0.05 * o["twoq"] and n["esp"] >= 0.97 * o["esp"])]
    lines.append(f"V1 gates: C48 2q within 5% of C47 and ESP >= 0.97 x C47 on every test: "
                 f"{'CONFIRMED' if not v1 else 'REFUTED'} ({len(pairs) - len(v1)}/{len(pairs)}; outside: {v1})")
    ok2 = sum(1 for o, n, _ in pairs if n["dur_us"] <= 1.02 * o["dur_us"])
    lines.append(f"V2 duration: C48 <= 1.02 x C47 on >= 90% of tests: {'CONFIRMED' if ok2 >= 0.9 * len(pairs) else 'REFUTED'} "
                 f"({ok2}/{len(pairs)})")
    tf = [(o, n) for o, n, _ in pairs if o["test"] in TFIM]
    bad3 = [(o["test"], o["device"], round(n["dur_us"] / o["dur_us"], 3)) for o, n in tf if n["dur_us"] > 0.5 * o["dur_us"]]
    lines.append(f"V3 Ising: C48 <= 0.5 x C47 duration on every Ising test and device: {'CONFIRMED' if not bad3 else 'REFUTED'} "
                 f"({len(tf) - len(bad3)}/{len(tf)}; above: {bad3})")
    ex = [n["exact"] for _, n, _ in pairs if n.get("exact") is not None]
    bad4 = [(n["test"], n["device"], n["exact"]) for _, n, _ in pairs if n.get("exact") is not None and n["exact"] > 1e-6]
    lines.append(f"V4 exact: every C48 output with <= {EXACT_MAX_ACTIVE} active qubits within 1e-6: "
                 f"{'CONFIRMED' if not bad4 else 'REFUTED'} ({len(ex) - len(bad4)}/{len(ex)} checked; largest "
                 f"{max(ex):.1e}; failures {bad4})")
    rat = [n["dur_us"] / o["dur_us"] for o, n, _ in pairs]
    lines.append(f"V5 reported: duration C48/C47 median {statistics.median(rat):.3f}, geometric mean "
                 f"{math.exp(sum(math.log(x) for x in rat) / len(rat)):.3f}, min {min(rat):.3f}, max {max(rat):.3f}")
    q47 = sum(1 for o, _, q in pairs if o["dur_us"] <= q["dur_us"])
    q48 = sum(1 for _, n, q in pairs if n["dur_us"] <= q["dur_us"])
    lines.append(f"V6 reported: duration <= QK3's: C47 {q47}/{len(pairs)}, C48 {q48}/{len(pairs)}; ESP >= QK3's: C48 "
                 f"{sum(1 for _, n, q in pairs if n['esp'] >= q['esp'])}/{len(pairs)}")
    slow = [(n["test"], n["device"], n["secs"], o["secs"]) for o, n, _ in pairs if n["secs"] > 1.5 * o["secs"] + 2.0
            or n["secs"] > COMPILE_CAP_S]
    lines.append(f"V7 compile time: C48 <= 1.5 x C47 + 2 s on every test: {'CONFIRMED' if not slow else 'REFUTED'} "
                 f"(outside: {slow})")
    rep = [r for r in rows if "sig_repeat" in r]
    lines.append(f"V8 same output on a second call: {'CONFIRMED' if all(r['sig'] == r['sig_repeat'] for r in rep) else 'REFUTED'} "
                 f"({sum(r['sig'] == r['sig_repeat'] for r in rep)}/{len(rep)})")
    lines.append("")
    lines.append(f"{'device':10s} {'test':16s} {'2q QK3/C47/C48':>17s} {'ESP C48/C47':>11s} {'dur us QK3 / C47 / C48':>26s} "
                 f"{'C48/C47':>7s} {'C48/QK3':>7s}")
    for o, n, q in pairs:
        lines.append(f"{o['device'][4:]:10s} {o['test']:16s} {q['twoq']:5d}/{o['twoq']:5d}/{n['twoq']:5d} "
                     f"{n['esp'] / o['esp']:11.3f} {q['dur_us']:8.2f} /{o['dur_us']:8.2f} /{n['dur_us']:7.2f} "
                     f"{n['dur_us'] / o['dur_us']:7.3f} {n['dur_us'] / q['dur_us']:7.3f}")
    lines.append(f"\nfinished {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    open(os.path.join(OUT, "c48_val_report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
