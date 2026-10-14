"""C56-VAL: prototype c56 (c55 with item 94 off + item 96, the scope gate: a circuit whose two-qubit interaction graph
does not embed in the device's working couplers is handed to Qiskit's level 3 unchanged) against c55 (item 94 off)
and Qiskit level 3. Offline; no real device. Built from C55-VAL's script; the same 28 tests and 5 devices.

Devices: FakeFez, FakeMarrakesh, FakeKingston, FakeTorino and ibm_marrakesh's HW2 snapshot (2026-10-13 11:37 CEST).
The fake Targets have no fractional gates; a fractional copy adds rzz on every cz coupler with a CZ's duration and rx
on every qubit with an sx's, as ibm_marrakesh reported them (rzz 68 ns, rx 36 ns). Arms, all with seed 0:
  QK3   Qiskit level 3 (generate_preset_pass_manager(3, target, seed 0)) on the Target without fractional gates
  QK3F  the same on the fractional Target
  C55   prototype c55's default call (item 94 off) on the Target without fractional gates
  C56   prototype c56's default call on the Target without fractional gates
  C55F  prototype c55's default call (item 94 off) on the fractional Target
  C56F  prototype c56's default call on the fractional Target
Tests: C49-VAL's 18, six layered classifiers with an odd number of CZ rings (W3's model; n = 4, 6; L = 3, 5, 9), and
C54-VAL's four embeddable ones.
Durations as soon as possible on each arm's own Target, measurements left out.

    python benchmarks/c56_val.py <out dir>   (refuses an existing out dir)"""
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
EMBED = ("hhI_n16", "hhI_n20", "ring12_tfim", "hhtree_heis_n16")
TESTS = TESTS + EMBED
ISING = ("tfim_n12", "tfim_n16", "tfimP_n12", "tfimP_n16", "tfim_n20_r3", "tfim_n40_r2")
REPEAT = ("qaoa_n20_p2", "tfim_n16", "ba_n20")
ARMS = ("QK3", "QK3F", "C55", "C56", "C55F", "C56F")
IN_SCOPE = ("tfim_n12", "tfim_n16", "ghz_n12", "tfimP_n12", "tfimP_n16", "tfim_n20_r3", "heis_n12_r2", "tfim_n40_r2",
            "hhI_n16", "hhI_n20", "ring12_tfim", "hhtree_heis_n16")  # V4: predicted in scope on every device
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
C55_FILE = os.path.join("patches", "psf_compile_c55_2026-10-14", "psf_compile.py")
C56_FILE = os.path.join("patches", "psf_compile_c56_2026-10-14", "psf_compile.py")
load("benchmarks/psf_smart_layout.py", "psf_smart_layout")
M49 = load(C49_FILE, "psf_c49")
M55 = load(C55_FILE, "psf_c55")
M56 = load(C56_FILE, "psf_c56")
M49.WARN_WITHOUT_TARGET = M55.WARN_WITHOUT_TARGET = M56.WARN_WITHOUT_TARGET = False
M55.ZZ_RELAYER = False  # item 94 dropped (Addendum 488); c56's default
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


def _hh_graph(n, kind, seed):
    """n qubits of FakeFez's coupling graph grown by BFS from qubit 0: all couplers among them ("patch") or the BFS
    tree ("tree"); labels shuffled with `seed`."""
    import random
    cm = fake_provider.FakeFez().target.build_coupling_map()
    adj = {}
    for a, b in cm.get_edges():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    order, par, seen, fr = [], {}, {0}, [0]
    while fr and len(order) < n:
        u = fr.pop(0)
        order.append(u)
        for v in sorted(adj[u]):
            if v not in seen:
                seen.add(v)
                par[v] = u
                fr.append(v)
    S = order[:n]
    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    idx = {q: perm[i] for i, q in enumerate(S)}
    if kind == "tree":
        return sorted((min(idx[v], idx[par[v]]), max(idx[v], idx[par[v]])) for v in S[1:])
    return sorted({(min(idx[a], idx[b]), max(idx[a], idx[b])) for a in S for b in adj[a] if b in S})


def _evolution(n, terms, t, steps):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import SuzukiTrotter
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(0.3 + 0.07 * q, q)
    qc.append(PauliEvolutionGate(SparsePauliOp.from_list(terms), time=t, synthesis=SuzukiTrotter(order=2, reps=steps)),
              range(n))
    qc.measure_all()
    return qc


def _lab(n, d):
    s = ["I"] * n
    for q, p in d.items():
        s[n - 1 - q] = p
    return "".join(s)


def build(name):
    if name in EMBED:
        import random
        rng = random.Random("C55-VAL|" + name)
        if name.startswith("hhI_n"):
            n = int(name.split("_n")[1])
            edges = _hh_graph(n, "patch", 5400 + n)
            terms = [(_lab(n, {a: "Z", b: "Z"}), rng.uniform(-1.5, 1.5)) for a, b in edges]
            terms += [(_lab(n, {q: "X"}), 0.7) for q in range(n)]
            return _evolution(n, terms, 1.0, 4)
        if name == "ring12_tfim":
            n = 12
            terms = [(_lab(n, {q: "Z", (q + 1) % n: "Z"}), rng.uniform(-1.5, 1.5)) for q in range(n)]
            terms += [(_lab(n, {q: "X"}), 0.7) for q in range(n)]
            return _evolution(n, terms, 1.0, 4)
        n = 16
        edges = _hh_graph(n, "tree", 5416)
        terms = [(_lab(n, {a: p, b: p}), rng.uniform(-1, 1)) for a, b in edges for p in "XYZ"]
        terms += [(_lab(n, {q: "Z"}), rng.uniform(-1, 1)) for q in range(n)]
        return _evolution(n, terms, 0.6, 2)
    if name.startswith("clf_"):
        n, L = int(name.split("_n")[1].split("_")[0]), int(name.split("_L")[1])
        return classifier(n, L, 52_000 + 100 * n + L)
    if name.startswith("tfimP"):
        return H2.build_test(name)[0]
    return H.build_test(name)[0] if name in H.TESTS else gens.build(name)


def compile_arm(arm, qc, t, ft):
    if arm in ("QK3", "QK3F"):
        return generate_preset_pass_manager(3, target=t if arm == "QK3" else ft, seed_transpiler=0).run(qc)
    mod, tg = {"C55": (M55, t), "C56": (M56, t), "C55F": (M55, ft), "C56F": (M56, ft)}[arm]
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


def in_scope(qc, t):
    e, r = M56._scope96_edges(qc)
    return r is None and (not e or M56._scope96_embeds(e, qc.num_qubits, t, 0.5))


def uses_failed(c, t):
    fe, fq = M56._failed_elements(t, 0.5)
    return M56._uses_failed(c, fe, fq)


if __name__ == "__main__":
    head = dict(started=time.strftime("%Y-%m-%d %H:%M:%S %Z"), c55_sha256=sha(C55_FILE), c56_sha256=sha(C56_FILE),
                script_sha256=sha(os.path.abspath(__file__)), devices=DEVICES, tests=TESTS)
    json.dump(head, open(os.path.join(OUT, "c56_val_head.json"), "w"), indent=1)
    rows = []
    f = open(os.path.join(OUT, "c56_val_rows.jsonl"), "w")
    for d in DEVICES:
        t, ft = device(d)
        for name in TESTS:
            qc = build(name)
            scope = in_scope(qc, t)
            for arm in ARMS:
                t0 = time.time()
                err = None
                try:
                    c = compile_arm(arm, qc, t, ft)
                except Exception as e:  # noqa: BLE001 - recorded
                    c, err = None, f"{type(e).__name__}: {str(e)[:200]}"
                secs = time.time() - t0
                tg = ft if arm.endswith("F") else t
                r = dict(device=d, test=name, arm=arm, in_scope=scope, secs=round(secs, 2), error=err)
                if c is not None:
                    r.update(sig=sig(c), uses_failed=uses_failed(c, tg), **metrics(c, tg))
                    if arm in ("C56", "C56F"):
                        r["exact"] = exactness(qc, c)
                    if arm == "C56F" and name in REPEAT + ("clf_n6_L5",):
                        r["sig_repeat"] = sig(compile_arm(arm, qc, t, ft))
                rows.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"{d[-10:]:10s} {name:16s} {arm:4s} {'IN ' if scope else 'out'} " + (
                    f"2q {r['twoq']:5d} rzz {r['rzz']:4d} dur {r['dur_us']:7.2f} exact {r.get('exact')} "
                    if c is not None else f"ERROR {err} ") + f"{secs:.1f}s", flush=True)
    f.close()
    by = {(r["device"], r["test"], r["arm"]): r for r in rows}
    cells = [(d, t) for d in DEVICES for t in TESTS]
    gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")  # noqa: E731
    ok = lambda d, t, a: by[(d, t, a)].get("error") is None  # noqa: E731
    lines = []
    bad1 = [(d, t, a, by[(d, t, a)].get("exact", by[(d, t, a)].get("error"))) for d, t in cells for a in ("C56", "C56F")
            if not ok(d, t, a) or (by[(d, t, a)].get("exact") or 0) > 1e-6]
    n1 = sum(1 for c in cells for a in ("C56", "C56F") if by[(*c, a)].get("exact") is not None)
    lines.append(f"V1 exact: every C56 and C56F output with <= {EXACT_MAX_ACTIVE} active qubits within 1e-6, no refusal: "
                 f"{'CONFIRMED' if not bad1 else 'REFUTED'} ({n1} checked; failures {bad1})")
    ins = [c for c in cells if by[(*c, "C56")]["in_scope"]]
    outs = [c for c in cells if not by[(*c, "C56")]["in_scope"]]
    bad2 = [(d, t, a) for d, t in ins for a, b in (("C56", "C55"), ("C56F", "C55F"))
            if not (ok(d, t, a) and ok(d, t, b) and by[(d, t, a)]["sig"] == by[(d, t, b)]["sig"])]
    lines.append(f"V2 in scope: C56 / C56F identical to C55 / C55F on every cell: {'CONFIRMED' if not bad2 else 'REFUTED'} "
                 f"({2 * len(ins) - len(bad2)}/{2 * len(ins)}; differing {bad2})")
    same3, excused, bad3 = 0, [], []
    for d, t in outs:
        for a, q in (("C56", "QK3"), ("C56F", "QK3F")):
            if ok(d, t, a) and ok(d, t, q) and by[(d, t, a)]["sig"] == by[(d, t, q)]["sig"]:
                same3 += 1
            elif ok(d, t, a) and ok(d, t, q) and by[(d, t, q)]["uses_failed"] and not by[(d, t, a)]["uses_failed"]:
                excused.append((d, t, a))
            else:
                bad3.append((d, t, a))
    lines.append(f"V3 out of scope: C56 / C56F identical to Qiskit level 3 (QK3 / QK3F), except where Qiskit's output "
                 f"uses a failed element and C56's does not: {'CONFIRMED' if not bad3 else 'REFUTED'} (identical {same3}; "
                 f"failed-element exceptions {excused}; other {bad3})")
    bad4 = [(d, t) for d, t in cells if by[(d, t, "C56")]["in_scope"] != (t in IN_SCOPE)]
    lines.append(f"V4 scope as predicted ({len(IN_SCOPE)} tests in, {len(TESTS) - len(IN_SCOPE)} out) on every device: "
                 f"{'CONFIRMED' if not bad4 else 'REFUTED'} (mismatches {bad4})")
    rep = [r for r in rows if "sig_repeat" in r]
    lines.append(f"V5 same output on a second call (C56F): {'CONFIRMED' if rep and all(r['sig'] == r['sig_repeat'] for r in rep) else 'REFUTED'} "
                 f"({sum(r['sig'] == r['sig_repeat'] for r in rep)}/{len(rep)})")
    bad6 = [(d, t) for d, t in cells if not (ok(d, t, "C56F") and by[(d, t, "C56F")].get("isa") and by[(d, t, "C56F")].get("rzz_in_range"))]
    lines.append(f"V6 every C56F output native to the fractional Target, every rzz angle in (0, pi/2]: "
                 f"{'CONFIRMED' if not bad6 else 'REFUTED'} ({len(cells) - len(bad6)}/{len(cells)}; failures {bad6})")
    ok7 = [c for c in cells if by[(*c, "C56F")]["secs"] <= 1.5 * by[(*c, "C55F")]["secs"] + 2.0]
    lines.append(f"V7 compile time C56F <= 1.5 x C55F + 2 s on >= 90% of cells: "
                 f"{'CONFIRMED' if len(ok7) >= 0.9 * len(cells) else 'REFUTED'} ({len(ok7)}/{len(cells)})")
    nf = [(d, t, a) for d, t in cells for a in ("C56", "C56F") if ok(d, t, a) and by[(d, t, a)]["uses_failed"]]
    lines.append(f"V8 no C56 / C56F output uses a failed element: {'CONFIRMED' if not nf else 'REFUTED'} ({nf})")
    def dur(d, t, a):
        return by[(d, t, a)]["dur_us"] if ok(d, t, a) else float("nan")
    for lab, cs in (("in scope", ins), ("out of scope", outs)):
        a = [dur(*c, "C56") / dur(*c, "C55") for c in cs]
        b = [dur(*c, "C56F") / dur(*c, "C55F") for c in cs]
        q = [dur(*c, "C56") / dur(*c, "QK3") for c in cs]
        qf = [dur(*c, "C56F") / dur(*c, "QK3F") for c in cs]
        lines.append(f"reported, {lab} ({len(cs)} cells): C56/C55 geo {gm(a):.3f}; C56F/C55F geo {gm(b):.3f}; "
                     f"C56/QK3 geo {gm(q):.3f}; C56F/QK3F geo {gm(qf):.3f}")
    lines.append(f"reported: SCOPE96_STATS {M56.SCOPE96_STATS}; EXACT93_STATS c56 {M56.EXACT93_STATS}")
    lines.append(f"{'device':10s} {'test':16s} {'scope':5s} {'dur us QK3 / C55 / C56 / QK3F / C55F / C56F':>52s}")
    for d, t in cells:
        lines.append(f"{d[-10:]:10s} {t:16s} {'IN' if by[(d, t, 'C56')]['in_scope'] else 'out':5s} "
                     + " /".join(f"{dur(d, t, a):8.2f}" for a in ARMS))
    lines.append(f"\nfinished {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    open(os.path.join(OUT, "c56_val_report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
