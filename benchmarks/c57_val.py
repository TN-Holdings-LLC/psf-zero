"""C57-VAL: prototype c57 (c56 + item 96 revised: on a Target with rzz, the handed-on output is folded into IBM's
range by item 86's _fold) against C56-VAL's recorded outputs and Qiskit level 3 folded the same way. Offline; no real
device. Built from C56-VAL's script; the same 28 tests and 5 devices. C56's and C56F's outputs are not recompiled:
their signatures are read from C56-VAL's rows (data/2026-10-14/c56_val/run/c56_val_rows.jsonl).

Devices: FakeFez, FakeMarrakesh, FakeKingston, FakeTorino and ibm_marrakesh's HW2 snapshot (2026-10-13 11:37 CEST).
The fake Targets have no fractional gates; a fractional copy adds rzz on every cz coupler with a CZ's duration and rx
on every qubit with an sx's, as ibm_marrakesh reported them (rzz 68 ns, rx 36 ns). Arms, all with seed 0:
  QK3F  Qiskit level 3 (generate_preset_pass_manager(3, target, seed 0)) on the fractional Target; its output folded by
        c57's _fold is recorded as QK3F+fold
  C57   prototype c57's default call on the Target without fractional gates
  C57F  prototype c57's default call on the fractional Target
Tests: C49-VAL's 18, six layered classifiers with an odd number of CZ rings (W3's model; n = 4, 6; L = 3, 5, 9), and
C54-VAL's four embeddable ones.
Durations as soon as possible on each arm's own Target, measurements left out.

    python benchmarks/c57_val.py <out dir>   (refuses an existing out dir)"""
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
ARMS = ("QK3F", "C57", "C57F")
C56_ROWS = os.path.join("data", "2026-10-14", "c56_val", "run", "c56_val_rows.jsonl")
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
C56_FILE = os.path.join("patches", "psf_compile_c56_2026-10-14", "psf_compile.py")
C57_FILE = os.path.join("patches", "psf_compile_c57_2026-10-14", "psf_compile.py")
load("benchmarks/psf_smart_layout.py", "psf_smart_layout")
M49 = load(C49_FILE, "psf_c49")
M57 = load(C57_FILE, "psf_c57")
M56 = M57  # the scope reading and failed-element helpers are unchanged from c56
M49.WARN_WITHOUT_TARGET = M57.WARN_WITHOUT_TARGET = False
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
    if arm == "QK3F":
        return generate_preset_pass_manager(3, target=ft, seed_transpiler=0).run(qc)
    mod, tg = {"C57": (M57, t), "C57F": (M57, ft)}[arm]
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
    head = dict(started=time.strftime("%Y-%m-%d %H:%M:%S %Z"), c56_sha256=sha(C56_FILE), c57_sha256=sha(C57_FILE),
                c56_rows_sha256=sha(C56_ROWS), script_sha256=sha(os.path.abspath(__file__)), devices=DEVICES,
                tests=TESTS)
    json.dump(head, open(os.path.join(OUT, "c57_val_head.json"), "w"), indent=1)
    old = {(r["device"], r["test"], r["arm"]): r for r in map(json.loads, open(C56_ROWS))}
    rows = []
    f = open(os.path.join(OUT, "c57_val_rows.jsonl"), "w")
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
                    if arm == "QK3F":
                        try:
                            r["sig_fold"] = sig(M57._fold(c, ft))
                        except Exception as e:  # noqa: BLE001
                            r["sig_fold"] = f"fold failed: {type(e).__name__}"
                    if arm in ("C57", "C57F"):
                        r["exact"] = exactness(qc, c)
                    if arm == "C57F" and name in REPEAT + ("clf_n6_L5",):
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
    ok = lambda d, t, a: by[(d, t, a)].get("error") is None  # noqa: E731
    lines = []
    bad1 = [(d, t, a, by[(d, t, a)].get("exact", by[(d, t, a)].get("error"))) for d, t in cells for a in ("C57", "C57F")
            if not ok(d, t, a) or (by[(d, t, a)].get("exact") or 0) > 1e-6]
    n1 = sum(1 for c in cells for a in ("C57", "C57F") if by[(*c, a)].get("exact") is not None)
    lines.append(f"V1 exact: every C57 and C57F output with <= {EXACT_MAX_ACTIVE} active qubits within 1e-6, no refusal: "
                 f"{'CONFIRMED' if not bad1 else 'REFUTED'} ({n1} checked; failures {bad1})")
    bad2 = [(d, t) for d, t in cells if not ok(d, t, "C57") or by[(d, t, "C57")]["sig"] != old[(d, t, "C56")]["sig"]]
    lines.append(f"V2 C57 identical to C56 (C56-VAL's rows) on every cell: {'CONFIRMED' if not bad2 else 'REFUTED'} "
                 f"({len(cells) - len(bad2)}/{len(cells)}; differing {bad2})")
    ins = [c for c in cells if by[(*c, "C57")]["in_scope"]]
    outs = [c for c in cells if not by[(*c, "C57")]["in_scope"]]
    bad3 = [(d, t) for d, t in ins if not ok(d, t, "C57F") or by[(d, t, "C57F")]["sig"] != old[(d, t, "C56F")]["sig"]]
    lines.append(f"V3 in scope: C57F identical to C56F (C56-VAL's rows): {'CONFIRMED' if not bad3 else 'REFUTED'} "
                 f"({len(ins) - len(bad3)}/{len(ins)}; differing {bad3})")
    same4, exc4, bad4 = 0, [], []
    for d, t in outs:
        if ok(d, t, "C57F") and by[(d, t, "C57F")]["sig"] == by[(d, t, "QK3F")].get("sig_fold"):
            same4 += 1
        elif old[(d, t, "QK3F")].get("uses_failed") and not by[(d, t, "C57F")].get("uses_failed", True):
            exc4.append((d, t))
        else:
            bad4.append((d, t))
    lines.append(f"V4 out of scope: C57F identical to Qiskit level 3 folded by _fold, except where Qiskit's output uses a "
                 f"failed element (C56-VAL's record) and C57F's does not: {'CONFIRMED' if not bad4 else 'REFUTED'} "
                 f"(identical {same4}; failed-element exceptions {exc4}; other {bad4})")
    rep = [r for r in rows if "sig_repeat" in r]
    lines.append(f"V5 same output on a second call (C57F): {'CONFIRMED' if rep and all(r['sig'] == r['sig_repeat'] for r in rep) else 'REFUTED'} "
                 f"({sum(r['sig'] == r['sig_repeat'] for r in rep)}/{len(rep)})")
    bad6 = [(d, t) for d, t in cells if not (ok(d, t, "C57F") and by[(d, t, "C57F")].get("isa") and by[(d, t, "C57F")].get("rzz_in_range"))]
    lines.append(f"V6 every C57F output native to the fractional Target, every rzz angle in (0, pi/2]: "
                 f"{'CONFIRMED' if not bad6 else 'REFUTED'} ({len(cells) - len(bad6)}/{len(cells)}; failures {bad6})")
    nf = [(d, t, a) for d, t in cells for a in ("C57", "C57F") if ok(d, t, a) and by[(d, t, a)]["uses_failed"]]
    lines.append(f"V7 no C57 / C57F output uses a failed element: {'CONFIRMED' if not nf else 'REFUTED'} ({nf})")
    ok8 = [c for c in cells if by[(*c, "C57F")]["secs"] <= 1.5 * old[(*c, "C56F")]["secs"] + 2.0]
    lines.append(f"V8 compile time C57F <= 1.5 x C56F (C56-VAL's) + 2 s on >= 90% of cells: "
                 f"{'CONFIRMED' if len(ok8) >= 0.9 * len(cells) else 'REFUTED'} ({len(ok8)}/{len(cells)})")
    gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")  # noqa: E731
    for lab, cs in (("in scope", ins), ("out of scope", outs)):
        q = [by[(*c, "C57F")]["dur_us"] / by[(*c, "QK3F")]["dur_us"] for c in cs if ok(*c, "C57F") and ok(*c, "QK3F")]
        p = [by[(*c, "C57F")]["dur_us"] / old[(*c, "C55F")]["dur_us"] for c in cs if ok(*c, "C57F")]
        lines.append(f"reported, {lab} ({len(cs)} cells): C57F/QK3F (unfolded) geo {gm(q):.3f}; C57F/C55F geo {gm(p):.3f}")
    lines.append(f"reported: SCOPE96_STATS {M57.SCOPE96_STATS}; EXACT93_STATS c57 {M57.EXACT93_STATS}")
    lines.append(f"\nfinished {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    open(os.path.join(OUT, "c57_val_report.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
