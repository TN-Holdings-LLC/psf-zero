"""family_map.py -- exploratory (not pre-registered), 2026-10-14: per circuit family, PSF-Zero's duration against a
load lower bound and against the best a knowledgeable Qiskit user can get, on FakeFez without (T) and with (F) IBM's
fractional gates (rzz with a CZ's length, rx with an sx's).

Arms (durations as soon as possible on the Target's gate lengths, measurements left out):
  PSF    c53's default call (classifiers: a16 on c53, the AI front end)
  QK3    Qiskit level 3, seed 0, given the circuit as written
  USER   the best of Qiskit level 3 with seeds 0-4, given the circuit as written and, where the circuit is a
         Pauli-term program, also given a hand-layered version: commuting two-qubit ZZ terms between single-qubit
         layers grouped by a greedy edge colouring into layers of disjoint pairs, written as rzz (what a user who
         knows the trick writes by hand); chains are already brickwork after that
  LB     a load lower bound: the busiest qubit's summed minimum time (after exact merging of repeated terms; a trailing run of
         diagonal terms before a measurement of every qubit, and a leading run of diagonal terms on qubits still in |0>,
         not charged), each two-qubit term at least one two-qubit
         pulse pair on T (2 CZ, 136 ns) or one rzz on F (68 ns), each X or Y rotation one sx pair on T (72 ns) or one rx
         on F (36 ns), Z rotations free, no SWAP and no dependency counted (so it is not reachable when routing is
         needed). For the classifiers, the circuit as written: its CZs and its non-Z rotations.
F's outputs are folded (FoldRzzAngle) before they are measured; correctness of PSF's outputs is checked on the
outcome distribution where at most 20 qubits are active.

    python benchmarks/family_map.py <out dir>"""
import contextlib
import copy
import io
import json
import math
import os
import random
import sys
import time
import warnings

warnings.simplefilter("ignore")
sys.path[:0] = ["benchmarks", ".", os.path.join("data", "2026-10-12", "c48_dev")]
OUT = sys.argv[1]
if os.path.exists(OUT):
    raise SystemExit(f"STOP: {OUT} exists")
os.makedirs(OUT)

import core_fix_c2_eval as H  # noqa: E402
H.load_module("benchmarks/psf_smart_layout.py", "psf_smart_layout")
C53 = H.load_module("patches/psf_compile_c53_2026-10-14/psf_compile.py", "psf_compile")  # a16 imports psf_compile
C53.WARN_WITHOUT_TARGET = False
A16 = H.load_module("patches/psf_ai_compile_a16_2026-10-14/psf_ai_compile.py", "a16_map")
V52 = None
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402
from qiskit.circuit.library import RXGate, RZZGate  # noqa: E402
from qiskit.transpiler import InstructionProperties, PassManager, generate_preset_pass_manager  # noqa: E402
from qiskit_ibm_runtime.fake_provider import FakeFez  # noqa: E402
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle  # noqa: E402

HW1 = H.load_module("benchmarks/hw1_real.py", "hw1_map")
HW1.KIT = os.path.join("data", "2026-10-12", "hw_kit")
HW2 = H.load_module(os.path.join("data", "2026-10-13", "hw2_kit", "hw2_real.py"), "hw2_map")
HW2.KIT = HW1.KIT
import gens  # noqa: E402

FAMILIES = [
    ("Ising chain", ["tfim_n12", "tfimP_n16", "tfim_n20_r3", "tfim_n40_r2"]),
    ("Heisenberg chain / 2D", ["heis_n12_r2", "heis2d_n16_r1"]),
    ("3-regular QAOA", ["qaoa_n12_p2", "qaoa_n20_p2", "qaoa3reg_n20_r3"]),
    ("Barabasi-Albert QAOA", ["ba_n14", "ba_n20"]),
    ("classifier (ring)", ["clf_n4_L4", "clf_n4_L5", "clf_n6_L8", "clf_n6_L9"]),
]


def classifier(n, layers, seed):
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
        return classifier(n, L, 61_000 + 100 * n + L)
    if name.startswith("tfimP"):
        return HW2.build_test(name)[0]
    return HW1.build_test(name)[0] if name in HW1.TESTS else gens.build(name)


def fractional(t):
    f = copy.deepcopy(t)
    f.add_instruction(RZZGate(Parameter("t")), {q: InstructionProperties(duration=p.duration, error=p.error)
                                                 for q, p in t["cz"].items()})
    f.add_instruction(RXGate(Parameter("t")), {q: InstructionProperties(duration=p.duration, error=p.error)
                                               for q, p in t["sx"].items()})
    return f


def dur(c, t):
    pos = {q: i for i, q in enumerate(c.qubits)}
    clock = {}
    for ins in c.data:
        nm = ins.operation.name
        if nm in ("barrier", "measure", "delay") or nm not in t.operation_names:
            continue
        qs = tuple(pos[q] for q in ins.qubits)
        p = t[nm].get(qs)
        d = (p.duration or 0.0) if p is not None else 0.0
        s = max(clock.get(q, 0.0) for q in qs)
        for q in qs:
            clock[q] = s + d
    return (max(clock.values()) if clock else 0.0) * 1e6


def n2(c):
    return sum(1 for i in c.data if len(i.qubits) == 2 and i.operation.name != "barrier")


def terms_of(qc):
    ev = C53._st_parse(qc)
    ev = ev + ([], None) if ev is not None else C53._st_parse_gates(qc)
    return ev


def load_bound(name, qc, frac):
    two, one = (68e-3, 36e-3) if frac else (136e-3, 72e-3)  # us
    load = {}
    if name.startswith("clf_"):
        for ins in qc.data:
            nm = ins.operation.name
            qs = [qc.find_bit(q).index for q in ins.qubits]
            if nm in ("barrier", "measure", "rz"):
                continue
            c = 68e-3 if len(qs) == 2 else one
            for q in qs:
                load[q] = load.get(q, 0.0) + c
        return max(load.values())
    ev = terms_of(qc)
    if ev is None:
        return None
    triples = list(C53._st_merge(list(ev[2])))  # exact merging of repeated terms (items 72-73)
    untouched = set(range(qc.num_qubits)) - {q for _, q in ev[0]}
    # a leading run of diagonal terms on qubits still in |0> only adds a phase: not charged
    while triples and set(triples[0][0]) <= {"I", "Z"} and \
            {len(triples[0][0]) - 1 - i for i, ch in enumerate(triples[0][0]) if ch != "I"} <= untouched:
        triples.pop(0)
    if {q for q, _ in ev[1]} == set(range(qc.num_qubits)) and not ev[4]:
        # every qubit measured and nothing after the terms: a trailing run of diagonal (Z-only) terms cannot change
        # the outcome distribution and needs no pulse (items 88-89 drop it), so it is not charged
        while triples and set(triples[-1][0]) <= {"I", "Z"}:
            triples.pop()
    for lb, a, _ in triples:
        n = len(lb)
        sup = [(n - 1 - i, ch) for i, ch in enumerate(lb) if ch != "I"]
        if abs(math.remainder(a, math.pi / 2)) < 1e-12:
            continue
        if len(sup) == 1:
            c = 0.0 if sup[0][1] == "Z" else one
        else:
            c = two
        for q, _ in sup:
            load[q] = load.get(q, 0.0) + c
    for op, q in ev[0]:  # leading single-qubit gates: one pulse pair on T, one on F, if not diagonal
        if op.name not in ("rz", "z", "s", "sdg", "t", "tdg", "p", "id"):
            load[q] = load.get(q, 0.0) + one
    return max(load.values()) if load else 0.0


def hand_layered(qc):
    """The circuit as a user who knows the trick writes it: its Pauli terms in order, each maximal run of pairwise
    commuting ZZ terms grouped by a greedy edge colouring into layers of disjoint pairs (rzz), other terms as their
    own PauliEvolutionGate; the reader's leading gates first, its final gates and measurements last. None when the
    circuit is not a Pauli-term program, or it permutes its qubits."""
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    ev = terms_of(qc)
    if ev is None:
        return None
    prefix, meas, triples, _, suffix, perm = ev
    if perm and list(perm) != list(range(qc.num_qubits)):
        return None
    n = qc.num_qubits
    out = QuantumCircuit(n, qc.num_clbits)
    for op, q in prefix:
        out.append(op, [q])

    def zz(lb):
        sup = [n - 1 - i for i, ch in enumerate(lb) if ch != "I"]
        return len(sup) == 2 and all(lb[n - 1 - q] == "Z" for q in sup) and tuple(sup)

    run = []

    def flush():
        layers = []
        for (a, b), ang in run:
            for lay in layers:
                if a not in lay["q"] and b not in lay["q"]:
                    lay["g"].append((a, b, ang))
                    lay["q"] |= {a, b}
                    break
            else:
                layers.append({"q": {a, b}, "g": [(a, b, ang)]})
        for lay in layers:
            for a, b, ang in lay["g"]:
                out.rzz(2 * ang, a, b)
        run.clear()

    for lb, ang, _ in triples:
        e = zz(lb)
        if e:
            run.append((e, ang))
            continue
        flush()
        sup = [n - 1 - i for i, ch in enumerate(lb) if ch != "I"]  # on its own support only
        out.append(PauliEvolutionGate(SparsePauliOp("".join(lb[n - 1 - q] for q in reversed(sup))), time=ang), sup)
    flush()
    for op, w in suffix:
        out.append(op, [w])
    for q, c in meas:
        out.measure(q, c)
    return out


def qk3(qc, t, seed):
    c = generate_preset_pass_manager(3, target=t, seed_transpiler=seed).run(qc)
    if "rzz" in t.operation_names:
        c = PassManager([FoldRzzAngle()]).run(c)
    return c


def exact(qc, c):
    act = len({q for i in c.data if i.operation.name != "barrier" for q in i.qubits})
    if act > 20:
        return None
    p0, _ = HW1.measured_distribution(HW1._unrolled(qc))
    p1, _ = HW1.measured_distribution(c)
    return max(abs(p0.get(k, 0) - p1.get(k, 0)) for k in set(p0) | set(p1))


def main():
    b = FakeFez()
    targets = {"T": b.target, "F": fractional(b.target)}
    rows = []
    f = open(os.path.join(OUT, "family_map_rows.jsonl"), "w")
    for fam, names in FAMILIES:
        for name in names:
            qc = build(name)
            hl = None if name.startswith("clf_") else hand_layered(qc)
            hl_ok = None
            if hl is not None:
                d = exact(qc, hl)
                hl_ok = d is None or d < 1e-6
                if not hl_ok:
                    hl = None
            for tk, t in targets.items():
                t0 = time.time()
                with contextlib.redirect_stdout(io.StringIO()):
                    if name.startswith("clf_"):
                        basis = [g for g in t.operation_names if g in ("cz", "rz", "sx", "x", "rx", "rzz")]
                        psf = A16.compile_for_model_circuit(qc, t.build_coupling_map(), basis, target=t)
                    else:
                        psf = C53.compile_for_hardware(qc, target=t, entangling_basis="cx", layout_search=True,
                                                       seed_transpiler=0)
                ps = time.time() - t0
                q0 = qk3(qc, t, 0)
                user = [(dur(qk3(qc, t, s), t), "as written", s) for s in range(5)]
                if hl is not None:
                    user += [(dur(qk3(hl, t, s), t), "hand-layered", s) for s in range(5)]
                ub = min(user)
                lb = load_bound(name, qc, tk == "F")
                r = dict(family=fam, test=name, target=tk, psf_us=dur(psf, t), psf_2q=n2(psf), psf_exact=exact(qc, psf),
                         psf_s=round(ps, 1), qk3_us=dur(q0, t), user_us=ub[0], user_how=f"{ub[1]}, seed {ub[2]}",
                         hand_layered_exact=hl_ok, lb_us=lb)
                rows.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"{tk} {name:16s} PSF {r['psf_us']:7.2f} QK3 {r['qk3_us']:7.2f} USER {r['user_us']:7.2f} "
                      f"({r['user_how']}) LB {lb if lb is None else round(lb, 3)} exact {r['psf_exact']}", flush=True)
    f.close()
    gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")  # noqa: E731
    lines = ["| family | target | PSF / LB | PSF / QK3 | PSF / user's best | user's best / LB |", "|---|---|---|---|---|---|"]
    for fam, names in FAMILIES:
        for tk in ("T", "F"):
            rr = [r for r in rows if r["family"] == fam and r["target"] == tk]
            pl = [r["psf_us"] / r["lb_us"] for r in rr if r["lb_us"]]
            ul = [r["user_us"] / r["lb_us"] for r in rr if r["lb_us"]]
            lines.append(f"| {fam} | {tk} | {gm(pl):.2f} | {gm([r['psf_us'] / r['qk3_us'] for r in rr]):.3f} | "
                         f"{gm([r['psf_us'] / r['user_us'] for r in rr]):.3f} | {gm(ul):.2f} |")
    open(os.path.join(OUT, "family_map.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
