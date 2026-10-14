"""Tests for the workplace proposal of item 91 on prototype psf_compile 2026-10-14.c53: a Pauli evolution whose
interaction graph embeds in the device, built by edge colouring (first VF2 embedding, no score; one layer per colour).
Targets: FakeFez's own (no rzz) and FakeFez's with rzz and rx added as in test_c51.py.

Run from the repository root with this file next to the built psf_compile.py:
    python -m pytest <dir>/test_91.py -q
"""
import contextlib
import copy
import io
import math
import os
import random
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("PSF_REPO", os.getcwd())
sys.path[:0] = [HERE, os.path.join(REPO, "benchmarks"), REPO]
C53_FILE = os.path.join(REPO, "patches", "psf_compile_c53_2026-10-14", "psf_compile.py")


def _load(path, name):
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    m = H.load_module(path, name)
    m.WARN_WITHOUT_TARGET = False
    return m


@pytest.fixture(scope="module")
def new():
    return _load(os.path.join(HERE, "psf_compile.py"), "psf_compile_c53_91_test")


@pytest.fixture(scope="module")
def c53():
    return _load(C53_FILE, "psf_compile_c53_for_91_test")


def fractional(target):
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RXGate, RZZGate
    from qiskit.transpiler import InstructionProperties
    t = copy.deepcopy(target)
    t.add_instruction(RZZGate(Parameter("t")), {qa: InstructionProperties(duration=p.duration, error=p.error)
                                                 for qa, p in target["cz"].items()})
    t.add_instruction(RXGate(Parameter("t")), {qa: InstructionProperties(duration=p.duration, error=p.error)
                                               for qa, p in target["sx"].items()})
    return t


@pytest.fixture(scope="module")
def fez():
    from qiskit_ibm_runtime.fake_provider import FakeFez
    b = FakeFez()
    return b.target, fractional(b.target)


def _lab(n, d):
    s = ["I"] * n
    for q, p in d.items():
        s[n - 1 - q] = p
    return "".join(s)


def _evo(n, terms, t, steps, measure, order=2):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import LieTrotter, SuzukiTrotter
    qc = QuantumCircuit(n)
    syn = SuzukiTrotter(order=2, reps=steps) if order == 2 else LieTrotter(reps=steps)
    qc.append(PauliEvolutionGate(SparsePauliOp.from_list(terms), time=t, synthesis=syn), range(n))
    if measure:
        qc.measure_all()
    return qc


def _adj(target):
    adj = {}
    for a, b in target.build_coupling_map().get_edges():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def patch(target, n, start=3, seed=0):
    """an n-qubit heavy-hex patch of the device (BFS from `start`, all couplers among them), labels shuffled"""
    adj = _adj(target)
    order, seen, fr = [], {start}, [start]
    while fr and len(order) < n:
        u = fr.pop(0)
        order.append(u)
        for v in sorted(adj[u]):
            if v not in seen:
                seen.add(v)
                fr.append(v)
    S = set(order[:n])
    perm = list(range(n))
    random.Random(seed).shuffle(perm)
    idx = {q: perm[i] for i, q in enumerate(order[:n])}
    return sorted({(min(idx[a], idx[b]), max(idx[a], idx[b])) for a in S for b in adj[a] if b in S})


def ising(n, edges, steps, measure=True, seed=5):
    rng = random.Random(seed)
    terms = [(_lab(n, {a: "Z", b: "Z"}), rng.choice([-1, 1]) * rng.uniform(0.3, 2.0)) for a, b in edges]
    terms += [(_lab(n, {q: "X"}), rng.uniform(-1, 1)) for q in range(n)]
    return _evo(n, terms, 1.0, steps, measure)


def xyz(n, edges, steps, measure=True, seed=7):
    rng = random.Random(seed)
    terms = [(_lab(n, {a: p, b: p}), rng.choice([-1, 1]) * rng.uniform(0.5, 1.5)) for p in "XYZ" for a, b in edges]
    terms += [(_lab(n, {q: "Z"}), rng.uniform(-1, 1)) for q in range(n)]
    return _evo(n, terms, 0.6, steps, measure, order=1)


def _call(mod, qc, target):
    with contextlib.redirect_stdout(io.StringIO()):
        return mod.compile_for_hardware(qc, target=target, seed_transpiler=0)


def _sig(c):
    from c29_identity import value_sig
    return value_sig(c)


def _probs(circ):
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    meas, body = {}, []
    for ins in circ.data:
        nm = ins.operation.name
        if nm == "measure":
            meas[circ.find_bit(ins.clbits[0]).index] = circ.find_bit(ins.qubits[0]).index
        elif nm not in ("barrier", "delay"):
            body.append((ins.operation, [circ.find_bit(q).index for q in ins.qubits]))
    active = sorted({q for _, qs in body for q in qs} | set(meas.values()))
    pos = {q: i for i, q in enumerate(active)}
    c = QuantumCircuit(len(active))
    for op, qs in body:
        c.append(op, [pos[q] for q in qs])
    c.save_probabilities([pos[meas[k]] for k in range(circ.num_clbits)])
    return np.asarray(AerSimulator(method="statevector").run(c).result().data()["probabilities"])


def _logical(qc):
    from qiskit import transpile
    return transpile(qc, basis_gates=["cx", "rz", "sx", "x", "h", "rzz", "rx", "ry", "u", "measure"],
                     optimization_level=0)


def same_distribution(out, qc):
    return 0.5 * float(np.abs(_probs(out) - _probs(_logical(qc))).sum()) < 1e-9


def same_unitary_with_phase(out, qc):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    n = qc.num_qubits
    lay = out.layout.initial_index_layout(filter_ancillas=True)
    if lay != out.layout.final_index_layout(filter_ancillas=True):
        return False
    pos = {p: i for i, p in enumerate(lay)}
    red = QuantumCircuit(n, global_phase=out.global_phase)
    for ins in out.data:
        qs = [out.find_bit(q).index for q in ins.qubits]
        if any(q not in pos for q in qs):
            return False
        red.append(ins.operation, [pos[q] for q in qs])
    return np.allclose(Operator(red).data, Operator(_logical(qc)).data, atol=1e-8)


def in_range(c):
    return all(0 < float(i.operation.params[0]) <= math.pi / 2 + 1e-12 for i in c.data if i.operation.name == "rzz")


def test_version(new):
    assert new.VERSION == "2026-10-14.c53+91"


def test_patch_ising_exact_in_range_and_not_longer_than_c53(new, c53, fez):
    t, ft = fez
    qc = ising(12, patch(t, 12), 3)
    for tg in (t, ft):
        k = new.GRAPH_STATS["used"]
        out = _call(new, qc, tg)
        assert new.GRAPH_STATS["used"] == k + 1
        assert same_distribution(out, qc)
        assert in_range(out) and new._isa_ok(out, tg)
        assert new._duration_s(out, tg) <= c53._duration_s(_call(c53, qc, tg), tg) + 1e-12
    assert any(i.operation.name == "rzz" for i in out.data)


def test_unmeasured_same_unitary_with_phase(new, fez):
    t, ft = fez
    tree = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)]  # degree-3 qubit 1, neighbours of degree <= 2: embeds
    for qc in (ising(10, patch(t, 10), 2, measure=False), xyz(8, patch(t, 8), 1, measure=False),
               ising(6, tree, 2, measure=False)):
        for tg in (t, ft):
            out = new._graph_brickwork(qc, tg, 0.5, 0)
            assert out is not None and in_range(out)
            assert same_unitary_with_phase(out, qc)


def test_colouring_proper_and_minimal(new):
    rng = random.Random(1)
    for _ in range(30):  # random bipartite graphs of max degree <= 3 (left: even labels, right: odd)
        edges = set()
        deg = {}
        for _ in range(25):
            a, b = 2 * rng.randrange(8), 2 * rng.randrange(8) + 1
            if deg.get(a, 0) < 3 and deg.get(b, 0) < 3 and (a, b) not in edges:
                edges.add((a, b))
                deg[a] = deg.get(a, 0) + 1
                deg[b] = deg.get(b, 0) + 1
        edges = sorted((min(e), max(e)) for e in edges)
        col = new._graph_colours(edges)
        assert col is not None and max(col.values()) + 1 == max(deg.values())
        for x in deg:
            cs = [col[e] for e in edges if x in e]
            assert len(cs) == len(set(cs))
        assert col == new._graph_colours(edges)
    assert new._graph_colours([(0, 1), (1, 2), (0, 2)]) is None  # a triangle is not bipartite


def test_not_embeddable_left_to_c53(new, c53, fez):
    _, ft = fez
    star = [(0, 1), (0, 2), (0, 3), (0, 4)]  # degree 4
    adjacent3 = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)]  # two degree-3 qubits side by side: not in heavy-hex
    square = [(0, 1), (1, 2), (2, 3), (0, 3), (3, 4)]  # a 4-cycle: heavy-hex has none shorter than 12
    for edges, n, why in ((star, 5, "degree"), (adjacent3, 6, "no_embedding"), (square, 5, "no_embedding")):
        k = new.GRAPH_STATS[why]
        qc = ising(n, edges, 2)
        assert new._graph_brickwork(qc, ft, 0.5, 0) is None
        assert new.GRAPH_STATS[why] == k + 1
        assert _sig(_call(new, qc, ft)) == _sig(_call(c53, qc, ft))


def test_twelve_ring_embeds_in_a_hexagon(new, fez):
    _, ft = fez
    ring = [(q, (q + 1) % 12) for q in range(12)]
    qc = ising(12, ring, 2)
    out = new._graph_brickwork(qc, ft, 0.5, 0)
    assert out is not None and same_distribution(out, qc)
    assert new.GRAPH_STATS["colours"].get(2, 0) >= 1


def test_chain_still_item_87(new, fez):
    _, ft = fez
    qc = ising(8, [(q, q + 1) for q in range(7)], 2)
    g, c = new.GRAPH_STATS["tried"], new.CHAIN_STATS["used"]
    _call(new, qc, ft)
    assert new.CHAIN_STATS["used"] == c + 1 and new.GRAPH_STATS["tried"] == g


def test_cz_path_without_rzz(new, fez):
    t, _ = fez
    qc = xyz(10, patch(t, 10), 1)
    out = new._graph_brickwork(qc, t, 0.5, 0)
    assert not any(i.operation.name == "rzz" for i in out.data)
    assert same_distribution(out, qc) and new._isa_ok(out, t)


def test_failed_qubit_avoided(new, fez):
    from qiskit.transpiler import InstructionProperties
    t, _ = fez
    qc = ising(10, patch(t, 10), 2)
    out = new._graph_brickwork(qc, t, 0.5, 0)
    used = sorted({out.find_bit(q).index for i in out.data for q in i.qubits if i.operation.name != "barrier"})
    bad = copy.deepcopy(t)
    p = bad["sx"][(used[2],)]
    bad.update_instruction_properties("sx", (used[2],), InstructionProperties(duration=p.duration, error=1.0))
    out2 = new._graph_brickwork(qc, bad, 0.5, 0)
    used2 = {out2.find_bit(q).index for i in out2.data for q in i.qubits if i.operation.name != "barrier"}
    assert used[2] not in used2 and same_distribution(out2, qc)


def test_deterministic(new, fez):
    _, ft = fez
    qc = ising(14, patch(ft, 14), 2)
    assert _sig(_call(new, qc, ft)) == _sig(_call(new, qc, ft))
