"""Tests for prototypes psf_compile 2026-10-15.c59 (item 98: the gate closed) and psf_ai_compile 2026-10-15.a20
(item 35: the ring template; item 36: the gate closed).

Run from the repository root:  python -m pytest patches/psf_ai_compile_a20_2026-10-15/test_a20_c59.py -q
"""
import contextlib
import copy
import io
import math
import os
import random
import sys
import warnings

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(REPO, "benchmarks"), REPO]
warnings.simplefilter("ignore")


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    pc = H.load_module(os.path.join(REPO, "patches", "psf_compile_c59_2026-10-15", "psf_compile.py"), "psf_compile")
    pc.WARN_WITHOUT_TARGET = False
    a = H.load_module(os.path.join(HERE, "psf_ai_compile.py"), "psf_ai_compile_a20_test")
    return dict(pc=pc, a=a)


@pytest.fixture(scope="module")
def fez():
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RXGate, RZZGate
    from qiskit.transpiler import InstructionProperties
    from qiskit_ibm_runtime.fake_provider import FakeFez
    t = FakeFez().target
    f = copy.deepcopy(t)
    f.add_instruction(RZZGate(Parameter("t")), {q: InstructionProperties(duration=p.duration, error=p.error)
                                                 for q, p in t["cz"].items()})
    f.add_instruction(RXGate(Parameter("t")), {q: InstructionProperties(duration=p.duration, error=p.error)
                                               for q, p in t["sx"].items()})
    return t, f


def classifier(n, layers, seed):
    from qiskit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            qc.ry(rng.uniform(-math.pi, math.pi), q)
            qc.rz(rng.uniform(-math.pi, math.pi), q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    for q in range(n):
        qc.ry(rng.uniform(-math.pi, math.pi), q)
    qc.measure_all()
    return qc


def _probs(c):
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    meas, body = {}, []
    for ins in c.data:
        if ins.operation.name == "measure":
            meas[c.find_bit(ins.clbits[0]).index] = c.find_bit(ins.qubits[0]).index
        elif ins.operation.name != "barrier":
            body.append((ins.operation, [c.find_bit(q).index for q in ins.qubits]))
    act = sorted({q for _, qs in body for q in qs} | set(meas.values()))
    pos = {q: i for i, q in enumerate(act)}
    s = QuantumCircuit(len(act))
    for op, qs in body:
        s.append(op, [pos[q] for q in qs])
    p = Statevector(s).probabilities()
    out = {}
    for idx in np.nonzero(p > 1e-15)[0]:
        k = "".join("1" if (int(idx) >> pos[meas[cb]]) & 1 else "0" for cb in sorted(meas, reverse=True))
        out[k] = out.get(k, 0) + float(p[idx])
    return out


def _tvd(a, b):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


def _with_values(qc, tpl, vals):
    ref = qc.copy_empty_like()
    k2v = {k: v for (k, _), v in zip(tpl.var_kinds, vals)}
    for k, ins in enumerate(qc.data):
        ref.append(ins.replace(operation=ins.operation.__class__(k2v[k])) if k in k2v else ins)
    return ref


def test_versions(mods):
    assert mods["pc"].VERSION == "2026-10-15.c59" and mods["pc"].GATE_CLOSED is True
    assert mods["a"].AI_COMPILE_VERSION == "2026-10-15.a20" and mods["a"].GATE_CLOSED is True


def test_compiler_refuses_out_of_scope_and_no_target(mods, fez):
    import networkx as nx
    from qiskit import QuantumCircuit
    pc = mods["pc"]
    t, _ = fez
    g = nx.random_regular_graph(3, 10, seed=1)
    qc = QuantumCircuit(10)
    for a, b in g.edges():
        qc.rzz(0.5, a, b)
    qc.measure_all()
    with pytest.raises(pc.OutOfScopeError, match="does not embed"):
        pc.compile_for_hardware(qc, target=t)
    with pytest.raises(pc.OutOfScopeError, match="Target"):
        pc.compile_for_hardware(qc, coupling_map=t.build_coupling_map(), basis_gates=["cz", "rz", "sx", "x"])
    ghz = QuantumCircuit(5)
    ghz.h(0)
    for q in range(4):
        ghz.cx(q, q + 1)
    ghz.measure_all()
    with pytest.raises(pc.OutOfScopeError, match="neither formula"):
        pc.compile_for_hardware(ghz, target=t)


def test_compiler_builds_a_chain(mods, fez):
    from qiskit import QuantumCircuit
    pc = mods["pc"]
    t, f = fez
    qc = QuantumCircuit(8)
    for q in range(8):
        qc.ry(0.3 + 0.1 * q, q)
    for q in range(7):
        qc.rzz(0.4, q, q + 1)
    qc.measure_all()
    for tg in (t, f):
        out = pc.compile_for_hardware(qc, target=tg, seed_transpiler=0)
        assert pc._isa_ok(out, tg) and pc._exact93(qc, out) is True


@pytest.mark.parametrize("n,L", [(4, 3), (5, 4), (6, 8), (7, 5)])
def test_template_exact_and_as_short_as_a19(mods, fez, n, L):
    a, pc = mods["a"], mods["pc"]
    for tg in fez:
        qc = classifier(n, L, 10 * n + L)
        tpl = a.ring_template(qc, tg)
        assert tpl is not None
        rnd = random.Random(n * L)
        for _ in range(3):
            vals = [v + rnd.uniform(-1, 1) for v in tpl.values_of(qc)]
            out = tpl.bind_values(vals)
            assert pc._isa_ok(out, tg)
            assert _tvd(_probs(out), _probs(_with_values(qc, tpl, vals))) < 1e-12
        basis = [g for g in tg.operation_names if g in ("cz", "rz", "sx", "x")]
        a19 = a.compile_for_model_circuit(qc, tg.build_coupling_map(), basis, target=tg)
        assert pc._duration_s(tpl.bind(qc), tg) <= 1.05 * pc._duration_s(a19, tg)


def test_template_rejects_another_structure(mods, fez):
    a = mods["a"]
    t, _ = fez
    tpl = a.ring_template(classifier(4, 3, 1), t)
    with pytest.raises(ValueError):
        tpl.bind(classifier(4, 4, 1))
    with pytest.raises(ValueError):
        tpl.bind_values([0.0])


def test_front_end_without_target_refuses(mods, fez):
    a = mods["a"]
    t, _ = fez
    with pytest.raises(a.OutOfScopeError):
        a.compile_for_model_circuit(classifier(4, 2, 2), t.build_coupling_map(), ["cz", "rz", "sx", "x"], target=None)
