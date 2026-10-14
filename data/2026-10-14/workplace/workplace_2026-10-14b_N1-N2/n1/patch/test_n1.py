"""Tests of item 86 (workplace proposal N1 on c49): chain Hamiltonians built as a brickwork.
    python -m pytest -q test_n1.py      (from this folder; the Rust core on PYTHONPATH as for c49's tests)"""
import importlib.util
import os
import sys
import warnings

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
warnings.simplefilter("ignore")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


_load(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
P = _load(os.path.join(HERE, "psf_compile.py"), "psf_compile_c49n1")
P.WARN_WITHOUT_TARGET = False


def _lab(n, d):
    s = ["I"] * n
    for q, p in d.items():
        s[n - 1 - q] = p
    return "".join(s)


def tfim(n, steps, ring=False):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import SuzukiTrotter
    edges = [(q, q + 1) for q in range(n - 1)] + ([(n - 1, 0)] if ring else [])
    terms = [(_lab(n, {a: "Z", b: "Z"}), 1.0) for a, b in edges] + [(_lab(n, {q: "X"}), 1.0) for q in range(n)]
    qc = QuantumCircuit(n)
    qc.append(PauliEvolutionGate(SparsePauliOp.from_list(terms), time=0.6, synthesis=SuzukiTrotter(order=2, reps=steps)),
              range(n))
    qc.measure_all()
    return qc


def xxz(n, steps):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import SuzukiTrotter
    terms = []
    for q in range(n - 1):
        terms += [(_lab(n, {q: "X", q + 1: "X"}), 1.0), (_lab(n, {q: "Y", q + 1: "Y"}), 1.0),
                  (_lab(n, {q: "Z", q + 1: "Z"}), 0.7)]
    terms += [(_lab(n, {q: "Z"}), 0.3 * (q % 3)) for q in range(n)] + [(_lab(n, {q: "X"}), 0.2) for q in range(n)]
    qc = QuantumCircuit(n)
    for q in range(0, n, 2):
        qc.x(q)
    qc.append(PauliEvolutionGate(SparsePauliOp.from_list(terms), time=0.5, synthesis=SuzukiTrotter(order=2, reps=steps)),
              range(n))
    qc.measure_all()
    return qc


def brickwork(n, steps):
    from qiskit import QuantumCircuit
    dt = 0.6 / steps
    qc = QuantumCircuit(n)

    def zz(t):
        for s in (0, 1):
            for q in range(s, n - 1, 2):
                qc.rzz(t, q, q + 1)
    zz(dt)
    for k in range(steps):
        for q in range(n):
            qc.rx(2 * dt, q)
        zz(2 * dt if k < steps - 1 else dt)
    qc.measure_all()
    return qc


def probs(circ):
    """exact outcome distribution over the classical bits (Aer statevector on the active qubits)"""
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    meas, body = {}, []
    for ins in circ.data:
        if ins.operation.name == "measure":
            meas[circ.find_bit(ins.clbits[0]).index] = circ.find_bit(ins.qubits[0]).index
        elif ins.operation.name not in ("barrier", "delay"):
            body.append((ins.operation, [circ.find_bit(q).index for q in ins.qubits]))
    act = sorted({q for _, qs in body for q in qs} | set(meas.values()))
    pos = {q: i for i, q in enumerate(act)}
    c = QuantumCircuit(len(act))
    for op, qs in body:
        c.append(op, [pos[q] for q in qs])
    c = transpile(c, basis_gates=["cx", "rz", "sx", "x", "h", "rzz", "rx", "ry", "u", "rxx", "ryy"], optimization_level=0)
    c.save_probabilities([pos[meas[k]] for k in range(circ.num_clbits)])
    return np.asarray(AerSimulator(method="statevector").run(c).result().data()["probabilities"])


def dur_cz(out, target):
    pos = {q: i for i, q in enumerate(out.qubits)}
    clock, cz = {}, 0
    for ins in out.data:
        nm = ins.operation.name
        if nm in ("barrier", "measure", "delay"):
            continue
        qs = tuple(pos[q] for q in ins.qubits)
        pr = target[nm].get(qs)
        d = (pr.duration or 0.0) if pr is not None else 0.0
        s = max(clock.get(q, 0.0) for q in qs)
        for q in qs:
            clock[q] = s + d
        cz += len(qs) == 2
    return max(clock.values()), cz


@pytest.fixture(scope="module")
def fez():
    from qiskit_ibm_runtime.fake_provider import FakeFez
    return FakeFez()


def _used(before):
    return P.CHAIN_STATS["used"] - before


@pytest.mark.parametrize("n,steps", [(8, 2), (12, 10)])
def test_tfim_exact_and_not_longer_than_brickwork(fez, n, steps):
    from qiskit.transpiler import generate_preset_pass_manager
    b = P.CHAIN_STATS["used"]
    out = P.compile_for_hardware(tfim(n, steps), backend=fez, seed_transpiler=0)
    assert _used(b) == 1
    assert 0.5 * np.abs(probs(out) - probs(tfim(n, steps).decompose(reps=3))).sum() < 1e-9
    bw = generate_preset_pass_manager(3, fez, seed_transpiler=0).run(brickwork(n, steps))
    d, cz = dur_cz(out, fez.target)
    d0, cz0 = dur_cz(bw, fez.target)
    assert cz <= cz0 and d <= d0 + 1e-12


def test_xxz_chain_exact(fez):
    qc = xxz(8, 3)
    b = P.CHAIN_STATS["used"]
    out = P.compile_for_hardware(qc, backend=fez, seed_transpiler=0)
    assert _used(b) == 1
    assert 0.5 * np.abs(probs(out) - probs(qc.decompose(reps=3))).sum() < 1e-9


def test_n100_ten_steps_no_size_limit(fez):
    b = P.CHAIN_STATS["used"]
    out = P.compile_for_hardware(tfim(100, 10), backend=fez, seed_transpiler=0)
    assert _used(b) == 1
    d, cz = dur_cz(out, fez.target)
    assert cz <= 2178 and d <= 5.74e-6 + 1e-12          # W2's QK3-BW on FakeFez


def test_ring_and_qaoa_not_applied_and_unchanged(fez):
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp
    ring = tfim(8, 2, ring=True)
    b = P.CHAIN_STATS["not_chain"]
    out1 = P.compile_for_hardware(ring, backend=fez, seed_transpiler=0)
    assert P.CHAIN_STATS["not_chain"] == b + 1
    P.CHAIN_BRICKWORK = False
    try:
        out0 = P.compile_for_hardware(ring, backend=fez, seed_transpiler=0)
    finally:
        P.CHAIN_BRICKWORK = True
    assert [(i.operation.name, [out1.find_bit(q).index for q in i.qubits]) for i in out1.data] == \
        [(i.operation.name, [out0.find_bit(q).index for q in i.qubits]) for i in out0.data]


def test_deterministic(fez):
    a = P.compile_for_hardware(tfim(16, 4), backend=fez, seed_transpiler=0)
    b = P.compile_for_hardware(tfim(16, 4), backend=fez, seed_transpiler=0)
    sig = lambda c: [(i.operation.name, [c.find_bit(q).index for q in i.qubits],  # noqa: E731
                      [round(float(p), 12) for p in i.operation.params]) for i in c.data]
    assert sig(a) == sig(b)


def test_layout_reads_logical_qubits(fez):
    out = P.compile_for_hardware(tfim(10, 2), backend=fez, seed_transpiler=0)
    lay = out.layout.final_index_layout()
    meas = {out.find_bit(i.clbits[0]).index: out.find_bit(i.qubits[0]).index for i in out.data if i.operation.name == "measure"}
    assert [lay[q] for q in range(10)] == [meas[q] for q in range(10)]


def test_not_cz_device_not_applied():
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    b = P.CHAIN_STATS["not_cz"]
    P.compile_for_hardware(tfim(6, 2), backend=FakeSherbrooke(), seed_transpiler=0)
    assert P.CHAIN_STATS["not_cz"] > b      # (also counted in the call's own internal recompiles)


def test_never_longer_than_c49_own_output():
    """On a generic backend (gate lengths differ per qubit) the brickwork can be longer than c49's own output; the
    comparison then returns c49's (item 84's rule)."""
    sys.path.insert(0, HERE)
    import test_c48_on_c49 as T
    import test_c49 as T9
    b = T9._small()
    qc = T._qaoa_gates(5, 21)
    k0 = P.CHAIN_STATS["kept_c49"]
    out = P.compile_for_hardware(qc, backend=b, seed_transpiler=0)
    P.CHAIN_BRICKWORK = False
    try:
        own = P.compile_for_hardware(qc, backend=b, seed_transpiler=0)
    finally:
        P.CHAIN_BRICKWORK = True
    assert P._duration_s(out, b.target) <= P._duration_s(own, b.target) + 1e-12
    assert P.CHAIN_STATS["kept_c49"] == k0 + 1
