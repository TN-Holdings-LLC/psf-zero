"""Release checks for psf_compile 2026-10-17.1 (prototype c67: item 104, devices whose two-qubit gate is ECR, by a CZ
view; C67-VAL, Addendum 516) and the AI front end a23 (Addendum 503), as installed: psf_compile.py at the repository root
and benchmarks/psf_ai_compile.py. The prototype's validation (data/2026-10-17/c67_val/) and the release's bit-for-bit
check against it (patches/psf_compile_c67_2026-10-17/check_release.py) test the same code; these check that the
installed file is the prototype with its docstring and version line changed (make_release.py), that it works from
where it is installed, the release's scope, item 103's promise on a fractional-gate target, and item 104 on an ECR
device: exact and native, templates refused.

Run from the repository root:  python -m pytest benchmarks/test_release_2026_10_17_1.py -q
"""
import contextlib
import copy
import io
import os
import re
import sys
import warnings

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

C67_DIR = os.path.join(REPO, "patches", "psf_compile_c67_2026-10-17")
A23 = os.path.join(REPO, "patches", "psf_ai_compile_a23_2026-10-16", "psf_ai_compile.py")


def _load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mods():
    with contextlib.redirect_stdout(io.StringIO()):
        rel = _load(os.path.join(REPO, "psf_compile.py"), "psf_compile")
        rel.WARN_WITHOUT_TARGET = False
        ai = _load(os.path.join(HERE, "psf_ai_compile.py"), "psf_ai_compile_release_test")
    return dict(rel=rel, ai=ai)


@pytest.fixture(scope="module")
def fez():
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import RXGate, RZZGate
    from qiskit.transpiler import InstructionProperties
    from qiskit_ibm_runtime.fake_provider import FakeFez
    t = FakeFez().target
    f = copy.deepcopy(t)
    f.add_instruction(RZZGate(Parameter("t")), {qa: InstructionProperties(duration=p.duration, error=p.error)
                                                 for qa, p in t["cz"].items()})
    f.add_instruction(RXGate(Parameter("t")), {qa: InstructionProperties(duration=p.duration, error=p.error)
                                               for qa, p in t["sx"].items()})
    return t, f


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _code(path, version_pattern):
    """The file without its module docstring and without its version line."""
    s = open(path, encoding="utf-8").read()
    s = s[s.index('"""', 3) + 3:]
    return re.sub(version_pattern, "", s, flags=re.M)


def _dist(circ):
    """Exact outcome distribution over the classical bits (final measurements only, active qubits only)."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    meas, body = {}, []
    for ins in circ.data:
        qs = [circ.find_bit(q).index for q in ins.qubits]
        if ins.operation.name == "measure":
            meas[circ.find_bit(ins.clbits[0]).index] = qs[0]
        elif ins.operation.name not in ("barrier", "delay"):
            body.append((ins.operation, qs))
    active = sorted({q for _, qs in body for q in qs} | set(meas.values()))
    pos = {q: i for i, q in enumerate(active)}
    small = QuantumCircuit(len(active))
    for op, qs in body:
        small.append(op, [pos[q] for q in qs])
    probs = Statevector(small).probabilities()
    out = {}
    for idx in np.nonzero(probs > 1e-15)[0]:
        key = tuple((int(idx) >> pos[meas[c]]) & 1 for c in sorted(meas))
        out[key] = out.get(key, 0.0) + float(probs[idx])
    return out


def _tvd(a, b):
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def _native(c, t):
    pos = {q: i for i, q in enumerate(c.qubits)}
    for ins in c.data:
        nm = ins.operation.name
        if nm in ("barrier", "measure", "delay"):
            continue
        qs = tuple(pos[q] for q in ins.qubits)
        if nm not in t.operation_names or (qs not in t[nm] and None not in t[nm]):
            return False
    return True


def _ghz(n):
    from qiskit import QuantumCircuit
    c = QuantumCircuit(n, n)
    c.h(0)
    for i in range(n - 1):
        c.cx(i, i + 1)
    c.measure(range(n), range(n))
    return c


def _ring(n, layers, seed):
    """A ring classifier as TOQB's model_circuit writes it: per layer an RY encoding, then RY RZ on every qubit and a
    CZ ring; a last RY layer; every qubit measured."""
    from qiskit import QuantumCircuit
    rng = np.random.default_rng(seed)
    c = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            c.ry(float(rng.uniform(-1.5, 1.5)), q)
        for q in range(n):
            c.ry(float(rng.uniform(-3, 3)), q)
            c.rz(float(rng.uniform(-3, 3)), q)
        for q in range(n):
            c.cz(q, (q + 1) % n)
    for q in range(n):
        c.ry(float(rng.uniform(-3, 3)), q)
    c.measure_all()
    return c


def test_versions(mods):
    assert mods["rel"].VERSION == "2026-10-17.1"  # current release
    assert mods["ai"].AI_COMPILE_VERSION == "2026-10-16.a23"


def test_installed_files_are_the_prototypes(tmp_path):
    import subprocess
    out = tmp_path / "psf_compile.py"
    subprocess.run([sys.executable, os.path.join(C67_DIR, "make_release.py"), os.path.join(C67_DIR, "psf_compile.py"),
                    str(out)], check=True, capture_output=True)
    assert open(os.path.join(REPO, "psf_compile.py"), encoding="utf-8").read() == out.read_text(encoding="utf-8")
    assert _code(os.path.join(HERE, "psf_ai_compile.py"), r'^AI_COMPILE_VERSION = .*$') == \
        _code(A23, r'^AI_COMPILE_VERSION = .*$')


@pytest.mark.parametrize("which", ["T", "F"])
def test_ghz_exact_and_native(mods, fez, which):
    t = fez[0] if which == "T" else fez[1]
    qc = _ghz(6)
    out = _quiet(mods["rel"].compile_for_hardware, qc, target=t)
    assert _native(out, t)
    assert _tvd(_dist(qc), _dist(out)) < 1e-9


def test_out_of_scope_is_refused(mods, fez):
    from qiskit.circuit.random import random_circuit
    rel = mods["rel"]
    with pytest.raises(rel.OutOfScopeError):
        _quiet(rel.compile_for_hardware, random_circuit(8, 6, max_operands=2, measure=True, seed=3), target=fez[0])
    with pytest.raises(rel.OutOfScopeError):
        _quiet(rel.compile_for_hardware, _ghz(4))  # no Target


@pytest.mark.parametrize("n", [6, 12])
def test_ai_ring_recompile_and_template(mods, fez, n):
    t = fez[0]
    ai = mods["ai"]
    qc = _ring(n, 2, seed=n)
    basis = [g for g in ("cz", "rz", "sx", "x") if g in t.operation_names]
    out = _quiet(ai.compile_for_model_circuit, qc, t.build_coupling_map(), basis, target=t)
    assert _native(out, t)
    assert _tvd(_dist(qc), _dist(out)) < 1e-9
    tp = _quiet(ai.model_template, qc, t)
    assert tp is not None
    b = _quiet(tp.bind, qc)
    assert _native(b, t)
    assert _tvd(_dist(qc), _dist(b)) < 1e-9


def _chain_tfim(n, signs, scale=1.0):
    """A TFIM evolution on a chain (C55-VAL's form): ZZ couplings with the given signs, X fields, measured."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import SuzukiTrotter

    def lab(d):
        s = ["I"] * n
        for q, p in d.items():
            s[n - 1 - q] = p
        return "".join(s)
    terms = [(lab({q: "Z", q + 1: "Z"}), scale * (0.4 + 0.1 * q) * signs[q]) for q in range(n - 1)]
    terms += [(lab({q: "X"}), 0.7 * scale) for q in range(n)]
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(0.3 + 0.07 * q, q)
    qc.append(PauliEvolutionGate(SparsePauliOp.from_list(terms), time=1.0, synthesis=SuzukiTrotter(order=2, reps=3)),
              range(n))
    qc.measure_all()
    return qc


def _shape(c):
    return [(ins.operation.name, tuple(c.find_bit(q).index for q in ins.qubits)) for ins in c.data]


def test_item103_fractional_template_keeps_its_shape(mods, fez):
    """Item 103 on a fractional-gate target: one template, bound with same-sign and mixed-sign couplings, is exact and
    native every time, never rebuilt or switched, and every output has the build's shape."""
    rel = mods["rel"]
    t = fez[1]
    n = 8
    base = _chain_tfim(n, [1] * (n - 1))
    tp = _quiet(rel.compile_template, base, t)
    shape0 = _shape(tp.bind(base))
    before = (rel.TEMPLATE100_STATS["rebuilt"], rel.TEMPLATE100_STATS["switched"])
    rng = np.random.default_rng(103)
    inputs = [_chain_tfim(n, [1] * (n - 1), 1.3), _chain_tfim(n, [-1] * (n - 1))]
    inputs += [_chain_tfim(n, list(rng.choice([-1, 1], n - 1)), float(rng.uniform(0.5, 1.5))) for _ in range(4)]
    for qc in inputs:
        out = _quiet(tp.bind, qc)
        assert _native(out, t)
        assert _shape(out) == shape0
        assert _tvd(_dist(qc.decompose()), _dist(out)) < 1e-9  # the Trotter circuit, not the exact evolution
    assert (rel.TEMPLATE100_STATS["rebuilt"], rel.TEMPLATE100_STATS["switched"]) == before


@pytest.fixture(scope="module")
def eagle():
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    return FakeSherbrooke().target


@pytest.mark.parametrize("name", ["ghz", "chain_tfim", "ring"])
def test_item104_ecr_exact_and_native(mods, eagle, name):
    """Item 104: on an ECR device (FakeSherbrooke) the default call compiles, natively (ecr, rz, sx, x) and exactly."""
    qc = {"ghz": _ghz(8), "chain_tfim": _chain_tfim(8, [1, -1, 1, 1, -1, 1, 1]), "ring": _ring(12, 2, 3)}[name]
    out = _quiet(mods["rel"].compile_for_hardware, qc, target=eagle)
    assert _native(out, eagle)
    assert "ecr" in out.count_ops() and "cz" not in out.count_ops()
    ref = qc.decompose() if name == "chain_tfim" else qc
    assert _tvd(_dist(ref), _dist(out)) < 1e-9


def test_item104_ecr_template_is_refused(mods, eagle):
    rel = mods["rel"]
    with pytest.raises(rel.OutOfScopeError):
        _quiet(rel.compile_template, _ghz(4), eagle)
