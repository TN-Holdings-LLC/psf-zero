"""Release checks for psf_compile 2026-10-16.1 (prototype c65; C63-VAL and C63A-VAL, Addenda 505-508) and the AI
front end a23 (Addendum 503), as installed: psf_compile.py at the repository root and benchmarks/psf_ai_compile.py.
The prototypes' own checks (patches/psf_compile_c65_2026-10-16/check_c65.py, patches/psf_ai_compile_a23_2026-10-16/
check_same_a22.py) test the same code; these check that the installed files are those prototypes, that they work from
where they are installed, and the release's scope: what does not embed in the device is refused.

Run from the repository root:  python -m pytest benchmarks/test_release_2026_10_16_1.py -q
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

C65 = os.path.join(REPO, "patches", "psf_compile_c65_2026-10-16", "psf_compile.py")
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
    assert mods["rel"].VERSION == "2026-10-16.1"  # current release
    assert mods["ai"].AI_COMPILE_VERSION == "2026-10-16.a23"


def test_installed_files_are_the_prototypes():
    assert _code(os.path.join(REPO, "psf_compile.py"), r'^VERSION = .*$') == _code(C65, r'^VERSION = .*$')
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
