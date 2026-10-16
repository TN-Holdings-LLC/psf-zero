"""Release checks for psf_compile 2026-10-14.1 (prototype c57, C57-VAL, Addendum 491) and the AI front end a18
(Addendum 489), as installed: psf_compile.py at the repository root and benchmarks/psf_ai_compile.py. The prototypes'
own suites (patches/psf_compile_c57_2026-10-14/, 122 tests; patches/psf_ai_compile_a18_2026-10-14/, 20 tests) test
the same code; these check that the installed files are those prototypes and work from where they are installed.

Run from the repository root:  python -m pytest benchmarks/test_release_2026_10_14_1.py -q
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
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture(scope="module")
def mods():
    import core_fix_c2_eval as H
    H.load_module(os.path.join(REPO, "benchmarks", "psf_smart_layout.py"), "psf_smart_layout")
    rel = H.load_module(os.path.join(REPO, "patches", "psf_compile_release_2026-10-14.1", "psf_compile.py"), "psf_compile")
    rel.WARN_WITHOUT_TARGET = False
    ai = H.load_module(os.path.join(REPO, "benchmarks", "psf_ai_compile_a18.py"), "psf_ai_compile_release_test")
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


def _sig(c):
    from c29_identity import value_sig
    return value_sig(c)


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _chain(n):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    for q in range(n):
        qc.ry(0.2 + 0.05 * q, q)
    for _ in range(2):
        for q in range(n - 1):
            qc.rzz(0.3 + 0.01 * q, q, q + 1)
        for q in range(n):
            qc.rx(0.25, q)
    qc.measure_all()
    return qc


def _regular3(n, seed):
    import networkx as nx
    from qiskit import QuantumCircuit
    g = nx.random_regular_graph(3, n, seed=seed)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for a, b in sorted(g.edges()):
        qc.rzz(0.7, a, b)
    for q in range(n):
        qc.rx(0.4, q)
    qc.measure_all()
    return qc


def _classifier(n, layers, seed):
    from qiskit import QuantumCircuit
    rng = random.Random(seed)
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            qc.ry(rng.uniform(-1, 1) * math.pi / 2, q)
        for q in range(n):
            qc.ry(rng.uniform(-math.pi, math.pi), q)
            qc.rz(rng.uniform(-math.pi, math.pi), q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    for q in range(n):
        qc.ry(rng.uniform(-math.pi, math.pi), q)
    qc.measure_all()
    return qc


def test_versions(mods):
    assert mods["rel"].VERSION == "2026-10-14.1"
    assert mods["ai"].AI_COMPILE_VERSION == "2026-10-14.a18"
    assert mods["rel"].SCOPE_GATE is True and mods["rel"].ZZ_RELAYER is False
    assert mods["ai"].SEARCH_FALLBACK is False and mods["ai"].HAND_OFF is True


def test_release_file_is_c57_with_version_lines_changed():
    rel = open(os.path.join(REPO, "patches", "psf_compile_release_2026-10-14.1", "psf_compile.py"), encoding="utf-8").read().splitlines()
    c57 = open(os.path.join(REPO, "patches", "psf_compile_c57_2026-10-14", "psf_compile.py"),
               encoding="utf-8").read().splitlines()
    assert len(rel) == len(c57)
    differ = [i for i, (a, b) in enumerate(zip(rel, c57)) if a != b]
    assert len(differ) == 2 and all("VERSION" in rel[i] for i in differ)


def test_in_scope_built_out_of_scope_handed_on(mods, fez):
    from qiskit.transpiler import generate_preset_pass_manager
    rel = mods["rel"]
    t, f = fez
    qc = _chain(12)
    out = _quiet(rel.compile_for_hardware, qc, target=t, seed_transpiler=0)
    qk3 = generate_preset_pass_manager(3, target=t, seed_transpiler=0).run(qc)
    assert rel._duration_s(out, t) <= rel._duration_s(qk3, t)  # in scope: never longer than Qiskit level 3 here
    assert rel._exact93(qc, out) is True
    qc = _regular3(12, 1)
    out = _quiet(rel.compile_for_hardware, qc, target=t, seed_transpiler=0)
    assert _sig(out) == _sig(generate_preset_pass_manager(3, target=t, seed_transpiler=0).run(qc))
    outf = _quiet(rel.compile_for_hardware, qc, target=f, seed_transpiler=0)
    assert rel._rzz_in_range(outf) and rel._isa_ok(outf, f) and rel._exact93(qc, outf) is True


def test_ai_front_end_from_benchmarks(mods, fez):
    ai, rel = mods["ai"], mods["rel"]
    t, _ = fez
    basis = [g for g in t.operation_names if g in ("cz", "rz", "sx", "x")]
    qc = _classifier(4, 3, 5)
    out, info = _quiet(ai.compile_for_model_circuit, qc, t.build_coupling_map(), basis, target=t, return_info=True)
    assert info["chosen"] in ("ring", "ring2") and rel._exact93(qc, out) is True
    qc = _chain(6)
    out, info = _quiet(ai.compile_for_model_circuit, qc, t.build_coupling_map(), basis, target=t, return_info=True)
    assert info["path"] == "hand-off" and info["chosen"] == "compiler"
