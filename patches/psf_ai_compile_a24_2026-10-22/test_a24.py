"""Tests of AI front end a24 (items 46-48), from the PSF-Zero root with rust/psfbind built:

    python -m pytest -q patches/psf_ai_compile_a24_2026-10-22/test_a24.py
"""
import contextlib
import copy
import importlib.util
import io
import os
import random
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip("qiskit_ibm_runtime")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.transpiler import InstructionProperties  # noqa: E402
from qiskit_ibm_runtime import fake_provider  # noqa: E402

warnings.simplefilter("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def mods():
    with contextlib.redirect_stdout(io.StringIO()):
        pc = _load(os.path.join(ROOT, "psf_compile.py"), "psf_compile")
        pc.WARN_WITHOUT_TARGET = False
        a24 = _load(os.path.join(ROOT, "patches", "psf_ai_compile_a24_2026-10-22", "psf_ai_compile.py"), "ai24_test")
    try:
        a24._psfbind()
    except RuntimeError as exc:
        pytest.skip(f"rust/psfbind not built: {exc}")
    return pc, a24


def ring(n, L, seed):
    """A CZ ring classifier (as a22's tests): per layer ry, rz on every qubit, then the ring of CZs; measured."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for _ in range(L):
        for q in range(n):
            qc.ry(float(rng.uniform(-3, 3)), q)
            qc.rz(float(rng.uniform(-3, 3)), q)
        for q in range(n):
            qc.cz(q, (q + 1) % n)
    for q in range(n):
        qc.ry(float(rng.uniform(-3, 3)), q)
    qc.measure_all()
    return qc


def test_zipper_order_and_cover(mods):
    _, a24 = mods
    order, rounds = a24.ring_rounds_zip(10)
    assert order == [0, 1, 2, 9, 8, 3, 4, 7, 6, 5]
    assert sum(len(c) for c, _ in rounds) == 10
    with pytest.raises(RuntimeError):
        a24.ring_rounds_zip(7)


def test_rust_template_by_default_and_checked(mods, monkeypatch):
    pc, a24 = mods
    t = fake_provider.FakeMarrakesh().target
    qc = ring(10, 4, 1)
    if a24._ring_embeds(qc, t):
        pytest.skip("embeds")
    monkeypatch.setenv("PSF_RUST_TEMPLATE", "check")
    monkeypatch.delenv("PSF_RING_ZIP", raising=False)
    tp = a24.model_template(qc, t)
    assert tp.schedule == "q4"
    monkeypatch.setenv("PSF_RING_ZIP", "1")
    tz = a24.model_template(qc, t)
    assert tz.schedule == "zip"
    cz = lambda tpl: sum(1 for i in tpl.circuit.data if i.operation.name == "cz")  # noqa: E731
    assert cz(tz) < cz(tp)


def test_missing_library_stops(mods, monkeypatch):
    _, a24 = mods
    saved = list(a24._PSFBIND)
    a24._PSFBIND.clear()
    monkeypatch.setenv("PSF_BIND_LIB", os.path.join(ROOT, "no_such_libpsfbind.so"))
    try:
        with pytest.raises(RuntimeError, match="item 46"):
            a24.model_template(ring(10, 2, 2), fake_provider.FakeMarrakesh().target)
    finally:
        a24._PSFBIND[:] = saved


def _cut(t, k, seed):
    t = copy.deepcopy(t)
    r = random.Random(seed)
    cz = sorted(qa for qa in t["cz"] if qa is not None)
    for qa in r.sample(cz, k):
        p = t["cz"][qa]
        t.update_instruction_properties("cz", qa, InstructionProperties(duration=p.duration, error=0.7))
    return t


def _find(pc, base, n, want):
    for seed in range(300):
        t = _cut(base, 14, seed)
        lat = pc._chain_lattice(t, 0.5)
        if lat is None:
            continue
        K, kadj, fadj, snake = lat
        hf, hk = pc._chain_place(n, K, fadj, snake)[1], pc._chain_place(n, K, kadj, snake)[1]
        if want == "second" and hf != "snake" and hk == "snake":
            return t, pc._chain_place(n, K, kadj, snake)[0]
        if want == "none" and hf == "vf2_path" and hk != "snake":
            return t, None
    pytest.skip(f"no seeded device for {want}")


def test_item47_refuses_where_only_vf2_places(mods, monkeypatch):
    pc, a24 = mods
    monkeypatch.setenv("PSF_RUST_TEMPLATE", "check")
    t, _ = _find(pc, fake_provider.FakeTorino().target, 14, "none")
    qc = ring(14, 4, 3)
    with pytest.raises(a24.OutOfScopeError, match="item 47"):
        a24.model_template(qc, t)
    basis = [g for g in ("cz", "rz", "sx", "x") if g in t.operation_names]
    with pytest.raises(a24.OutOfScopeError):
        with contextlib.redirect_stdout(io.StringIO()):
            a24.compile_for_model_circuit(qc, t.build_coupling_map(), basis, target=t)


def test_item47_second_snake(mods, monkeypatch):
    pc, a24 = mods
    monkeypatch.setenv("PSF_RUST_TEMPLATE", "check")
    t, line = _find(pc, fake_provider.FakeTorino().target, 14, "second")
    tp = a24.model_template(ring(14, 4, 4), t)
    assert sorted(tp.layout) == sorted(line)
