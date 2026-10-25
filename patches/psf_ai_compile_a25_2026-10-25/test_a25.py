"""Tests of AI front end a25 (item 49, the zipper for odd n), from the PSF-Zero root with rust/psfbind (ring r3)
built:

    python -m pytest -q patches/psf_ai_compile_a25_2026-10-25/test_a25.py
"""
import contextlib
import importlib.util
import io
import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip("qiskit_ibm_runtime")
from qiskit import QuantumCircuit  # noqa: E402
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
        a24 = _load(os.path.join(ROOT, "patches", "psf_ai_compile_a24_2026-10-22", "psf_ai_compile.py"), "ai24_t25")
        a25 = _load(os.path.join(ROOT, "patches", "psf_ai_compile_a25_2026-10-25", "psf_ai_compile.py"), "ai25_test")
    try:
        a25._psfbind()
    except RuntimeError as exc:
        pytest.skip(f"rust/psfbind ring r3 not built: {exc}")
    return pc, a24, a25


def ring(n, L, seed):
    """A CZ ring classifier (as a24's tests): per layer ry, rz on every qubit, then the ring of CZs; measured."""
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


def sig(tpl):
    """The template's gate list as plain values (parameters by name), so that two builds compare equal."""
    c = tpl.circuit
    return [(i.operation.name, tuple(c.find_bit(q).index for q in i.qubits), tuple(str(p) for p in i.operation.params))
            for i in c.data]


def cz(tpl):
    return sum(1 for i in tpl.circuit.data if i.operation.name == "cz")


def test_even_rounds_unchanged(mods):
    _, a24, a25 = mods
    for n in range(4, 41, 2):
        assert a25.ring_rounds_zip(n) == a24.ring_rounds_zip(n)


def test_odd_rounds(mods):
    _, _, a25 = mods
    order, rounds = a25.ring_rounds_zip(9)
    assert order == [0, 1, 2, 8, 7, 3, 4, 6, 5]
    for n in range(5, 42, 2):
        order, rounds = a25.ring_rounds_zip(n)
        assert sorted(order) == list(range(n))
        assert sum(len(c) for c, _ in rounds) == n
        assert sum(len(s) for _, s in rounds) == 1 + (n - 3) // 2
    with pytest.raises(RuntimeError):
        a25.ring_rounds_zip(3)


def test_version_needs_r3(mods):
    _, _, a25 = mods
    assert a25.AI_COMPILE_VERSION.startswith("2026-10-25.a25")
    assert a25._psfbind().version.startswith("psfbind ring r3")


@pytest.mark.parametrize("n", [5, 7, 9, 11, 13])
def test_odd_zipper_checked_against_python(mods, monkeypatch, n):
    """Rust (ring r3) and the Python build the same template (PSF_RUST_TEMPLATE=check stops on any difference); the
    zipper is chosen with (5n - 7)/2 CZ per layer, fewer than the schedule without the switch."""
    _, _, a25 = mods
    t = fake_provider.FakeMarrakesh().target
    L = 4
    qc = ring(n, L, n)
    if a25._ring_embeds(qc, t):
        pytest.skip("embeds")
    monkeypatch.setenv("PSF_RUST_TEMPLATE", "check")
    monkeypatch.delenv("PSF_RING_ZIP", raising=False)
    base = a25.model_template(qc, t)
    monkeypatch.setenv("PSF_RING_ZIP", "1")
    tz = a25.model_template(qc, t)
    assert tz.schedule == "zip"
    assert cz(tz) == L * (5 * n - 7) // 2
    assert cz(tz) < cz(base)


@pytest.mark.parametrize("n", [5, 7, 9, 10])
def test_without_switch_a25_is_a24(mods, monkeypatch, n):
    _, a24, a25 = mods
    t = fake_provider.FakeMarrakesh().target
    qc = ring(n, 3, 100 + n)
    monkeypatch.delenv("PSF_RING_ZIP", raising=False)
    monkeypatch.delenv("PSF_RUST_TEMPLATE", raising=False)
    t24, t25 = a24.model_template(qc, t), a25.model_template(qc, t)
    assert t24.schedule == t25.schedule
    assert sig(t24) == sig(t25)
    assert list(t24.layout) == list(t25.layout)


def test_even_zipper_a25_is_a24(mods, monkeypatch):
    _, a24, a25 = mods
    t = fake_provider.FakeMarrakesh().target
    qc = ring(10, 4, 1)
    monkeypatch.setenv("PSF_RING_ZIP", "1")
    monkeypatch.setenv("PSF_RUST_TEMPLATE", "check")
    t24, t25 = a24.model_template(qc, t), a25.model_template(qc, t)
    assert t24.schedule == t25.schedule == "zip"
    assert sig(t24) == sig(t25)
