"""Tests for psf_compile.py VERSION 2026-09-27.1 (Addendum 207): the polish
of Addendum 186 batched over all blocks of a circuit (changelog item 20).

The batched functions must compute what the per-block ones compute: the
same reconstruction and Jacobian, the same decision on which blocks to step,
and residuals as small; compile() must give the same unitary with either
setting of USE_BATCHED_POLISH.
"""
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

import psf_compile as pc


def _random_params(rng, n):
    return np.column_stack([rng.uniform(-1.0, 1.0, (n, 3)), rng.uniform(-np.pi, np.pi, (n, 13))])


def test_reconstruct_batch_matches_per_block():
    rng = np.random.default_rng(1)
    p = _random_params(rng, 50)
    u = pc._reconstruct_batch(p)
    for i in range(50):
        assert np.max(np.abs(u[i] - pc._reconstruct(*pc._unpack(p[i])))) < 1e-14


def test_jacobian_batch_matches_per_block():
    rng = np.random.default_rng(2)
    p = _random_params(rng, 10)
    u, jac = pc._reconstruct_batch(p, with_jacobian=True)
    for i in range(10):
        u1, j1 = pc._reconstruct_with_jacobian(p[i])
        assert np.max(np.abs(u[i] - u1)) < 1e-14
        for k in range(16):
            assert np.max(np.abs(jac[i, k] - j1[k])) < 1e-13


@pytest.mark.parametrize("scale", [1e-15, 1e-12, 1e-10])
def test_refine_batch_matches_per_block(scale):
    rng = np.random.default_rng(3)
    n = 40
    p = _random_params(rng, n)
    targets = pc._reconstruct_batch(p)
    p0 = p + rng.normal(size=p.shape) * scale
    p1, before, after = pc._refine_batch(targets, p0)
    for i in range(n):
        _, b, a = pc._refine_decomposition(targets[i], *pc._unpack(p0[i]))
        assert (before[i] > pc.REFINE_THRESHOLD) == (b > pc.REFINE_THRESHOLD)
        assert after[i] <= max(2.0 * a, 1e-13)
        if before[i] <= pc.REFINE_THRESHOLD:
            assert np.array_equal(p1[i], p0[i])


def test_refine_batch_empty_step_set_is_identity():
    rng = np.random.default_rng(4)
    p = _random_params(rng, 5)
    targets = pc._reconstruct_batch(p)
    p1, before, after = pc._refine_batch(targets, p)
    assert np.array_equal(p1, p)
    assert np.array_equal(before, after)


def _dense_pairs(seed, n=6, depth=20):
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for a in range(0, n - 1, 2):
        for _ in range(depth):
            qc.rz(rng.uniform(-np.pi, np.pi), a)
            qc.ry(rng.uniform(-np.pi, np.pi), a + 1)
            qc.rxx(rng.uniform(-np.pi, np.pi), a, a + 1)
            qc.rzz(rng.uniform(-np.pi, np.pi), a, a + 1)
    return qc


@pytest.mark.parametrize("basis", ["canonical", "cx"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_compile_same_unitary_with_and_without_batching(basis, seed):
    qc = _dense_pairs(seed)
    ref = Operator(qc).data
    outs = {}
    try:
        for flag in (False, True):
            pc.USE_BATCHED_POLISH = flag
            pc._CX_CORE_CACHE.clear()
            outs[flag] = pc.compile(qc, entangling_basis=basis)
    finally:
        pc.USE_BATCHED_POLISH = True
    for flag, out in outs.items():
        u = Operator(out).data
        tr = np.trace(ref.conj().T @ u)
        assert abs(abs(tr) / ref.shape[0] - 1.0) < 1e-12, flag
    two = [sum(1 for i in o.data if len(i.qubits) == 2) for o in outs.values()]
    assert two[0] == two[1]


def test_synthesize_single_block_unchanged():
    rng = np.random.default_rng(5)
    z = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    q, r = np.linalg.qr(z)
    u = q * (np.diag(r) / np.abs(np.diag(r)))
    synth = pc.SU4GeodesicPSFSynthesizer(pc.GeodesicPSFHyper(entangling_basis="cx"))
    qc = synth.synthesize(u)
    tr = np.trace(u.conj().T @ Operator(qc).data)
    assert abs(abs(tr) / 4.0 - 1.0) < 1e-12
