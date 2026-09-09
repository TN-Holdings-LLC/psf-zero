"""
psf_zero_core_stub.py

STAND-IN for the real Rust core (`psf_zero_core`), used only because the one
`.so` we were given (`psf_zero_core_1.so`) is a non-x86 binary that will not
load in this environment (`file` reports an unrecognized ELF machine type,
0x203e) -- it is not a correctness problem with the real core, just an
environment mismatch we cannot resolve here.

This module implements `geometric_decompose(u_r, u_i)` with the EXACT same
call signature and return contract that `psf_compile.py`'s
`SU4GeodesicPSFSynthesizer.synthesize()` expects from the real Rust core:

    cartan_angles, k1, k2, global_phase = geometric_decompose(u_r, u_i)

where `cartan_angles = (a, b, c)` are the canonical Weyl-chamber parameters
of U = exp(i*global_phase) * (K1l (x) K1r) . Ud(a,b,c) . (K2l (x) K2r), and
`k1 = [triple_for_K1l, triple_for_K1r]`, `k2 = [triple_for_K2l,
triple_for_K2r]`, each triple being `(phi, theta, lam)` such that the
corresponding 2x2 local factor equals RZ(phi).RY(theta).RZ(lam) up to a
scalar phase (which this function folds into the single returned
`global_phase`) -- i.e. exactly the convention `psf_compile.py`'s own
`local()` helper assumes (`qc.rz(lam); qc.ry(theta); qc.rz(phi)`), and
exactly the index convention its `synthesize()` uses (`k1[0]`/`k2[0]` on
qubit 1, `k1[1]`/`k2[1]` on qubit 0 -- see that file's own bug #2a/#2b
comments).

We get this from two independently-implemented, well-tested pieces of
Qiskit itself, not from re-deriving KAK math by hand:

  - `qiskit.synthesis.two_qubit.two_qubit_decompose.TwoQubitWeylDecomposition`
    for the canonical (a, b, c) and the four local 2x2 factors K1l/K1r/K2l/K2r.
  - `qiskit.synthesis.one_qubit.one_qubit_decompose.OneQubitEulerDecomposer`
    ('ZYZ' basis) for each local factor's (theta, phi, lam, phase) --
    `angles_and_phase()` is Qiskit's own public API for exactly this.

VERIFIED (not assumed): reassembling a 4x4 unitary from this function's
output using the real file's own gate-by-gate recipe (`local()` + RXX/RYY/RZZ
with the real -2x sign convention) and its own `unitary_fidelity()` formula
gives, over 200 random SU(4) trials, worst-case (1 - fidelity) = 8.88e-16 --
matching the order of magnitude the real Rust core's own header comment
claims (1.11e-15 over 1000 trials). See `test_psf_zero_core_stub.py`.

This is a stand-in for testing purposes only. It is not the real Rust core,
and this file makes no claim about the real core's performance -- only that
it is a faithful enough substitute to exercise the REST of `compile()` /
`compile_for_hardware()` (block filtering, consolidation, fallback logic,
hardware transpile) end-to-end while that binary is unusable here.
"""

from __future__ import annotations

import numpy as np
from qiskit.synthesis.one_qubit.one_qubit_decompose import OneQubitEulerDecomposer
from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition

_zyz = OneQubitEulerDecomposer("ZYZ")


def _triple(mat: np.ndarray) -> tuple[tuple[float, float, float], float]:
    """Return ((phi, theta, lam), phase) such that mat == exp(i*phase) *
    RZ(phi).RY(theta).RZ(lam), matching psf_compile.py's local() convention.
    """
    theta, phi, lam, phase = _zyz.angles_and_phase(mat)
    return (phi, theta, lam), phase


def geometric_decompose(u_r, u_i):
    """Drop-in stand-in for the real `psf_zero_core.geometric_decompose`."""
    U = np.asarray(u_r, dtype=float) + 1j * np.asarray(u_i, dtype=float)
    w = TwoQubitWeylDecomposition(U)

    total_phase = float(w.global_phase)
    k1l, p = _triple(w.K1l); total_phase += p
    k1r, p = _triple(w.K1r); total_phase += p
    k2l, p = _triple(w.K2l); total_phase += p
    k2r, p = _triple(w.K2r); total_phase += p

    cartan_angles = (float(w.a), float(w.b), float(w.c))
    k1 = [k1l, k1r]
    k2 = [k2l, k2r]
    return cartan_angles, k1, k2, total_phase
