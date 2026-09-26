"""PSF-Zero -- the compiler. **This file is the latest version of it.**

VERSION: 2026-09-26.4 (previous revision: 2026-09-26.3)

Where to look for what
----------------------
This module is the compiler. It defines exactly two entry points --
`compile()` and `compile_for_hardware()` -- and a file that does not define
those two is not this module, whatever it happens to be named. The
layout-search prototype (`smart_vf2_layout()`) is a separate thing that lives
in `psf_smart_layout.py`. The contents of those two files have been swapped by
accident more than once, so the `VERSION` line above and this paragraph are
here to answer "is this the current compiler?" the moment the file is opened.

Versioning is by content, not by filename: when this file changes, bump
`VERSION` (and `__version__`, which mirrors it). Nothing has to be renamed to
mark a newer copy, and no second file has to exist alongside this one.

Changes in the 2026-09-15 revision
----------------------------------
1. FIXED: the Rust core import. The previous revision of this file imported
   `psf_zero_core_stub` -- the Qiskit-based Python stand-in written for a
   sandbox that could not load the real `.so` -- instead of `psf_zero_core`.
   With that import in place nothing in the Rust core ran at all: the
   degeneracy handling was never exercised, and every "PSF-Zero vs Qiskit"
   measurement was really Qiskit's own TwoQubitWeylDecomposition being
   compared against Qiskit, while the debug line still announced "PSF-Zero
   Rust Core executed for N blocks". The import is now the real core, and a
   missing core raises immediately with an explanation rather than silently
   substituting something else.

2. Verification is no longer the dominant cost, so it no longer has to be
   switched off to be competitive. The old `verify=True` path rebuilt the
   synthesized circuit with `Operator(qc)` and compared -- measured at
   ~1.35 ms per block against ~0.22 ms for everything else put together,
   i.e. ~87% of the total, which is the entire reason this project's
   measured speed advantage was only available with `verify=False`. The
   core now returns the reconstruction infidelity itself
   (`geometric_decompose_checked`), computed from a handful of 4x4 products
   on values it already has. Measured on 400 random SU(4) blocks:

       verify via Operator(qc)                 1.352 ms/block
       verify via numpy reconstruction         0.111 ms/block
       verify via the core's own check        ~0.000 ms/block (in the FFI call)

   `verify=True` therefore stays the default and is now essentially free.
   `verify="strict"` keeps the old `Operator(qc)` behavior for anyone who
   wants the circuit object itself checked rather than the decomposition
   (see `_verify_block` for exactly what each one does and does not cover).

3. `logging` instead of `print`. A library writing to stdout on every call
   forced this project's own benchmark scripts to wrap it in
   `contextlib.redirect_stdout`, and would do the same to anything else that
   embeds it. The per-block debug line is now a single `logger.debug`.

4. Fallbacks are counted and reported once at the end instead of raising a
   `warnings.warn` per block, and they are now classified: a legitimately
   degenerate input and an unexpected core failure are different events, and
   the core's new exception types let them be told apart.

5. `on_unsupported` is exposed on `compile()` (it was hardcoded to "keep"),
   and `compile_for_hardware()` accepts `seed_transpiler` so its internal
   routing call can be pinned -- the asymmetry this project's own benchmarks
   ran into, where an unpinned `optimization_level>=2` transpile returns a
   different answer and a different runtime on every call.

6. The entangling core is appended directly rather than built as a separate
   QuantumCircuit and composed, and the CX-basis form of a given canonical
   triple is cached -- Trotter and QAOA layers repeat the same (a, b, c)
   across many blocks, and each miss otherwise costs an `Operator()` build
   plus a full Qiskit KAK.

Changes in this (2026-09-16) revision
-------------------------------------
None of the following changes any published number. Items 7 and 10 change
behaviour only in cases that previously either crashed or were misreported;
item 9 swaps one computation of a quantity for a numerically equivalent,
cheaper one. **None of them has been benchmarked** -- the claims below about
cost are structural (fewer calls, fewer allocations), not measured, and should
be treated as such until someone times them.

7. The output circuit is built with `qc_blocked.copy_empty_like()` instead of
   `QuantumCircuit(qc.num_qubits, qc.num_clbits)`. The old construction made a
   circuit with fresh registers, which meant the source circuit's registers,
   loose bits, name and metadata were all dropped, and the per-instruction
   `find_bit` lookups in the main loop existed only to work around that. Two
   consequences, both silent:
     - a circuit carrying a classical condition on one of its own registers
       could not be rebuilt, because the target register no longer existed;
     - any named or split register came back as a single anonymous "q".
   `copy_empty_like()` keeps the registers, the bits, the name, the metadata
   and the global phase, so instructions can be re-appended with the original
   bit objects and the two `find_bit` calls per instruction disappear from the
   hot loop.

8. `verify`, `entangling_basis` and `on_unsupported` are validated at entry
   instead of being interpreted loosely. Previously `verify="Strict"` (capital
   S) or `verify=1` silently selected the cheap check rather than the strict
   one, and `entangling_basis="CX"` silently emitted the canonical basis --
   three ways to think you had asked for something you had not. They now raise
   `ValueError` naming the accepted values. This is the same reasoning as this
   project's "no silent fallback" rule, applied to its own arguments.

9. The core's own infidelity is now used for `entangling_basis="cx"` as well.
   Both `verify=True` paths check the same quantity -- the decomposition
   (`cartan`, `k1`, `k2`, phase) against the target unitary -- and the core
   already computed it during the FFI call, so recomputing the identical
   number in numpy on the cx path (~0.111 ms/block) bought nothing. What the
   cx path does *not* check is the CX substitution itself, and that was
   already true before this change: the substitution is Qiskit's own exact
   decomposer applied to an exact matrix, and `verify="strict"` remains the
   only mode that validates the emitted circuit object.

10. The `try` around the core call no longer also covers circuit
    construction. A failure inside `_build_circuit` -- i.e. a bug in this
    file, not in the input -- was being reported as "Decomposition failed or
    degenerate" and, when the core's typed exceptions are unavailable,
    counted as an *expected* degeneracy. The two are now separate blocks with
    separate messages, and a construction failure is always counted as
    unexpected.

11. Housekeeping: an unused eigenvalue array is no longer bound, `__all__`
    names the public surface, and `compile()` has the docstring it was
    missing.

Additional change, same day (2026-09-16), item 12
--------------------------------------------------
12. **NEW, opt-in: `layout_search` on `compile_for_hardware()`.** Addenda
    24-25 (spare-qubit-cliff series) measured that at the spare-qubit cliff,
    `compile_for_hardware()`'s advantage over plain Qiskit L3 comes almost
    entirely from running its internal routing call at a lower
    `routing_optimization_level`, not from avoiding the cliff mechanism
    itself (`VF2Layout` failing, falling back to `SabreLayout`) -- raising
    the level toward 3 made the cliff, and PSF-Zero's own advantage, both
    disappear together. This item addresses the cliff at its actual cause
    instead: when `layout_search=True`, `compile_for_hardware()` now runs
    this project's own `smart_vf2_layout()` (previously a `prototypes/`-only
    tool a caller had to invoke and wire in by hand, per Addendum 17) against
    the *compressed* circuit's own interaction graph before calling
    `transpile()`, and threads a found layout through `initial_layout`
    itself. On a topology/interaction-graph combination the search can
    solve near-instantly (grids and lines with a matching-type interaction
    pattern, per Addendum 13), this is intended to let `routing_optimization_level`
    stay low without inheriting a slow `VF2Layout` failure to get there.

    Deliberately conservative about failure modes, per this project's
    "no silent fallback" rule:
      - `layout_search=True` together with an explicit `initial_layout` is a
        contradiction (which one should win?) and raises `ValueError` rather
        than silently picking one.
      - `psf_smart_layout` not being importable raises `ImportError` with an
        explicit message, only when `layout_search=True` is actually
        requested -- normal `compile_for_hardware()` calls that never ask for
        this are completely unaffected, and nothing about this failure is
        swallowed into "fell back to default behaviour" the way a bare
        `except Exception` would.
      - If the search does not find a layout within its budget (this can and
        does happen -- see `psf_smart_layout.py`'s own docstring on `brick`
        and other hard topologies), `compile_for_hardware()` falls through to
        Qiskit's own default layout stage, exactly as if `layout_search` had
        been left `False` -- but the time already spent searching is real and
        is not hidden from the caller's own wall-clock measurement of this
        call, matching this project's Addendum 14 principle that a failed
        search's cost must be counted, not absorbed silently.

    **Not yet benchmarked in this file's own revision history** -- this
    changelog entry describes what the code now does, not a measured result.
    The pre-registered prediction and the benchmark built to test it live
    separately, in this project's spare-qubit-cliff addenda (see Addendum 26
    once it exists), not in this docstring, per this project's own
    convention of keeping code changelog entries and measured findings in
    separate documents.

Additional change, same day (2026-09-16), item 13
--------------------------------------------------
13. **NEW, opt-in: `callback` on `compile_for_hardware()`.** Addendum 27
    found that PSF-Zero's own default (`layout_search=False`) path still has
    rare, large, unexplained slowdowns exactly at the spare-qubit cliff's
    peak (spare=0) -- 3 of 30 runs across 5 rounds, up to 8x the median, not
    tied to one seed. This project's own established technique for seeing
    *where inside a transpile call* time actually goes is Qiskit's own
    `transpile(callback=...)` (already used, from outside this file, in
    Addendum 10 -- see the `initial_layout` docstring above for that
    finding). Until now there was no way to use it from *inside* a
    `compile_for_hardware()` call, since this function builds its own
    internal `transpile()` call and gave the caller no hook into it.

    `callback`, when given, is forwarded verbatim to that internal
    `transpile()` call -- one parameter, one line of forwarding, the same
    minimal-and-additive shape as `initial_layout` (Addendum 17) and
    `layout_search` (item 12 above). Default `None`, so every existing call
    is unaffected. It is Qiskit's own `callback(pass_, dag, time, property_set,
    count, running_time=...)`-style callable (see Qiskit's `transpile()`
    documentation for the exact signature Qiskit invokes it with); this file
    does nothing with it beyond passing it through, so anyone already
    familiar with the plain-`transpile()` version needs to learn nothing new
    to use it here.

    **Not yet used to explain anything** -- this changelog entry describes
    the hook, not a finding. The investigation it exists to support (the
    Addendum 27 spare=0 outlier hunt) lives separately, per this project's
    changelog/findings separation convention.

Changes in the 2026-09-21 revision
----------------------------------
14. The CX-basis decomposer is configured for the {rz, sx} basis:
    `TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")` instead of
    `TwoQubitBasisDecomposer(CXGate())`. Constructed without an Euler basis,
    it emitted a general rotation between each pair of its three CXs, which
    became two `sx` pulses per qubit per gap after basis translation; Qiskit's
    own transpilation places at most one. On a repeated-pair variational
    ansatz this left PSF-Zero with 1.6x Qiskit's `sx` count at the same CX
    count (spare-qubit-cliff Addendum 114). With this setting, `sx` count, sx
    depth and total depth equal Qiskit's own re-compile exactly (128 -> 80
    `sx`, depth 23 -> 16 at 16 qubits; 336 -> 210, 23 -> 16 at 42); CX count
    unchanged; 300/300 random blocks still exact (Addendum 116).
    `pulse_optimize=True` changed nothing measurable and is left out.

    Scope and caveats: the benefit is for devices whose single-qubit basis is
    {rz, sx}; other bases are still translated by `transpile()`, untested
    here. This decomposer also serves the degenerate-block fallback, which
    Addendum 116 did not exercise. What changes and what does not:
    `compile()` and `compile_for_hardware()` default to
    `entangling_basis="canonical"`, which uses this decomposer only on that
    fallback, so default-basis output -- including the 10,000-iteration
    gate-synthesis speed benchmark (`test_cumulative_compile_scale.py`) --
    is essentially unaffected. Output with `entangling_basis="cx"` changes:
    same two-qubit counts, fewer `sx`, lower total depth than the 2026-09-16
    revision (e.g. the README's coupling-map cliff table, measured with
    "cx", reports depth 23 for `layout_search=True`; expected 16 now). A
    per-circuit speed change of order 10-20% on the "cx" path could not be
    ruled in or out (Addendum 116, P4).

Changes in the 2026-09-26.3 revision (spare-qubit-cliff Addenda 191-194)
------------------------------------------------------------------------
15. **CX-basis core in closed form.** With `entangling_basis="cx"`, each
    block's canonical core exp(i(a XX + b YY + c ZZ)) used to be built as a
    circuit, turned into an `Operator`, and decomposed again by Qiskit's
    `TwoQubitBasisDecomposer` -- a second KAK decomposition of a matrix whose
    decomposition was already known. On random blocks the cache never hits,
    and this was 23% of `compile_for_hardware`'s time on the Nighthawk cliff
    (Addendum 191). The core is now emitted directly as three CXs
    (Vatan-Williams form) with the rotations in the two middle gaps moved
    through the CXs so that each gap is exactly one `sx` after translation
    (an `rx(pi/2)` on the target side of the middle CX; see
    `_append_cx_core_closed_form`). Checked in isolation against
    `expm(i(aXX+bYY+cZZ))` on 20,000 random triples over [-pi, pi]^3:
    worst Frobenius distance 2.7e-15 including global phase; both middle
    gaps have |off-diagonal| = 1/sqrt(2) to 2.2e-16 (one `sx` each).
    Degenerate triples -- any coordinate within 1e-6 of a multiple of pi/2,
    where fewer than three CXs suffice -- keep the previous path, so the CX
    count can only stay equal. `USE_CX_CLOSED_FORM = False` restores the
    previous behavior everywhere (used for A/B validation).

16. **NEW, opt-in: `layout_edge_errors` on `compile_for_hardware()`.** A
    `{(p, q): error}` map of the native 2-qubit gate's error per physical
    edge (`edge_errors_from_target(backend.target)` builds one). When given
    with `layout_search=True`, and the interaction graph is a set of disjoint
    pairs (the matching shortcut of `psf_smart_layout`, LAYOUT_VERSION
    2026-09-26.m1, Addenda 192-193), the layout is the maximum-weight
    matching with weight log(1 - error) per edge, i.e. it maximizes the
    product of the edge fidelities the pairs land on. Ignored for any other
    interaction graph (VF2 is unweighted). Single-qubit and readout errors
    are not considered. Like any `initial_layout`, it skips VF2PostLayout.

Changes in the 2026-09-26.4 revision (spare-qubit-cliff Addenda 195-196)
------------------------------------------------------------------------
17. **FIX (correctness): every block left to Qiskit's CX decomposer is now
    checked.** Addendum 195 found that Qiskit's
    `TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")` (this file's
    `_CX_DECOMPOSER`, Qiskit 2.5.2) returns a wrong circuit -- average gate
    infidelity 6.99e-2 -- for unitaries whose smallest canonical coordinate
    is about 3e-8 to 3e-7, and that plain `transpile()` to basis
    [cx, rz, sx, x] fails the same way. Every earlier revision emitted such
    blocks unchanged on the `entangling_basis="cx"` path (the degenerate
    core via `_cx_core_cached`, and whole blocks via the fallback), and
    `verify=True` did not catch it because it checks the decomposition, not
    the emitted circuit. Now `_guarded_cx_synthesis` computes the emitted
    circuit's unitary and accepts it only if its average gate infidelity is
    <= 1e-8 (ten times Qiskit's own default requested fidelity, so its
    documented approximations still pass), correcting its global phase to
    match exactly. On rejection it retries
    with the default Euler basis (exact on every input tested), and for a
    canonical core finally with the closed form of item 15 regardless of
    degeneracy (three CXs, exact). A whole block that fails both raises
    `RuntimeError` rather than being emitted. The check costs one 4x4
    `Operator` per decomposer call, which after item 15 happens only on
    degenerate blocks. `USE_CX_GUARD = False` restores the unchecked
    behavior (for A/B validation only). Counts in `GUARD_STATS`.

18. **Closed-form core with native middle gaps.** The 2026-09-26.3 form
    left `ry`/`rx` in the two middle gaps for Qiskit to translate; drift
    under repeated recompilation rose about 2.6x (Addendum 195, C5). The
    gaps are now emitted directly in {rz, sx}, using
    rx(pi/2) ry(t) = rz(t) rx(pi/2), ry(t) rx(-pi/2) = rx(-pi/2) rz(t),
    rx(pi/2) = e^{-i pi/4} sx and rx(-pi/2) = -e^{-i pi/4} rz(pi) sx rz(pi):
    gap 1 is `sx, rz(2a - pi/2)`, gap 2 is `rz(3pi/2 - 2b), sx, rz(pi)`,
    global phase 3pi/4 in total. Checked in isolation on 20,000 random
    triples: worst 3.6e-15 including global phase. `USE_NATIVE_GAPS =
    False` restores the 2026-09-26.3 gaps.

19. **Single-qubit errors in the weighted layout.** `layout_qubit_errors`
    (`{q: error}` of the single-qubit `sx`; `qubit_errors_from_target`
    builds it) adds each endpoint's single-qubit log-fidelity to the edge
    weight. This is exact for a matching, where every matched edge covers
    exactly its two qubits. Gate counts per pair are estimated from the
    circuit: n2 = mean 2-qubit gates per interacting pair, n1 = 2 n2 + 1
    `sx` per qubit (one per middle gap plus the outer layers; matches the
    measured 7 per qubit for 3-CX blocks). Addendum 195 (Q3): without it,
    the weighted layout placed pairs on a qubit whose `sx` has error 1.0.
"""
from __future__ import annotations

import logging
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Union

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Operator
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks

try:
    import psf_zero_core
    from psf_zero_core import geometric_decompose
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise ImportError(
        "psf_zero_core (the compiled Rust core) could not be imported. Build it "
        "with `maturin develop --release`. Do NOT substitute psf_zero_core_stub "
        "here: it is a Qiskit-based stand-in for environments that cannot load "
        "the real extension, and silently swapping it in makes every benchmark "
        "in this project measure Qiskit against Qiskit."
    ) from exc

VERSION = "2026-09-26.4"
__version__ = VERSION

__all__ = [
    "VERSION",
    "compile",
    "compile_for_hardware",
    "GeodesicPSFHyper",
    "SU4GeodesicPSFSynthesizer",
    "unitary_fidelity",
    "edge_errors_from_target",
    "qubit_errors_from_target",
    "GUARD_STATS",
]

# Optional, newer core entry points. Older builds of the core have neither;
# the code below degrades to an equivalent (slower) path rather than failing.
_CORE_CHECKED = getattr(psf_zero_core, "geometric_decompose_checked", None)
_PSF_DEGENERATE_ERRORS = tuple(
    err
    for err in (
        getattr(psf_zero_core, "PsfDegenerateError", None),
        getattr(psf_zero_core, "PsfSU2SingularError", None),
        getattr(psf_zero_core, "PsfNumericError", None),
    )
    if err is not None
)

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_GATE_FLOOR = 12

_VALID_ENTANGLING_BASES = ("canonical", "cx")
_VALID_ON_UNSUPPORTED = ("keep", "raise")

# Reuses Qiskit's own exact CX-optimal decomposer, configured for the
# {rz, sx} basis so the single-qubit layers between its CXs are minimal
# (changelog item 14).
_CX_DECOMPOSER = TwoQubitBasisDecomposer(CXGate(), euler_basis="ZSX")
# Changelog item 17: the retry decomposer, and the guard's switch and counts.
_CX_DECOMPOSER_DEFAULT_EULER = TwoQubitBasisDecomposer(CXGate())
USE_CX_GUARD = True
# Average gate infidelity accepted from Qiskit's decomposer: ten times its own
# default requested fidelity (1 - 1e-9), so its documented approximations
# (dropping a tiny interaction to save a CX) pass and the 7e-2 failures do not.
_GUARD_TOL = 1e-8
GUARD_STATS = {"checked": 0, "zsx_rejected": 0, "default_rejected": 0, "closed_form_forced": 0}


def _aligned_distance(u: np.ndarray, circ: QuantumCircuit):
    """Average gate infidelity between `u` and the circuit's unitary
    (1 - (4 f^2 + 1) / 5 with f = |tr(V^dag U)| / 4), and the phase angle p
    with u ~= exp(i p) * unitary(circ)."""
    v = Operator(circ).data
    t = np.trace(v.conj().T @ u)
    ph = t / abs(t) if abs(t) > 0 else 1.0
    f = abs(t) / 4.0
    return float(1.0 - (4.0 * f * f + 1.0) / 5.0), float(np.angle(ph))


def _guarded_cx_synthesis(u: np.ndarray):
    """Qiskit's CX-basis synthesis of `u`, checked (changelog item 17).

    Returns (circuit, ok). An accepted circuit has its global phase corrected
    so that its unitary matches `u`, not only up to phase. ok is False only
    when both the ZSX and the default-Euler decomposers exceed `_GUARD_TOL`
    in average gate infidelity; the caller decides what to do then.
    """
    circ = _CX_DECOMPOSER(u)
    if not USE_CX_GUARD:
        return circ, True
    GUARD_STATS["checked"] += 1
    dist, phase = _aligned_distance(u, circ)
    if dist <= _GUARD_TOL:
        circ.global_phase += phase
        return circ, True
    GUARD_STATS["zsx_rejected"] += 1
    logger.debug("PSF-Zero guard: ZSX decomposer infidelity %.3e; retrying", dist)
    circ = _CX_DECOMPOSER_DEFAULT_EULER(u)
    dist, phase = _aligned_distance(u, circ)
    if dist <= _GUARD_TOL:
        circ.global_phase += phase
        return circ, True
    GUARD_STATS["default_rejected"] += 1
    return circ, False

# XX, YY and ZZ commute and share one eigenbasis, so the canonical core's
# matrix exponential is a diagonal scaling in a basis that can be computed
# once at import instead of per block. The 1/2/4 coefficients are there to
# make the combined spectrum non-degenerate, so `eigh` returns a basis that
# simultaneously diagonalises all three rather than an arbitrary rotation
# inside a degenerate eigenspace. Only used by the numpy verification
# fallback, for cores too old to have `geometric_decompose_checked`.
_XX = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=complex)
_YY = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=complex)
_ZZ = np.diag([1, -1, -1, 1]).astype(complex)
_, _CORE_BASIS = np.linalg.eigh(_XX + 2.0 * _YY + 4.0 * _ZZ)
_CORE_BASIS_H = _CORE_BASIS.conj().T
_WX = np.real(np.diag(_CORE_BASIS_H @ _XX @ _CORE_BASIS))
_WY = np.real(np.diag(_CORE_BASIS_H @ _YY @ _CORE_BASIS))
_WZ = np.real(np.diag(_CORE_BASIS_H @ _ZZ @ _CORE_BASIS))


def _validate_verify(verify: Union[bool, str]) -> Union[bool, str]:
    """Accept exactly True, False or "strict".

    Deliberately strict about types: `verify=1` is not `verify=True` here, and
    `"Strict"` is not `"strict"`. Both used to fall through to the cheap check
    without saying so, which is the failure mode this project's own rules are
    written against.
    """
    if verify is True or verify is False or verify == "strict":
        return verify
    raise ValueError(
        f"verify must be True, False or 'strict' (got {verify!r}). "
        "True = check the decomposition using the core's own reconstruction; "
        "'strict' = rebuild the emitted circuit with Operator(qc) and check "
        "that; False = no check."
    )


@dataclass
class GeodesicPSFHyper:
    tol: float = 1e-5
    phase_fix: bool = True
    on_unsupported: str = "keep"
    entangling_basis: str = "canonical"  # "canonical" (default) | "cx" (native CX output directly)

    def __post_init__(self) -> None:
        if self.entangling_basis not in _VALID_ENTANGLING_BASES:
            raise ValueError(
                f"entangling_basis must be one of {_VALID_ENTANGLING_BASES} "
                f"(got {self.entangling_basis!r})."
            )
        if self.on_unsupported not in _VALID_ON_UNSUPPORTED:
            raise ValueError(
                f"on_unsupported must be one of {_VALID_ON_UNSUPPORTED} "
                f"(got {self.on_unsupported!r})."
            )
        if not self.tol > 0.0:
            raise ValueError(f"tol must be positive (got {self.tol!r}).")


def unitary_fidelity(U_target: np.ndarray, qc: QuantumCircuit) -> float:
    """Average gate fidelity between a target unitary and a circuit.

    Unchanged, and still the most independent check available -- it builds the
    circuit's operator with Qiskit rather than trusting anything this package
    computed. It is also the most expensive one by more than an order of
    magnitude, which is why it is no longer on the default path; see
    `_verify_block`.
    """
    U_out = Operator(qc).data
    tr = np.trace(U_target.conj().T @ U_out)
    d = 4.0
    return float((np.abs(tr) ** 2 + d) / (d * (d + 1)))


def _zyz_matrix(triple) -> np.ndarray:
    phi, theta, lam = triple
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    ep, em = np.exp(-0.5j * phi), np.exp(0.5j * phi)
    lp, lm = np.exp(-0.5j * lam), np.exp(0.5j * lam)
    return np.array([[ep * c * lp, -ep * s * lm], [em * s * lp, em * c * lm]], dtype=complex)


def _reconstruct(cartan, k1, k2, phase: float) -> np.ndarray:
    """Rebuild the 4x4 from the values the core returned, following
    `synthesize()`'s own recipe (k2 locals, canonical core, k1 locals, global
    phase) so that what gets checked is the gate about to be emitted."""
    a, b, c = cartan
    core = (_CORE_BASIS * np.exp(1j * (a * _WX + b * _WY + c * _WZ))) @ _CORE_BASIS_H
    left = np.kron(_zyz_matrix(k1[0]), _zyz_matrix(k1[1]))
    right = np.kron(_zyz_matrix(k2[0]), _zyz_matrix(k2[1]))
    return np.exp(1j * phase) * (left @ core @ right)


def _infidelity(U_target: np.ndarray, U_out: np.ndarray) -> float:
    tr = np.trace(U_target.conj().T @ U_out)
    d = 4.0
    return float(1.0 - (np.abs(tr) ** 2 + d) / (d * (d + 1)))


REFINE_THRESHOLD = 1e-13
_REFINE_TARGET = 1e-14
_HALF_Z = np.diag([-0.5j, 0.5j])


def _pack(cartan, k1, k2, phase):
    return np.array([*cartan, *k1[0], *k1[1], *k2[0], *k2[1], phase], dtype=float)


def _unpack(p):
    return ((p[0], p[1], p[2]),
            ((p[3], p[4], p[5]), (p[6], p[7], p[8])),
            ((p[9], p[10], p[11]), (p[12], p[13], p[14])),
            p[15])


def _zyz_and_derivatives(triple):
    """_zyz_matrix(triple) and its derivatives in phi, theta and lam."""
    phi, theta, lam = triple
    m = _zyz_matrix(triple)
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    ep, em = np.exp(-0.5j * phi), np.exp(0.5j * phi)
    lp, lm = np.exp(-0.5j * lam), np.exp(0.5j * lam)
    d_theta = 0.5 * np.array([[-ep * s * lp, -ep * c * lm], [em * c * lp, -em * s * lm]], dtype=complex)
    return m, (_HALF_Z @ m, d_theta, m @ _HALF_Z)


def _reconstruct_with_jacobian(p):
    """_reconstruct(*_unpack(p)) and its derivative with respect to each of
    the 16 parameters, computed in closed form (Addendum 189)."""
    (a, b, c), k1, k2, phase = _unpack(p)
    diag = np.exp(1j * (a * _WX + b * _WY + c * _WZ))
    core = (_CORE_BASIS * diag) @ _CORE_BASIS_H
    l0, dl0 = _zyz_and_derivatives(k1[0])
    l1, dl1 = _zyz_and_derivatives(k1[1])
    r0, dr0 = _zyz_and_derivatives(k2[0])
    r1, dr1 = _zyz_and_derivatives(k2[1])
    left, right = np.kron(l0, l1), np.kron(r0, r1)
    g = np.exp(1j * phase)
    cr = core @ right
    lc = left @ core
    u = g * (left @ cr)
    jac = []
    for w in (_WX, _WY, _WZ):
        jac.append(g * (left @ ((_CORE_BASIS * (1j * w * diag)) @ _CORE_BASIS_H) @ right))
    for d in dl0:
        jac.append(g * (np.kron(d, l1) @ cr))
    for d in dl1:
        jac.append(g * (np.kron(l0, d) @ cr))
    for d in dr0:
        jac.append(g * (lc @ np.kron(d, r1)))
    for d in dr1:
        jac.append(g * (lc @ np.kron(r0, d)))
    jac.append(1j * u)
    return u, jac


def _refine_decomposition(U_target, cartan, k1, k2, phase, threshold=REFINE_THRESHOLD, max_iter=3):
    """Polish the core's decomposition in parameter space (Gauss-Newton on
    the 16 real parameters, residual = _reconstruct(...) - U_target).

    Addendum 185 traced PSF-Zero's per-block error to the core's returned
    parameters: up to ~1.8e-11 (Frobenius) on inputs near the Weyl-chamber
    face c = 0, where Qiskit's decomposer stays near 1e-14. Each Newton step
    roughly squares the error, so one step reaches machine precision.
    Addendum 189: the residual check uses the plain reconstruction; the
    closed-form Jacobian is computed only when a step is actually taken, and
    iteration stops once the residual is <= 1e-14 or a step fails to halve
    it. Returns the (possibly unchanged) decomposition and the residual norms
    before and after.
    """
    p = _pack(cartan, k1, k2, phase)
    d = (_reconstruct(cartan, k1, k2, phase) - U_target).ravel()
    before = float(np.linalg.norm(d))
    if before <= threshold:
        return (cartan, k1, k2, phase), before, before
    norm_r = before
    for _ in range(max_iter):
        _, jac = _reconstruct_with_jacobian(p)
        jm = np.stack([j.ravel() for j in jac], axis=1)
        jr = np.concatenate([jm.real, jm.imag], axis=0)
        r = np.concatenate([d.real, d.imag])
        q = p + np.linalg.lstsq(jr, -r, rcond=None)[0]
        d_q = (_reconstruct(*_unpack(q)) - U_target).ravel()
        norm_q = float(np.linalg.norm(d_q))
        if norm_q >= norm_r:
            break
        progress = norm_q < 0.5 * norm_r
        p, d, norm_r = q, d_q, norm_q
        if norm_r <= _REFINE_TARGET or not progress:
            break
    return _unpack(p), before, norm_r


class SU4GeodesicPSFSynthesizer:
    """Synthesize a 2-qubit unitary via the Rust core's Cartan decomposition.

    `verify` selects what, if anything, is checked before a synthesized block
    is accepted:

      True     -- check the decomposition, using the core's own reconstruction
                  when the core provides one and a numpy reconstruction
                  otherwise. Catches a bad decomposition. Does not independently
                  re-derive the circuit object, so it would not catch a bug in
                  this file's circuit construction (the core's reconstruction
                  mirrors that construction deliberately, which is what makes
                  the check meaningful, and also what makes it not independent
                  of it). On `entangling_basis="cx"` it checks the same
                  decomposition; the CX substitution applied afterwards is
                  Qiskit's own exact decomposer and is not re-checked here.
      "strict" -- build `Operator(qc)` from the emitted circuit and compare.
                  Independent of both the core and this file's own recipe, and
                  the only mode that validates the actual circuit object. ~12x
                  the cost of the default; worth it in CI, rarely in production.
      False    -- no check. The core still reports genuine failures as
                  exceptions, which are still caught and fall back.
    """

    def __init__(self, hyper: GeodesicPSFHyper, verify: Union[bool, str] = True):
        self.hyper = hyper
        self.verify = _validate_verify(verify)
        self.fallback_count = 0
        # Blocks whose core decomposition was polished by _refine_decomposition
        # (Addendum 186); the maximum residual seen before and after polishing.
        self.refine_count = 0
        self.refine_max_before = 0.0
        self.refine_max_after = 0.0
        self.degenerate_count = 0
        self.unexpected_count = 0
        self._last_reasons: list[str] = []

    def _fallback(self, U_target: np.ndarray, msg: str, expected: bool) -> QuantumCircuit:
        self.fallback_count += 1
        if expected:
            self.degenerate_count += 1
        else:
            self.unexpected_count += 1
        if self.hyper.on_unsupported == "raise":
            raise RuntimeError(msg)
        # Collected rather than warned per block: a large circuit can produce
        # thousands of these, and one summary at the end is both cheaper and
        # more useful than a flood of identical lines.
        if len(self._last_reasons) < 5:
            self._last_reasons.append(msg)
        logger.debug("PSF-Zero fallback: %s", msg)
        circ, ok = _guarded_cx_synthesis(U_target)
        if not ok:
            # Never observed; emitting a block known to be wrong is worse
            # than stopping (changelog item 17).
            raise RuntimeError(
                "CX-basis synthesis of a fallback block failed verification "
                "with both Euler bases"
            )
        return circ

    def fallback_summary(self) -> str:
        if not self.fallback_count:
            return ""
        parts = [
            f"{self.fallback_count} block(s) fell back to CX-basis synthesis "
            f"({self.degenerate_count} degenerate/numeric, "
            f"{self.unexpected_count} unexpected)"
        ]
        parts.extend(f"  e.g. {r}" for r in self._last_reasons)
        return "\n".join(parts)

    def _entangling_core(self, qc: QuantumCircuit, a: float, b: float, c: float) -> None:
        """Append the canonical entangling core directly to `qc`.

        The previous version built a second QuantumCircuit and composed it in;
        appending straight to the target skips an object allocation and a
        compose per block for an identical result.
        """
        if self.hyper.entangling_basis == "cx":
            if USE_CX_CLOSED_FORM and _append_cx_core_closed_form(qc, a, b, c):
                return
            sub = _cx_core_cached(a, b, c)
            if sub is not None:
                qc.compose(sub, [0, 1], inplace=True)
            return
        if abs(a) > 1e-10:
            qc.rxx(-2 * a, 0, 1)
        if abs(b) > 1e-10:
            qc.ryy(-2 * b, 0, 1)
        if abs(c) > 1e-10:
            qc.rzz(-2 * c, 0, 1)

    def _build_circuit(self, cartan, k1, k2, global_phase: float) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.global_phase = global_phase

        def local(triple, qubit):
            phi, theta, lam = triple
            qc.rz(lam, qubit)
            qc.ry(theta, qubit)
            qc.rz(phi, qubit)

        local(k2[0], 1)
        local(k2[1], 0)
        self._entangling_core(qc, *cartan)
        local(k1[0], 1)
        local(k1[1], 0)
        return qc

    def _verify_block(self, U_target, qc, cartan, k1, k2, phase, core_infid):
        """Return the infidelity to test against `tol`, or None to skip."""
        if self.verify is False:
            return None
        if self.verify == "strict":
            return 1.0 - unitary_fidelity(U_target, qc)
        if core_infid is not None:
            # The core already reconstructed this decomposition and told us how
            # far off it was; nothing further to compute. This is the same
            # quantity the numpy path below produces, for either entangling
            # basis -- what is being validated is the decomposition, not the
            # emitted circuit (only "strict" does that).
            return core_infid
        # Older core with no self-check: recompute the same reconstruction here.
        return _infidelity(U_target, _reconstruct(cartan, k1, k2, phase))

    def synthesize(self, U_target: np.ndarray) -> QuantumCircuit:
        if U_target.shape != (4, 4):
            raise ValueError("Input must be a 4x4 unitary matrix.")

        u_r = U_target.real.tolist()
        u_i = U_target.imag.tolist()

        core_infid = None
        try:
            # "strict" rebuilds the circuit with Operator(qc) and ignores
            # core_infid entirely, so asking the core for it is pure waste.
            want_core_check = self.verify is True
            if _CORE_CHECKED is not None and want_core_check:
                cartan, k1, k2, global_phase, core_infid = _CORE_CHECKED(u_r, u_i)
            else:
                cartan, k1, k2, global_phase = geometric_decompose(u_r, u_i)
        except Exception as exc:
            # A degenerate or numerically singular input is an expected event
            # that the CX path handles correctly; anything else means the core
            # itself misbehaved, and the two are worth counting separately.
            expected = isinstance(exc, _PSF_DEGENERATE_ERRORS) if _PSF_DEGENERATE_ERRORS else True
            return self._fallback(U_target, f"Decomposition failed or degenerate: {exc}", expected)

        # Polish the core's parameters before building the circuit (Addendum
        # 186). Near the Weyl-chamber face c = 0 the core's output can be off
        # by ~1e-11; one Gauss-Newton step on the 16 parameters brings it to
        # machine precision. A no-op when the residual is already <= 1e-13.
        (cartan, k1, k2, global_phase), res_before, res_after = _refine_decomposition(
            U_target, cartan, k1, k2, global_phase
        )
        if res_before > REFINE_THRESHOLD:
            self.refine_count += 1
            self.refine_max_before = max(self.refine_max_before, res_before)
            self.refine_max_after = max(self.refine_max_after, res_after)
            if core_infid is not None:
                # The core's self-check described the unpolished parameters;
                # re-derive it for the parameters actually emitted.
                core_infid = _infidelity(U_target, _reconstruct(cartan, k1, k2, global_phase))

        try:
            qc = self._build_circuit(cartan, k1, k2, global_phase)
        except Exception as exc:
            # The core returned a decomposition and this file failed to turn it
            # into a circuit. That is a bug here, not a property of the input,
            # and is never an expected degeneracy.
            return self._fallback(
                U_target,
                f"Circuit construction failed after a successful decomposition: {exc}",
                expected=False,
            )

        infid = self._verify_block(U_target, qc, cartan, k1, k2, global_phase, core_infid)
        if infid is not None and infid > self.hyper.tol:
            return self._fallback(
                U_target, f"Fidelity loss exceeded tolerance: {infid:.2e}", expected=False
            )
        return qc


def _cx_core_cached(a: float, b: float, c: float):
    """CX-basis form of the canonical core for one (a, b, c).

    Cached on the exact triple: Trotter steps and QAOA layers apply the same
    canonical angles to many pairs, and every miss costs an `Operator()` build
    plus a full Qiskit KAK on a matrix whose decomposition we already know.
    Keyed on the exact floats, so a cache hit is bit-identical -- no rounding,
    no accuracy traded for the speedup.

    LRU rather than "stop caching once full": the previous version kept the
    first 4096 triples forever and silently degraded to no caching at all
    after that, which is the wrong way round for a long-lived process whose
    working set moves.

    Not synchronised. Two threads racing here can each build the same entry,
    which wastes a decomposition but cannot produce a wrong one; the cache is
    keyed on the inputs and the value is a function of them alone.
    """
    key = (a, b, c)
    if key in _CX_CORE_CACHE:
        _CX_CORE_CACHE.move_to_end(key)
        return _CX_CORE_CACHE[key]
    core = QuantumCircuit(2)
    if abs(a) > 1e-10:
        core.rxx(-2 * a, 0, 1)
    if abs(b) > 1e-10:
        core.ryy(-2 * b, 0, 1)
    if abs(c) > 1e-10:
        core.rzz(-2 * c, 0, 1)
    if len(core.data) == 0:
        result = None
    else:
        result, ok = _guarded_cx_synthesis(Operator(core).data)
        if not ok:
            # Both decomposers missed; the closed form is exact for any
            # triple, at the price of three CXs (changelog item 17).
            GUARD_STATS["closed_form_forced"] += 1
            result = QuantumCircuit(2)
            _append_cx_core_closed_form(result, a, b, c, force=True)
    _CX_CORE_CACHE[key] = result
    if len(_CX_CORE_CACHE) > _CX_CORE_CACHE_MAX:
        _CX_CORE_CACHE.popitem(last=False)
    return result


_CX_CORE_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_CX_CORE_CACHE_MAX = 4096

# Changelog item 15. Module-level so a validation run can switch it off.
USE_CX_CLOSED_FORM = True
# Changelog item 18: middle gaps emitted in {rz, sx}.
USE_NATIVE_GAPS = True
# A coordinate this close to a multiple of pi/2 makes the core reducible to
# fewer than three CXs; those triples keep the decomposer path.
_CLOSED_FORM_DEGENERATE = 1e-6
_HALF_PI = float(np.pi / 2)


def _append_cx_core_closed_form(qc: QuantumCircuit, a: float, b: float, c: float,
                                force: bool = False) -> bool:
    """Append exp(i(a XX + b YY + c ZZ)) to qubits (0, 1) of `qc` as three
    CXs, exactly (including global phase). Returns False, appending nothing,
    when the triple is degenerate (see `_CLOSED_FORM_DEGENERATE`).

    Vatan-Williams form, adapted: q1 is the control of the outer CXs and the
    target of the middle one, so an `rx` on q1 commutes with the middle CX.
    Inserting rx(pi/2) rx(-pi/2) around it turns each middle-gap rotation
    into rx(pi/2) ry(t) or ry(t) rx(-pi/2), whose off-diagonal magnitude is
    1/sqrt(2) for every t -- one `sx` after translation, instead of two.
    With USE_NATIVE_GAPS (changelog item 18) the two gaps are written in
    {rz, sx} directly. `force=True` skips the degeneracy check (used when
    Qiskit's decomposer failed on a degenerate triple; the result is still
    exact, with three CXs).
    """
    if not force:
        for x in (a, b, c):
            if abs(x - _HALF_PI * round(x / _HALF_PI)) < _CLOSED_FORM_DEGENERATE:
                return False
    half_pi = _HALF_PI
    if USE_NATIVE_GAPS:
        qc.rz(-half_pi, 1)
        qc.cx(1, 0)
        qc.rz(half_pi - 2.0 * c, 0)
        qc.sx(1)
        qc.rz(2.0 * a - half_pi, 1)
        qc.cx(0, 1)
        qc.rz(3.0 * half_pi - 2.0 * b, 1)
        qc.sx(1)
        qc.rz(np.pi, 1)
        qc.cx(1, 0)
        qc.rz(half_pi, 0)
        qc.global_phase += 3.0 * np.pi / 4
        return True
    qc.rz(-half_pi, 1)
    qc.cx(1, 0)
    qc.rz(half_pi - 2.0 * c, 0)
    qc.ry(2.0 * a - half_pi, 1)
    qc.rx(half_pi, 1)
    qc.cx(0, 1)
    qc.rx(-half_pi, 1)
    qc.ry(half_pi - 2.0 * b, 1)
    qc.cx(1, 0)
    qc.rz(half_pi, 0)
    qc.global_phase += np.pi / 4
    return True


def compile(
    qc: QuantumCircuit,
    block_gate_floor: int = DEFAULT_BLOCK_GATE_FLOOR,
    verify: Union[bool, str] = True,
    entangling_basis: str = "canonical",
    on_unsupported: str = "keep",
    tol: float = 1e-5,
) -> QuantumCircuit:
    """Collect 2-qubit blocks, consolidate them, and re-synthesize each one
    through the Rust core's Cartan (KAK) decomposition.

    Only runs of more than `block_gate_floor` gates on the same qubit pair are
    collected, so a wide-and-shallow circuit (`random_circuit()`, say) reports
    "0/0 blocks" and comes back untouched by design rather than by accident.
    Everything that is not a 2-qubit `unitary` block is copied through as-is,
    keeping the input's registers, bits, name and metadata.

    Args:
        qc: the circuit to compile.
        block_gate_floor: minimum run length before a same-pair block is worth
            consolidating and re-synthesizing.
        verify: True (default, checks the decomposition via the core's own
            reconstruction -- effectively free), "strict" (rebuilds the emitted
            circuit with `Operator(qc)` and checks that, ~12x the cost), or
            False (no check). See `SU4GeodesicPSFSynthesizer`.
        entangling_basis: "canonical" emits RXX/RYY/RZZ directly; "cx"
            re-expresses the entangling core through Qiskit's exact CX-basis
            decomposer, which costs 2x fewer native gates on hardware whose
            native 2-qubit gate is CX-like (see docs/findings/entangling-basis.md).
        on_unsupported: "keep" (default) falls back to CX-basis synthesis for a
            block the core cannot decompose; "raise" turns it into an error.
        tol: infidelity above which a synthesized block is rejected and falls
            back instead of being emitted.

    Returns:
        A new circuit with the same structure and qubit/clbit layout as `qc`.
    """
    verify = _validate_verify(verify)

    def worth_consolidating(dag, block):
        return len(block) > block_gate_floor

    pm_consolidate = PassManager([
        Collect2qBlocks(filter_fn=worth_consolidating),
        ConsolidateBlocks(kak_basis_gate=None, force_consolidate=True),
    ])
    qc_blocked = pm_consolidate.run(qc)

    hyper = GeodesicPSFHyper(
        tol=tol, on_unsupported=on_unsupported, entangling_basis=entangling_basis
    )
    synth = SU4GeodesicPSFSynthesizer(hyper, verify=verify)

    # `copy_empty_like()` keeps the registers, loose bits, name, metadata and
    # global phase of the input. The previous version built
    # `QuantumCircuit(qc.num_qubits, qc.num_clbits)` and then mapped every bit
    # back through `find_bit`, which dropped register structure (so a
    # classically-conditioned instruction had nowhere to point) and paid two
    # lookups per instruction to do it.
    qc_psf = qc_blocked.copy_empty_like()
    qc_psf.global_phase = qc_blocked.global_phase

    blocks_processed = 0
    blocks_seen = 0

    for inst in qc_blocked.data:
        op = inst.operation

        if len(inst.qubits) == 2 and op.name == "unitary":
            mat = op.to_matrix()
            if mat is not None and mat.shape == (4, 4):
                blocks_seen += 1
                before = synth.fallback_count
                synthesized_block = synth.synthesize(mat)
                if synth.fallback_count == before:
                    blocks_processed += 1
                qc_psf.compose(synthesized_block, inst.qubits, inplace=True)
                continue

        qc_psf.append(op, inst.qubits, inst.clbits)

    logger.debug(
        "PSF-Zero Rust Core executed for %d/%d blocks (%d fell back); "
        "block_gate_floor=%d; verify=%s; entangling_basis=%s.",
        blocks_processed,
        blocks_seen,
        synth.fallback_count,
        block_gate_floor,
        verify,
        entangling_basis,
    )
    if synth.fallback_count:
        # One summary, once, instead of a warning per block.
        warnings.warn(synth.fallback_summary(), UserWarning, stacklevel=2)
    return qc_psf


def edge_errors_from_target(target, gate_names=("cz", "ecr", "cx")) -> dict:
    """`{(p, q): error}` for the first of `gate_names` the target supports,
    undirected (p < q), the smaller error when both directions are listed.
    Edges without an error value are left out. For `layout_edge_errors`
    (changelog item 16)."""
    name = next((g for g in gate_names if g in target.operation_names), None)
    if name is None:
        raise ValueError(f"target supports none of {gate_names}")
    out = {}
    for qargs, props in target[name].items():
        if qargs is None or props is None or props.error is None:
            continue
        key = (min(qargs), max(qargs))
        out[key] = min(out.get(key, props.error), props.error)
    return out


def qubit_errors_from_target(target, gate_name: str = "sx") -> dict:
    """`{q: error}` of the single-qubit `gate_name` (default `sx`, the only
    non-virtual single-qubit gate on IBM devices). Qubits without an error
    value are left out. For `layout_qubit_errors` (changelog item 19)."""
    out = {}
    if gate_name not in target.operation_names:
        return out
    for qargs, props in target[gate_name].items():
        if qargs is None or props is None or props.error is None:
            continue
        out[qargs[0]] = props.error
    return out


def _edge_weights_from_errors(edge_errors: dict, qubit_errors: dict | None = None,
                              n2: float = 1.0, n1: float = 0.0) -> dict:
    """Integer matching weights, larger is better:
    round(1e6 * (K + n2 log(1 - e_pq) + n1 (log(1 - e_p) + log(1 - e_q)))),
    every error capped at 0.5 and K chosen so that every weight is positive.
    The total weight of a matching then tracks the log of the product of the
    fidelities of the gates it will carry. Edges absent from `edge_errors`
    get weight 0 in the matching (treated as worst); qubits absent from
    `qubit_errors` contribute nothing. With the defaults (n2=1, n1=0, no
    qubit errors) edges rank exactly as in the 2026-09-26.3 weighting
    (the constant offset differs)."""
    import math

    def lf(err):
        return math.log(1.0 - min(float(err), 0.5))

    k = 1.0 + (n2 + 2.0 * n1) * math.log(2.0)
    out = {}
    for (p, q), err in edge_errors.items():
        total = n2 * lf(err)
        if qubit_errors is not None and n1:
            total += n1 * (lf(qubit_errors.get(p, 0.0)) + lf(qubit_errors.get(q, 0.0)))
        out[(p, q)] = int(round(1e6 * (k + total)))
    return out


def _layout_weights(qc_compressed: QuantumCircuit, pairs, edge_errors: dict,
                    qubit_errors: dict | None) -> dict:
    """Matching weights for `compile_for_hardware` (changelog items 16, 19).
    Without qubit errors: the 2026-09-26.3 weighting. With them: n2 = mean
    number of 2-qubit gates per interacting pair in the compressed circuit,
    n1 = 2 n2 + 1 estimated `sx` per qubit."""
    if qubit_errors is None:
        return _edge_weights_from_errors(edge_errors)  # item 16 behaviour
    n_twoq = sum(1 for inst in qc_compressed.data if len(inst.qubits) == 2)
    n2 = n_twoq / max(1, len(pairs))
    return _edge_weights_from_errors(edge_errors, qubit_errors, n2=n2, n1=2.0 * n2 + 1.0)


def _layout_map_to_list(layout_map: dict, num_logical: int, num_physical: int) -> list[int]:
    """Convert `smart_vf2_layout()`'s `{logical: physical}` dict to the
    virtual-qubit-index-ordered list `transpile(initial_layout=...)` expects.

    Inlined from `benchmarks/benchmark_smart_layout_vs_default.py`'s
    `layout_map_to_list()` (same logic, copied rather than imported so this
    module does not depend on a benchmark script's internals) -- see that
    file's own docstring for why unused logical qubits get the remaining
    physical qubits assigned in index order rather than left unassigned.
    """
    used = set(layout_map.values())
    spare_iter = (p for p in range(num_physical) if p not in used)
    out = []
    for q in range(num_logical):
        if q in layout_map:
            out.append(layout_map[q])
        else:
            out.append(next(spare_iter))
    return out


def compile_for_hardware(
    qc: QuantumCircuit,
    coupling_map: CouplingMap,
    basis_gates: list[str] | None = None,
    block_gate_floor: int = DEFAULT_BLOCK_GATE_FLOOR,
    routing_optimization_level: int = 1,
    verify: Union[bool, str] = True,
    entangling_basis: str = "canonical",
    seed_transpiler: int | None = None,
    initial_layout: list[int] | None = None,
    on_unsupported: str = "keep",
    tol: float = 1e-5,
    layout_search: bool = False,
    layout_search_time_budget_s: float = 2.0,
    layout_search_call_limit: int = 50_000,
    layout_search_fallback_call_limit: int = 2_000_000,
    layout_search_use_fallback: bool = True,
    layout_edge_errors: dict | None = None,
    layout_qubit_errors: dict | None = None,
    callback=None,
) -> QuantumCircuit:
    """Compress with PSF-Zero, then route (and, if `basis_gates` is given,
    translate) with Qiskit.

    `basis_gates` matters more than it looks: with only a `coupling_map`,
    `transpile()` lays out and routes but never targets a gate set, so the
    RXX/RYY/RZZ this pass emits pass straight through and the result is not
    ISA-submittable.

    `routing_optimization_level` defaults to 1. It used to default to 2, on the
    stated grounds that translation "only happens" at level 2 -- that is not
    true, and was measured: with `basis_gates=["rz","sx","x","cx"]` the output
    contains nothing outside that set at level 0, 1 and 2 alike. What level 2
    actually does here is undo this pass. Qiskit's preset pipeline re-runs
    ConsolidateBlocks (init stage) and UnitarySynthesis (init + translation
    stages) over input it has no reason to trust, so at level 2 the result is
    bit-identical to plain `transpile(optimization_level=2)` -- verified gate
    for gate, qubit for qubit, parameter for parameter at 4, 6 and 7 qubits --
    for essentially the same wall time (100q dense-pair blocks: 1224 ms here
    vs 1232 ms for plain Qiskit). Every millisecond PSF-Zero spends is thrown
    away at that level. Addenda 24-25 (spare-qubit-cliff series) measured the
    same effect on a saturated coupling map specifically: raising this
    parameter from 1 toward 3 makes PSF-Zero's own spare-qubit cliff grow from
    ~7-8x to ~250-275x -- landing inside plain Qiskit L3's own 263-300x cliff
    -- and by level 3, `compile_for_hardware` is no longer faster than plain
    Qiskit L3 anywhere in that sweep, cliff or not (0.83x-0.96x). That result
    is what motivated `layout_search` below: rather than relying on level 1's
    internal routing call inheriting a cheaper `VF2Layout` failure, address
    the failure itself.

    The trade at level 1, measured on 50-156 qubit dense-pair-block circuits
    over a grid coupling map, is: the same 2-qubit gate count as Qiskit's
    optimization_level 2 and 3 (150 at 100q, 240 at 156q) for 1/20th to 1/59th
    of their time, at roughly 30-40% more depth (23 vs 16 at 100q, 44 vs 35 at
    156q). Against optimization_level 1 it wins outright: 1.8x faster, 20x
    fewer 2-qubit gates, 10x shallower. If minimum depth is what matters and
    compile time is not a constraint, call Qiskit's optimization_level=2
    directly -- routing_optimization_level=2 here gives you exactly that
    circuit and charges PSF-Zero's synthesis on top.

    This pass only helps circuits with deep interaction on the same qubit pair,
    the ones `Collect2qBlocks` can gather into blocks longer than
    `block_gate_floor`. On wide-and-shallow input (`random_circuit`, say) it
    reports "0/0 blocks", returns the circuit untouched by design, and is then
    pure overhead ahead of the Qiskit call.

    `tol` is forwarded to `compile()`. It previously was not, so the
    acceptance threshold was pinned at its default on this path with no way to
    reach it -- `seed_transpiler` and `on_unsupported` were both plumbed
    through and this one was simply missed.

    `seed_transpiler` pins the internal routing search. Leaving it unset means
    an `optimization_level >= 2` transpile returns a different circuit, and
    takes a different amount of time, on every call for identical input --
    which is exactly the effect that made this project's own coupling-map
    timings unreproducible until it was pinned on the comparison side.

    `initial_layout` pins the logical-to-physical qubit assignment, bypassing
    Qiskit's own layout search entirely. It is forwarded verbatim to
    `transpile()`, so it takes the same forms `transpile()` accepts -- most
    usefully a list of physical qubit indices ordered by virtual qubit index.

    **Passing this skips the whole layout stage, not just the layout search.**
    Measured with `transpile(callback=...)` on 2026-09-14:

        default:              SetLayout, VF2Layout, SabreLayout, VF2PostLayout
        with initial_layout:  SetLayout, ApplyLayout

    Note `VF2PostLayout` is skipped too. With only a `coupling_map` (no error
    rates) that pass has nothing to score against and its absence changes
    nothing, which is the configuration this project's benchmarks use. **With
    a real hardware `Target` carrying error rates it does have something to
    score, and skipping it can therefore cost fidelity** -- a layout that is
    valid (every interacting pair lands on a coupled edge) is not
    automatically a layout that is good on a noisy device. Supplying this
    argument moves that judgement to the caller. `layout_search=True` below
    also skips `VF2PostLayout` whenever it finds a layout, for the same
    reason: it, too, threads its result through `initial_layout`.

    The motivating use is feeding in a layout found by a separate search: this
    project's `psf_smart_layout.smart_vf2_layout()` returns `{logical:
    physical}`, which `layout_map_to_list()` in
    `benchmarks/benchmark_smart_layout_vs_default.py` converts to the list
    form expected here. Until this parameter existed there was no way to do
    that, so the end-to-end comparison that motivated the search could not be
    run at all (recorded in spare-qubit-cliff addendum 15, section 4).

    `layout_search` (new, item 12 in this file's changelog): when True, runs
    that same search *inside* this call instead of requiring the caller to
    invoke `smart_vf2_layout()` and build `initial_layout` by hand. The
    interaction graph handed to the search is read directly off the
    PSF-Zero-compressed circuit's own 2-qubit instructions (deduplicated,
    undirected pairs of qubit indices) -- the same circuit that is about to be
    routed, not the pre-compression input, since consolidation does not
    change which qubit pairs interact. Mutually exclusive with passing
    `initial_layout` directly (raises `ValueError` if both are given -- there
    is no principled way to silently prefer one over the other). Requires
    `psf_smart_layout` to be importable (raises `ImportError` naming the
    problem if it is not, rather than silently skipping the search); this is
    a new hard dependency only for callers who opt into `layout_search=True`.
    `layout_search_time_budget_s`, `layout_search_call_limit` and
    `layout_search_fallback_call_limit` are forwarded to
    `smart_vf2_layout()`'s `time_budget_s`, `per_attempt_call_limit` and
    `fallback_call_limit` respectively; `layout_search_use_fallback` disables
    its more expensive stage-2 search (see `psf_smart_layout.py`) if set to
    False. If the search does not find a layout within budget, this function
    falls through to Qiskit's own default layout stage exactly as if
    `layout_search` had been False -- the time already spent searching is
    real and is included in this call's own wall-clock cost, not hidden.

    `layout_edge_errors` (new, item 16): `{(p, q): error}` for the native
    2-qubit gate; see the changelog. Used only with `layout_search=True` and
    only when the interaction graph is a set of disjoint pairs.
    `layout_qubit_errors` (new, item 19): `{q: error}` of `sx`, added to the
    same weights; ignored unless `layout_edge_errors` is also given.

    `callback` (new, item 13): forwarded verbatim to the internal
    `transpile()` call below, unchanged from what plain `transpile(callback=
    ...)` accepts. `None` by default -- passing nothing here changes nothing
    about this function's behavior or cost. See item 13 in this file's
    changelog for why this exists.
    """
    if layout_search and initial_layout is not None:
        raise ValueError(
            "layout_search=True and an explicit initial_layout were both "
            "given -- ambiguous which should be used. Pass one or the other, "
            "not both."
        )

    qc_compressed = compile(
        qc,
        block_gate_floor=block_gate_floor,
        verify=verify,
        entangling_basis=entangling_basis,
        on_unsupported=on_unsupported,
        tol=tol,
    )

    if layout_search:
        try:
            from psf_smart_layout import smart_vf2_layout
        except ImportError as exc:
            raise ImportError(
                "layout_search=True requires psf_smart_layout.py to be "
                "importable (it is normally under prototypes/ in this "
                "project's repository layout) -- it was not found. Either "
                "make it importable (e.g. on PYTHONPATH or copied next to "
                "this file) or call with layout_search=False."
            ) from exc

        pairs = set()
        for inst in qc_compressed.data:
            if len(inst.qubits) == 2:
                i = qc_compressed.find_bit(inst.qubits[0]).index
                j = qc_compressed.find_bit(inst.qubits[1]).index
                if i != j:
                    pairs.add((i, j) if i < j else (j, i))

        layout_map, _search_info = smart_vf2_layout(
            coupling_map,
            sorted(pairs),
            qc_compressed.num_qubits,
            per_attempt_call_limit=layout_search_call_limit,
            time_budget_s=layout_search_time_budget_s,
            fallback_call_limit=layout_search_fallback_call_limit,
            use_fallback=layout_search_use_fallback,
            **({} if layout_edge_errors is None
               else {"edge_weights": _layout_weights(qc_compressed, pairs, layout_edge_errors,
                                                     layout_qubit_errors)}),
        )
        if layout_map is not None:
            initial_layout = _layout_map_to_list(
                layout_map, qc_compressed.num_qubits, coupling_map.size()
            )
        # else: fall through to Qiskit's own default layout stage below,
        # exactly as if layout_search had been False. The search time already
        # spent is real and is not subtracted from this call's wall time.

    return transpile(
        qc_compressed,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        optimization_level=routing_optimization_level,
        seed_transpiler=seed_transpiler,
        initial_layout=initial_layout,
        callback=callback,
    )
