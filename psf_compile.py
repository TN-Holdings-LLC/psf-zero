# psf_compile.py -- PSF-Zero's Qiskit transpiler pass.
#
# Changes in this revision
# ------------------------
# 1. FIXED: the Rust core import. The previous revision of this file imported
#    `psf_zero_core_stub` -- the Qiskit-based Python stand-in written for a
#    sandbox that could not load the real `.so` -- instead of `psf_zero_core`.
#    With that import in place nothing in the Rust core ran at all: the
#    degeneracy handling was never exercised, and every "PSF-Zero vs Qiskit"
#    measurement was really Qiskit's own TwoQubitWeylDecomposition being
#    compared against Qiskit, while the debug line still announced "PSF-Zero
#    Rust Core executed for N blocks". The import is now the real core, and a
#    missing core raises immediately with an explanation rather than silently
#    substituting something else.
#
# 2. Verification is no longer the dominant cost, so it no longer has to be
#    switched off to be competitive. The old `verify=True` path rebuilt the
#    synthesized circuit with `Operator(qc)` and compared -- measured at
#    ~1.35 ms per block against ~0.22 ms for everything else put together,
#    i.e. ~87% of the total, which is the entire reason this project's
#    measured speed advantage was only available with `verify=False`. The
#    core now returns the reconstruction infidelity itself
#    (`geometric_decompose_checked`), computed from a handful of 4x4 products
#    on values it already has. Measured on 400 random SU(4) blocks:
#
#        verify via Operator(qc)                 1.352 ms/block
#        verify via numpy reconstruction         0.111 ms/block
#        verify via the core's own check        ~0.000 ms/block (in the FFI call)
#
#    `verify=True` therefore stays the default and is now essentially free.
#    `verify="strict"` keeps the old `Operator(qc)` behavior for anyone who
#    wants the circuit object itself checked rather than the decomposition
#    (see `_verify_block` for exactly what each one does and does not cover).
#
# 3. `logging` instead of `print`. A library writing to stdout on every call
#    forced this project's own benchmark scripts to wrap it in
#    `contextlib.redirect_stdout`, and would do the same to anything else that
#    embeds it. The per-block debug line is now a single `logger.debug`.
#
# 4. Fallbacks are counted and reported once at the end instead of raising a
#    `warnings.warn` per block, and they are now classified: a legitimately
#    degenerate input and an unexpected core failure are different events, and
#    the core's new exception types let them be told apart.
#
# 5. `on_unsupported` is exposed on `compile()` (it was hardcoded to "keep"),
#    and `compile_for_hardware()` accepts `seed_transpiler` so its internal
#    routing call can be pinned -- the asymmetry this project's own benchmarks
#    ran into, where an unpinned `optimization_level>=2` transpile returns a
#    different answer and a different runtime on every call.
#
# 6. The entangling core is appended directly rather than built as a separate
#    QuantumCircuit and composed, and the CX-basis form of a given canonical
#    triple is cached -- Trotter and QAOA layers repeat the same (a, b, c)
#    across many blocks, and each miss otherwise costs an `Operator()` build
#    plus a full Qiskit KAK.
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

# Qiskitの厳密なCX最適分解器を再利用
_CX_DECOMPOSER = TwoQubitBasisDecomposer(CXGate())

# XX, YY and ZZ commute and share one eigenbasis, so the canonical core's
# matrix exponential is a diagonal scaling in a basis that can be computed
# once at import instead of per block. Only used by the numpy verification
# fallback, for cores too old to have `geometric_decompose_checked`.
_XX = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=complex)
_YY = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=complex)
_ZZ = np.diag([1, -1, -1, 1]).astype(complex)
_CORE_EIGVALS, _CORE_BASIS = np.linalg.eigh(_XX + 2.0 * _YY + 4.0 * _ZZ)
_CORE_BASIS_H = _CORE_BASIS.conj().T
_WX = np.real(np.diag(_CORE_BASIS_H @ _XX @ _CORE_BASIS))
_WY = np.real(np.diag(_CORE_BASIS_H @ _YY @ _CORE_BASIS))
_WZ = np.real(np.diag(_CORE_BASIS_H @ _ZZ @ _CORE_BASIS))


@dataclass
class GeodesicPSFHyper:
    tol: float = 1e-5
    phase_fix: bool = True
    on_unsupported: str = "keep"
    entangling_basis: str = "canonical"  # "canonical" (デフォルト) | "cx" (native CX直接出力)


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
                  of it).
      "strict" -- build `Operator(qc)` from the emitted circuit and compare.
                  Independent of both the core and this file's own recipe, and
                  the only mode that validates the actual circuit object. ~12x
                  the cost of the default; worth it in CI, rarely in production.
      False    -- no check. The core still reports genuine failures as
                  exceptions, which are still caught and fall back.
    """

    def __init__(self, hyper: GeodesicPSFHyper, verify: Union[bool, str] = True):
        self.hyper = hyper
        self.verify = verify
        self.fallback_count = 0
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
        return _CX_DECOMPOSER(U_target)

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
        if core_infid is not None and self.hyper.entangling_basis != "cx":
            # The core already reconstructed exactly this gate and told us how
            # far off it was; nothing further to compute.
            return core_infid
        # Either an older core with no self-check, or the CX-basis path, where
        # the emitted core differs from what the Rust side reconstructed (the
        # substitution itself is Qiskit's own exact decomposer applied to an
        # exact matrix, so what is being validated here is the decomposition).
        return _infidelity(U_target, _reconstruct(cartan, k1, k2, phase))

    def synthesize(self, U_target: np.ndarray) -> QuantumCircuit:
        if U_target.shape != (4, 4):
            raise ValueError("Input must be a 4x4 unitary matrix.")

        u_r = U_target.real.tolist()
        u_i = U_target.imag.tolist()

        try:
            core_infid = None
            # "strict" rebuilds the circuit with Operator(qc) and ignores
            # core_infid entirely, so asking the core for it is pure waste.
            want_core_check = self.verify is True
            if _CORE_CHECKED is not None and want_core_check:
                cartan, k1, k2, global_phase, core_infid = _CORE_CHECKED(u_r, u_i)
            else:
                cartan, k1, k2, global_phase = geometric_decompose(u_r, u_i)
            qc = self._build_circuit(cartan, k1, k2, global_phase)
        except Exception as exc:
            # A degenerate or numerically singular input is an expected event
            # that the CX path handles correctly; anything else means the core
            # itself misbehaved, and the two are worth counting separately.
            expected = isinstance(exc, _PSF_DEGENERATE_ERRORS) if _PSF_DEGENERATE_ERRORS else True
            return self._fallback(U_target, f"Decomposition failed or degenerate: {exc}", expected)

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
    result = None if len(core.data) == 0 else _CX_DECOMPOSER(Operator(core).data)
    _CX_CORE_CACHE[key] = result
    if len(_CX_CORE_CACHE) > _CX_CORE_CACHE_MAX:
        _CX_CORE_CACHE.popitem(last=False)
    return result


_CX_CORE_CACHE: "OrderedDict[tuple, object]" = OrderedDict()
_CX_CORE_CACHE_MAX = 4096


def compile(
    qc: QuantumCircuit,
    block_gate_floor: int = DEFAULT_BLOCK_GATE_FLOOR,
    verify: Union[bool, str] = True,
    entangling_basis: str = "canonical",
    on_unsupported: str = "keep",
    tol: float = 1e-5,
) -> QuantumCircuit:
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

    qc_psf = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    qc_psf.global_phase = qc_blocked.global_phase

    blocks_processed = 0
    blocks_seen = 0

    for inst in qc_blocked.data:
        op = inst.operation

        # Resolve to integer positions rather than passing the source
        # circuit's Bit objects through. `qc_psf` was constructed with fresh
        # registers, and Qiskit compares Qubits by (register, index) with
        # QuantumRegister compared by (name, size) -- so handing it a Bit from
        # `qc_blocked` only works when the input happens to use a single
        # register named "q", i.e. exactly what `QuantumCircuit(n)` produces.
        # Any named or split register raised
        #   CircuitError: Bit '<Qubit register=(4, "data"), index=0>' is not
        #   in the circuit
        # Every benchmark in this project builds `QuantumCircuit(n)`, so this
        # never fired here and would fire immediately for anyone else.
        qidx = [qc_blocked.find_bit(q).index for q in inst.qubits]
        cidx = [qc_blocked.find_bit(c).index for c in inst.clbits]

        if len(qidx) == 2 and op.name == "unitary":
            mat = op.to_matrix()
            if mat is not None and mat.shape == (4, 4):
                blocks_seen += 1
                before = synth.fallback_count
                synthesized_block = synth.synthesize(mat)
                if synth.fallback_count == before:
                    blocks_processed += 1
                qc_psf.compose(synthesized_block, qidx, inplace=True)
                continue

        qc_psf.append(op, qidx, cidx)

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


def compile_for_hardware(
    qc: QuantumCircuit,
    coupling_map: CouplingMap,
    basis_gates: list[str] | None = None,
    block_gate_floor: int = DEFAULT_BLOCK_GATE_FLOOR,
    routing_optimization_level: int = 1,
    verify: Union[bool, str] = True,
    entangling_basis: str = "canonical",
    seed_transpiler: int | None = None,
    on_unsupported: str = "keep",
    tol: float = 1e-5,
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
    away at that level.

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
    """
    qc_compressed = compile(
        qc,
        block_gate_floor=block_gate_floor,
        verify=verify,
        entangling_basis=entangling_basis,
        on_unsupported=on_unsupported,
        tol=tol,
    )
    return transpile(
        qc_compressed,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        optimization_level=routing_optimization_level,
        seed_transpiler=seed_transpiler,
    )
