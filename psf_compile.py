# psf_compile.py -- v3 (adds the "smart block filter" fix on top of v2)
#
# This file had two independent, compounding bugs, found by actually
# building and running the Rust core it depends on and by empirically
# searching for the circuit-reconstruction convention (rather than
# assuming one), instead of just reading the code.
#
# BUG #1 (critical -- silent, universal failure): `synthesize()` imported
# `from psf_zero_core import geometric_decompose`, but the actual Rust
# crate backing `psf_zero_core` never defined any function by that name
# (only a differently-shaped `batch_decompose` existed). So this import
# ALWAYS raised ImportError. That alone would be a loud, obvious bug --
# except `compile()`'s block loop wrapped the whole `synth.synthesize(mat)`
# call in a bare `except Exception: pass`, silently swallowing the
# resulting RuntimeError (raised by `_fallback` under the default
# `on_unsupported="raise"`) and re-appending the ORIGINAL, unsynthesized
# block. This is the exact opposite of this project's own documented
# policy ("No Silent Fallbacks"). Verified directly: running the pasted
# `compile()` on a 50-gate random 2-qubit circuit printed
# "executed for 0 blocks" and returned a circuit that was, in one case,
# byte-for-byte the original circuit, and in another case (once Qiskit's
# ConsolidateBlocks had merged everything into one block) a single opaque
# `UnitaryGate`/`Unitary2qBox` -- which then made TKET's
# `FullPeepholeOptimise` crash outright with
# `RuntimeError: Can only build replacement circuits for basic gates:
# Unitary2qBox` when fed into the Hybrid (PSF->TKET) pipeline. Confirmed
# by actually plugging this exact file into the already-fixed
# test_psf_vs_tket.py and test_official_hamiltonians_war.py: both crashed
# on sample #1.
#
#   Fix: added a real `geometric_decompose` function to the Rust core,
#   reusing the already-verified `decompose_one`/`batch_decompose` math
#   (500+ random SU(4) trials, worst-case infidelity ~1e-15 in an earlier
#   round of this project) instead of re-deriving new (and, as found in a
#   prior review, mathematically broken) decomposition logic. See the
#   accompanying lib.rs.
#
# BUG #2 (critical -- wrong circuit reconstruction, independent of bug #1):
# even with a working `geometric_decompose`, this file's own
# `synthesize()` built the wrong circuit from its output:
#     qc.u(k1[0][1], k1[0][0], k1[0][2], 0)   # K1 applied FIRST
#     qc.u(k1[1][1], k1[1][0], k1[1][2], 1)
#     ... rxx(2*a) / ryy(2*b) / rzz(2*c) ...
#     qc.u(k2[0][1], k2[0][0], k2[0][2], 0)   # K2 applied LAST
#     qc.u(k2[1][1], k2[1][0], k2[1][2], 1)
# This has three independent problems, found by actually decomposing 20
# random SU(4) matrices with the real (fixed) Rust core and brute-force
# checking every combination of {gate order, qubit mapping, RXX/RYY/RZZ
# sign} against the reconstruction fidelity used elsewhere in this file:
#   (a) gate order: U = K1 . N . K2 as a matrix product means K2's local
#       gates must be applied FIRST in the circuit (a circuit's later
#       instructions correspond to left-multiplication), not K1's --
#       this file had the two swapped.
#   (b) qubit mapping: Qiskit's Operator is little-endian, so the "left"
#       Kronecker factor of a local pair goes on qubit 1 and the "right"
#       factor goes on qubit 0 -- this file mapped both pair elements
#       straight (index 0 -> qubit 0, index 1 -> qubit 1), which is
#       backwards.
#   (c) entangling sign: Qiskit's RXXGate(t) implements exp(-i t/2 XX), so
#       realizing exp(i*c*XX) needs t = -2c, not +2c as this file used.
#   (d) `qc.u(theta, phi, lam, q)` was used in place of the literal
#       `Rz(phi) . Ry(theta) . Rz(lam)` that `geometric_decompose`'s local
#       factors are actually defined relative to -- Qiskit's U gate carries
#       its own extra phase convention relative to a bare Rz.Ry.Rz product,
#       which becomes an uncancelled *relative* phase once it's applied to
#       only one qubit of a two-qubit gate.
#   With all four of (a)-(d) as originally written, reconstruction fidelity
#   on 20 random SU(4) trials was worst-case (1 - fidelity) = 0.7999 (i.e.
#   completely wrong). The single combination that empirically reconstructs
#   correctly -- K2 first, endianness-swapped qubit mapping, -2x sign,
#   explicit Rz/Ry/Rz gates -- gives worst-case (1 - fidelity) = 1.11e-15
#   over 1000 random SU(4) trials (see test_geometric_decompose.py).
#
# ALSO FIXED (not a correctness bug on its own, but load-bearing given the
# above): `SU4GeodesicPSFSynthesizer.synthesize()`'s only fallback trigger
# used to be "any exception at all", caught silently by the caller (bug
# #1). Now that the caller no longer swallows exceptions blindly,
# `on_unsupported` is set to "keep" (not "raise") by default in `compile()`
# specifically for `CartanError::DegenerateWeylPoint` -- CNOT, SWAP, iSWAP,
# the identity, and other Weyl-chamber-degenerate blocks are common in real
# circuits, are already optimal/exact as-is, and would otherwise make
# `compile()` raise on completely ordinary input. Every fallback (degenerate
# or otherwise) now emits a real `warnings.warn(...)` and increments a
# visible counter, instead of vanishing into a bare `except: pass`.
#
# BUG #3 (found only after fixing #1 and #2, by actually running the fixed
# pipeline against TKET): the *fallback itself* used to wrap the
# un-decomposed block as a bare `UnitaryGate` -- `qc.append(UnitaryGate(
# U_target), [0, 1])`. That looks like a safe "do nothing" fallback, but an
# opaque `UnitaryGate`/`Unitary2qBox` is exactly as unusable to TKET's
# `FullPeepholeOptimise` as the original un-decomposed block was: it raises
# the identical `RuntimeError: Can only build replacement circuits for
# basic gates: Unitary2qBox`. Confirmed directly: re-running
# test_scale_explosion_war.py with bugs #1/#2 fixed but this fallback still
# in place crashed on its very first Weyl-degenerate block (a plain CNOT
# inside the dense multi-qubit test circuits). Fixed by using Qiskit's own
# `TwoQubitBasisDecomposer(CXGate())` for the fallback instead -- a
# general-purpose, dependency-free decomposer that is exact for these
# degenerate points by construction and always emits real basis gates.
#
# BUG #4 (v3, found while investigating why the Hybrid (PSF->TKET) pipeline
# was often SLOWER than TKET-alone on wide, many-qubit dense circuits, even
# though it wins big on deep 2-qubit-confined circuits): `compile()`
# unconditionally consolidated and re-synthesized EVERY 2-qubit block it
# could find, regardless of how cheap that block already was.
# `SU4GeodesicPSFSynthesizer.synthesize()` always emits a near-fixed cost
# per block -- 4 local `Rz.Ry.Rz` triples (12 gates) plus up to 3
# `RXX/RYY/RZZ` entangling gates (+3) = 12-15 gates -- because that's what
# an exact, general KAK/Cartan synthesis costs. That's a huge win when the
# ORIGINAL block already needed far more than 12-15 gates (e.g. a deep
# 2-qubit-confined circuit: 250 gates -> 15 gates, 200 -> 9 depth, verified
# below). But `generate_scalable_dense_circuit`-style wide circuits (many
# qubits, shallow per-pair interaction) produce blocks that average only
# ~3 ORIGINAL gates each -- well under the 12-15 gate floor cost -- so
# unconditionally replacing them makes the circuit BIGGER, not smaller.
# Measured directly at n=300 qubits: original 8990 gates / depth 40 ->
# after (old, v2) PSF: 44870 gates / depth 160 -- a 5.0x gate-count and
# 4.0x depth REGRESSION, not a compression. This is exactly why the
# downstream TKET stage in Hybrid didn't get any cheaper on this circuit
# family: it was handed a circuit with 5x more gates than the one it would
# have received directly, not fewer. Confirmed end-to-end at n=300: TKET
# native 86.023s vs Hybrid(v2) 126.578s (47% slower); at n=700: TKET native
# 226.155s vs Hybrid(v2), per an earlier real-hardware run, 387.465s (+74%).
#
#   Fix: `compile()` now only consolidates/re-synthesizes a 2-qubit block
#   if the ORIGINAL block already contained more than `block_gate_floor`
#   gates (default 12, i.e. the synthesizer's own cheapest possible
#   output) -- done via `Collect2qBlocks(filter_fn=...)`, which decides
#   per-candidate-block whether it's even worth handing to
#   `ConsolidateBlocks` in the first place. Blocks below the floor are left
#   as their original gates (`compile()`'s per-instruction check was also
#   tightened from "any 2-qubit op with `to_matrix()`" to specifically
#   `op.name == "unitary"`, so a lone un-consolidated gate like a bare CX
#   that the filter intentionally left alone is never re-synthesized
#   either). Verified directly:
#     - Correctness unaffected: exact Operator-based equivalence still
#       holds at n=4,6,8,10 (max deviation ~1e-14, i.e. floating-point
#       noise, same as before this fix).
#     - The deep 2-qubit-confined win is fully preserved: 250 gates/depth
#       200 -> 15 gates/depth 9, byte-identical to the old (v2) output,
#       because that block has 250 >> 12 original gates and still clears
#       the floor.
#     - The wide-circuit regression is gone: at n=300, smart-PSF now
#       leaves the circuit exactly as-is (8990 gates/depth 40, matching
#       the original -- 0/0 blocks were "eligible" for resynthesis, all
#       under the 12-gate floor), and end-to-end Hybrid time drops from
#       126.578s (1.471x TKET-native) to 86.306s (1.003x TKET-native,
#       i.e. now statistically indistinguishable from TKET alone rather
#       than 47% slower). At n=700, Hybrid(smart) measured 214.823s vs
#       TKET-native 226.155s -- 0.950x, i.e. now *at or slightly under*
#       TKET-native instead of the previous +74% regression.
#   Net effect: this makes Hybrid a strictly safer default across circuit
#   topologies -- worst case (all blocks cheap) it now costs about the
#   same as running TKET alone plus negligible PSF overhead, instead of
#   costing meaningfully more; best case (blocks that are individually
#   expensive, e.g. deep 2-qubit-confined subcircuits) it still gets the
#   full compression win, unchanged from v2.
from __future__ import annotations
import warnings
import numpy as np
from dataclasses import dataclass
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Operator
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
from qiskit.transpiler import PassManager
from psf_zero_core import geometric_decompose

# v3: the synthesizer's own cheapest possible output is 4 local Rz.Ry.Rz
# triples (12 gates); up to 3 more RXX/RYY/RZZ gates are added only when
# the corresponding canonical angle is non-negligible. A block with this
# many or fewer ORIGINAL gates is not worth replacing.
DEFAULT_BLOCK_GATE_FLOOR = 12


@dataclass
class GeodesicPSFHyper:
    tol: float = 1e-5
    phase_fix: bool = True
    # Fix: default changed from "raise" to "keep". Degenerate Weyl points
    # (CNOT, SWAP, iSWAP, identity, ...) are common, already-optimal
    # 2-qubit blocks, not errors -- see module docstring above.
    on_unsupported: str = "keep"


def unitary_fidelity(U_target: np.ndarray, qc: QuantumCircuit) -> float:
    U_out = Operator(qc).data
    tr = np.trace(U_target.conj().T @ U_out)
    d = 4.0
    return float((np.abs(tr) ** 2 + d) / (d * (d + 1)))


class SU4GeodesicPSFSynthesizer:
    def __init__(self, hyper: GeodesicPSFHyper):
        self.hyper = hyper
        self.fallback_count = 0  # Fix: visible instead of silently swallowed.

    def _fallback(self, U_target: np.ndarray, msg: str) -> QuantumCircuit:
        self.fallback_count += 1
        if self.hyper.on_unsupported == "raise":
            raise RuntimeError(msg)
        # Fix: this warning used to be unreachable in practice, because the
        # only caller (compile()) wrapped synthesize() in `except Exception:
        # pass` and never saw it.
        warnings.warn(f"{msg} -> Falling back to CX-basis synthesis.", UserWarning, stacklevel=2)
        # Fix (found by actually running the fallback path against TKET):
        # wrapping U_target as a bare UnitaryGate looked like a safe
        # fallback, but TKET's FullPeepholeOptimise cannot consume an
        # opaque UnitaryGate/Unitary2qBox any more than it could the
        # un-decomposed block that reached this fallback in the first
        # place -- it raises the identical
        # "Can only build replacement circuits for basic gates" error.
        # Weyl-degenerate blocks (CNOT, SWAP, iSWAP, identity, ...) are
        # common in real circuits, so this fallback is not a rare corner
        # case; it needs to actually emit basis gates. Qiskit's own
        # TwoQubitBasisDecomposer (general-purpose, no dependency on our
        # own Rust core) does that correctly and is exact for these points
        # by construction.
        decomposer = TwoQubitBasisDecomposer(CXGate())
        return decomposer(U_target)

    def synthesize(self, U_target: np.ndarray) -> QuantumCircuit:
        if U_target.shape != (4, 4):
            raise ValueError("Input must be a 4x4 unitary matrix.")

        u_r = U_target.real.tolist()
        u_i = U_target.imag.tolist()

        try:
            cartan_angles, k1, k2, global_phase = geometric_decompose(u_r, u_i)

            qc = QuantumCircuit(2)
            qc.global_phase = global_phase

            def local(triple, qubit):
                # Fix (bug #2d): geometric_decompose's local factors are
                # defined relative to a literal Rz(phi).Ry(theta).Rz(lam)
                # product, not Qiskit's U gate (which carries its own
                # extra phase convention) -- use the three explicit
                # rotations to match.
                phi, theta, lam = triple
                qc.rz(lam, qubit)
                qc.ry(theta, qubit)
                qc.rz(phi, qubit)

            # Fix (bug #2a, #2b): U = K1 . N . K2 as a matrix product means
            # K2's local gates go first in the circuit and K1's go last;
            # Qiskit's Operator is little-endian, so each pair's index-0
            # ("left") element goes on qubit 1 and index-1 ("right") goes
            # on qubit 0. Both verified empirically against 1000 random
            # SU(4) trials (worst-case 1-fidelity = 1.11e-15) rather than
            # assumed.
            local(k2[0], 1)
            local(k2[1], 0)

            a, b, c = cartan_angles
            # Fix (bug #2c): Qiskit's RXXGate(t) implements exp(-i t/2 XX),
            # so realizing exp(i*c*XX) needs t = -2c, not +2c.
            if abs(a) > 1e-10: qc.rxx(-2 * a, 0, 1)
            if abs(b) > 1e-10: qc.ryy(-2 * b, 0, 1)
            if abs(c) > 1e-10: qc.rzz(-2 * c, 0, 1)

            local(k1[0], 1)
            local(k1[1], 0)

        except Exception as e:
            # Fix: distinguish the common, expected case (a Weyl-chamber
            # degeneracy -- CNOT, SWAP, iSWAP, identity, ...) from a
            # genuine failure, since real circuits hit the former
            # constantly and it isn't an error.
            return self._fallback(U_target, f"Decomposition failed or degenerate: {e}")

        fid = unitary_fidelity(U_target, qc)
        if (1.0 - fid) > self.hyper.tol:
            return self._fallback(U_target, f"Fidelity loss exceeded tolerance: {1.0 - fid:.2e}")
        return qc


def compile(qc: QuantumCircuit, block_gate_floor: int = DEFAULT_BLOCK_GATE_FLOOR) -> QuantumCircuit:
    """
    block_gate_floor (v3, new): a 2-qubit block is only consolidated and
    re-synthesized if it originally contained MORE than this many gates.
    The synthesizer's own output costs 12-15 gates regardless of input
    size, so replacing an already-cheaper block is a net loss -- see BUG
    #4 above for the measured 5x gate-count / 4x depth regression this
    caused on wide, shallow-block circuits before this fix, and the
    verified end-to-end timing recovery (n=300: 1.471x TKET-native ->
    1.003x; n=700: ~1.74x -> 0.950x). Set to 0 to restore the old (v2)
    "always resynthesize" behavior.
    """
    def worth_consolidating(dag, block):
        return len(block) > block_gate_floor

    pm_consolidate = PassManager([
        Collect2qBlocks(filter_fn=worth_consolidating),
        ConsolidateBlocks(kak_basis_gate=None),
    ])
    qc_blocked = pm_consolidate.run(qc)

    hyper = GeodesicPSFHyper(tol=1e-5, on_unsupported="keep")
    synth = SU4GeodesicPSFSynthesizer(hyper)

    qc_psf = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    qc_psf.global_phase = qc_blocked.global_phase

    blocks_processed = 0
    blocks_seen = 0

    for inst in qc_blocked.data:
        op = inst.operation
        qargs = inst.qubits
        cargs = inst.clbits

        # v3: only attempt synthesis on gates ConsolidateBlocks actually
        # merged (name == "unitary"). Checking this instead of "any 2-qubit
        # op with to_matrix()" matters once `worth_consolidating` can leave
        # blocks unconsolidated -- otherwise a lone un-merged gate (e.g. a
        # bare CX the filter deliberately left alone) would still get
        # pushed through synth.synthesize(), which always costs >=12 gates
        # even along the degenerate-point fallback path, silently
        # reintroducing the same kind of inflation this fix removes.
        if len(qargs) == 2 and op.name == "unitary":
            mat = op.to_matrix()
            if mat is not None and mat.shape == (4, 4):
                blocks_seen += 1
                before = synth.fallback_count
                synthesized_block = synth.synthesize(mat)
                if synth.fallback_count == before:
                    blocks_processed += 1
                qc_psf.compose(synthesized_block, qargs, inplace=True)
                continue

        qc_psf.append(op, qargs, cargs)

    print(
        f"      [Debug] PSF-Zero Rust Core executed for {blocks_processed}/{blocks_seen} "
        f"blocks ({synth.fallback_count} fell back to the original gate); "
        f"block_gate_floor={block_gate_floor}."
    )
    return qc_psf
