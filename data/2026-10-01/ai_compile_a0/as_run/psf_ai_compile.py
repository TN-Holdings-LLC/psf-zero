"""psf_ai_compile.py -- PSF-Zero front end for circuits written by a language model (prototype).

The compiler itself is not changed: this module only calls `psf_compile.compile_for_hardware()` (and its
Rust core) and works around it. It is meant for what a model writes in the vLLM loop: small circuits
(a handful of qubits), many SWAPs and redundant gates, often given as PennyLane-style `unitary` gates,
compiled many times but each in milliseconds. Small circuits leave time for things a large-circuit path
cannot afford:

  1. a commutation-aware clean-up of the input (CommutativeCancellation) as a second starting point,
  2. several routing seeds per starting point, keeping the circuit with the fewest 2-qubit gates,
  3. a polish of the routed circuit: commutative cancellation plus re-synthesis (by the PSF-Zero core) of
     every 2-qubit block whose optimal CX count is below what it holds, repeated while it helps.

Circuits above `SMALL_MAX_QUBITS` go straight to `compile_for_hardware()` unchanged.
The layout of the routed circuit (initial and final) is preserved by every step.
"""
import contextlib
import io
import warnings

from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import (BasisTranslator, Collect2qBlocks, CommutativeCancellation,
                                      ConsolidateBlocks, Optimize1qGatesDecomposition)

import psf_compile as pc

AI_COMPILE_VERSION = "2026-10-01.a0"  # prototype (workplace)
SMALL_MAX_QUBITS = 8
DEFAULT_SEEDS = (0, 1, 2, 3)
POLISH_ROUNDS = 3


def _two_q(c):
    return sum(1 for inst in c.data if len(inst.qubits) == 2)


def _depth2q(c):
    return c.depth(lambda inst: len(inst.qubits) == 2)


def _resynthesize_blocks(circ, verify, tol, on_unsupported):
    """Consolidate every 2-qubit block that costs more CX than its optimum and re-synthesise it with the
    PSF-Zero core (CX basis). Returns a new circuit (layout not yet restored)."""
    pm = PassManager([
        Collect2qBlocks(filter_fn=lambda dag, block: pc._block_saves_cx(block)),
        ConsolidateBlocks(kak_basis_gate=None, force_consolidate=True),
    ])
    blocked = pm.run(circ)
    idx = [i for i, inst in enumerate(blocked.data)
           if len(inst.qubits) == 2 and inst.operation.name == "unitary"]
    if not idx:
        return circ
    synth = pc.SU4GeodesicPSFSynthesizer(
        pc.GeodesicPSFHyper(tol=tol, on_unsupported=on_unsupported, entangling_basis="cx"), verify=verify)
    done = dict(zip(idx, synth.synthesize_many([blocked.data[i].operation.to_matrix() for i in idx])))
    out = blocked.copy_empty_like()
    out.global_phase = blocked.global_phase
    for i, inst in enumerate(blocked.data):
        if i in done:
            out.compose(done[i][0], inst.qubits, inplace=True)
        else:
            out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def polish(routed, basis_gates, rounds=POLISH_ROUNDS, verify=True, tol=1e-5, on_unsupported="keep"):
    """Commutative cancellation + block re-synthesis + translation, repeated while the 2-qubit count drops.
    Gates stay on the qubits they were routed to, so the result needs no re-routing."""
    layout = routed._layout
    tidy = PassManager([
        BasisTranslator(SessionEquivalenceLibrary, list(basis_gates)),
        Optimize1qGatesDecomposition(basis=list(basis_gates)),
    ])
    best = routed
    for _ in range(rounds):
        c = PassManager([CommutativeCancellation(basis_gates=list(basis_gates))]).run(best)
        c = _resynthesize_blocks(c, verify, tol, on_unsupported)
        c = tidy.run(c)
        c._layout = layout
        if _two_q(c) < _two_q(best):
            best = c
        else:
            break
    return best


def compile_for_model_circuit(qc, coupling_map, basis_gates, entangling_basis="cx", seeds=DEFAULT_SEEDS,
                              commutation_start=True, do_polish=True, layout_search=True, return_info=False,
                              **kwargs):
    """Best of several PSF-Zero compiles of a small circuit (fewest 2-qubit gates, then 2-qubit depth).

    kwargs are forwarded to `psf_compile.compile_for_hardware()`."""
    def cfh(circ, seed):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pc.compile_for_hardware(circ, coupling_map=coupling_map, basis_gates=basis_gates,
                                           entangling_basis=entangling_basis, layout_search=layout_search,
                                           seed_transpiler=seed, **kwargs)

    if qc.num_qubits > SMALL_MAX_QUBITS or basis_gates is None:
        out = cfh(qc, seeds[0] if seeds else None)
        return (out, {"path": "fast"}) if return_info else out

    starts = [("input", qc)]
    if commutation_start:
        cc = PassManager([CommutativeCancellation()]).run(qc)
        if _two_q(cc) < _two_q(qc) or [i.operation.name for i in cc.data] != [i.operation.name for i in qc.data]:
            starts.append(("commuted", cc))
    tried = []
    best, best_key = None, None
    for sname, circ in starts:
        last = None
        for seed in seeds:
            out = cfh(circ, seed)
            if do_polish:
                out = polish(out, basis_gates)
            key = (_two_q(out), _depth2q(out))
            tried.append((sname, seed, key[0]))
            if best_key is None or key < best_key:
                best, best_key = out, key
            # identical result from two seeds in a row: the layout search fixed the layout, stop early
            sig = (key, tuple(out.layout.initial_index_layout()[:qc.num_qubits]))
            if sig == last:
                break
            last = sig
    info = {"path": "small", "tried": tried, "best": best_key, "version": AI_COMPILE_VERSION}
    return (best, info) if return_info else best
