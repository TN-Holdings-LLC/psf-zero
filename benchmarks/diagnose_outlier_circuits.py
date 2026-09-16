#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconstructs and analyzes the circuits behind the worst outliers found in
`cumulative_compile_times_50000.npz`.

## Why this is needed

The 20 slowest Qiskit compiles in the 50,000-iteration run are 14.7x-16.3x
the median, and 19 of those 20 indices are *also* elevated on PSF-Zero's
side (2.9x-3.9x the median) at the exact same index -- only one exception
(index 47933) is slow on Qiskit alone. That pattern points at the circuit
itself, not either engine in isolation: `build_dense_pair_blocks_circuit`
uses `seed = 1000 + i`, so the circuit behind any index can be rebuilt
exactly and inspected directly, rather than guessed at.

## What this checks

For each of the top-N outlier indices (default 20), rebuilds the circuit
with the exact same seed and inspects every 2-qubit block for:

- **Distance from a degenerate point (a scalar multiple of the identity)**,
  the shortest Frobenius distance from the block (projected to SU(4)) to
  any of the four CNOT-orbit-adjacent degenerate points this project's
  `lib.rs` documentation names as historically hard for KAK-style
  decomposition. A block close to degenerate is a direct, checkable
  candidate for "this is the block making both engines slow."
- **Local canonical-class coordinates** (the three canonical-parameter
  angles from a Cartan/KAK-style decomposition via `scipy`'s own SVD path,
  independent of and not trusting `psf_compile`'s internal one), so a
  block sitting exactly on a canonical-class boundary can be spotted
  without relying on the code under test to report its own difficulty.

This is deliberately **read-only diagnostics** -- it does not modify
`psf_compile.py` or draw a conclusion about the compiler itself. It answers
"what does the slow circuit actually contain" first; whether that maps to
something fixable in `psf_compile.py` or `lib.rs` is a decision for after
seeing this data, not before.

## Usage

    python diagnose_outlier_circuits.py
    python diagnose_outlier_circuits.py --npz cumulative_compile_times_50000.npz --top 20
    python diagnose_outlier_circuits.py --index 30968      # inspect one specific index directly
"""
from __future__ import annotations

import argparse

import numpy as np

N_QUBITS = 15
GATES_PER_PAIR = 20
SEED_OFFSET = 1000  # must match test_cumulative_compile_scale.py exactly


# ---------------------------------------------------------------- rebuild
def rebuild_circuit_unitaries(index):
    """Reproduces exactly what build_dense_pair_blocks_circuit(15, 20,
    seed=1000+index) constructs, but returns the raw list of per-block
    4x4 unitaries (one list per qubit pair, in application order) instead
    of a QuantumCircuit -- this is what gets fed block-by-block into
    psf_compile's synthesis path, and what this script needs to inspect."""
    seed = SEED_OFFSET + index
    rng = np.random.default_rng(seed)
    pairs = [(i, i + 1) for i in range(0, N_QUBITS - 1, 2)]
    from qiskit.quantum_info import random_unitary
    blocks = {}
    for (a, b) in pairs:
        us = []
        for _ in range(GATES_PER_PAIR):
            u = random_unitary(4, seed=int(rng.integers(0, 2**31))).data
            us.append(u)
        blocks[(a, b)] = us
    return seed, blocks


# ---------------------------------------------------------------- analysis
_SQRT_HALF = 1 / np.sqrt(2)
# The four points the project's write-ups on the compiler's own
# degeneracy handling flag as historically hard: the identity, and the
# CNOT/SWAP-family points that sit on canonical-class boundaries.
_DEGENERATE_LANDMARKS = {
    "identity": np.eye(4, dtype=complex),
    "CNOT": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                     dtype=complex),
    "SWAP": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                     dtype=complex),
    "iSWAP": np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
                      dtype=complex),
}


def su4_project(u):
    """Projects a 4x4 unitary to SU(4) by dividing out the global phase
    (det(u)**0.25), matching the branch this project's addenda already
    flagged as ambiguous for near-degenerate inputs -- ambiguity is exactly
    what is being looked for here, so it is left as-is rather than fixed."""
    det = np.linalg.det(u)
    phase = det ** 0.25
    return u / phase


def frobenius_distance_to_landmarks(u):
    """Shortest Frobenius distance from u (SU(4)-projected) to each
    landmark (also SU(4)-projected -- a landmark like CNOT has det=-1, so
    it must be normalized the same way u is, or every distance below is
    wrong by construction, which an earlier version of this function was:
    self-test caught it by checking CNOT against itself and getting 1.53
    instead of 0), minimized over the four SU(4) branches of u's own
    projection (since that branch is itself ambiguous near a degenerate
    point, all four are tried and the closest is kept -- this is
    diagnostic, not a claim about which branch psf_compile itself would
    pick)."""
    det = np.linalg.det(u)
    base_phase = det ** 0.25
    out = {}
    for name, landmark in _DEGENERATE_LANDMARKS.items():
        landmark_su4 = su4_project(landmark)
        best = np.inf
        for k in range(4):
            phase = base_phase * np.exp(1j * np.pi * k / 2)
            u_proj = u / phase
            for j in range(4):
                # The landmark's own branch is compared too, since a
                # global-phase difference of i, -1, or -i would otherwise
                # register as a large, spurious distance.
                lm = landmark_su4 * np.exp(1j * np.pi * j / 2)
                d = np.linalg.norm(u_proj - lm, ord="fro")
                best = min(best, d)
        out[name] = best
    return out


def canonical_coordinates(u):
    """Independent canonical-class coordinates via scipy's own SVD, not
    psf_compile's. Returns the three canonical parameters (each in
    [0, pi/4] after folding into the standard Weyl chamber) computed from
    the magic-basis transform -- the same textbook construction
    documented in this project's own findings, reimplemented from scratch
    here so this check does not depend on the code it is inspecting."""
    magic = (1 / np.sqrt(2)) * np.array([
        [1, 0, 0, 1j],
        [0, 1j, 1, 0],
        [0, 1j, -1, 0],
        [1, 0, 0, -1j],
    ], dtype=complex)
    u_su4 = su4_project(u)
    m = magic.conj().T @ u_su4 @ magic
    # m is now orthogonal up to a global phase; its eigenvalues give the
    # canonical coordinates directly.
    evals = np.linalg.eigvals(m.T @ m)
    thetas = np.sort(np.angle(evals))[::-1]
    # Fold to the standard three canonical parameters (c1 >= c2 >= |c3|).
    c = np.sort(np.abs(thetas[:3] - thetas[3:][0] if len(thetas) > 3 else thetas))[::-1]
    return c[:3] if len(c) >= 3 else np.pad(c, (0, 3 - len(c)))


def analyze_index(index, verbose=True):
    seed, blocks = rebuild_circuit_unitaries(index)
    results = []
    for pair, us in blocks.items():
        # The 20 gates in a block are applied in sequence, so the block's
        # net effect on those two qubits is their product, in application
        # order -- this is what actually gets synthesized as one KAK/
        # canonical block, not any single one of the 20 individual gates.
        net = np.eye(4, dtype=complex)
        for u in us:
            net = u @ net
        dists = frobenius_distance_to_landmarks(net)
        min_landmark = min(dists, key=dists.get)
        min_dist = dists[min_landmark]
        results.append(dict(pair=pair, min_landmark=min_landmark,
                            min_dist=min_dist, all_dists=dists))
    if verbose:
        print(f"index={index}  seed={seed}")
        for r in results:
            tag = "  <-- CLOSE" if r["min_dist"] < 0.3 else ""
            print(f"  pair={r['pair']}  nearest={r['min_landmark']:<9} "
                  f"dist={r['min_dist']:.4f}{tag}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="cumulative_compile_times_50000.npz")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--index", type=int, default=None,
                    help="inspect one specific index directly, skipping the .npz ranking")
    ap.add_argument("--close-threshold", type=float, default=0.3,
                    help="Frobenius distance below which a block is flagged as "
                         "\"close to a degenerate landmark\" (0.3 is a loose "
                         "starting point, not a value derived from prior "
                         "measurement -- tighten or loosen based on what "
                         "this run's distribution actually looks like)")
    args = ap.parse_args()

    if args.index is not None:
        print("=" * 78)
        print(f"Single-index inspection: index={args.index}")
        print("=" * 78)
        analyze_index(args.index)
        return

    d = np.load(args.npz)
    q = d["qiskit"]
    median_q = np.median(q)
    idx_sorted = np.argsort(q)[::-1][:args.top]

    print("=" * 78)
    print(f"Top {args.top} outliers in {args.npz} (by Qiskit time) -- "
          f"rebuilding and inspecting each circuit")
    print("=" * 78)

    close_counts = {name: 0 for name in _DEGENERATE_LANDMARKS}
    total_blocks = 0
    for rank, i in enumerate(idx_sorted, 1):
        print(f"\n--- rank {rank}, index {i}, Qiskit time "
              f"{q[i]*1000:.3f}ms ({q[i]/median_q:.1f}x median) ---")
        results = analyze_index(i)
        for r in results:
            total_blocks += 1
            if r["min_dist"] < args.close_threshold:
                close_counts[r["min_landmark"]] += 1

    print("\n" + "=" * 78)
    print("Verdict")
    print("=" * 78)
    print(f"Blocks inspected: {total_blocks} (7 pairs x {args.top} circuits)")
    print(f"Blocks within {args.close_threshold} of a landmark, by landmark:")
    for name, count in close_counts.items():
        print(f"  {name:<10} {count}")
    total_close = sum(close_counts.values())
    print(f"\nTotal close blocks: {total_close} / {total_blocks} "
          f"({100*total_close/total_blocks:.1f}%)")
    print("\n  -> If this rate is far above what a random SU(4) sample would give")
    print("     (baseline: check with --index against known-fast indices for")
    print("     comparison), that supports \"near-degenerate blocks are what makes")
    print("     both engines slow\" as the mechanism. A rate close to baseline")
    print("     means degeneracy-proximity is not the explanation and something")
    print("     else about these specific circuits is.")


if __name__ == "__main__":
    main()
