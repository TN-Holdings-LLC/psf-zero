# Exploratory candidate psf_compile 2026-10-09.c28 (not pre-registered, not proposed for adoption)

Release 2026-10-07.1 with changelog items 53 (candidate c26), 56 and 57 (part a). Tried once, on nine development
tests, with nothing predicted: [Addendum 421](../../docs/findings/spare-qubit-cliff-combined-383.md), data in
[`data/2026-10-09/c28_try/`](../../data/2026-10-09/c28_try/), script [`benchmarks/c28_try.py`](../../benchmarks/c28_try.py).

- **Item 56 (feasibility first).** A candidate is used only if item 39's exactness check passes, and that check cannot
  be made on a circuit of more than `EXACT_MAX_OPS` instructions or more than `RESYNTH_MAX_QUBITS` touched qubits.
  Where counts show this before any matrix is formed, the candidate is not built or estimated, and the circuit in
  hand is kept: the release's output, by construction.
- **Item 57a.** `excitation_cost` and `hybrid_cost` form the 4x4 reduced state of a two-qubit gate's pair in one
  product over the state (`_rho_pair`), instead of two one-qubit passes. The same numbers summed in another order:
  values may differ in the last bits, so this part is not identity by construction.

This file is kept as tried (sha256 `cf10b54700ce258a4a3e414a973aec17530caf059c37cdebcf8a82e9a245af6f`). A
pre-registered candidate carrying item 56 without item 57a is to follow.
