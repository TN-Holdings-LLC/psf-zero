# Addendum 186 -- Pre-registration: a residual-polishing step in PSF-Zero's block synthesizer, and a check that it brings PSF-Zero's precision to Qiskit's level without costing its speed (2026-09-26)

**Status: pre-registration of the fix's expected effect. The fix is written
(below); no validation run has been made.**

## 1. The problem (Addendum 185)

PSF-Zero's Rust core returns decompositions that are off by up to 1.76e-11
(phase-aligned Frobenius distance) on some inputs; the worst pairs cluster
near the Weyl-chamber face c = 0 (the ten worst all have |c| <= 0.11, the
worst c = -0.0022). Qiskit's decomposer stays below 2e-13 on the same
inputs. Re-synthesizing the same circuit lap after lap accumulates the
core's error linearly (Addendum 184: 6.47e-10 after 10 laps; Qiskit
~4e-13).

## 2. The fix (`psf_compile.py`, VERSION 2026-09-26)

`_refine_decomposition()`: after the core returns (a, b, c), the two local
ZYZ triples on each side and the global phase -- 16 real parameters -- the
residual `_reconstruct(params) - U_target` (the file's own rebuild of the
gate about to be emitted) is computed. If its norm exceeds 1e-13, up to three
Gauss-Newton steps on the 16 parameters (forward-difference Jacobian, step
1e-7, least-squares solve; a step is kept only if it reduces the residual)
polish the parameters; the circuit is then built from the polished values,
and the verification value is re-derived for them. Below 1e-13 nothing
changes. The Rust core is untouched. A counter
(`refine_count`, `refine_max_before`, `refine_max_after`) records use.

Checked before any validation run, in isolation (numpy only, the file's own
`_reconstruct`): a decomposition perturbed by 1e-11 in every parameter was
polished from 3.3e-11 to 7.5e-16 (c = 0.2), from 5.4e-11 to 5.2e-16
(c = 0.0022) and from 6.0e-11 to 5.7e-16 (c = 0); an exact decomposition was
left unchanged.

Diff against VERSION 2026-09-21: 76 lines added, 2 changed (version
strings). New file: 47,606 bytes, SHA-256
`a1a207813d8bfd03b969c5b224ec08d99086b19811704129b64a1c6174122baf`.
The old file is kept as `psf_compile_2026-09-21.py`.

## 3. Validation plan (scripts unchanged)

1. `diag_core_worst_pairs.py` (Addendum 185's follow-up diagnostic) with
   the new `psf_compile.py`.
2. The six connection test suites (36 tests, Addendum 166).
3. `deadline_compound_chain.py`, hash-locked in Addendum 183
   (`8ffabd4e...`), `--laps 10`, unchanged.

## 4. Pre-registered predictions

**V1 (precision fixed at the source).** In `diag_core_worst_pairs.py`, the
maximum PSF-Zero error over the 60 pairs is <= 1e-13 (was 1.76e-11).
**If V1 fails, the error is not in the core's parameters (it would have to
be in the circuit construction), and the fix is wrong-headed -- reported as
such.**

**V2 (nothing broken).** All 36 tests pass.

**V3 (drift gone in the real loop).** In `deadline_compound_chain.py`,
PSF-Zero's maximum per-pair distance is <= 1e-12 at every lap, at both
spares (was 1.76e-11 at lap 1, 6.47e-10 at lap 10).

**V4 (speed kept).** PSF-Zero at spare 0 meets the 1 s deadline on 10/10
laps, and its median compile time is at most 0.12 s (twice the 0.061 s of
Addendum 184).

**V5 (Qiskit untouched).** Qiskit's per-lap distances are identical to
Addendum 184's, to the last digit (the fix does not touch Qiskit's path).
