# Addendum 189 -- Pre-registration: a faster polishing step (closed-form Jacobian, cheaper stopping rule) and a single copy of `psf_compile.py`; checks that precision is kept and most of the lost speed returns (2026-09-26)

**Status: pre-registration. The new file is written; no validation run has
been made.**

## 1. Why

Addendum 188: the polishing step fixed PSF-Zero's precision (per-call error
1.76e-11 -> 6.2e-14) but raised the median compile time at spare 0 from
0.061 s to 0.135 s. Two causes were found in the code: the Jacobian was
built from 16 forward differences, and the loop stopped only at 1e-15 --
machine precision, rarely reached -- so almost every polished block ran all
three iterations. And the mistake of Addendum 187 came from two copies of
`psf_compile.py` (repository root and `benchmarks/`).

## 2. Changes

**`psf_compile.py`, VERSION 2026-09-26.2** (49,442 bytes, SHA-256
`c6621f7136f3cad7af2ca708bab35b2b62365835c78ec96e1c0bb72b559ab5ca`; against
VERSION 2026-09-21: 117 lines added, 2 changed):
- `_reconstruct_with_jacobian()`: the 16 derivatives in closed form (core
  angles, the four ZYZ triples, global phase).
- The residual check uses the plain `_reconstruct`; the Jacobian is
  computed only when a step is taken; iteration stops once the residual is
  <= 1e-14 or a step fails to halve it.

Checked in isolation before any run (numpy, the file's own functions): the
closed-form Jacobian matches central differences to 3.2e-10 (the size of
the finite-difference error itself); on 200 decompositions perturbed by
1e-11, the polished residual is at most 1.7e-15 (previous version 1.5e-15);
time per polished block 0.588 ms (previous 1.590 ms); time per check when
no polishing is needed 0.067 ms (previous 0.071 ms).

**One copy of `psf_compile.py`**: `benchmarks/psf_compile.py` is removed.
The package is installed editable from the repository root, so
`import psf_compile` resolves to the root file from any directory. Every
validation log records which file was loaded, with the scripts' own import
order.

## 3. Predictions (same scripts as Addendum 188)

**W0 (one copy).** With `benchmarks/psf_compile.py` removed, the scripts'
import order loads the root `psf_compile.py`, VERSION 2026-09-26.2.

**W1 (precision kept).** `diag_core_worst_pairs.py`: maximum PSF-Zero
error over the 60 pairs <= 1e-13.

**W2.** 36 tests pass.

**W3 (drift kept low).** `deadline_compound_chain.py` (hash-locked,
Addendum 183): PSF-Zero's maximum per-pair distance <= 1e-12 at every lap.

**W4 (speed returns).** PSF-Zero at spare 0: 10/10 laps within 1 s, median
compile time <= 0.10 s (Addendum 188: 0.135 s; before polishing: 0.061 s).

**W5.** Qiskit's per-lap values identical to Addendum 184.
