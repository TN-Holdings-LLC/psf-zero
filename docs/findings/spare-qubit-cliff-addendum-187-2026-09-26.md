# Addendum 187 -- The parameter-polishing fix of Addendum 186 did nothing: PSF-Zero's error was unchanged to the last digit, because the core's parameters were already accurate -- the error enters when the circuit is built from them. V1 and V3 refuted; the fix is withdrawn (2026-09-26)

> **CORRECTION (Addendum 188)**: this run never loaded the fix -- the scripts imported an older `benchmarks/psf_compile.py`. The conclusions below (parameters accurate, error at circuit construction, fix withdrawn) are withdrawn. The valid validation is in Addendum 188: the fix works.

**Pre-registered in**:
`spare-qubit-cliff-addendum-186-preregistration-2026-09-26.md`. Run on WSL2
(home). The log begins: `psf_compile.py` 47,606 bytes, SHA-256
`a1a20781...`, VERSION 2026-09-26 (the fixed file, as registered). The
log's second hash, `2e32646d...`, is the raw `sha256sum` of
`deadline_compound_chain.py`; the lock in Addendum 183 is a normalized hash
(`8ffabd4e...`), so the two are not comparable -- the script's identity is
supported instead by Qiskit's per-lap values reproducing Addendum 184
exactly (V5). Log and both CSVs received as files.

## 0. In one line

V2, V4 and V5 hold; **V1 and V3 are refuted, and in the most informative
way**: PSF-Zero's errors were identical to the last printed digit to their
values before the fix -- the diagnostic's summary line and all ten listed
worst pairs, and every lap of the timed loop. The polishing step only runs when the residual of the
core's parameters, rebuilt by `_reconstruct`, exceeds 1e-13; that it never
changed anything means those parameters were already accurate, and the
1.76e-11 is introduced afterwards, when the circuit is built. Addendum 186
named exactly this outcome as "the fix is wrong-headed".

## 1. Scoring (Addendum 186)

| Prediction | Result |
|---|---|
| V1: max PSF-Zero error over 60 pairs <= 1e-13 | **REFUTED** -- 1.76e-11, unchanged; summary line and all ten listed worst pairs identical to the pre-fix run (the pre-fix CSV was not received, so the other 50 pairs are not compared) |
| V2: 36 tests pass | CONFIRMED -- 36 passed |
| V3: PSF-Zero per-pair distance <= 1e-12 every lap | **REFUTED** -- 1.76e-11 at lap 1 to 6.47e-10 at lap 10, identical to Addendum 184 |
| V4: speed kept (10/10 within 1 s, median <= 0.12 s) | CONFIRMED -- 10/10, median 0.061 s |
| V5: Qiskit untouched | CONFIRMED -- every per-lap value identical to Addendum 184 |

## 2. What this corrects

Addendum 185 attributed the error to "the Rust core's synthesis itself".
Its stage S1 called the whole block synthesizer -- the core's
decomposition followed by this file's Python circuit construction -- and did
not separate the two. The more careful statement is: the error arises in
the block synthesizer, and this run shows it is not in the core's returned
parameters. Addendum 185 is left as written; this is its correction.

## 3. Other observation

Across all 60 pairs, PSF-Zero's error correlates with small |c| (Spearman
-0.47), confirming the pattern Addendum 185 read off its ten worst pairs.
In `entangling_basis="cx"` mode the middle, entangling part of each block
is not built by PSF-Zero: `_cx_core_cached(a, b, c)` asks Qiskit's
`TwoQubitBasisDecomposer` to decompose the canonical gate. That is the
leading candidate; `diag_construction.py` separates it from the local
rotations and from the canonical-basis construction.

## 4. Disposition of the fix

Withdrawn: it never engaged, so it only adds code and a per-block check.
`psf_compile.py` returns to VERSION 2026-09-21 (kept as
`psf_compile_2026-09-21.py` before the run). The polishing function is
preserved in this project's records (`psf_compile_fix_2026-09-26/`) in case
a future path produces parameter-level error.

## 5. Files

| File | What it is |
|---|---|
| `fix_validation.txt` | raw log of the three validation runs |
| `diag_core_worst_pairs_2026-09-26.csv` | the diagnostic with the fixed file (identical errors) |
| `deadline_compound_chain_2026-09-26_fixrun.csv` | the timed loop with the fixed file |
| `diag_construction.py` | the next diagnostic |
