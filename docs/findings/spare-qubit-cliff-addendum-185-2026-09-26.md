# Addendum 185 -- Diagnosis: PSF-Zero's drift in Addendum 184 comes from the Rust core's synthesis itself (1.76e-11 on the worst pair, identical every call), not from the pipeline; the PennyLane round trip adds nothing (2026-09-26)

**Status**: exploratory diagnosis, not a pre-registered test. Run on WSL2
(home) with the same `psf_compile.py` Addendum 184 used (44,535 bytes,
VERSION 2026-09-21, SHA-256 `66a705ea...`, identical to the copy read at
home). Raw output observed as text in the conversation.

## 0. In one line

Rebuilding Addendum 184's 60 pairs exactly and switching the pipeline on
stage by stage, the full-lap stage (S4) reproduces Addendum 184's PSF-Zero
numbers to the printed digits at every lap, and the synthesizer alone (S1)
already carries the whole lap-1 error, 1.76e-11, growing by exactly that
amount every lap. The PennyLane round trip adds nothing (S3 = S4). Qiskit's
default carries 3.4e-13 at lap 1 and 7.7e-15 per lap.

## 1. Results (maximum over the 60 pairs; phase-aligned Frobenius distance)

| stage | lap 1 | lap 2 | lap 3 | lap 5 | lap 10 | per-lap growth |
|---|---:|---:|---:|---:|---:|---:|
| S1 PSF-Zero synthesizer only (Rust core) | 1.76e-11 | 3.52e-11 | 5.27e-11 | 8.78e-11 | 1.76e-10 | 1.8e-11 |
| S2 `psf_compile.compile()` | 1.76e-11 | 1.90e-11 | 3.80e-11 | 2.81e-11 | 2.01e-10 | 2.0e-11 |
| S3 S2 + level-1 translation to (cz, rz, sx, x) | 1.76e-11 | 5.54e-11 | 3.64e-11 | 2.21e-10 | 6.47e-10 | 7.0e-11 |
| S4 S3 + PennyLane round trip (one Addendum 184 lap) | 1.76e-11 | 5.54e-11 | 3.64e-11 | 2.21e-10 | 6.47e-10 | 7.0e-11 |
| Q4 Qiskit opt 3 + PennyLane round trip (control) | 3.37e-13 | 3.38e-13 | 3.39e-13 | 3.40e-13 | 4.06e-13 | 7.7e-15 |

## 2. Reading

- **The diagnosis is faithful**: S4 equals Addendum 184's PSF-Zero column
  (1.76e-11, 5.54e-11, 3.64e-11, 1.66e-10, 2.21e-10, 3.51e-10, 3.32e-10,
  4.62e-10, 5.17e-10, 6.47e-10) lap for lap.
- **The source is the Rust core.** Its output for the worst pair is
  1.76e-11 away from its input on the first call, and each further call adds
  the same amount in the same direction (S1 is exactly linear: 1.76, 3.52,
  5.27, ... e-11). A deterministic per-call error, not noise.
- **The pipeline does not create error.** S3 = S4 exactly: the PennyLane
  <-> Qiskit conversion (fixed in Addenda 165-166) contributes nothing
  measurable. The level-1 translation does not add error of its own either;
  it changes the circuit's form, so the next lap's `compile()` re-synthesizes
  from a different starting circuit and the core's error is added afresh
  each lap (growth 7e-11 per lap in S3 versus 1.8e-11 in S1).
- **The gap to Qiskit is in the core's per-call precision**: 1.76e-11 versus
  3.37e-13 at lap 1, about 50x. Addendum 182's smaller PSF-Zero figure
  (1.4e-13 per lap) came from 3 blocks on 4 qubits; here the maximum is over
  60 pairs, so the core's error is input-dependent and reaches ~1e-11 on
  some inputs.

## 3. Next

1. Characterize the worst pairs: whether large core errors coincide with
   near-degenerate inputs, where the core's degeneracy handling uses
   loose tolerances (`GROUP_TOL_CANDIDATES` starting at 1e-4,
   `ANGLE_SUM_TOL` = 1e-6 in `lib.rs`).
2. Candidate fix, least likely to break anything else: a final
   residual-correction step in the core (or in the Python wrapper) that
   measures the remaining error of the synthesized circuit and absorbs it,
   then re-run Addendum 183's hash-locked test with a pre-registered
   prediction that PSF-Zero's drift falls to Qiskit's level while its speed
   is unchanged.

Physically the error is negligible (as a fidelity loss, of order its
square, ~1e-22 per call); the point is that Qiskit achieves ~50x better
precision on the same inputs, so PSF-Zero can too.

## 4. Files

| File | What it is |
|---|---|
| `diag_psf_drift.py` | the diagnostic script |
| `diag_psf_drift.txt` | its raw output (received; matches Section 1) |
