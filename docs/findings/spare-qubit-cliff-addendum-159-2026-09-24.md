# Addendum 159 -- Settled: Addendum 157's 30% depth reduction is the euler_basis="ZSX" configuration, not PSF-Zero. A ZSX-configured Qiskit decomposer produces exactly PSF-Zero's depth and size on every tape, and synthesizes 5.6-7.6x (per tape; 5.64-7.60 from the CSV's unrounded medians) faster per block in this path (2026-09-24 night)

**Pre-registered in**:
`spare-qubit-cliff-addendum-158-preregistration-2026-09-24.md`. Raw
output observed as text in the conversation.

## 0. In one line

All four predictions hold. Arm C (`TwoQubitBasisDecomposer(CXGate(),
euler_basis="ZSX")`) matched PSF-Zero (Arm B) exactly -- depth 16, size 56,
6 CX -- on all 5 tapes, and their noisy TVDs differed by at most 0.0042.
Addendum 157's depth/size difference was therefore entirely the
unconfigured stand-in decomposer (Arm A), as that addendum suspected but
had not tested. At this scale PSF-Zero offers no advantage in this
connection; its per-block synthesis was also slower than the equivalently
configured Qiskit decomposer (median ~0.56-0.67 ms vs ~0.09-0.10 ms, 10
blocks per arm, no timing claim).

## 1. Results

| tape | arm | routed CX | depth | size | GPU diff | TVD noisy | TVD ideal | synth ms (median) | fallbacks |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | A Qiskit default | 6 | 23 | 84 | 5.94e-15 | 0.1333 | 0.0145 | 0.074 | -- |
| 0 | B PSF-Zero | 6 | 16 | 56 | 7.34e-13 | 0.1298 | 0.0145 | 0.564 | 0 |
| 0 | C Qiskit ZSX | 6 | 16 | 56 | 5.61e-15 | 0.1303 | 0.0145 | 0.088 | -- |
| 1 | A Qiskit default | 6 | 23 | 84 | 6.94e-15 | 0.0649 | 0.0199 | 0.094 | -- |
| 1 | B PSF-Zero | 6 | 16 | 56 | 7.99e-15 | 0.0635 | 0.0199 | 0.583 | 0 |
| 1 | C Qiskit ZSX | 6 | 16 | 56 | 7.02e-15 | 0.0635 | 0.0199 | 0.103 | -- |
| 2 | A Qiskit default | 6 | 23 | 84 | 5.11e-15 | 0.1398 | 0.0210 | 0.069 | -- |
| 2 | B PSF-Zero | 6 | 16 | 56 | 8.66e-15 | 0.1441 | 0.0210 | 0.591 | 0 |
| 2 | C Qiskit ZSX | 6 | 16 | 56 | 5.27e-15 | 0.1448 | 0.0210 | 0.086 | -- |
| 3 | A Qiskit default | 6 | 23 | 84 | 3.61e-15 | 0.0740 | 0.0238 | 0.069 | -- |
| 3 | B PSF-Zero | 6 | 16 | 56 | 2.28e-14 | 0.0666 | 0.0238 | 0.589 | 0 |
| 3 | C Qiskit ZSX | 6 | 16 | 56 | 3.66e-15 | 0.0708 | 0.0238 | 0.098 | -- |
| 4 | A Qiskit default | 6 | 23 | 84 | 1.22e-15 | 0.0593 | 0.0218 | 0.068 | -- |
| 4 | B PSF-Zero | 6 | 16 | 56 | 4.16e-15 | 0.0643 | 0.0218 | 0.674 | 0 |
| 4 | C Qiskit ZSX | 6 | 16 | 56 | 1.55e-15 | 0.0633 | 0.0218 | 0.089 | -- |

## 2. Scoring

**P1 (C's depth and size equal B's on every tape) -- CONFIRMED**, 16/56 on
all 5. Addendum 157 Section 3's attribution stands, now tested.

**P2 (CX count 6 in all arms) -- CONFIRMED.**

**P3 (|TVD_C - TVD_B| < 0.05) -- CONFIRMED**, 0.0000-0.0042.

**P4 (A and B reproduce Addendum 157 exactly) -- CONFIRMED** for depth,
size, TVD noisy and TVD ideal, to the printed precision. (Synthesis times
differ slightly between runs, as expected for timing; not part of P4.)

## 3. What this means

- In this connection, at this scale, PSF-Zero's block synthesizer is
  interchangeable with a correctly configured Qiskit decomposer on every
  metric that reaches the device, and slower per block.
- The prototype's stand-in (`reference_cpu_synthesize`, no
  `euler_basis`) is the only arm that is worse, and only in single-qubit
  gate count and depth -- the same configuration issue Addenda 114-116 found
  and fixed inside PSF-Zero itself. If the prototype connection is kept,
  its stand-in should use `euler_basis="ZSX"`.
- This agrees with this project's standing conclusion (Addendum 124 and
  others): benefits that come from re-synthesizing bound blocks are not
  PSF-Zero-specific; PSF-Zero's own established advantage is compile speed
  at the saturated large-scale layout cliff, which this 4-qubit / 5-qubit
  comparison does not reach.

## 4. Files

| File | What it is |
|---|---|
| `compare_zsx_arm.py` | this run's script (reuses compare_with_without_psf.py unchanged) |
| `compare_zsx_arm_2026-09-24.csv` | raw results, 15 rows; received and checked against every figure in Section 1 (0 mismatches) |
