# Addendum 182 -- Compound test result: the fixed pipeline never changes a circuit's meaning over 20,000 laps (the broken control is caught at lap 1), but floating-point error accumulates linearly in every arm, including Qiskit's own default -- R1 refuted; PSF-Zero's per-lap error is about 15x Qiskit's (2026-09-25)

**Pre-registered in**:
`spare-qubit-cliff-addendum-181-preregistration-2026-09-25.md`. Run on
WSL2 (home), Python 3.12.13, Qiskit 2.5.2, PennyLane 0.45.1, four arms in
parallel processes (10-12 min each). All four logs and checkpoint CSVs
received as files and read from disk; every figure below recomputed from
the CSVs. The hash re-check before running is not visible in the received
logs -- recorded as not shown.

## 0. In one line

R2, R3 and R4 hold; **R1 is refuted**. No lap in any fixed arm changed the
circuit's meaning, the operation count never grew, nothing failed, and the
deliberately broken control arm was caught at lap 1 (distance 5.63). But in
all three fixed arms -- Qiskit's ZSX decomposer, PSF-Zero, and Qiskit's own
optimization level 3 -- the distance from the original operator grew
**linearly** with the number of laps (alpha 0.93, 1.03, 1.03), reaching
5.1e-11, 2.75e-9 and 1.4e-10 after 20,000 laps. PSF-Zero's per-lap error is
about 15 times Qiskit's, so its drift is about 50 times larger.

## 1. Results

| arm | alpha | delta_1 | delta_20000 | per-lap median | delta_20000 / 20000 | ops lap 1 -> 20000 | fallbacks |
|---|---:|---:|---:|---:|---:|---|---:|
| A Qiskit ZSX (without PSF-Zero) | +0.927 | 1.20e-14 | 5.13e-11 | 8.98e-15 | 2.56e-15 | 98 -> 98 | -- |
| P PSF-Zero | +1.031 | 8.95e-13 | 2.75e-09 | 1.40e-13 | 1.38e-13 | 98 -> 98 | 0 |
| Q3 Qiskit default (opt 3) | +1.028 | 1.38e-14 | 1.43e-10 | 1.19e-14 | 7.17e-15 | 86 -> 86 | -- |
| C control (pre-fix bug) | +0.000 | 5.63e+00 | 5.63e+00 | 8.22e-15 | -- | 98 -> 98 | -- |

Full trajectories: `compound_chain_{A,P,Q3,C}_checkpoints_2026-09-25.csv`;
figure: `compound_chain_2026-09-25.png` (log-log, with slope-0.5 and slope-1
guides).

## 2. Scoring

- **R1 (alpha < 0.75 and delta_20000 < 1e-10 for A, P, Q3) -- REFUTED.**
  alpha >= 0.93 in all three; delta_20000 below 1e-10 only for A.
- **R2 (control caught, delta_1 > 0.1) -- CONFIRMED.** 5.63 from lap 1,
  constant thereafter: the meaning check detects a real bug immediately.
- **R3 (no growth in circuit size) -- CONFIRMED.** Operation counts
  unchanged from lap 1 to lap 20,000 in every arm.
- **R4 (pipeline healthy) -- CONFIRMED.** Every lap completed in every arm;
  no SWAP was needed (the check would have raised); 0 PSF-Zero fallbacks.

## 3. What the refutation means

**Not a bug in the pipeline.** A bug that changes meaning looks like arm C:
a large distance from lap 1. The fixed arms stay at 1e-9 to 1e-11 after
20,000 laps -- far below anything physical (as a fidelity loss, of order
delta squared, below 1e-17).

**The pre-registration's reading of alpha was too strong.** Section 3 of
Addendum 181 said alpha >= 0.75 would indicate "a hidden bug or bias". The
linear growth is what iterating a deterministic, slightly lossy map near a
fixed input produces: each lap's input differs from the last by ~1e-14, so
each lap makes nearly the same rounding error in nearly the same direction,
and those errors add coherently rather than cancelling. That Qiskit's own
default (Q3) shows the same slope is consistent with this being a property
of repeated floating-point computation, not of any one component. This is
recorded as an error in the pre-registration's interpretation, not moved
after the fact: the prediction as written is refuted.

**One PSF-Zero-specific finding.** PSF-Zero's per-lap error (median
1.40e-13) is about 15 times Qiskit's decomposer's (8.98e-15), and because
it too accumulates linearly, its drift after 20,000 laps is about 50 times
larger (2.75e-9 vs 5.13e-11). This agrees with Addendum 157, where the
real-GPU check's difference was 7.34e-13 for PSF-Zero against ~5e-15 for
Qiskit. PSF-Zero's synthesis is exact to about 1e-13 per call in this
metric, Qiskit's to about 1e-14 -- a real, measurable, and improvable
difference, though far below any physical consequence.

## 4. What this means

The pipeline fixed on 2026-09-24 is sound in the sense that matters: over
20,000 feedback laps it never changed what the circuit does, never grew
the circuit, and never failed, while the test demonstrably catches the kind
of bug that was found and fixed. What does accumulate is ordinary
floating-point drift, in every arm, at levels nine orders of magnitude
below physical relevance -- with PSF-Zero's synthesis about one order of
magnitude less precise per call than Qiskit's.

## 5. Files

| File | What it is |
|---|---|
| `compound_pipeline_chain.py` | the script (hash-locked in Addendum 181) |
| `compound_chain_A_checkpoints_2026-09-25.csv` | arm A checkpoints |
| `compound_chain_P_checkpoints_2026-09-25.csv` | arm P checkpoints |
| `compound_chain_Q3_checkpoints_2026-09-25.csv` | arm Q3 checkpoints |
| `compound_chain_C_checkpoints_2026-09-25.csv` | arm C checkpoints |
| `chain_A.txt`, `chain_P.txt`, `chain_Q3.txt`, `chain_C.txt` | raw logs |
| `compound_chain_2026-09-25.png` | the figure |
