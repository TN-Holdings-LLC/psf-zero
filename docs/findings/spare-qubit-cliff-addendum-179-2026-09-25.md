# Addendum 179 -- Depth sweep result: the "PSF-Zero wins with depth" hypothesis is refuted -- no win for either side at any depth on any of 4 backends; the comparison's own sanity checks all hold (2026-09-25)

**Pre-registered in**:
`spare-qubit-cliff-addendum-177-preregistration-2026-09-25.md`. Run on
WSL2 (home), Python 3.12.13, Qiskit 2.5.2. Script hash re-checked on the
machine before running: `cbcb93fa...` (matches). Raw output observed as
text; every figure below recomputed from the CSV.

## 0. In one line

**H refuted.** P (PSF-Zero) won at 0 of 4 backends at L = 8 and at L = 16;
no (backend, L) cell produced a win for either side under the
pre-registered rule (|d| >= 0.02 and >= 3 SE). The largest paired
difference anywhere was d = +0.0043 (FakeBrisbane, L = 16). All four
sanity predictions (S1-S4) hold, so the null is a property of the
compilers, not of a broken comparison. As expected in advance (Section 3
of Addendum 177).

## 1. Results (mean over 10 circuits; d = mean paired TVD_Q3 - TVD_P)

| backend | L | d | SE | verdict | size Q3 / P / Z | depth Q3 / P / Z |
|---|---:|---:|---:|---|---|---|
| FakeBrisbane | 1 | +0.0001 | 0.0006 | none | 84 / 78 / 78 | 24 / 22 / 22 |
| FakeBrisbane | 2 | +0.0011 | 0.0006 | none | 114 / 103 / 103 | 42 / 36 / 36 |
| FakeBrisbane | 4 | +0.0027 | 0.0017 | none | 204 / 182 / 182 | 78 / 70 / 70 |
| FakeBrisbane | 8 | +0.0018 | 0.0011 | none | 384 / 340 / 340 | 150 / 134 / 134 |
| FakeBrisbane | 16 | +0.0043 | 0.0011 | none | 744 / 624 / 624 | 294 / 230 / 230 |
| FakeFez | 1 | +0.0001 | 0.0002 | none | 68 / 76 / 76 | 20 / 24 / 24 |
| FakeFez | 2 | -0.0008 | 0.0003 | none | 90 / 102 / 102 | 34 / 42 / 42 |
| FakeFez | 4 | -0.0005 | 0.0004 | none | 156 / 180 / 180 | 62 / 78 / 78 |
| FakeFez | 8 | +0.0005 | 0.0005 | none | 288 / 336 / 336 | 118 / 150 / 150 |
| FakeFez | 16 | -0.0009 | 0.0006 | none | 551.9 / 648 / 648 | 230 / 294 / 294 |
| FakeKingston | 1 | +0.0002 | 0.0002 | none | 68 / 76 / 76 | 20 / 24 / 24 |
| FakeKingston | 2 | -0.0001 | 0.0002 | none | 90 / 102 / 102 | 34 / 42 / 42 |
| FakeKingston | 4 | +0.0002 | 0.0002 | none | 156 / 180 / 180 | 62 / 78 / 78 |
| FakeKingston | 8 | +0.0001 | 0.0001 | none | 288 / 336 / 336 | 118 / 150 / 150 |
| FakeKingston | 16 | -0.0000 | 0.0004 | none | 551.9 / 648 / 648 | 230 / 294 / 294 |
| FakeMarrakesh | 1 | -0.0000 | 0.0003 | none | 68 / 76 / 76 | 20 / 24 / 24 |
| FakeMarrakesh | 2 | -0.0002 | 0.0004 | none | 90 / 102 / 102 | 34 / 42 / 42 |
| FakeMarrakesh | 4 | -0.0000 | 0.0003 | none | 156 / 180 / 180 | 62 / 78 / 78 |
| FakeMarrakesh | 8 | +0.0002 | 0.0005 | none | 288 / 336 / 336 | 118 / 150 / 150 |
| FakeMarrakesh | 16 | -0.0000 | 0.0003 | none | 551.9 / 648 / 648 | 230 / 294 / 294 |

Mean noisy TVD at L = 16 ranged from about 0.06 (Marrakesh) to 0.17
(Brisbane): noise did accumulate enough for a difference to show, had one
existed.

## 2. Scoring

- **H (P wins on >= 3 of 4 backends at both L = 8 and L = 16)**: P won on
  0 at L = 8 and 0 at L = 16 -> **REFUTED** (rule: at most 1 at L = 16).
- **S1 -- CONFIRMED.** P and Z identical in two-qubit count, depth and size
  on 200 of 200 circuits; 0 PSF-Zero fallbacks.
- **S2 -- CONFIRMED.** Q3's two-qubit count equals P's on 200 of 200.
- **S3 -- CONFIRMED.** Mean noisy TVD non-decreasing from L = 1 to 16 for
  all 12 (backend, arm) series.
- **S4 -- CONFIRMED.** Noiseless TVD at most 0.028 (< 0.06).
- Q3, P and Z used the same physical layout on 200 of 200 circuits, as the
  design intended.

## 3. Observations (not pre-registered)

- **Circuit size depends on the device's native gate, in opposite
  directions.** On the CZ devices (Fez, Marrakesh, Kingston), Q3 --
  synthesizing directly in CZ -- is smaller and shallower (L = 16: size 552
  vs 648, depth 230 vs 294), the cost of P and Z emitting CX that level 1
  then translates, as Addendum 177 anticipated. On the ECR device
  (Brisbane) the direction reverses: P and Z are smaller and shallower (624
  vs 744, 230 vs 294).
- **Neither size difference moves the noisy score.** The differences are
  single-qubit gates; the two-qubit count, which dominates the error, is
  identical in every circuit (S2).
- **Brisbane's small, consistent lean toward P (up to +0.0043, 3.9 SE at
  L = 16) is not PSF-Zero-specific**: Z, a plain Qiskit decomposer given
  the same treatment, has the same mean TVD (0.1667 vs P's 0.1667 at
  L = 16). It is a property of the CX-then-translate path versus Q3's
  direct ECR synthesis, and in any case an order of magnitude below the
  pre-registered 0.02 bar.
- **Equivalence, reported after the fact (not pre-registered; see
  Addendum 177's closing note)**: across all 20 cells, |d| <= 0.0043. A
  formal equivalence criterion is part of the pre-registration of the next
  experiment (Addendum 178), not of this one.

## 4. What this means

Where the layout cliff does not occur, PSF-Zero's exact synthesis and
Qiskit's default compilation give circuits of the same noisy quality, at
every depth tested, on both native-gate families. This is now shown with
depth varied systematically, closing the gap Addendum 177 named. Together
with Addenda 157, 159 and 172: PSF-Zero's value is not better output
quality; it is producing the same quality faster where Qiskit is slow --
which is what Addendum 178 (timed, deadline-scored) is designed to measure.

## 5. Files

| File | What it is |
|---|---|
| `psf_vs_qiskit_depth_sweep.py` | the script (hash-locked in Addendum 177) |
| `psf_vs_qiskit_depth_sweep_2026-09-25.csv` | raw results, 600 rows |
