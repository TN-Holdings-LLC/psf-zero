# Workplace exploration (2026-10-13): can past hardware data show that ESP is no better than chance for choosing between compilations? On HW1's 18 usable points, analysed within each test, ESP's correlation with the device is -0.40 (95% range -0.77 to +0.16): no useful positive signal, while 88% of the differences between arms are real, not shot noise

> Committed as the owner pasted it into the home session's chat on 2026-10-13. The script it names
> (`esp_random_check.py`) was not part of the delivery and is not in the repository yet.

**Status: exploratory.** ESP was the pre-specified predictor of HW1 (H1, H2 in Addendum 454), so its result is a fair
test. The a14 estimate and the exposure measure were looked at after HW1's data (exposure only today), so their
numbers are optimistic. Script `esp_random_check.py`; inputs `data/2026-10-12/hw1/` and `req1_out.json`.

## Which past data can be used

| data | usable for this question? |
|---|---|
| 2026-09-04 jobs on ibm_marrakesh / ibm_fez (README section 7-8, `real-hardware.md`) | no: neither the circuits nor that day's calibration are in the repository, so ESP cannot be computed |
| 2026-10-12 Stage 2 XOR (Addendum 212) | not for ESP; the calibration-based Aer prediction for the pinned control was M = 0.969, the device gave 0.236 (one large miss, cause not established) |
| 2026-10-12 HW1 (Addendum 455) | yes: 20 circuits; 18 usable (tfim_n16 has F > 1, GHZ arms are identical) in 7 tests, 2-3 arms each |

## Method

Log F and log ESP (and the other predictors) minus their mean within each test, so only differences between
compilations of the same circuit count (11 degrees of freedom). Correlation of the two; exact permutation test that
shuffles the arms within each test (10,368 permutations); Fisher-z range as an approximation. Measurement noise from
the two repeats of each circuit.

## Results

- **The differences are real:** within-test variance of log F 0.0111, of which the repeat noise of a two-repeat mean
  is 0.0014; about 88% is real differences between arms.
- **ESP:** within-test r = -0.40; permutation p(r >= observed) = 0.91, two-sided 0.18; approximate 95% range -0.77 to
  +0.16.
- a14 estimate (exact, idle included): r = +0.42, permutation p = 0.003 (Fisher range -0.15 to +0.78).
- Exposure, sum of (window / T2) over used qubits: r = +0.58, permutation p = 0.005 (range +0.07 to +0.85).

## Reading

1. "ESP is random" cannot be proved: no test can show a correlation of exactly zero, and the data do not even
   exclude a moderately negative one.
2. What the data do show, at the 95% level: within a test, ESP's correlation with the device is at most about
   +0.16. For choosing between compilations of the same circuit, ESP has no useful positive predictive value on this
   device and day, and this is not because the device's differences were noise.
3. The two time-aware predictors point the right way, but were chosen after seeing the data; they need a
   pre-registered test with more pairs (20-30 pairs would narrow the ranges to about +-0.35).

## Addition (same day): the same data expressed as probabilities

Flat prior on the Fisher-z of the within-test correlation (11 degrees of freedom); Bayes factor with a uniform prior
on rho > 0 ("helps") against rho = 0; replication = a new test with 25 pairs; hit rate = winner right over all arm
pairs within a test (15 pairs), uniform prior.

| | ESP | a14 (post hoc) | exposure (post hoc) |
|---|---|---|---|
| P(true correlation > 0) | 0.08 | 0.93 | 0.99 |
| P(true correlation > 0.2) | 0.02 | 0.79 | 0.94 |
| Bayes factor, helps vs none | 0.16 (about 6 to 1 for "none") | 1.6 | 5.5 |
| P(a 25-pair replication shows r > 0) | 0.13 | 0.89 | 0.96 |
| winner right | 7 of 15 | 13 of 15 | - |
| P(hit rate > 0.6) | 0.14 | 0.98 | - |

The ESP column rests on a predictor fixed before the data; the others were chosen after the data and are optimistic.
