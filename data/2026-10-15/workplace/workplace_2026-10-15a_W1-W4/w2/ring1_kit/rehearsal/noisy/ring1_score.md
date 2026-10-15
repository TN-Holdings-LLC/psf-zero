# RING1 scores (2026-10-15T00:04:21+00:00 (2026-10-15 02:04:21 CEST))

device fake_fez; calibration None; job rehearsal; usage {'quantum_seconds': 0}

Per point: z = P(0) - P(1) of logical qubit 0 over both repeats' pooled counts; correct if sign(z) = y (z = 0 counts half); margin = y z (the correct class's probability minus the other's); F = normalised linear XEB against the exact distribution.

| n | L | arm | points | accuracy | margin (SE) | ideal margin | F (mean) | 2q | duration us | read on |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 4 | A18 | 24 | 0.875 | 0.5515 (0.0664) | 0.6359 | 0.8675 | 28.0 | 2.33 | 87 |
| 4 | 4 | QK3 | 24 | 0.875 | 0.5529 (0.0672) | 0.6359 | 0.8631 | 37.0 | 3.95 | 117, 126 |
| 4 | 8 | A18 | 24 | 0.875 | 0.4686 (0.0503) | 0.6118 | 0.7710 | 56.0 | 4.65 | 87 |
| 4 | 8 | QK3 | 24 | 0.875 | 0.4769 (0.0504) | 0.6118 | 0.7378 | 77.0 | 8.19 | 117, 126 |
| 6 | 4 | A18 | 24 | 0.958 | 0.5491 (0.0449) | 0.6625 | 0.7827 | 60.0 | 3.63 | 86 |
| 6 | 4 | QK3 | 24 | 0.958 | 0.4910 (0.0412) | 0.6625 | 0.7171 | 70.0 | 6.98 | 144 |
| 6 | 8 | A18 | 24 | 0.958 | 0.5671 (0.0422) | 0.7248 | 0.6473 | 120.0 | 7.26 | 86 |
| 6 | 8 | QK3 | 24 | 0.958 | 0.3799 (0.0274) | 0.7248 | 0.4737 | 142.0 | 15.08 | 144 |

## Predictions

- RA (pooled margin ratio A18 / QK3 >= 1.05): 0.5341 / 0.4752 = 1.124 -> CONFIRMED
- RA2 (reported): A18's margin higher in 2 of 4 configurations (not independent: the configurations of one n share each arm's placement)
- RB (pooled accuracy A18 >= QK3 over 96 points each): 0.917 vs 0.917 -> CONFIRMED
- RC (reported): F ratio A18 / QK3 per configuration: 1.005, 1.045, 1.091, 1.366; geometric mean 1.119
- RD (reported): paired margin difference A18 - QK3 over 96 points: +0.0589 (SE 0.0096)
- RE (control ghz_n6, the same circuit twice: |F ratio - 1| <= 0.10): 0.9445 / 0.9432 -> CONFIRMED
- **RING1 (RA and RB confirmed): CONFIRMED**
