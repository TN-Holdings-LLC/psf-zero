# RING1 scores (2026-10-15T00:01:21+00:00 (2026-10-15 02:01:21 CEST))

device fake_fez; calibration None; job rehearsal; usage {'quantum_seconds': 0}

Per point: z = P(0) - P(1) of logical qubit 0 over both repeats' pooled counts; correct if sign(z) = y (z = 0 counts half); margin = y z (the correct class's probability minus the other's); F = normalised linear XEB against the exact distribution.

| n | L | arm | points | accuracy | margin (SE) | ideal margin | F (mean) | 2q | duration us | read on |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 4 | A18 | 24 | 0.875 | 0.6421 (0.0760) | 0.6359 | 1.0032 | 28.0 | 2.33 | 87 |
| 4 | 4 | QK3 | 24 | 0.875 | 0.6330 (0.0776) | 0.6359 | 0.9953 | 37.0 | 3.95 | 117, 126 |
| 4 | 8 | A18 | 24 | 0.875 | 0.6162 (0.0655) | 0.6118 | 1.0062 | 56.0 | 4.65 | 87 |
| 4 | 8 | QK3 | 24 | 0.875 | 0.6167 (0.0657) | 0.6118 | 0.9996 | 77.0 | 8.19 | 117, 126 |
| 6 | 4 | A18 | 24 | 1.000 | 0.6634 (0.0533) | 0.6625 | 1.0051 | 60.0 | 3.63 | 86 |
| 6 | 4 | QK3 | 24 | 0.958 | 0.6623 (0.0536) | 0.6625 | 1.0038 | 70.0 | 6.98 | 144 |
| 6 | 8 | A18 | 24 | 0.958 | 0.7299 (0.0539) | 0.7248 | 0.9986 | 120.0 | 7.26 | 86 |
| 6 | 8 | QK3 | 24 | 0.958 | 0.7252 (0.0544) | 0.7248 | 0.9979 | 142.0 | 15.08 | 144 |

## Predictions

- RA (pooled margin ratio A18 / QK3 >= 1.05): 0.6629 / 0.6593 = 1.005 -> REFUTED
- RA2 (reported): A18's margin higher in 3 of 4 configurations (not independent: the configurations of one n share each arm's placement)
- RB (pooled accuracy A18 >= QK3 over 96 points each): 0.927 vs 0.917 -> CONFIRMED
- RC (reported): F ratio A18 / QK3 per configuration: 1.008, 1.007, 1.001, 1.001; geometric mean 1.004
- RD (reported): paired margin difference A18 - QK3 over 96 points: +0.0036 (SE 0.0022)
- RE (control ghz_n6, the same circuit twice: |F ratio - 1| <= 0.10): 1.0000 / 1.0000 -> CONFIRMED
- **RING1 (RA and RB confirmed): NOT CONFIRMED**
