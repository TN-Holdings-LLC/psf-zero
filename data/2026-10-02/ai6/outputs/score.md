# ai6 score

P0: PASS -- files 90 of 90 (missing []); noiseless infidelity max 1.9e-08 (<= 1e-6); too wide 0 of 12690; model circuits converted 153 (>= 150)

| set | device | A6/A5 | A6/C5 | A6F/A6 | A6/L3T | A5/L3T | C5/L3T | mean infidelity C5 / A5 / A6 / A6F / L3T |
|---|---|---|---|---|---|---|---|---|
| GAP | FakeAuckland | 1.000 | 0.967 | 1.018 | 1.024 | 1.024 | 1.059 | 0.3307 / 0.3198 / 0.3198 / 0.3257 / 0.3124 |
| GAP | FakeTorino | 1.000 | 0.963 | 1.002 | 0.984 | 0.984 | 1.022 | 0.1789 / 0.1723 / 0.1723 / 0.1726 / 0.1750 |
| GAP | FakeKingston | 1.000 | 0.945 | 1.006 | 0.983 | 0.983 | 1.040 | 0.0889 / 0.0840 / 0.0840 / 0.0845 / 0.0855 |
| MODEL | FakeAuckland | 1.000 | 0.731 | 1.284 | 0.740 | 0.741 | 1.013 | 0.0575 / 0.0420 / 0.0420 / 0.0540 / 0.0568 |
| MODEL | FakeTorino | 0.999 | 0.921 | 1.020 | 0.932 | 0.933 | 1.012 | 0.0211 / 0.0195 / 0.0194 / 0.0198 / 0.0208 |
| MODEL | FakeKingston | 1.000 | 0.914 | 1.021 | 0.934 | 0.934 | 1.022 | 0.0106 / 0.0097 / 0.0097 / 0.0099 / 0.0104 |

| set | device | A6/A5 | A6/C5 | A6/L3T | mean 2q C5 / A5 / A6 / A6F / L3T |
|---|---|---|---|---|---|
| F1 | FakeAuckland | 1.000 | 0.983 | 0.988 | 53.5 / 53.5 / 53.5 / 53.3 / 53.8 |
| F1 | FakeTorino | 1.000 | 0.925 | 0.991 | 59.2 / 53.7 / 53.7 / 53.7 / 53.7 |
| F1 | FakeKingston | 1.000 | 0.933 | 0.989 | 57.3 / 53.5 / 53.5 / 53.5 / 53.8 |
| F2 | FakeAuckland | 1.000 | 0.915 | 0.952 | 47.2 / 44.5 / 44.5 / 44.4 / 45.4 |
| F2 | FakeTorino | 1.000 | 0.962 | 0.976 | 48.4 / 45.5 / 45.5 / 45.5 / 46.4 |
| F2 | FakeKingston | 1.000 | 0.940 | 0.974 | 48.1 / 45.1 / 45.1 / 45.1 / 45.8 |
| F3 | FakeAuckland | 1.000 | 0.999 | 1.178 | 87.0 / 87.0 / 87.0 / 87.0 / 87.0 |
| F3 | FakeTorino | 1.000 | 0.996 | 1.002 | 87.0 / 87.0 / 87.0 / 87.0 / 87.0 |
| F3 | FakeKingston | 1.000 | 0.943 | 0.995 | 87.0 / 87.0 / 87.0 / 87.0 / 87.0 |
| F4 | FakeAuckland | 1.000 | 0.935 | 0.935 | 42.2 / 41.0 / 41.0 / 41.0 / 41.7 |
| F4 | FakeTorino | 1.000 | 0.974 | 0.947 | 42.2 / 41.1 / 41.1 / 41.1 / 42.0 |
| F4 | FakeKingston | 1.000 | 0.970 | 0.955 | 42.5 / 41.1 / 41.1 / 41.1 / 41.7 |
| F5 | FakeAuckland | 1.000 | 0.932 | 0.932 | 5.0 / 5.0 / 5.0 / 5.0 / 5.0 |
| F5 | FakeTorino | 1.000 | 0.997 | 0.997 | 5.0 / 5.0 / 5.0 / 5.0 / 5.0 |
| F5 | FakeKingston | 1.000 | 0.999 | 0.999 | 5.0 / 5.0 / 5.0 / 5.0 / 5.0 |
| MODEL | FakeAuckland | 1.000 | 0.731 | 0.740 | 6.3 / 5.8 / 5.8 / 5.8 / 6.3 |
| MODEL | FakeTorino | 0.999 | 0.921 | 0.932 | 6.3 / 5.9 / 5.9 / 5.8 / 6.3 |
| MODEL | FakeKingston | 1.000 | 0.914 | 0.934 | 6.4 / 5.9 / 5.9 / 5.9 / 6.3 |

## Predictions

- H1 (A6/A5 <= 1.00 on every device, GAP and MODEL): **AMBIGUOUS** (GAP/FakeAuckland 1.000, GAP/FakeTorino 1.000, GAP/FakeKingston 1.000, MODEL/FakeAuckland 1.000, MODEL/FakeTorino 0.999, MODEL/FakeKingston 1.000)
- H2 (GAP: A6/A5 <= 0.98 on both Heron devices): **AMBIGUOUS** (FakeTorino 1.000, FakeKingston 1.000)
- H3 (MODEL: A6/C5 <= 0.95 on every device): **CONFIRMED** (FakeAuckland 0.731, FakeTorino 0.921, FakeKingston 0.914)
- H4 (GAP: A6/C5 <= 1.00 on every device): **CONFIRMED** (FakeAuckland 0.967, FakeTorino 0.963, FakeKingston 0.945)
- H5 (fast mode: A6F/A6 <= 1.05 on the Heron devices, GAP and MODEL, and median time A6F <= 0.5 x A6): **CONFIRMED** (GAP/FakeTorino 1.002, GAP/FakeKingston 1.006, MODEL/FakeTorino 1.020, MODEL/FakeKingston 1.021; median A6F 0.297 s, A6 0.611 s)
- H6 (A6/L3T <= 1.00 on every device, GAP and MODEL): **AMBIGUOUS** (GAP/FakeAuckland 1.024, GAP/FakeTorino 0.984, GAP/FakeKingston 0.983, MODEL/FakeAuckland 0.740, MODEL/FakeTorino 0.932, MODEL/FakeKingston 0.934)
- H7 (C5, A6 and A6F never use a failed coupler or qubit): **CONFIRMED** ({'C5': 0, 'A5': 0, 'A6': 0, 'A6F': 0, 'L3T': 0})
- H8 (median compile time A6 <= 1.3 x A5): **CONFIRMED** (A5 0.613 s, A6 0.611 s)

Reported: median compile s C5 0.028, A5 0.613, A6 0.611, A6F 0.297, L3T 0.014
Reported: circuits with an off-target instruction, by arm: C5 0, A5 0, A6 0, A6F 0, L3T 0
