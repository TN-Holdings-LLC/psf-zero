# c3 score (SMOKE -- not a result)

P0: PASS -- files 45 of 45 (missing []); noiseless infidelity max 4.8e-15 (<= 1e-6); too wide 0 of 144

| family | device | C2/L3T | C3/L3T | C3/C2 | failed-edge uses C2 / C3 / L3T |
|---|---|---|---|---|---|
| F1 | FakeAuckland | 1.170 | 1.170 | 1.000 | 0 / 0 / 0 |
| F1 | FakeTorino | 3.267 | 1.507 | 0.461 | 33 / 0 / 0 |
| F1 | FakeKingston | 1.114 | 1.114 | 1.000 | 0 / 0 / 0 |
| F2 | FakeAuckland | 1.095 | 1.095 | 1.000 | 0 / 0 / 0 |
| F2 | FakeTorino | 1.186 | 1.186 | 1.000 | 0 / 0 / 0 |
| F2 | FakeKingston | 1.149 | 1.149 | 1.000 | 0 / 0 / 0 |
| F3o | FakeAuckland | 1.422 | 1.422 | 1.000 | 0 / 0 / 0 |
| F3o | FakeTorino | 1.236 | 1.236 | 1.000 | 0 / 0 / 0 |
| F3o | FakeKingston | 1.475 | 1.475 | 1.000 | 0 / 0 / 0 |
| F3p | FakeAuckland | 1.318 | 1.318 | 1.000 | 0 / 0 / 0 |
| F3p | FakeTorino | 1.053 | 1.053 | 1.000 | 0 / 0 / 0 |
| F3p | FakeKingston | 1.224 | 1.224 | 1.000 | 0 / 0 / 0 |
| F4 | FakeAuckland | 0.931 | 0.931 | 1.000 | 0 / 0 / 0 |
| F4 | FakeTorino | 2.924 | 2.613 | 0.894 | 15 / 0 / 0 |
| F4 | FakeKingston | 1.227 | 1.227 | 1.000 | 0 / 0 / 0 |
| F5 | FakeAuckland | 1.244 | 1.244 | 1.000 | 0 / 0 / 0 |
| F5 | FakeTorino | 1.299 | 1.299 | 1.000 | 0 / 0 / 0 |
| F5 | FakeKingston | 1.538 | 1.538 | 1.000 | 0 / 0 / 0 |

C2 circuits using a failed element: 4 of 48; C3 recompiled 4

## Predictions

- H1 (C3 never uses a failed coupler or qubit): **CONFIRMED** (0 uses)
- H2 (the FakeTorino outlier is gone: C3/L3T <= 1.30 on F1, F2, F4): **REFUTED** (F1 1.507, F2 1.186, F4 2.613)
- H3 (where C2 used no failed element, C3 is identical to C2): **CONFIRMED** (44 of 44)
- H4 (where C2 used a failed element, C3 has lower infidelity in >= 90%): **CONFIRMED** (4 of 4 = 1.000)
- H5 (the placement gap remains: chains C3/L3T >= 1.10 on every device): **CONFIRMED** (FakeAuckland 1.365, FakeTorino 1.253, FakeKingston 1.492)

Reported: median compile s C2 0.036, C3 0.041, L3T 0.020
