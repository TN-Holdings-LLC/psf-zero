# c3 score

P0: PASS -- files 45 of 45 (missing []); noiseless infidelity max 1.9e-08 (<= 1e-6); too wide 0 of 6237

| family | device | C2/L3T | C3/L3T | C3/C2 | failed-edge uses C2 / C3 / L3T |
|---|---|---|---|---|---|
| F1 | FakeAuckland | 1.156 | 1.156 | 1.000 | 0 / 0 / 0 |
| F1 | FakeTorino | 3.252 | 1.511 | 0.465 | 1188 / 0 / 0 |
| F1 | FakeKingston | 1.115 | 1.115 | 1.000 | 0 / 0 / 0 |
| F2 | FakeAuckland | 1.147 | 1.147 | 1.000 | 0 / 0 / 0 |
| F2 | FakeTorino | 2.058 | 1.189 | 0.578 | 183 / 0 / 0 |
| F2 | FakeKingston | 1.148 | 1.148 | 1.000 | 0 / 0 / 0 |
| F3o | FakeAuckland | 1.377 | 1.377 | 1.000 | 0 / 0 / 0 |
| F3o | FakeTorino | 1.240 | 1.240 | 1.000 | 0 / 0 / 0 |
| F3o | FakeKingston | 1.459 | 1.459 | 1.000 | 0 / 0 / 0 |
| F3p | FakeAuckland | 1.252 | 1.252 | 1.000 | 0 / 0 / 0 |
| F3p | FakeTorino | 1.051 | 1.051 | 1.000 | 0 / 0 / 0 |
| F3p | FakeKingston | 1.233 | 1.233 | 1.000 | 0 / 0 / 0 |
| F4 | FakeAuckland | 1.031 | 1.031 | 1.000 | 0 / 0 / 0 |
| F4 | FakeTorino | 1.872 | 1.696 | 0.906 | 261 / 0 / 0 |
| F4 | FakeKingston | 1.195 | 1.195 | 1.000 | 0 / 0 / 0 |
| F5 | FakeAuckland | 1.244 | 1.244 | 1.000 | 0 / 0 / 0 |
| F5 | FakeTorino | 1.299 | 1.299 | 1.000 | 0 / 0 / 0 |
| F5 | FakeKingston | 1.538 | 1.538 | 1.000 | 0 / 0 / 0 |

C2 circuits using a failed element: 153 of 2079; C3 recompiled 153

## Predictions

- H1 (C3 never uses a failed coupler or qubit): **CONFIRMED** (0 uses)
- H2 (the FakeTorino outlier is gone: C3/L3T <= 1.30 on F1, F2, F4): **AMBIGUOUS** (F1 1.511, F2 1.189, F4 1.696)
- H3 (where C2 used no failed element, C3 is identical to C2): **CONFIRMED** (1926 of 1926)
- H4 (where C2 used a failed element, C3 has lower infidelity in >= 90%): **CONFIRMED** (153 of 153 = 1.000)
- H5 (the placement gap remains: chains C3/L3T >= 1.10 on every device): **CONFIRMED** (FakeAuckland 1.360, FakeTorino 1.246, FakeKingston 1.467)

Reported: median compile s C2 0.026, C3 0.027, L3T 0.015
