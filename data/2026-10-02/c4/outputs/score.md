# c4 score

P0: PASS -- files 45 of 45 (missing []); noiseless infidelity max 1.9e-08 (<= 1e-6); too wide 0 of 6237

| family | device | C3/L3T | C4/L3T | C4/C3 | mean 2q C3 / C4 / L3T |
|---|---|---|---|---|---|
| F1 | FakeAuckland | 1.156 | 1.106 | 0.957 | 53.5 / 53.5 / 53.8 |
| F1 | FakeTorino | 1.511 | 1.291 | 0.855 | 59.2 / 59.2 / 53.7 |
| F1 | FakeKingston | 1.115 | 1.528 | 1.371 | 57.3 / 57.3 / 53.8 |
| F2 | FakeAuckland | 1.147 | 1.045 | 0.911 | 47.2 / 47.2 / 45.4 |
| F2 | FakeTorino | 1.189 | 1.214 | 1.021 | 48.1 / 48.4 / 46.4 |
| F2 | FakeKingston | 1.148 | 1.488 | 1.296 | 48.1 / 48.1 / 45.8 |
| F3o | FakeAuckland | 1.377 | 1.288 | 0.936 | 60.0 / 60.0 / 60.0 |
| F3o | FakeTorino | 1.240 | 1.313 | 1.059 | 60.0 / 60.0 / 60.0 |
| F3o | FakeKingston | 1.459 | 1.269 | 0.870 | 60.0 / 60.0 / 60.0 |
| F3p | FakeAuckland | 1.252 | 1.381 | 1.103 | 114.0 / 114.0 / 114.0 |
| F3p | FakeTorino | 1.051 | 1.606 | 1.527 | 114.0 / 114.0 / 114.0 |
| F3p | FakeKingston | 1.233 | 1.634 | 1.326 | 114.0 / 114.0 / 114.0 |
| F4 | FakeAuckland | 1.031 | 1.039 | 1.007 | 42.2 / 42.2 / 41.7 |
| F4 | FakeTorino | 1.696 | 1.407 | 0.830 | 42.2 / 42.2 / 42.0 |
| F4 | FakeKingston | 1.195 | 1.608 | 1.346 | 42.5 / 42.5 / 41.7 |
| F5 | FakeAuckland | 1.244 | 1.048 | 0.842 | 5.0 / 5.0 / 5.0 |
| F5 | FakeTorino | 1.299 | 1.375 | 1.059 | 5.0 / 5.0 / 5.0 |
| F5 | FakeKingston | 1.538 | 1.578 | 1.026 | 5.0 / 5.0 / 5.0 |

C4 backstop recompiles (item 31 path): 0

## Predictions

- H1 (chains: C4/L3T <= 1.05 on every device): **REFUTED** (FakeAuckland 1.258, FakeTorino 1.320, FakeKingston 1.301)
- H2 (C4 <= C3 in >= 16 of 18 cells and no cell > 1.10): **REFUTED** (7 of 18 <= 1.0; max 1.527)
- H3 (C4/L3T <= 1.10 in every cell): **REFUTED** (max 1.634)
- H4 (C4 never uses a failed coupler or qubit): **CONFIRMED** (0 uses)
- H5 (median compile time C4 <= 2 x C3): **CONFIRMED** (C3 0.028 s, C4 0.014 s)

Reported: median compile s C3 0.028, C4 0.014, L3T 0.015
