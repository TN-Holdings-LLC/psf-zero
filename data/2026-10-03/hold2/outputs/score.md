# hold2 score

P0: PASS -- files 216 of 216 (missing []); noiseless infidelity max 2.1e-09 (<= 1e-6); too wide 0 of 54216

| device | C6/C5 | C6/L3T | C5/L3T | chains C6/L3T | chains C5/L3T | A7/C6 | A7/L3T |
|---|---|---|---|---|---|---|---|
| FakeAuckland (cx) | 0.984 | 1.039 | 1.056 | 1.089 | 1.097 | 0.909 | 0.944 |
| FakeTorino | 1.000 | 1.025 | 1.025 | 0.996 | 0.996 | 0.961 | 0.985 |
| FakeKingston | 0.998 | 1.037 | 1.039 | 0.982 | 0.998 | 0.945 | 0.981 |
| FakeHanoiV2 (cx) | 0.996 | 1.059 | 1.064 | 1.138 | 1.140 | 0.924 | 0.979 |
| FakeAlgiers (cx) | 0.970 | 1.077 | 1.111 | 1.101 | 1.194 | 0.908 | 0.978 |
| FakeGeneva (cx) | 0.994 | 1.006 | 1.011 | 1.036 | 1.077 | 0.940 | 0.945 |
| FakeFez | 0.999 | 1.040 | 1.040 | 0.991 | 0.991 | 0.948 | 0.986 |
| FakeMarrakesh | 0.992 | 1.017 | 1.026 | 0.963 | 1.022 | 0.943 | 0.960 |
| FakeAachen | 1.000 | 1.055 | 1.055 | 0.992 | 0.992 | 0.930 | 0.981 |

| cell C6/C5 | Auckland | Torino | Kingston | HanoiV2 | Algiers | Geneva | Fez | Marrakesh | Aachen |
|---|---|---|---|---|---|---|---|---|---|
| F1 | 0.999 | 1.000 | 1.000 | 0.988 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| F2 | 0.967 | 1.000 | 1.000 | 1.000 | 0.988 | 0.997 | 1.000 | 0.999 | 1.000 |
| F3o | 1.000 | 1.000 | 0.981 | 0.999 | 0.921 | 1.000 | 1.000 | 0.940 | 1.000 |
| F3p | 1.000 | 1.000 | 1.000 | 1.000 | 0.953 | 1.000 | 1.000 | 1.000 | 1.000 |
| F4 | 0.957 | 1.000 | 1.000 | 0.997 | 0.948 | 1.000 | 0.997 | 0.995 | 1.000 |
| F5 | 0.934 | 1.000 | 1.000 | 1.001 | 0.931 | 0.721 | 1.000 | 0.950 | 1.000 |
| F6 | 0.961 | 1.000 | 1.000 | 1.004 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Predictions

- H1 (C6/C5 <= 1.00 on every device): **AMBIGUOUS** (Auckland 0.984, Torino 1.000, Kingston 0.998, HanoiV2 0.996, Algiers 0.970, Geneva 0.994, Fez 0.999, Marrakesh 0.992, Aachen 1.000)
- H2 (cx devices: C6/C5 <= 0.98 on at least 3 of 4): **AMBIGUOUS** (Auckland 0.984, HanoiV2 0.996, Algiers 0.970, Geneva 0.994)
- H3 (cz devices: C6/C5 within 0.98-1.02 on all 5): **CONFIRMED** (Torino 1.000, Kingston 0.998, Fez 0.999, Marrakesh 0.992, Aachen 1.000)
- H4 (cx chains: C6/L3T <= 1.05 on at least 3 of 4): **AMBIGUOUS** (Auckland 1.089, HanoiV2 1.138, Algiers 1.101, Geneva 1.036)
- H5 (C6 never uses a failed direction or qubit): **CONFIRMED** (0 uses)
- H6 (median compile time C6 <= 3 x C5): **CONFIRMED** (C5 0.026 s, C6 0.050 s)
- H7 (cx devices: C6/C5 <= 1.00 in >= 80% of the 28 cell-device pairs): **CONFIRMED** (0.929)

Reported: median compile s C5 0.026, C6 0.050, A7 0.689, L3T 0.014
Reported: failed uses by coupler {'C5': 0, 'C6': 0, 'A7': 0, 'L3T': 0}; by direction {'C5': 0, 'C6': 0, 'A7': 0, 'L3T': 0}
Reported: circuits with an off-target instruction, by arm: C5 0, C6 0, A7 0, L3T 0
