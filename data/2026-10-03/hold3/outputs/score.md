# hold3 score

P0: PASS -- files 270 of 270 (missing []); noiseless infidelity max 3.7e-09 (<= 1e-6); too wide 0 of 67770

| device | C8/C5 | C8/L3T | C5/L3T | F3 open C8/L3T | F3 open C5/L3T | chains C8/L3T | chains C5/L3T | A7/C8 | A7/L3T |
|---|---|---|---|---|---|---|---|---|---|
| FakeAuckland (cx) | 0.983 | 1.037 | 1.055 | 1.014 | 1.107 | 1.012 | 1.093 | 0.913 | 0.947 |
| FakeTorino | 0.998 | 1.021 | 1.023 | 0.992 | 0.997 | 0.992 | 0.997 | 0.964 | 0.985 |
| FakeKingston | 0.987 | 1.026 | 1.039 | 0.987 | 0.999 | 0.989 | 0.999 | 0.956 | 0.981 |
| FakeHanoiV2 (cx) | 0.971 | 1.034 | 1.065 | 1.042 | 1.156 | 1.037 | 1.137 | 0.949 | 0.981 |
| FakeAlgiers (cx) | 0.969 | 1.075 | 1.109 | 1.051 | 1.214 | 1.045 | 1.186 | 0.910 | 0.978 |
| FakeGeneva (cx) | 0.986 | 0.992 | 1.006 | 1.011 | 1.089 | 1.009 | 1.076 | 0.949 | 0.941 |
| FakeFez | 0.995 | 1.034 | 1.040 | 0.987 | 0.993 | 0.988 | 0.993 | 0.953 | 0.985 |
| FakeMarrakesh | 0.989 | 1.015 | 1.026 | 0.989 | 1.017 | 0.990 | 1.016 | 0.946 | 0.960 |
| FakeAachen | 0.993 | 1.048 | 1.055 | 0.989 | 0.994 | 0.990 | 0.995 | 0.937 | 0.981 |

| cell C8/C5 | Auckland | Torino | Kingston | HanoiV2 | Algiers | Geneva | Fez | Marrakesh | Aachen |
|---|---|---|---|---|---|---|---|---|---|
| F1 | 0.997 | 1.000 | 0.999 | 0.990 | 1.001 | 1.001 | 0.999 | 0.999 | 0.999 |
| F2 | 1.002 | 1.000 | 1.000 | 1.005 | 1.004 | 1.001 | 1.000 | 1.000 | 1.000 |
| F3o | 0.916 | 0.995 | 0.988 | 0.901 | 0.866 | 0.928 | 0.994 | 0.972 | 0.995 |
| F3p | 0.967 | 0.995 | 0.947 | 0.931 | 0.925 | 0.967 | 0.981 | 0.969 | 0.974 |
| F4 | 0.997 | 0.998 | 0.999 | 0.988 | 0.993 | 1.000 | 0.997 | 0.997 | 0.997 |
| F5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| F6 | 0.993 | 1.000 | 1.001 | 0.993 | 1.003 | 1.000 | 0.999 | 1.000 | 0.999 |

| cell C8/L3T | Auckland | Torino | Kingston | HanoiV2 | Algiers | Geneva | Fez | Marrakesh | Aachen |
|---|---|---|---|---|---|---|---|---|---|
| F1 | 1.002 | 1.072 | 1.060 | 0.991 | 1.028 | 0.991 | 1.066 | 0.949 | 1.078 |
| F2 | 1.022 | 1.020 | 1.029 | 1.020 | 1.019 | 0.978 | 1.030 | 0.999 | 1.017 |
| F3o | 1.014 | 0.992 | 0.987 | 1.042 | 1.051 | 1.011 | 0.987 | 0.989 | 0.989 |
| F3p | 1.192 | 1.008 | 1.032 | 1.183 | 1.402 | 1.120 | 1.054 | 1.211 | 1.142 |
| F4 | 0.995 | 0.974 | 0.978 | 0.988 | 0.994 | 0.995 | 0.987 | 0.953 | 0.968 |
| F5 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| F6 | 1.054 | 1.001 | 1.047 | 1.047 | 1.060 | 0.709 | 1.051 | 1.003 | 1.003 |

## Predictions

- H1 (C8/C5 <= 1.00 on every device): **CONFIRMED** (Auckland 0.983, Torino 0.998, Kingston 0.987, HanoiV2 0.971, Algiers 0.969, Geneva 0.986, Fez 0.995, Marrakesh 0.989, Aachen 0.993)
- H2 (cx devices: C8/C5 <= 0.99 on at least 3 of 4): **CONFIRMED** (Auckland 0.983, HanoiV2 0.971, Algiers 0.969, Geneva 0.986)
- H3 (cz devices: C8/C5 within 0.98-1.02 on all 5): **CONFIRMED** (Torino 0.998, Kingston 0.987, Fez 0.995, Marrakesh 0.989, Aachen 0.993)
- H4 (cx devices, F3 open: C8/L3T <= 1.05 on at least 3 of 4): **CONFIRMED** (Auckland 1.014, HanoiV2 1.042, Algiers 1.051, Geneva 1.011)
- H5 (cx chains F3 open + F5: C8/L3T <= 1.05 on at least 3 of 4): **CONFIRMED** (Auckland 1.012, HanoiV2 1.037, Algiers 1.045, Geneva 1.009)
- H6 (C8 never uses a failed direction or qubit, nor an off-target instruction): **CONFIRMED** (0 failed uses, 0 circuits off target)
- H7 (median compile time C8 <= 3 x C5): **CONFIRMED** (C5 0.026 s, C8 0.042 s)
- H8 (cx devices: C8/C5 <= 1.00 in >= 80% of the 28 cell-device pairs): **AMBIGUOUS** (0.714)
- H9 (where C7F and C5 differ, C8 picks the one with the lower measured infidelity in >= 75%): **CONFIRMED** (11977 of 13554, 0.884)

Reported: C8 choices (RESYNTH_STATS counters that moved): {'applied,selected_original': 9428, 'applied,selected_resynthesised': 4126}
Reported: C7F/C5 by device: Auckland 0.999, Torino 1.013, Kingston 1.012, HanoiV2 0.987, Algiers 0.985, Geneva 1.013, Fez 1.015, Marrakesh 1.013, Aachen 1.023

Reported: median compile s C5 0.026, C7F 0.029, C8 0.042, A7 0.693, L3T 0.015
Reported: failed uses by coupler {'C5': 0, 'C7F': 0, 'C8': 0, 'A7': 0, 'L3T': 0}; by direction {'C5': 0, 'C7F': 0, 'C8': 0, 'A7': 0, 'L3T': 0}
Reported: circuits with an off-target instruction, by arm: C5 0, C7F 0, C8 0, A7 0, L3T 0
