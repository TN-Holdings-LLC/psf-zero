# A0 score: reported gate error below the T1/T2 floor

versions {"qiskit": "2.5.2", "aer": "0.17.2", "runtime": "0.49.0", "python": "3.12.13"}

P0: PASS {"rows": 13982, "recompute_max": 0.0, "floor_vs_aer_max": 1.9984014443252818e-15, "floor_vs_aer_n": 13982, "applied_vs_max_max": 3.3584246494910985e-15, "applied_n": 13545, "applied_missing": 0}; devices 67; named devices present True; skipped 0

## Per device

| device | qubits | snapshot | 2q gate | 2q usable | 2q below | frac | max floor/err 2q | sx usable | sx below | median T2 (us) |
|---|---|---|---|---|---|---|---|---|---|---|
| fake_armonk | 1 | 2021-03-15 | none | 0 | 0 | - | - | 1 | 0 | 238 |
| fake_algiers | 27 | 2024-02-28 | cx | 54 | 15 | 0.278 | 9.43 | 27 | 7 | 87 |
| fake_almaden | 20 | 2020-08-10 | cx | 46 | 11 | 0.239 | 1.38 | 0 | 0 | 34 |
| fake_athens | 5 | 2021-03-15 | cx | 8 | 0 | 0.000 | 0.71 | 5 | 1 | 112 |
| fake_auckland | 27 | 2024-05-27 | cx | 56 | 16 | 0.286 | 1.61 | 27 | 5 | 134 |
| fake_belem | 5 | 2021-03-15 | cx | 8 | 0 | 0.000 | 0.92 | 5 | 1 | 64 |
| fake_boeblingen | 20 | 2021-02-03 | cx | 46 | 2 | 0.043 | 1.39 | 0 | 0 | 79 |
| fake_bogota | 5 | 2021-03-15 | cx | 8 | 0 | 0.000 | 0.78 | 5 | 1 | 109 |
| fake_brooklyn | 65 | 2021-07-26 | cx | 144 | 12 | 0.083 | 1.64 | 65 | 8 | 84 |
| fake_burlington | 5 | 2020-06-11 | cx | 8 | 0 | 0.000 | 0.90 | 0 | 0 | 93 |
| fake_cambridge | 28 | 2020-07-16 | cx | 50 | 0 | 0.000 | 0.92 | 0 | 0 | 52 |
| fake_casablanca | 7 | 2021-03-15 | cx | 12 | 0 | 0.000 | 0.61 | 7 | 0 | 77 |
| fake_essex | 5 | 2020-08-10 | cx | 8 | 0 | 0.000 | 0.43 | 0 | 0 | 105 |
| fake_geneva | 27 | 2022-06-24 | cx | 54 | 4 | 0.074 | 4.58 | 27 | 2 | 321 |
| fake_guadalupe | 16 | 2021-04-20 | cx | 32 | 2 | 0.062 | 2.09 | 16 | 1 | 88 |
| fake_hanoi | 27 | 2025-02-26 | cx | 54 | 13 | 0.241 | 1.60 | 27 | 6 | 117 |
| fake_jakarta | 7 | 2024-05-27 | cx | 12 | 6 | 0.500 | 1.59 | 7 | 4 | 34 |
| fake_johannesburg | 20 | 2020-08-09 | cx | 46 | 42 | 0.913 | 2.29 | 0 | 0 | 14 |
| fake_kolkata | 27 | 2021-12-09 | cx | 56 | 16 | 0.286 | 1.61 | 27 | 12 | 83 |
| fake_lagos | 7 | 2024-05-27 | cx | 12 | 4 | 0.333 | 1.28 | 7 | 2 | 72 |
| fake_lima | 5 | 2021-03-15 | cx | 8 | 2 | 0.250 | 1.22 | 5 | 3 | 94 |
| fake_london | 5 | 2020-08-10 | cx | 8 | 1 | 0.125 | 1.05 | 0 | 0 | 75 |
| fake_manhattan | 65 | 2021-03-15 | cx | 100 | 6 | 0.060 | 1.38 | 65 | 10 | 80 |
| fake_manila | 5 | 2024-05-27 | cx | 8 | 2 | 0.250 | 1.15 | 5 | 2 | 54 |
| fake_melbourne | 15 | 2021-03-15 | cx | 40 | 3 | 0.075 | 1.27 | 15 | 3 | 55 |
| fake_montreal | 27 | 2021-03-15 | cx | 56 | 12 | 0.214 | 1.66 | 27 | 9 | 78 |
| fake_mumbai | 27 | 2021-03-13 | cx | 56 | 4 | 0.071 | 1.16 | 27 | 4 | 124 |
| fake_nairobi | 7 | 2024-05-27 | cx | 12 | 7 | 0.583 | 2.16 | 7 | 2 | 106 |
| fake_oslo | 7 | 2022-07-17 | cx | 12 | 0 | 0.000 | 0.87 | 7 | 2 | 47 |
| fake_ourense | 5 | 2021-01-20 | cx | 8 | 6 | 0.750 | 1.31 | 5 | 2 | 91 |
| fake_paris | 27 | 2021-03-15 | cx | 56 | 0 | 0.000 | 0.98 | 27 | 2 | 77 |
| fake_perth | 7 | 2024-05-27 | cx | 12 | 3 | 0.250 | 1.41 | 7 | 0 | 95 |
| fake_poughkeepsie | 20 | 2020-02-29 | cx | 46 | 15 | 0.326 | 2.89 | 0 | 0 | 80 |
| fake_quito | 5 | 2021-03-15 | cx | 8 | 0 | 0.000 | 0.54 | 5 | 1 | 27 |
| fake_rochester | 53 | 2020-07-16 | cx | 107 | 0 | 0.000 | 0.94 | 0 | 0 | 52 |
| fake_rome | 5 | 2021-03-15 | cx | 8 | 0 | 0.000 | 0.76 | 5 | 0 | 116 |
| fake_santiago | 5 | 2021-03-15 | cx | 8 | 1 | 0.125 | 1.01 | 5 | 1 | 99 |
| fake_singapore | 20 | 2020-08-07 | cx | 46 | 0 | 0.000 | 0.71 | 0 | 0 | 104 |
| fake_sydney | 27 | 2021-03-15 | cx | 56 | 10 | 0.179 | 1.37 | 27 | 6 | 113 |
| fake_toronto | 27 | 2021-03-15 | cx | 56 | 54 | 0.964 | 13.69 | 27 | 25 | 117 |
| fake_valencia | 5 | 2021-01-20 | cx | 8 | 2 | 0.250 | 1.18 | 5 | 1 | 49 |
| fake_vigo | 5 | 2021-01-20 | cx | 8 | 2 | 0.250 | 1.36 | 5 | 1 | 69 |
| fake_washington | 127 | 2022-04-12 | cx | 278 | 36 | 0.129 | 2.50 | 127 | 26 | 90 |
| fake_yorktown | 5 | 2021-03-15 | cx | 12 | 4 | 0.333 | 4.06 | 5 | 2 | 25 |
| fake_cairo | 27 | 2024-03-25 | mixed | 25 | 4 | 0.160 | 2.21 | 27 | 5 | 81 |
| fake_aachen | 156 | 2026-04-17 | cz | 348 | 0 | 0.000 | 0.56 | 154 | 1 | 255 |
| fake_berlin | 120 | 2026-04-17 | cz | 424 | 0 | 0.000 | 0.68 | 120 | 0 | 248 |
| fake_boston | 156 | 2026-04-17 | cz | 350 | 4 | 0.011 | 1.27 | 155 | 3 | 353 |
| fake_fez | 156 | 2025-02-26 | cz | 338 | 4 | 0.012 | 1.27 | 156 | 26 | 88 |
| fake_kingston | 156 | 2026-04-15 | cz | 338 | 18 | 0.053 | 1.47 | 151 | 24 | 144 |
| fake_marrakesh | 156 | 2025-02-26 | cz | 326 | 6 | 0.018 | 1.23 | 156 | 34 | 118 |
| fake_miami | 120 | 2026-04-17 | cz | 422 | 0 | 0.000 | 0.85 | 120 | 0 | 242 |
| fake_nighthawk | 120 | 2025-12-08 | cz | 436 | 6 | 0.014 | 1.50 | 119 | 2 | 342 |
| fake_pittsburgh | 156 | 2026-04-17 | cz | 342 | 0 | 0.000 | 0.67 | 155 | 1 | 317 |
| fake_prague | 33 | 2023-01-12 | cz | 68 | 0 | 0.000 | 0.59 | 33 | 2 | 118 |
| fake_torino | 133 | 2025-02-26 | cz | 278 | 0 | 0.000 | 0.47 | 133 | 7 | 141 |
| fake_brisbane | 127 | 2025-02-26 | ecr | 143 | 37 | 0.259 | 3.73 | 127 | 41 | 150 |
| fake_brussels | 127 | 2026-04-17 | ecr | 138 | 51 | 0.370 | 2.45 | 123 | 32 | 136 |
| fake_cusco | 127 | 2025-02-26 | ecr | 125 | 5 | 0.040 | 2.59 | 127 | 17 | 80 |
| fake_kawasaki | 127 | 2025-02-26 | ecr | 133 | 19 | 0.143 | 27.49 | 127 | 23 | 151 |
| fake_kyiv | 127 | 2025-02-26 | ecr | 139 | 19 | 0.137 | 3.18 | 127 | 37 | 118 |
| fake_kyoto | 127 | 2024-02-28 | ecr | 0 | 0 | - | - | 127 | 50 | 110 |
| fake_osaka | 127 | 2024-02-28 | ecr | 137 | 52 | 0.380 | 6.62 | 127 | 44 | 140 |
| fake_peekskill | 27 | 2024-02-28 | ecr | 28 | 0 | 0.000 | 0.88 | 27 | 1 | 296 |
| fake_quebec | 127 | 2025-02-26 | ecr | 135 | 22 | 0.163 | 4.79 | 127 | 28 | 171 |
| fake_sherbrooke | 127 | 2025-02-26 | ecr | 135 | 26 | 0.193 | 3.73 | 127 | 37 | 170 |
| fake_strasbourg | 127 | 2026-04-17 | ecr | 142 | 50 | 0.352 | 3.08 | 126 | 44 | 119 |

## By two-qubit gate class (devices with at least one usable 2q gate)

| class | devices | devices with any 2q below | median device frac | pooled frac | pooled sx frac |
|---|---|---|---|---|---|
| cx | 43 | 30 | 0.125 | 0.181 | 0.230 |
| ecr | 10 | 9 | 0.178 | 0.224 | 0.261 |
| cz | 11 | 5 | 0.000 | 0.010 | 0.069 |
| mixed | 1 | 1 | 0.160 | 0.160 | 0.185 |

## Predictions

- H1: **CONFIRMED** -- cz class median device frac 0.000 (11 devices)
- H2: **CONFIRMED** -- cx class median device frac 0.125 (43 devices)
- H3: **CONFIRMED** -- cx vs cz: median 0.125 vs 0.000, pooled 0.181 vs 0.010
- H4: **CONFIRMED** -- 0.928 of 636 below-floor 2q gates have a qubit with T2 below its device median
- H5: **AMBIGUOUS** -- Kingston cz below 18 of 338 (0.053)
- R1: **CONFIRMED** -- Auckland cx below 16 of 56; max floor/err cx 1.613, sx 2.096
- R2: **CONFIRMED** -- Torino cz below 0 of 278

## Reported without prediction

- cx: summed 2q error under max(reported, floor) / reported - 1: median 0.0205, max 2.8407
- ecr: summed 2q error under max(reported, floor) / reported - 1: median 0.0841, max 0.3590
- cz: summed 2q error under max(reported, floor) / reported - 1: median 0.0000, max 0.0031
- mixed: summed 2q error under max(reported, floor) / reported - 1: median 0.0802, max 0.0802
- Spearman rank correlation of snapshot date with 2q below-floor fraction: 0.069 (65 devices)
- rows excluded from fractions (no error, error >= 0.5, no duration or missing T1/T2): 437
- qubits with T2 > 2*T1 (truncated, as qiskit-aer does): 58
