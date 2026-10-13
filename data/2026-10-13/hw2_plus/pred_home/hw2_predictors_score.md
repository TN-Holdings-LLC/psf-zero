# HW2's additional predictors (Addendum 466)

## P1

| test | predicted ratio PSF / QK3 | device |
|---|---|---|
| qaoa_n12_p2 | 1.123 | 1.398 |
| qaoa_n20_p2 | 1.091 | 0.585 |
| ba_n14 | 1.317 | 1.113 |
| ba_n16 | 0.994 | 1.302 |
| ba_n18 | 1.740 | 0.590 |
| ba_n20 | 0.904 | 0.276 |
| tfim_n12 | 1.059 | 1.156 |

- same winner 4 of 7; rank correlation +0.143; useful (>= 5 of 7 and >= 0.5): NO
- part S (S1 / S0, predicted vs device): qaoa_n12_p2 0.922 vs 1.004; qaoa_n20_p2 0.657 vs 0.912; tfimP_n12 1.168 vs 1.297; tfimP_n16 1.224 vs 1.165

## P2

| test | predicted ratio PSF / QK3 | device |
|---|---|---|
| qaoa_n12_p2 | 1.130 | 1.398 |
| qaoa_n20_p2 | 0.969 | 0.585 |
| ba_n14 | 0.962 | 1.113 |
| ba_n16 | 0.953 | 1.302 |
| ba_n18 | 2.067 | 0.590 |
| ba_n20 | 0.582 | 0.276 |
| tfim_n12 | 1.069 | 1.156 |

- same winner 4 of 7; rank correlation +0.321; useful (>= 5 of 7 and >= 0.5): NO
- part S (S1 / S0, predicted vs device): qaoa_n12_p2 0.934 vs 1.004; qaoa_n20_p2 0.676 vs 0.912; tfimP_n12 1.171 vs 1.297; tfimP_n16 1.303 vs 1.165

## P4

| test | predicted ratio PSF / QK3 | device |
|---|---|---|
| qaoa_n12_p2 | 1.461 | 1.398 |
| qaoa_n20_p2 | 0.227 | 0.585 |
| ba_n14 | 0.072 | 1.113 |
| ba_n16 | 0.161 | 1.302 |
| ba_n18 | 2.655 | 0.590 |
| ba_n20 | 0.002 | 0.276 |
| tfim_n12 | 2.724 | 1.156 |

- same winner 4 of 7; rank correlation +0.393; useful (>= 5 of 7 and >= 0.5): NO
- part S (S1 / S0, predicted vs device): qaoa_n12_p2 1.117 vs 1.004; qaoa_n20_p2 1.346 vs 0.912; tfimP_n12 2.648 vs 1.297; tfimP_n16 4.653 vs 1.165

- H-T2* (P2 ranks at least as well as P1): rho P2 +0.321 vs P1 +0.143 -> CONFIRMED
