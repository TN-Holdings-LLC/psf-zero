# HW2's additional predictors (Addendum 466)

## P1c

| test | predicted ratio PSF / QK3 | device |
|---|---|---|
| qaoa_n12_p2 | 1.101 | 1.398 |
| qaoa_n20_p2 | 0.760 | 0.585 |
| ba_n14 | 1.233 | 1.113 |
| ba_n16 | 1.027 | 1.302 |
| ba_n18 | 1.626 | 0.590 |
| ba_n20 | 0.758 | 0.276 |
| tfim_n12 | 1.057 | 1.156 |

- same winner 6 of 7; rank correlation +0.393; useful (>= 5 of 7 and >= 0.5): NO
- part S (S1 / S0, predicted vs device): qaoa_n12_p2 0.963 vs 1.004; qaoa_n20_p2 0.841 vs 0.912; tfimP_n12 1.165 vs 1.297; tfimP_n16 1.273 vs 1.165

## P2c

| test | predicted ratio PSF / QK3 | device |
|---|---|---|
| qaoa_n12_p2 | 1.125 | 1.398 |
| qaoa_n20_p2 | 0.699 | 0.585 |
| ba_n14 | 0.953 | 1.113 |
| ba_n16 | 0.992 | 1.302 |
| ba_n18 | 1.932 | 0.590 |
| ba_n20 | 0.493 | 0.276 |
| tfim_n12 | 1.066 | 1.156 |

- same winner 4 of 7; rank correlation +0.607; useful (>= 5 of 7 and >= 0.5): NO
- part S (S1 / S0, predicted vs device): qaoa_n12_p2 0.960 vs 1.004; qaoa_n20_p2 0.680 vs 0.912; tfimP_n12 1.163 vs 1.297; tfimP_n16 1.369 vs 1.165


- Q2c (P2c ranks at least as well as P1c): rho P2c +0.607 vs P1c +0.393 -> CONFIRMED
