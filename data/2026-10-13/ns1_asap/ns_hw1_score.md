# NS1 scores

| test | arm | F_sim (SE) | F_real | ESP | F_sim / F_real |
|---|---|---|---|---|---|
| qaoa_n12_p2 | QK3 | 0.6438 (0.0281) | 0.4990 | 0.6624 | 1.290 |
| qaoa_n12_p2 | PSF | 0.6631 (0.0242) | 0.4939 | 0.6975 | 1.343 |
| tfim_n12 | QK3 | 0.8597 (0.0346) | 0.7208 | 0.7598 | 1.193 |
| tfim_n12 | PSF | 0.8402 (0.0311) | 0.8611 | 0.7929 | 0.976 |
| ghz_n12 | QK3 | 0.8617 (0.0315) | 0.8373 | 0.9023 | 1.029 |
| ghz_n12 | PSF | 0.8617 (0.0315) | 0.8384 | 0.9023 | 1.028 |
| ba_n14 | PSF | 0.2746 (0.0131) | 0.2052 | 0.4811 | 1.338 |
| ba_n14 | QK3 | 0.3528 (0.0173) | 0.2632 | 0.3742 | 1.340 |
| ba_n16 | QK3 | 0.3821 (0.0156) | 0.2800 | 0.2976 | 1.365 |
| ba_n16 | PSF | 0.4273 (0.0213) | 0.2840 | 0.3983 | 1.505 |
| ba_n18 | QK3 | 0.2050 (0.0101) | 0.1049 | 0.2418 | 1.954 |
| ba_n18 | PSF | 0.1703 (0.0094) | 0.0780 | 0.2535 | 2.183 |
| ba_n20 | QK3 | 0.3011 (0.0147) | 0.1741 | 0.1772 | 1.729 |
| qaoa_n20_p2 | QK3 | 0.4542 (0.0217) | 0.3536 | 0.4546 | 1.285 |
| ba_n20 | PSF | 0.3085 (0.0154) | 0.1407 | 0.2168 | 2.193 |
| qaoa_n20_p2 | PSF | 0.5158 (0.0329) | 0.4250 | 0.4786 | 1.214 |

| test | ESP ratio | F_sim ratio | F_real ratio |
|---|---|---|---|
| qaoa_n12_p2 | 1.053 | 1.030 | 0.990 |
| qaoa_n20_p2 | 1.053 | 1.136 | 1.202 |
| ba_n14 | 1.286 | 0.778 | 0.780 |
| ba_n16 | 1.338 | 1.118 | 1.014 |
| ba_n18 | 1.048 | 0.831 | 0.744 |
| ba_n20 | 1.223 | 1.025 | 0.808 |
| tfim_n12 | 1.044 | 0.977 | 1.195 |

## Predictions (fixed in NS1_preregistration.md)

- NS1a (same winner as the device on >= 6 of 7 reading tests): 4 of 7 -> REFUTED
- NS1b (rank correlation of F_sim ratio and F_real ratio >= 0.6): +0.750 (ESP's -0.179) -> CONFIRMED
- NS1c (F_sim within 0.5-2 x F_real on >= 80% of circuits, tfim_n16 left out): 14 of 16 -> CONFIRMED
- NS1d (F_real / F_sim per arm: |ln(PSF / QK3)| < 0.153, ESP's gap): QK3 0.729, PSF 0.708; gap 0.028 -> CONFIRMED
