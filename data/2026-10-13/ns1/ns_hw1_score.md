# NS1 scores

| test | arm | F_sim (SE) | F_real | ESP | F_sim / F_real |
|---|---|---|---|---|---|
| qaoa_n12_p2 | QK3 | 0.6373 (0.0257) | 0.4990 | 0.6624 | 1.277 |
| qaoa_n12_p2 | PSF | 0.6986 (0.0332) | 0.4939 | 0.6975 | 1.414 |
| tfim_n12 | QK3 | 0.8657 (0.0293) | 0.7208 | 0.7598 | 1.201 |
| tfim_n12 | PSF | 0.8415 (0.0321) | 0.8611 | 0.7929 | 0.977 |
| ghz_n12 | QK3 | 0.8766 (0.0263) | 0.8373 | 0.9023 | 1.047 |
| ghz_n12 | PSF | 0.8766 (0.0263) | 0.8384 | 0.9023 | 1.046 |
| ba_n14 | PSF | 0.3235 (0.0161) | 0.2052 | 0.4811 | 1.577 |
| ba_n14 | QK3 | 0.3659 (0.0171) | 0.2632 | 0.3742 | 1.390 |
| ba_n16 | QK3 | 0.4085 (0.0174) | 0.2800 | 0.2976 | 1.459 |
| ba_n16 | PSF | 0.4111 (0.0175) | 0.2840 | 0.3983 | 1.448 |
| ba_n18 | QK3 | 0.2398 (0.0113) | 0.1049 | 0.2418 | 2.285 |
| ba_n18 | PSF | 0.2462 (0.0122) | 0.0780 | 0.2535 | 3.155 |
| ba_n20 | QK3 | 0.3014 (0.0142) | 0.1741 | 0.1772 | 1.731 |
| qaoa_n20_p2 | QK3 | 0.5006 (0.0240) | 0.3536 | 0.4546 | 1.416 |
| ba_n20 | PSF | 0.2587 (0.0128) | 0.1407 | 0.2168 | 1.839 |
| qaoa_n20_p2 | PSF | 0.4967 (0.0268) | 0.4250 | 0.4786 | 1.169 |

| test | ESP ratio | F_sim ratio | F_real ratio |
|---|---|---|---|
| qaoa_n12_p2 | 1.053 | 1.096 | 0.990 |
| qaoa_n20_p2 | 1.053 | 0.992 | 1.202 |
| ba_n14 | 1.286 | 0.884 | 0.780 |
| ba_n16 | 1.339 | 1.006 | 1.014 |
| ba_n18 | 1.048 | 1.026 | 0.744 |
| ba_n20 | 1.224 | 0.858 | 0.808 |
| tfim_n12 | 1.044 | 0.972 | 1.195 |

## Predictions (fixed in NS1_preregistration.md)

- NS1a (same winner as the device on >= 6 of 7 reading tests): 3 of 7 -> REFUTED
- NS1b (rank correlation of F_sim ratio and F_real ratio >= 0.6): +0.000 (ESP's -0.179) -> REFUTED
- NS1c (F_sim within 0.5-2 x F_real on >= 80% of circuits, tfim_n16 left out): 14 of 16 -> CONFIRMED
- NS1d (F_real / F_sim per arm: |ln(PSF / QK3)| < 0.153, ESP's gap): QK3 0.696, PSF 0.678; gap 0.025 -> CONFIRMED
