# STEINER5 (exploratory)

git head 5ca55be; 67 HamLib FakeTorino tests on FakeTorino, FakeKingston; 268 jobs (arms P, PE). Others: ESP-FT (QK2, QK3, PSFR), STEINER4's O (S4O).

## FakeTorino

- P: 67 eligible; exact 23, not checked 44, NOT exact 0; on failed elements 0; placement chosen {'greedy': 40, 'vf2': 27}; errors 0; time 176.3 s summed
- PE: 67 eligible; exact 23, not checked 44, NOT exact 0; on failed elements 0; placement chosen {'vf2': 44, 'greedy': 23}; errors 0; time 190.3 s summed

| arm / other | q2 ratio (gmean, +1) | fewer / equal / more | ESP ratio (gmean, both > 0; best >= 0.01) | ESP 10% better / worse |
|---|---|---|---|---|
| P / S4O | 0.927 | 23 / 40 / 4 | 1.114 (22) | 4 / 1 |
| P / QK2 | 1.506 | 9 / 8 / 50 | 0.808 (22) | 3 / 7 |
| P / QK3 | 1.559 | 7 / 8 / 52 | 0.767 (23) | 0 / 12 |
| P / PSFR | 1.448 | 10 / 8 / 49 | 0.769 (23) | 1 / 12 |
| PE / S4O | 0.972 | 20 / 12 / 35 | 1.258 (22) | 11 / 0 |
| PE / QK2 | 1.580 | 8 / 8 / 51 | 0.918 (22) | 3 / 4 |
| PE / QK3 | 1.636 | 6 / 8 / 53 | 0.878 (23) | 1 / 7 |
| PE / PSFR | 1.518 | 7 / 8 / 52 | 0.881 (23) | 1 / 6 |

| test | n | terms | S4O | P | PE | QK2 | QK3 | PSFR | PE log10 ESP | QK3 log10 ESP | P placement | PE placement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ham_enc_stdbinary_dvalues_d-level-4 | 2 | 10 | 3 | 3 | 3 | 3 | 3 | 3 | -0.004296 | -0.003176 | greedy | vf2-0 |
| ham_mu_y_prime_enc_stdbinary_dvalues_8-8 | 3 | 12 | 19 | 19 | 19 | 18 | 18 | 18 | -0.025959 | -0.022311 | greedy | vf2-0 |
| ham_enc_gray_dvalues_d-level-16 | 4 | 99 | 223 | 223 | 223 | 174 | 172 | 202 | -0.275576 | -0.229393 | greedy | vf2-4 |
| ham_gnp-k_2_n-4_rinst-05 | 4 | 13 | 33 | 33 | 33 | 24 | 24 | 24 | -0.042554 | -0.033633 | greedy | vf2-0 |
| ham_ham_parity-4 | 4 | 27 | 62 | 71 | 73 | 62 | 62 | 63 | -0.096842 | -0.082195 | vf2-0 | vf2-1 |
| ham_mu_y_prime_enc_gray_dvalues_16-16-16 | 4 | 32 | 105 | 110 | 110 | 92 | 92 | 92 | -0.139366 | -0.12304 | vf2-0 | vf2-4 |
| ham_mu_y_prime_enc_stdbinary_dvalues_16- | 4 | 32 | 57 | 57 | 57 | 72 | 72 | 72 | -0.09208 | -0.088981 | greedy | vf2-0 |
| ham_mu_y_prime_enc_unary_dvalues_4-4-4 | 4 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | -0.009142 | -0.009248 | greedy | vf2-0 |
| ham_mu_z_prime_enc_gray_dvalues_4-4-4-4 | 4 | 8 | 4 | 4 | 4 | 4 | 4 | 4 | -0.007426 | -0.0041 | greedy | vf2-0 |
| ham_gnp-k_4_n-6_rinst-15 | 6 | 46 | 277 | 277 | 279 | 134 | 134 | 139 | -0.410824 | -0.213018 | greedy | vf2-1 |
| ham_mu_y_prime_enc_stdbinary_dvalues_8-8 | 6 | 24 | 38 | 38 | 38 | 36 | 36 | 36 | -0.063418 | -0.04537 | greedy | vf2-0 |
| ham_7-uf20-0384.cnf-8-res | 8 | 30 | 116 | 116 | 140 | 75 | 78 | 78 | -0.212533 | -0.121822 | greedy | vf2-0 |
| ham_ham_JW-8 | 8 | 185 | 1298 | 1298 | 1436 | 1367 | 1338 | 1341 | -2.208815 | -1.926593 | greedy | vf2-0 |
| ham_mu_y_prime_enc_unary_dvalues_8-8-8-8 | 8 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | -0.024773 | -0.023562 | greedy | vf2-0 |
| ham_tsp_rand-002_Ncity-4_enc-stdbinary | 8 | 49 | 228 | 228 | 230 | 133 | 133 | 136 | -0.340702 | -0.210147 | greedy | vf2-0 |
| ham_complbipart-n-10_a-5_b-5 | 10 | 76 | 699 | 634 | 634 | 287 | 307 | 271 | -1.001437 | -0.536625 | vf2-0 | vf2-0 |
| ham_ham_JW10 | 10 | 276 | 2748 | 2571 | 2545 | 2408 | 2338 | 2369 | -4.216838 | -4.203198 | vf2-0 | vf2-0 |
| ham_mu_x_prime_enc_stdbinary_dvalues_8-8 | 11 | 40 | 59 | 59 | 59 | 56 | 56 | 56 | -0.13314 | -0.080581 | greedy | greedy |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-4 | 12 | 1177 | 7813 | 7813 | 10335 | 4373 | 4359 | 3970 | -16.296966 | -7.000261 | greedy | vf2-0 |
| ham_enc_unary_dvalues_4-4-4 | 12 | 1789 | 22329 | 22329 | 23613 | 12015 | 10344 | 11414 | -38.875279 | -18.026196 | greedy | vf2-1 |
| ham_ham_BK12 | 12 | 631 | 8539 | 6601 | 6890 | 7201 | 6410 | 6713 | -12.656569 | -11.292162 | vf2-1 | vf2-0 |
| ham_ham_JW12 | 12 | 631 | 7257 | 5294 | 5311 | 4636 | 4319 | 4313 | -9.248726 | -8.511668 | vf2-0 | vf2-0 |
| ham_ham_parity12 | 12 | 631 | 6472 | 6396 | 5458 | 4748 | 4239 | 4239 | -10.412238 | -8.508074 | vf2-0 | vf2-4 |
| ham_ham_BK-14 | 14 | 1086 | 13923 | 13721 | 13925 | 15462 | 12831 | 13820 | -24.504422 | -22.684466 | vf2-0 | vf2-3 |
| ham_ham_JW14 | 14 | 670 | 9103 | 8183 | 7522 | 6284 | 5785 | 5818 | -12.079034 | -11.283427 | vf2-0 | vf2-0 |
| ham_ham_parity-14 | 14 | 670 | 9403 | 6238 | 6339 | 7137 | 6398 | 6342 | -11.426271 | -13.037159 | vf2-1 | vf2-0 |
| ham_graph-1D-grid-pbc-qubitnodes_Lx-16_h | 16 | 32 | 76 | 48 | 48 | 73 | 61 | 71 | -0.103705 | -0.119841 | vf2-0 | vf2-3 |
| ham_mu_z_prime_enc_unary_dvalues_16-16-1 | 16 | 30 | 61 | 30 | 30 | 30 | 30 | 30 | -0.056078 | -0.056349 | vf2-0 | vf2-0 |
| ham_ham_BK22 | 22 | 5466 | 104323 | 95384 | 97605 | 138536 | 112608 | 134458 | -209.538056 | -245.832959 | vf2-2 | vf2-5 |
| ham_gnp-k_5_n-24_rinst-04 | 24 | 189 | 2788 | 2788 | 2836 | 1026 | 1057 | 1060 | -5.890098 | -2.300604 | greedy | greedy |
| ham_ham_JW24 | 24 | 6509 | 96807 | 88875 | 89144 | 191950 | 124642 | 211033 | -168.758 | -274.356115 | vf2-0 | vf2-1 |
| ham_mu_y_prime_enc_stdbinary_dvalues_4-4 | 24 | 48 | 24 | 24 | 24 | 24 | 24 | 24 | -0.070733 | -0.0652 | greedy | vf2-0 |
| ham_tsp_prob-ulysses22_Ncity-8_enc-stdbi | 24 | 449 | 3218 | 2766 | 2766 | 1924 | 1938 | 1967 | -5.79576 | -4.324392 | vf2-0 | vf2-1 |
| ham_graph-1D-grid-pbc-qubitnodes_Lx-26_h | 26 | 52 | 86 | 58 | 60 | 110 | 95 | 109 | -0.119896 | -0.171325 | vf2-0 | vf2-0 |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-1 | 28 | 491 | 1669 | 1669 | 1669 | 1051 | 1041 | 1365 | -3.712098 | -2.037921 | greedy | greedy |
| ham_bh_graph-2D-triag-nonpbc-qubitnodes_ | 28 | 939 | 8106 | 7493 | 8130 | 3665 | 3753 | 4077 | -16.113187 | -8.543885 | vf2-0 | greedy |
| ham_reg-5_n-10_rinst-07 | 30 | 1296 | 14510 | 15706 | 14516 | 6215 | 6283 | 6425 | -29.285749 | -14.241333 | vf2-0 | greedy |
| ham_bh_graph-3D-grid-nonpbc-qubitnodes_L | 32 | 881 | 8408 | 7591 | 10350 | 5202 | 5338 | 5533 | -20.136725 | -11.863325 | vf2-0 | vf2-0 |
| ham_mu_y_prime_enc_unary_dvalues_16-16-1 | 32 | 60 | 131 | 60 | 60 | 60 | 60 | 60 | -0.131288 | -0.113742 | vf2-0 | vf2-0 |
| ham_0-uuf100-0810.cnf-40-res | 40 | 188 | 2018 | 2018 | 2082 | 1340 | 1299 | 1372 | -4.295111 | -3.1545 | greedy | greedy |
| ham_8-uuf100-0509.cnf-40-cc | 40 | 207 | 2366 | 2366 | 2396 | 1549 | 1549 | 1582 | -4.975435 | -3.644281 | greedy | greedy |
| ham_gnp-k_4_n-40_rinst-07 | 40 | 805 | 13972 | 13972 | 14399 | 6123 | 5974 | 6144 | -29.719398 | -14.977603 | greedy | greedy |
| ham_2-uf100-0396.cnf-46-res | 46 | 272 | 3582 | 3582 | 4420 | 2144 | 2164 | 2382 | -8.547338 | -7.714806 | greedy | vf2-0 |
| ham_2-uf100-0601.cnf-46-res | 46 | 285 | 4032 | 4032 | 4098 | 2405 | 2457 | 2638 | -8.568711 | -6.45959 | greedy | greedy |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-2 | 48 | 841 | 4142 | 4142 | 6057 | 1781 | 1728 | 2309 | -14.935002 | -4.056781 | greedy | vf2-1 |
| ham_tsp_prob-lin105_Ncity-7_enc-unary | 49 | 344 | 5678 | 5678 | 5724 | 1637 | 1572 | 1670 | -11.892653 | -3.473306 | greedy | greedy |
| ham_bh_graph-2D-triag-pbc-qubitnodes_Lx- | 60 | 3271 | 40421 | 52899 | 42173 | 21725 | 21894 | 24433 | -85.243153 | None | vf2-0 | greedy |
| ham_mu_x_prime_enc_unary_dvalues_16-16-1 | 64 | 120 | 268 | 120 | 120 | 120 | 120 | 120 | -0.272939 | -0.262179 | vf2-0 | vf2-0 |
| ham_tsp_prob-d198_Ncity-8_enc-unary | 64 | 513 | 9318 | 9318 | 9672 | 2545 | 2452 | 2591 | -19.447501 | None | greedy | greedy |
| ham_tsp_prob-fl417_Ncity-16_enc-stdbinar | 64 | 3841 | 44694 | 44694 | 43512 | 18489 | 18417 | 19284 | -118.619285 | None | greedy | greedy |
| ham_tsp_prob-kroD100_Ncity-16_enc-stdbin | 64 | 3841 | 44684 | 44684 | 43502 | 18700 | 18602 | 19284 | -118.615967 | None | greedy | greedy |
| ham_4-uf100-0246.cnf-70-res | 70 | 689 | 15334 | 15334 | 15998 | 8557 | 8559 | 8867 | -35.83624 | None | greedy | greedy |
| ham_dsjc1000.1,n-36,rinst-4 | 72 | 676 | 8306 | 8306 | 9534 | 3851 | 3808 | 3827 | -18.466681 | None | greedy | vf2-4 |
| ham_graph-2D-grid-nonpbc-qubitnodes_Lx-5 | 75 | 205 | 1858 | 1636 | 1834 | 1055 | 998 | 1116 | -4.062653 | None | vf2-0 | vf2-1 |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-1 | 80 | 3981 | 52947 | 52947 | 71182 | 24295 | 23536 | 26357 | -164.176294 | None | greedy | vf2-0 |
| ham_4-flat100-7.cnf-90-cc | 90 | 224 | 2060 | 1948 | 2078 | 1097 | 1011 | 1198 | -5.088991 | None | vf2-0 | vf2-0 |
| ham_gnp-k_5_n-90_rinst-02 | 90 | 2959 | 89754 | 89754 | 94987 | 29467 | 29516 | 31324 | -271.067458 | None | greedy | greedy |
| ham_reg-4_n-90_rinst-07 | 90 | 541 | 9405 | 9405 | 9968 | 3928 | 3802 | 4202 | -26.808989 | None | greedy | greedy |
| ham_graph-2D-triag-pbc-qubitnodes_Lx-13_ | 91 | 364 | 3768 | 3768 | 4150 | 2687 | 2377 | 3057 | -11.159857 | None | greedy | greedy |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-4 | 92 | 1611 | 9931 | 8830 | 9436 | 5245 | 5369 | 6594 | -35.995975 | None | vf2-0 | vf2-0 |
| ham_bh_graph-2D-grid-nonpbc-qubitnodes_L | 98 | 2836 | 41467 | 39012 | 43646 | 14839 | 13953 | 16763 | -129.304123 | None | vf2-0 | greedy |
| ham_fh-graph-1D-grid-pbc-qubitnodes_Lx-5 | 100 | 351 | 4705 | 3222 | 3626 | 2971 | 3001 | 3823 | -9.727893 | None | vf2-0 | vf2-0 |
| ham_tsp_prob-pr76_Ncity-10_enc-unary | 100 | 1001 | 22182 | 22182 | 24550 | 5245 | 5032 | 6209 | -59.583839 | None | greedy | greedy |
| ham_tsp_prob-st70_Ncity-10_enc-unary | 100 | 1001 | 22184 | 22184 | 24554 | 5278 | 5396 | 6209 | -59.588982 | None | greedy | greedy |
| ham_queen13_13,n-28,rinst-0 | 112 | 413 | 3578 | 3578 | 4994 | 2355 | 2189 | 2714 | -13.711002 | None | greedy | vf2-0 |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-6 | 120 | 2101 | 15339 | 15339 | 17159 | 4877 | 4769 | 6690 | -56.107487 | None | greedy | greedy |
| ham_will199gpia,n-60,rinst-8 | 120 | 1513 | 20766 | 20766 | 22190 | 9665 | 9229 | 9928 | -67.031184 | None | greedy | greedy |

## FakeKingston

- P: 67 eligible; exact 23, not checked 44, NOT exact 0; on failed elements 0; placement chosen {'vf2': 22, 'greedy': 45}; errors 0; time 178.7 s summed
- PE: 67 eligible; exact 23, not checked 44, NOT exact 0; on failed elements 0; placement chosen {'vf2': 45, 'greedy': 22}; errors 0; time 189.2 s summed

| arm / other | q2 ratio (gmean, +1) | fewer / equal / more | ESP ratio (gmean, both > 0; best >= 0.01) | ESP 10% better / worse |
|---|---|---|---|---|
| P / S4O | 0.949 | 19 / 45 / 3 | 0.998 (26) | 0 / 2 |
| P / QK2 | 1.450 | 10 / 8 / 49 | 0.367 (30) | 4 / 10 |
| P / QK3 | 1.503 | 8 / 8 / 51 | 0.246 (33) | 2 / 14 |
| P / PSFR | 1.421 | 8 / 8 / 51 | 0.321 (32) | 0 / 15 |
| PE / S4O | 1.013 | 19 / 12 / 36 | 1.121 (26) | 8 / 0 |
| PE / QK2 | 1.547 | 7 / 8 / 52 | 0.487 (31) | 5 / 8 |
| PE / QK3 | 1.604 | 6 / 8 / 53 | 0.311 (33) | 3 / 11 |
| PE / PSFR | 1.517 | 5 / 8 / 54 | 0.417 (32) | 1 / 12 |

| test | n | terms | S4O | P | PE | QK2 | QK3 | PSFR | PE log10 ESP | QK3 log10 ESP | P placement | PE placement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ham_enc_stdbinary_dvalues_d-level-4 | 2 | 10 | 3 | 3 | 3 | 3 | 3 | 3 | -0.002654 | -0.001967 | greedy | vf2-0 |
| ham_mu_y_prime_enc_stdbinary_dvalues_8-8 | 3 | 12 | 19 | 19 | 19 | 18 | 18 | 18 | -0.012894 | -0.011264 | greedy | vf2-0 |
| ham_enc_gray_dvalues_d-level-16 | 4 | 99 | 223 | 223 | 223 | 174 | 176 | 180 | -0.148026 | -0.1145 | greedy | vf2-4 |
| ham_gnp-k_2_n-4_rinst-05 | 4 | 13 | 33 | 33 | 33 | 24 | 22 | 24 | -0.01852 | -0.014875 | greedy | vf2-0 |
| ham_ham_parity-4 | 4 | 27 | 62 | 71 | 72 | 62 | 62 | 63 | -0.047751 | -0.036485 | vf2-0 | vf2-1 |
| ham_mu_y_prime_enc_gray_dvalues_16-16-16 | 4 | 32 | 125 | 110 | 107 | 85 | 92 | 85 | -0.067651 | -0.054784 | vf2-0 | vf2-4 |
| ham_mu_y_prime_enc_stdbinary_dvalues_16- | 4 | 32 | 57 | 57 | 87 | 72 | 72 | 72 | -0.062082 | -0.041915 | greedy | vf2-0 |
| ham_mu_y_prime_enc_unary_dvalues_4-4-4 | 4 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | -0.004396 | -0.004396 | greedy | vf2-0 |
| ham_mu_z_prime_enc_gray_dvalues_4-4-4-4 | 4 | 8 | 4 | 4 | 4 | 4 | 4 | 4 | -0.00328 | -0.002554 | greedy | vf2-0 |
| ham_gnp-k_4_n-6_rinst-15 | 6 | 46 | 277 | 277 | 279 | 132 | 132 | 137 | -0.172448 | -0.086889 | greedy | vf2-1 |
| ham_mu_y_prime_enc_stdbinary_dvalues_8-8 | 6 | 24 | 38 | 38 | 38 | 36 | 36 | 36 | -0.027148 | -0.022773 | greedy | vf2-0 |
| ham_7-uf20-0384.cnf-8-res | 8 | 30 | 112 | 112 | 140 | 78 | 81 | 78 | -0.08332 | -0.056105 | greedy | vf2-0 |
| ham_ham_JW-8 | 8 | 185 | 1298 | 1298 | 1436 | 1370 | 1341 | 1341 | -0.85843 | -0.857261 | greedy | vf2-1 |
| ham_mu_y_prime_enc_unary_dvalues_8-8-8-8 | 8 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | -0.010587 | -0.010946 | greedy | vf2-0 |
| ham_tsp_rand-002_Ncity-4_enc-stdbinary | 8 | 49 | 228 | 228 | 230 | 133 | 133 | 133 | -0.146334 | -0.090518 | greedy | vf2-0 |
| ham_complbipart-n-10_a-5_b-5 | 10 | 76 | 649 | 649 | 634 | 271 | 287 | 287 | -0.424063 | -0.213089 | greedy | vf2-0 |
| ham_ham_JW10 | 10 | 276 | 2893 | 2573 | 2505 | 2414 | 2340 | 2365 | -1.694175 | -1.736033 | vf2-0 | vf2-4 |
| ham_mu_x_prime_enc_stdbinary_dvalues_8-8 | 11 | 40 | 59 | 59 | 59 | 56 | 56 | 56 | -0.080812 | -0.036706 | greedy | vf2-0 |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-4 | 12 | 1177 | 7811 | 7811 | 10021 | 4501 | 4320 | 4205 | -6.823329 | -3.393586 | greedy | vf2-1 |
| ham_enc_unary_dvalues_4-4-4 | 12 | 1789 | 20769 | 20769 | 20769 | 11742 | 10641 | 10305 | -13.934736 | -7.614111 | greedy | greedy |
| ham_ham_BK12 | 12 | 631 | 7888 | 6612 | 6633 | 7035 | 6141 | 6598 | -4.382591 | -4.421564 | vf2-1 | vf2-0 |
| ham_ham_JW12 | 12 | 631 | 7017 | 5296 | 5314 | 4616 | 4311 | 4313 | -4.166386 | -3.422182 | vf2-0 | vf2-0 |
| ham_ham_parity12 | 12 | 631 | 6420 | 6396 | 5518 | 4738 | 4243 | 4245 | -4.160661 | -3.349908 | vf2-0 | vf2-0 |
| ham_ham_BK-14 | 14 | 1086 | 14734 | 13641 | 13847 | 15564 | 14647 | 15648 | -9.084133 | -11.848897 | vf2-0 | vf2-0 |
| ham_ham_JW14 | 14 | 670 | 8741 | 8184 | 7553 | 6365 | 5772 | 5784 | -6.050225 | -4.596224 | vf2-0 | vf2-0 |
| ham_ham_parity-14 | 14 | 670 | 7995 | 6226 | 6133 | 7145 | 6376 | 6403 | -4.855493 | -5.198179 | vf2-0 | vf2-0 |
| ham_graph-1D-grid-pbc-qubitnodes_Lx-16_h | 16 | 32 | 70 | 48 | 48 | 78 | 71 | 43 | -0.047769 | -0.065284 | vf2-0 | vf2-1 |
| ham_mu_z_prime_enc_unary_dvalues_16-16-1 | 16 | 30 | 49 | 30 | 30 | 30 | 30 | 30 | -0.033978 | -0.029233 | vf2-0 | vf2-0 |
| ham_ham_BK22 | 22 | 5466 | 103894 | 95078 | 94531 | 131161 | 115495 | 135917 | -83.444906 | -105.152201 | vf2-4 | vf2-2 |
| ham_gnp-k_5_n-24_rinst-04 | 24 | 189 | 2738 | 2738 | 3024 | 986 | 972 | 988 | -2.843157 | -0.828789 | greedy | vf2-0 |
| ham_ham_JW24 | 24 | 6509 | 83180 | 85951 | 81629 | 189813 | 118553 | 207548 | -66.247689 | -106.000445 | vf2-0 | vf2-1 |
| ham_mu_y_prime_enc_stdbinary_dvalues_4-4 | 24 | 48 | 24 | 24 | 24 | 24 | 24 | 24 | -0.036865 | -0.033935 | greedy | vf2-0 |
| ham_tsp_prob-ulysses22_Ncity-8_enc-stdbi | 24 | 449 | 3218 | 2760 | 3230 | 1912 | 1909 | 1944 | -3.196058 | -1.671318 | vf2-0 | vf2-2 |
| ham_graph-1D-grid-pbc-qubitnodes_Lx-26_h | 26 | 52 | 72 | 60 | 60 | 127 | 112 | 113 | -0.081544 | -0.101572 | vf2-0 | vf2-1 |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-1 | 28 | 491 | 1945 | 1945 | 1945 | 1068 | 999 | 1415 | -2.360478 | -0.902822 | greedy | greedy |
| ham_bh_graph-2D-triag-nonpbc-qubitnodes_ | 28 | 939 | 7576 | 7576 | 7934 | 3892 | 3845 | 4302 | -8.734426 | -3.71839 | greedy | vf2-2 |
| ham_reg-5_n-10_rinst-07 | 30 | 1296 | 14442 | 14442 | 16916 | 6234 | 6319 | 6337 | -16.774103 | -6.009068 | greedy | vf2-2 |
| ham_bh_graph-3D-grid-nonpbc-qubitnodes_L | 32 | 881 | 7929 | 7593 | 8104 | 5202 | 5364 | 5929 | -7.117487 | -5.29124 | vf2-0 | vf2-0 |
| ham_mu_y_prime_enc_unary_dvalues_16-16-1 | 32 | 60 | 93 | 60 | 60 | 60 | 60 | 60 | -0.080805 | -0.06311 | vf2-0 | vf2-0 |
| ham_0-uuf100-0810.cnf-40-res | 40 | 188 | 1962 | 1962 | 2518 | 1285 | 1321 | 1357 | -2.523064 | -1.311317 | greedy | vf2-0 |
| ham_8-uuf100-0509.cnf-40-cc | 40 | 207 | 2366 | 2366 | 3082 | 1582 | 1563 | 1603 | -2.74876 | -1.582954 | greedy | vf2-4 |
| ham_gnp-k_4_n-40_rinst-07 | 40 | 805 | 13743 | 13743 | 14999 | 5871 | 5953 | 6207 | -15.780788 | -6.086552 | greedy | greedy |
| ham_2-uf100-0396.cnf-46-res | 46 | 272 | 3514 | 3514 | 3844 | 2274 | 2150 | 2387 | -4.061125 | -2.2466 | greedy | greedy |
| ham_2-uf100-0601.cnf-46-res | 46 | 285 | 3966 | 3966 | 4370 | 2483 | 2398 | 2471 | -4.333911 | -2.662641 | greedy | greedy |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-2 | 48 | 841 | 3787 | 3787 | 3859 | 1835 | 1794 | 2366 | -4.895475 | -1.746346 | greedy | greedy |
| ham_tsp_prob-lin105_Ncity-7_enc-unary | 49 | 344 | 5304 | 5304 | 5854 | 1755 | 1559 | 1825 | -6.436945 | -1.70703 | greedy | greedy |
| ham_bh_graph-2D-triag-pbc-qubitnodes_Lx- | 60 | 3271 | 37831 | 50255 | 48151 | 21822 | 21946 | 24138 | -47.861979 | -25.297006 | vf2-0 | vf2-0 |
| ham_mu_x_prime_enc_unary_dvalues_16-16-1 | 64 | 120 | 188 | 120 | 120 | 120 | 120 | 120 | -0.151203 | -0.155606 | vf2-0 | vf2-0 |
| ham_tsp_prob-d198_Ncity-8_enc-unary | 64 | 513 | 8818 | 8818 | 9796 | 2422 | 2574 | 2633 | -10.144312 | -2.802264 | greedy | greedy |
| ham_tsp_prob-fl417_Ncity-16_enc-stdbinar | 64 | 3841 | 32210 | 32210 | 37844 | 18746 | 18495 | 19132 | -35.849339 | -19.81359 | greedy | greedy |
| ham_tsp_prob-kroD100_Ncity-16_enc-stdbin | 64 | 3841 | 32212 | 32212 | 37844 | 18676 | 18401 | 19132 | -35.845907 | -24.174651 | greedy | greedy |
| ham_4-uf100-0246.cnf-70-res | 70 | 689 | 14506 | 14506 | 16160 | 8488 | 8508 | 8755 | -16.809998 | -10.575809 | greedy | greedy |
| ham_dsjc1000.1,n-36,rinst-4 | 72 | 676 | 7870 | 7870 | 8826 | 3860 | 3811 | 4038 | -8.382712 | -4.88523 | greedy | vf2-0 |
| ham_graph-2D-grid-nonpbc-qubitnodes_Lx-5 | 75 | 205 | 1564 | 1310 | 1480 | 1063 | 931 | 1127 | -1.466488 | None | vf2-0 | vf2-0 |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-1 | 80 | 3981 | 54184 | 54184 | 67306 | 24235 | 24391 | 27020 | -65.967509 | None | greedy | greedy |
| ham_4-flat100-7.cnf-90-cc | 90 | 224 | 1622 | 1622 | 1696 | 1143 | 977 | 1091 | -1.770051 | None | greedy | greedy |
| ham_gnp-k_5_n-90_rinst-02 | 90 | 2959 | 82356 | 82356 | 90254 | 29433 | 29438 | 30143 | -89.377531 | None | greedy | greedy |
| ham_reg-4_n-90_rinst-07 | 90 | 541 | 8386 | 8386 | 9558 | 3796 | 3762 | 4014 | -9.497222 | None | greedy | greedy |
| ham_graph-2D-triag-pbc-qubitnodes_Lx-13_ | 91 | 364 | 3456 | 3456 | 4012 | 2623 | 2565 | 2959 | -4.266163 | None | greedy | greedy |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-4 | 92 | 1611 | 8857 | 6973 | 7699 | 5473 | 5342 | 5979 | -8.216568 | -6.880577 | vf2-0 | vf2-0 |
| ham_bh_graph-2D-grid-nonpbc-qubitnodes_L | 98 | 2836 | 34730 | 34730 | 39822 | 14606 | 14689 | 15829 | -40.529171 | None | greedy | greedy |
| ham_fh-graph-1D-grid-pbc-qubitnodes_Lx-5 | 100 | 351 | 4021 | 3158 | 3591 | 3170 | 2645 | 3165 | -3.488831 | None | vf2-0 | vf2-4 |
| ham_tsp_prob-pr76_Ncity-10_enc-unary | 100 | 1001 | 20894 | 20894 | 23534 | 5274 | 4896 | 5629 | -22.508021 | None | greedy | greedy |
| ham_tsp_prob-st70_Ncity-10_enc-unary | 100 | 1001 | 20894 | 20894 | 23532 | 5281 | 5104 | 5629 | -22.506649 | None | greedy | greedy |
| ham_queen13_13,n-28,rinst-0 | 112 | 413 | 3506 | 3506 | 4054 | 2335 | 2189 | 2414 | -4.326289 | None | greedy | greedy |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-6 | 120 | 2101 | 14459 | 14459 | 14883 | 4193 | 4193 | 6646 | -21.71327 | None | greedy | greedy |
| ham_will199gpia,n-60,rinst-8 | 120 | 1513 | 18182 | 18182 | 20350 | 9335 | 9375 | 9563 | -20.873973 | None | greedy | greedy |

