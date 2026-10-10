# BP-REC (exploratory, nothing predicted)

git head 0b7f790, uncommitted tracked changes: none; 106 FakeTorino tests of BP-FINAL, 212 jobs, 4 at a time, 12 CPUs; psf_zero_core57 2026-10-10.c30; the real clock

- both finished: 106; failed: R071 0, R102 0
- R102 / R071 compile time: geometric mean 0.839, median 0.962; summed 2013 s -> 567 s
- the same output by value: 106 of 106; the same two-qubit count: 106
- R071's two-qubit count equals BP-FINAL's RECR (the same release, work PC) on 103 of 105
- R102's estimate and check calls: rust 720, python 0, fallback 0
- two-qubit count / Qiskit level 2 (BP-FINAL's QK), geometric mean of (q2 + 1) ratios on 106: R071 1.035, R102 1.035
- tests with gates on failed elements: R071 0, R102 0
- not valid (Benchpress): R071 0, R102 0

| test | qubits | R071 s | R102 s | R102 / R071 | q2 R071 / R102 | same value |
|---|---|---|---|---|---|---|
| hwb11.qasm | 15 | 1207.70 | 89.17 | 0.07 | 331859 / 331859 | yes |
| ham_ham_BK12 | 12 | 180.31 | 23.08 | 0.13 | 6713 / 6713 | yes |
| hwb12.qasm | 20 | 72.00 | 84.89 | 1.18 | 646431 / 646431 | yes |
| ham_ham_parity-14 | 14 | 64.52 | 17.52 | 0.27 | 6342 / 6342 | yes |
| ham_ham_JW14 | 14 | 60.56 | 23.26 | 0.38 | 5818 / 5818 | yes |
| hwb8.qasm | 12 | 47.89 | 23.82 | 0.50 | 18813 / 18813 | yes |
| ham_ham_JW24 | 24 | 47.16 | 70.54 | 1.50 | 211033 / 211033 | yes |
| ham_ham_BK-14 | 14 | 47.11 | 47.29 | 1.00 | 13820 / 13820 | yes |
| ham_enc_unary_dvalues_4-4-4 | 12 | 33.77 | 23.26 | 0.69 | 11414 / 11414 | yes |
| ham_ham_BK22 | 22 | 29.42 | 24.15 | 0.82 | 134458 / 134458 | yes |
| csla_mux_3.qasm | 15 | 26.76 | 0.65 | 0.02 | 147 / 147 | yes |
| ham_graph-1D-grid-pbc-qubitnodes_Lx-16_h-2 | 16 | 20.40 | 1.19 | 0.06 | 71 / 71 | yes |
| ham_mu_z_prime_enc_unary_dvalues_16-16-16-16- | 16 | 15.24 | 0.37 | 0.02 | 30 / 30 | yes |
| ham_tsp_prob-fl417_Ncity-16_enc-stdbinary | 64 | 15.05 | 8.05 | 0.53 | 19284 / 19284 | yes |
| ham_ham_parity12 | 12 | 14.24 | 9.81 | 0.69 | 4239 / 4239 | yes |
| ham_ham_JW12 | 12 | 13.37 | 7.57 | 0.57 | 4313 / 4313 | yes |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-4_U-40 | 12 | 13.24 | 7.54 | 0.57 | 3970 / 3970 | yes |
| ham_bh_graph-2D-grid-nonpbc-qubitnodes_Lx-7_L | 98 | 9.87 | 5.30 | 0.54 | 16763 / 16763 | yes |
| ham_bh_graph-2D-triag-pbc-qubitnodes_Lx-5_Ly- | 60 | 8.27 | 5.03 | 0.61 | 24433 / 24433 | yes |
| ham_tsp_prob-kroD100_Ncity-16_enc-stdbinary | 64 | 7.56 | 7.69 | 1.02 | 19284 / 19284 | yes |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-10_U-3 | 80 | 6.70 | 6.61 | 0.99 | 26357 / 26357 | yes |
| ham_ham_JW10 | 10 | 6.68 | 9.18 | 1.37 | 2369 / 2369 | yes |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-60_U-1 | 120 | 4.75 | 4.42 | 0.93 | 6690 / 6690 | yes |
| gf2^32_mult.qasm | 96 | 4.58 | 4.98 | 1.09 | 21160 / 21160 | yes |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-46_U-2 | 92 | 3.76 | 2.93 | 0.78 | 6594 / 6594 | yes |
| ham_ham_JW-8 | 8 | 3.29 | 4.42 | 1.34 | 1341 / 1341 | yes |
| ham_will199gpia,n-60,rinst-8 | 120 | 2.71 | 2.64 | 0.98 | 9928 / 9928 | yes |
| ham_gnp-k_5_n-90_rinst-02 | 90 | 2.03 | 1.53 | 0.75 | 31324 / 31324 | yes |
| gf2^4_mult.qasm | 12 | 1.99 | 1.43 | 0.71 | 213 / 213 | yes |
| ham_mu_y_prime_enc_stdbinary_dvalues_8-8-8-8- | 6 | 1.59 | 0.44 | 0.28 | 36 / 36 | yes |
| ham_tsp_prob-st70_Ncity-10_enc-unary | 100 | 1.40 | 0.62 | 0.44 | 6209 / 6209 | yes |
| ham_dsjc1000.1,n-36,rinst-4 | 72 | 1.33 | 3.64 | 2.74 | 3827 / 3827 | yes |
| ham_reg-5_n-10_rinst-07 | 30 | 1.29 | 1.60 | 1.24 | 6425 / 6425 | yes |
| ham_4-uf100-0246.cnf-70-res | 70 | 1.25 | 1.74 | 1.39 | 8867 / 8867 | yes |
| ham_mu_x_prime_enc_stdbinary_dvalues_8-8-8-8- | 11 | 1.21 | 3.64 | 3.00 | 56 / 56 | yes |
| ham_bh_graph-2D-triag-nonpbc-qubitnodes_Lx-3_ | 28 | 1.19 | 0.97 | 0.82 | 4077 / 4077 | yes |
| ham_fh-graph-1D-grid-pbc-qubitnodes_Lx-50_U-2 | 100 | 1.12 | 0.42 | 0.38 | 3823 / 3823 | yes |
| ham15-high.qasm | 20 | 1.06 | 0.48 | 0.46 | 4787 / 4787 | yes |
| ham_enc_gray_dvalues_d-level-16 | 4 | 1.03 | 2.12 | 2.05 | 202 / 202 | yes |
| ham_bh_graph-3D-grid-nonpbc-qubitnodes_Lx-2_L | 32 | 0.97 | 2.02 | 2.09 | 5533 / 5533 | yes |
| barenco_tof_4.qasm | 7 | 0.97 | 0.31 | 0.32 | 74 / 74 | yes |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-24_U-2 | 48 | 0.94 | 0.97 | 1.03 | 2309 / 2309 | yes |
| qcla_mod_7.qasm | 26 | 0.93 | 0.36 | 0.39 | 800 / 800 | yes |
| ham_queen13_13,n-28,rinst-0 | 112 | 0.88 | 0.34 | 0.39 | 2714 / 2714 | yes |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-14_U-1 | 28 | 0.87 | 0.44 | 0.50 | 1365 / 1365 | yes |
| vbe_adder_3.qasm | 10 | 0.86 | 0.37 | 0.43 | 102 / 102 | yes |
| ham_graph-1D-grid-pbc-qubitnodes_Lx-26_h-6 | 26 | 0.81 | 0.78 | 0.96 | 109 / 109 | yes |
| qft_4.qasm | 5 | 0.78 | 0.27 | 0.35 | 76 / 76 | yes |
| ham_complbipart-n-10_a-5_b-5 | 10 | 0.77 | 0.45 | 0.58 | 271 / 271 | yes |
| qft.qasm | 4 | 0.77 | 0.24 | 0.31 | 17 / 17 | yes |
| ham_tsp_prob-pr76_Ncity-10_enc-unary | 100 | 0.71 | 0.71 | 1.00 | 6209 / 6209 | yes |
| rc_adder_6.qasm | 14 | 0.70 | 0.67 | 0.95 | 148 / 148 | yes |
| hwb6.qasm | 7 | 0.65 | 1.01 | 1.56 | 200 / 200 | yes |
| gf2^9_mult.qasm | 27 | 0.63 | 0.25 | 0.39 | 1564 / 1564 | yes |
| mod_adder_1024.qasm | 28 | 0.62 | 0.66 | 1.06 | 4371 / 4371 | yes |
| gf2^10_mult.qasm | 30 | 0.60 | 0.48 | 0.81 | 1878 / 1878 | yes |
| ham_2-uf100-0601.cnf-46-res | 46 | 0.60 | 2.26 | 3.77 | 2638 / 2638 | yes |
| ham_tsp_prob-d198_Ncity-8_enc-unary | 64 | 0.59 | 0.62 | 1.05 | 2591 / 2591 | yes |
| ham_tsp_prob-ulysses22_Ncity-8_enc-stdbinary | 24 | 0.55 | 0.54 | 0.97 | 1967 / 1967 | yes |
| ham_gnp-k_5_n-24_rinst-04 | 24 | 0.54 | 0.30 | 0.55 | 1060 / 1060 | yes |
| ham_mu_y_prime_enc_gray_dvalues_16-16-16 | 4 | 0.53 | 0.39 | 0.73 | 92 / 92 | yes |
| ham_gnp-k_4_n-6_rinst-15 | 6 | 0.53 | 1.01 | 1.92 | 139 / 139 | yes |
| ham_tsp_rand-002_Ncity-4_enc-stdbinary | 8 | 0.52 | 0.37 | 0.71 | 136 / 136 | yes |
| adder.qasm | 10 | 0.49 | 0.39 | 0.79 | 94 / 94 | yes |
| ham_7-uf20-0384.cnf-8-res | 8 | 0.48 | 0.89 | 1.86 | 78 / 78 | yes |
| tof_5.qasm | 9 | 0.43 | 0.27 | 0.63 | 60 / 60 | yes |
| gf2^16_mult.qasm | 48 | 0.43 | 0.39 | 0.90 | 4945 / 4945 | yes |
| ham_ham_parity-4 | 4 | 0.42 | 0.44 | 1.03 | 63 / 63 | yes |
| ham_mu_y_prime_enc_stdbinary_dvalues_16-16-16 | 4 | 0.39 | 0.80 | 2.04 | 72 / 72 | yes |
| ham_mu_x_prime_enc_unary_dvalues_16-16-16-16- | 64 | 0.39 | 0.92 | 2.39 | 120 / 120 | yes |
| ham_reg-4_n-90_rinst-07 | 90 | 0.38 | 0.44 | 1.17 | 4202 / 4202 | yes |
| gf2^5_mult.qasm | 15 | 0.37 | 0.33 | 0.90 | 405 / 405 | yes |
| ham_2-uf100-0396.cnf-46-res | 46 | 0.36 | 0.36 | 1.02 | 2382 / 2382 | yes |
| ham_mu_y_prime_enc_stdbinary_dvalues_8-8-4-4 | 3 | 0.35 | 0.26 | 0.74 | 18 / 18 | yes |
| ham_8-uuf100-0509.cnf-40-cc | 40 | 0.34 | 0.34 | 0.99 | 1582 / 1582 | yes |
| ham_tsp_prob-lin105_Ncity-7_enc-unary | 49 | 0.34 | 0.94 | 2.80 | 1670 / 1670 | yes |
| barenco_tof_3.qasm | 5 | 0.34 | 0.28 | 0.82 | 36 / 36 | yes |
| tof_4.qasm | 7 | 0.33 | 0.27 | 0.83 | 41 / 41 | yes |
| ham_graph-2D-triag-pbc-qubitnodes_Lx-13_Ly-13 | 91 | 0.31 | 0.30 | 0.97 | 3057 / 3057 | yes |
| ham_gnp-k_4_n-40_rinst-07 | 40 | 0.31 | 0.30 | 0.98 | 6144 / 6144 | yes |
| qcla_adder_10.qasm | 36 | 0.31 | 0.73 | 2.39 | 528 / 528 | yes |
| ham_4-flat100-7.cnf-90-cc | 90 | 0.31 | 0.34 | 1.11 | 1198 / 1198 | yes |
| ham_mu_y_prime_enc_unary_dvalues_16-16-16-16- | 32 | 0.30 | 0.79 | 2.64 | 60 / 60 | yes |
| ham_mu_y_prime_enc_unary_dvalues_8-8-8-8-8-8- | 8 | 0.30 | 0.23 | 0.77 | 14 / 14 | yes |
| tof_3.qasm | 5 | 0.28 | 0.59 | 2.08 | 26 / 26 | yes |
| teleportv2.qasm | 3 | 0.28 | 0.20 | 0.72 | 2 / 2 | yes |
| W-state.qasm | 3 | 0.27 | 0.28 | 1.04 | 10 / 10 | yes |
| ham_gnp-k_2_n-4_rinst-05 | 4 | 0.26 | 0.33 | 1.29 | 24 / 24 | yes |
| ham15-med.qasm | 17 | 0.25 | 0.25 | 0.98 | 1238 / 1238 | yes |
| adder_8.qasm | 24 | 0.24 | 0.31 | 1.31 | 784 / 784 | yes |
| inverseqft1.qasm | 4 | 0.22 | 0.56 | 2.49 | 0 / 0 | yes |
| ham_graph-2D-grid-nonpbc-qubitnodes_Lx-5_Ly-1 | 75 | 0.22 | 1.02 | 4.62 | 1116 / 1116 | yes |
| ham_0-uuf100-0810.cnf-40-res | 40 | 0.22 | 0.23 | 1.04 | 1372 / 1372 | yes |
| inverseqft2.qasm | 4 | 0.22 | 0.21 | 0.95 | 0 / 0 | yes |
| ham_mu_y_prime_enc_unary_dvalues_4-4-4 | 4 | 0.22 | 0.21 | 0.97 | 6 / 6 | yes |
| ham15-low.qasm | 17 | 0.22 | 0.18 | 0.84 | 839 / 839 | yes |
| csum_mux_9.qasm | 30 | 0.21 | 0.42 | 2.00 | 436 / 436 | yes |
| ham_mu_y_prime_enc_stdbinary_dvalues_4-4-4-4- | 24 | 0.21 | 0.30 | 1.45 | 24 / 24 | yes |
| qec.qasm | 5 | 0.21 | 0.28 | 1.31 | 4 / 4 | yes |
| tof_10.qasm | 19 | 0.20 | 0.17 | 0.81 | 145 / 145 | yes |
| teleport.qasm | 3 | 0.19 | 0.20 | 1.04 | 2 / 2 | yes |
| qpt.qasm | 1 | 0.18 | 0.24 | 1.34 | 0 / 0 | yes |
| gf2^7_mult.qasm | 21 | 0.18 | 0.46 | 2.56 | 905 / 905 | yes |
| bigadder.qasm | 18 | 0.18 | 0.58 | 3.27 | 246 / 246 | yes |
| ham_mu_z_prime_enc_gray_dvalues_4-4-4-4 | 4 | 0.16 | 0.13 | 0.81 | 4 / 4 | yes |
| ham_enc_stdbinary_dvalues_d-level-4 | 2 | 0.13 | 0.13 | 0.98 | 3 / 3 | yes |
