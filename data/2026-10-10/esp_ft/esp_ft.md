# ESP-FT (exploratory, nothing predicted)

git head 0b7f790, uncommitted tracked changes: none; 106 FakeTorino tests of BP-FINAL on FakeTorino, FakeKingston; 848 jobs, 4 at a time; {'qiskit': '2.5.2', 'qiskit_ibm_runtime': '0.49.0'}

ESP: product of (1 - error) over gates and measurements as the Target reports them; no idling term. Failed element: error >= 0.5.

## FakeTorino

- tests with all four arms finished: 106 of 106; failed jobs: QK2 0, QK3 0, PSFD 0, PSFR 0

| arm | tests with ops on failed elements | ops on failed elements | ESP = 0 (an op with error 1) | not in Target | q2 / QK2 (gmean, +1) | time (s, summed) |
|---|---|---|---|---|---|---|
| QK2 | 27 | 9167 | 27 | 25 | 1.000 | 60 |
| QK3 | 22 | 9218 | 22 | 25 | 0.976 | 98 |
| PSFD | 63 | 75209 | 63 | 25 | 1.056 | 207 |
| PSFR | 0 | 0 | 0 | 25 | 1.027 | 446 |

**Tests where the best arm's ESP >= 0.01: 52.** ESP ratios on them (PSFR / other; an ESP of 0 counts as a ratio over 10^6):

| other | median ratio | ratio >= 2 | ratio >= 10 | ratio <= 0.5 | ratio in (0.5, 2) |
|---|---|---|---|---|---|
| QK2 | 1.05 | 6 | 2 | 0 | 46 |
| QK3 | 1 | 0 | 0 | 1 | 51 |
| PSFD | 1.22 | 19 | 19 | 0 | 33 |

**Tests where the best arm's ESP >= 1e-06: 66.** ESP ratios on them (PSFR / other; an ESP of 0 counts as a ratio over 10^6):

| other | median ratio | ratio >= 2 | ratio >= 10 | ratio <= 0.5 | ratio in (0.5, 2) |
|---|---|---|---|---|---|
| QK2 | 1.05 | 12 | 5 | 3 | 51 |
| QK3 | 1 | 4 | 3 | 5 | 57 |
| PSFD | 1.42 | 32 | 32 | 0 | 34 |

Per test (log10 ESP; '0' = an op with error 1; failed = ops on failed elements):

| test | qubits | QK2 | QK3 | PSFD | PSFR | failed QK2 / QK3 / PSFD / PSFR |
|---|---|---|---|---|---|---|
| qpt.qasm | 1 | -0.00 | -0.00 | -0.08 | -0.00 | 0 / 0 / 0 / 0 |
| ham_enc_stdbinary_dvalues_d-level-4 | 2 | -0.00 | -0.00 | -0.03 | -0.00 | 0 / 0 / 0 / 0 |
| ham_mu_z_prime_enc_gray_dvalues_4-4-4-4 | 4 | -0.01 | -0.00 | -0.01 | -0.00 | 0 / 0 / 0 / 0 |
| ham_mu_y_prime_enc_unary_dvalues_4-4-4 | 4 | -0.01 | -0.01 | -0.01 | -0.01 | 0 / 0 / 0 / 0 |
| inverseqft1.qasm | 4 | -0.01 | -0.01 | -0.12 | -0.12 | 0 / 0 / 0 / 0 |
| inverseqft2.qasm | 4 | -0.01 | -0.01 | -0.12 | -0.12 | 0 / 0 / 0 / 0 |
| teleport.qasm | 3 | -0.02 | -0.02 | -0.10 | -0.02 | 0 / 0 / 0 / 0 |
| teleportv2.qasm | 3 | -0.02 | -0.02 | -0.10 | -0.02 | 0 / 0 / 0 / 0 |
| ham_mu_y_prime_enc_stdbinary_dvalues_8-8-4-4 | 3 | -0.04 | -0.02 | -0.04 | -0.02 | 0 / 0 / 0 / 0 |
| ham_mu_y_prime_enc_unary_dvalues_8-8-8-8-8-8 | 8 | -0.02 | -0.02 | -0.03 | -0.02 | 0 / 0 / 0 / 0 |
| W-state.qasm | 3 | -0.03 | -0.03 | -0.07 | -0.03 | 0 / 0 / 0 / 0 |
| ham_gnp-k_2_n-4_rinst-05 | 4 | -0.03 | -0.03 | -0.14 | -0.03 | 0 / 0 / 0 / 0 |
| tof_3.qasm | 5 | -0.05 | -0.03 | -0.06 | -0.04 | 0 / 0 / 0 / 0 |
| qec.qasm | 5 | -0.03 | -0.03 | -0.16 | -0.03 | 0 / 0 / 0 / 0 |
| ham_mu_y_prime_enc_stdbinary_dvalues_8-8-8-8 | 6 | -0.07 | -0.05 | -0.10 | -0.05 | 0 / 0 / 0 / 0 |
| barenco_tof_3.qasm | 5 | -0.08 | -0.05 | -0.08 | -0.05 | 0 / 0 / 0 / 0 |
| qft.qasm | 4 | -0.05 | -0.05 | -0.09 | -0.05 | 0 / 0 / 0 / 0 |
| ham_mu_z_prime_enc_unary_dvalues_16-16-16-16 | 16 | -0.06 | -0.06 | 0 | -0.05 | 0 / 0 / 4 / 0 |
| tof_4.qasm | 7 | -0.10 | -0.06 | 0 | -0.06 | 0 / 0 / 7 / 0 |
| ham_mu_y_prime_enc_stdbinary_dvalues_4-4-4-4 | 24 | -0.07 | -0.07 | 0 | -0.07 | 0 / 0 / 4 / 0 |
| ham_mu_x_prime_enc_stdbinary_dvalues_8-8-8-8 | 11 | -0.10 | -0.08 | -0.16 | -0.08 | 0 / 0 / 0 / 0 |
| ham_ham_parity-4 | 4 | -0.09 | -0.08 | -0.11 | -0.08 | 0 / 0 / 0 / 0 |
| ham_mu_y_prime_enc_stdbinary_dvalues_16-16-1 | 4 | -0.12 | -0.09 | -0.17 | -0.09 | 0 / 0 / 0 / 0 |
| tof_5.qasm | 9 | -0.15 | -0.09 | 0 | -0.09 | 0 / 0 / 13 / 0 |
| qft_4.qasm | 5 | -0.11 | -0.11 | -0.15 | -0.10 | 0 / 0 / 0 / 0 |
| ham_mu_y_prime_enc_unary_dvalues_16-16-16-16 | 32 | 0 | -0.11 | 0 | -0.11 | 4 / 0 / 4 / 0 |
| barenco_tof_4.qasm | 7 | -0.19 | -0.13 | -0.15 | -0.12 | 0 / 0 / 0 / 0 |
| ham_graph-1D-grid-pbc-qubitnodes_Lx-16_h-2 | 16 | -0.14 | -0.12 | 0 | -0.12 | 0 / 0 / 9 / 0 |
| ham_7-uf20-0384.cnf-8-res | 8 | -0.14 | -0.12 | -0.14 | -0.13 | 0 / 0 / 0 / 0 |
| ham_mu_y_prime_enc_gray_dvalues_16-16-16 | 4 | -0.15 | -0.12 | -0.18 | -0.12 | 0 / 0 / 0 / 0 |
| vbe_adder_3.qasm | 10 | -0.24 | -0.17 | -0.20 | -0.16 | 0 / 0 / 0 / 0 |
| ham_graph-1D-grid-pbc-qubitnodes_Lx-26_h-6 | 26 | -0.23 | -0.17 | 0 | -0.22 | 0 / 0 / 14 / 0 |
| adder.qasm | 10 | -0.28 | -0.17 | 0 | -0.18 | 0 / 0 / 13 / 0 |
| ham_gnp-k_4_n-6_rinst-15 | 6 | -0.21 | -0.21 | -0.25 | -0.22 | 0 / 0 / 0 / 0 |
| ham_tsp_rand-002_Ncity-4_enc-stdbinary | 8 | -0.22 | -0.21 | -0.27 | -0.22 | 0 / 0 / 0 / 0 |
| ham_enc_gray_dvalues_d-level-16 | 4 | -0.29 | -0.23 | -0.46 | -0.28 | 0 / 0 / 0 / 0 |
| rc_adder_6.qasm | 14 | -0.37 | -0.24 | -0.32 | -0.27 | 0 / 0 / 0 / 0 |
| tof_10.qasm | 19 | -0.61 | -0.36 | 0 | -0.25 | 0 / 0 / 13 / 0 |
| csla_mux_3.qasm | 15 | -0.32 | -0.31 | 0 | -0.25 | 0 / 0 / 41 / 0 |
| ham_mu_x_prime_enc_unary_dvalues_16-16-16-16 | 64 | 0 | -0.26 | 0 | -0.25 | 6 / 0 / 14 / 0 |
| hwb6.qasm | 7 | -0.35 | -0.33 | -0.40 | -0.32 | 0 / 0 / 0 / 0 |
| gf2^4_mult.qasm | 12 | -0.54 | -0.37 | -0.47 | -0.35 | 0 / 0 / 0 / 0 |
| ham_complbipart-n-10_a-5_b-5 | 10 | -0.48 | -0.54 | 0 | -0.46 | 0 / 0 / 11 / 0 |
| bigadder.qasm | 18 | -0.63 | -0.65 | 0 | -0.48 | 0 / 0 / 24 / 0 |
| gf2^5_mult.qasm | 15 | -0.98 | -0.75 | -0.78 | -0.77 | 0 / 0 / 0 / 0 |
| csum_mux_9.qasm | 30 | -0.98 | -0.82 | 0 | -0.92 | 0 / 0 / 43 / 0 |
| qcla_adder_10.qasm | 36 | -1.10 | -1.02 | -1.33 | -1.22 | 0 / 0 / 0 / 0 |
| adder_8.qasm | 24 | -1.90 | -1.35 | 0 | -1.61 | 0 / 0 / 81 / 0 |
| ham15-low.qasm | 17 | -1.97 | -1.41 | 0 | -1.49 | 0 / 0 / 81 / 0 |
| gf2^7_mult.qasm | 21 | -1.80 | -1.62 | 0 | -1.92 | 0 / 0 / 185 / 0 |
| qcla_mod_7.qasm | 26 | -2.19 | -1.82 | 0 | -1.67 | 0 / 0 / 98 / 0 |
| ham_ham_JW-8 | 8 | -2.31 | -1.93 | 0 | -1.93 | 0 / 0 / 401 / 0 |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-14_U- | 28 | -2.54 | -2.04 | 0 | -2.59 | 0 / 0 / 165 / 0 |
| ham_gnp-k_5_n-24_rinst-04 | 24 | -2.11 | -2.30 | 0 | -2.22 | 0 / 0 / 9 / 0 |
| ham15-med.qasm | 17 | -2.68 | -2.26 | 0 | -2.45 | 0 / 0 / 161 / 0 |
| gf2^9_mult.qasm | 27 | -3.38 | -3.01 | 0 | -3.69 | 0 / 0 / 23 / 0 |
| ham_0-uuf100-0810.cnf-40-res | 40 | -3.95 | -3.15 | -3.37 | -3.37 | 0 / 0 / 0 / 0 |
| ham_tsp_prob-lin105_Ncity-7_enc-unary | 49 | -4.17 | -3.47 | 0 | -5.07 | 0 / 0 / 35 / 0 |
| ham_tsp_prob-ulysses22_Ncity-8_enc-stdbinary | 24 | -4.09 | -4.32 | 0 | -3.50 | 0 / 0 / 85 / 0 |
| ham_8-uuf100-0509.cnf-40-cc | 40 | -3.67 | -3.64 | 0 | -3.78 | 0 / 0 / 135 / 0 |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-24_U- | 48 | -4.23 | -4.06 | 0 | -4.92 | 0 / 0 / 178 / 0 |
| ham_ham_JW10 | 10 | -4.84 | -4.20 | 0 | -4.32 | 0 / 0 / 373 / 0 |
| gf2^10_mult.qasm | 30 | -4.51 | -4.21 | 0 | -4.43 | 0 / 0 / 91 / 0 |
| ham_4-flat100-7.cnf-90-cc | 90 | 0 | 0 | 0 | -4.47 | 69 / 54 / 83 / 0 |
| ham_graph-2D-grid-nonpbc-qubitnodes_Lx-5_Ly- | 75 | 0 | 0 | 0 | -4.73 | 30 / 57 / 64 / 0 |
| ham_2-uf100-0396.cnf-46-res | 46 | 0 | -7.71 | 0 | -5.69 | 9 / 0 / 49 / 0 |
| ham_2-uf100-0601.cnf-46-res | 46 | 0 | -6.46 | 0 | -6.03 | 15 / 0 / 13 / 0 |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-4_U-4 | 12 | -8.86 | -7.00 | -16.82 | -6.99 | 0 / 0 / 0 / 0 |
| ham_tsp_prob-d198_Ncity-8_enc-unary | 64 | 0 | 0 | 0 | -7.27 | 38 / 82 / 92 / 0 |
| ham_queen13_13,n-28,rinst-0 | 112 | 0 | 0 | 0 | -8.27 | 187 / 261 / 132 / 0 |
| ham_bh_graph-2D-triag-nonpbc-qubitnodes_Lx-3 | 28 | -9.35 | -8.54 | -10.09 | -8.31 | 0 / 0 / 0 / 0 |
| mod_adder_1024.qasm | 28 | 0 | -8.37 | 0 | -9.28 | 3 / 0 / 564 / 0 |
| ham_ham_JW12 | 12 | -9.50 | -8.51 | 0 | -8.51 | 0 / 0 / 979 / 0 |
| ham_ham_parity12 | 12 | -9.65 | -8.51 | 0 | -8.51 | 0 / 0 / 835 / 0 |
| ham15-high.qasm | 20 | -12.63 | -9.34 | -25.72 | -9.84 | 0 / 0 / 0 / 0 |
| ham_graph-2D-triag-pbc-qubitnodes_Lx-13_Ly-1 | 91 | 0 | 0 | 0 | -9.90 | 129 / 137 / 202 / 0 |
| ham_ham_JW14 | 14 | -12.60 | -11.28 | 0 | -10.43 | 0 / 0 / 825 / 0 |
| ham_ham_BK12 | 12 | -13.61 | -11.29 | 0 | -10.83 | 0 / 0 / 1397 / 0 |
| ham_bh_graph-3D-grid-nonpbc-qubitnodes_Lx-2_ | 32 | -11.89 | -11.86 | 0 | -12.93 | 0 / 0 / 241 / 0 |
| gf2^16_mult.qasm | 48 | 0 | 0 | 0 | -12.28 | 60 / 40 / 62 / 0 |
| ham_ham_parity-14 | 14 | -15.14 | -13.04 | 0 | -12.97 | 0 / 0 / 1557 / 0 |
| ham_reg-5_n-10_rinst-07 | 30 | -13.76 | -14.24 | -14.83 | -14.83 | 0 / 0 / 0 / 0 |
| ham_reg-4_n-90_rinst-07 | 90 | 0 | 0 | 0 | -13.79 | 247 / 192 / 333 / 0 |
| ham_fh-graph-1D-grid-pbc-qubitnodes_Lx-50_U- | 100 | 0 | 0 | 0 | -13.95 | 203 / 189 / 268 / 0 |
| ham_dsjc1000.1,n-36,rinst-4 | 72 | 0 | 0 | 0 | -14.08 | 201 / 138 / 309 / 0 |
| ham_gnp-k_4_n-40_rinst-07 | 40 | -14.32 | -14.98 | 0 | -16.02 | 0 / 0 / 308 / 0 |
| ham_tsp_prob-pr76_Ncity-10_enc-unary | 100 | 0 | 0 | 0 | -17.32 | 466 / 302 / 183 / 0 |
| ham_tsp_prob-st70_Ncity-10_enc-unary | 100 | 0 | 0 | 0 | -17.33 | 108 / 361 / 183 / 0 |
| ham_enc_unary_dvalues_4-4-4 | 12 | -29.60 | -18.03 | -24.15 | -19.66 | 0 / 0 / 0 / 0 |
| ham_ham_BK-14 | 14 | -35.93 | -22.68 | -46.91 | -25.11 | 0 / 0 / 0 / 0 |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-60_U- | 120 | 0 | 0 | 0 | -22.80 | 359 / 301 / 368 / 0 |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-46_U- | 92 | 0 | 0 | 0 | -27.80 | 334 / 243 / 512 / 0 |
| ham_4-uf100-0246.cnf-70-res | 70 | 0 | 0 | 0 | -28.20 | 307 / 216 / 259 / 0 |
| hwb8.qasm | 12 | -45.84 | -33.14 | -42.99 | -31.44 | 0 / 0 / 0 / 0 |
| ham_will199gpia,n-60,rinst-8 | 120 | 0 | 0 | 0 | -32.60 | 751 / 952 / 653 / 0 |
| ham_bh_graph-2D-grid-nonpbc-qubitnodes_Lx-7_ | 98 | 0 | 0 | 0 | -53.87 | 982 / 635 / 1053 / 0 |
| ham_tsp_prob-kroD100_Ncity-16_enc-stdbinary | 64 | 0 | 0 | 0 | -58.28 | 318 / 123 / 768 / 0 |
| ham_tsp_prob-fl417_Ncity-16_enc-stdbinary | 64 | 0 | 0 | 0 | -58.29 | 560 / 538 / 768 / 0 |
| ham_bh_graph-2D-triag-pbc-qubitnodes_Lx-5_Ly | 60 | 0 | 0 | 0 | -66.06 | 31 / 866 / 315 / 0 |
| gf2^32_mult.qasm | 96 | 0 | 0 | 0 | -74.08 | 1134 / 1156 / 825 / 0 |
| ham_gnp-k_5_n-90_rinst-02 | 90 | 0 | 0 | 0 | -100.07 | 1874 / 1035 / 1588 / 0 |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-10_U- | 80 | 0 | 0 | 0 | -118.51 | 742 / 1340 / 2369 / 0 |
| ham_ham_BK22 | 22 | -303.13 | -245.83 | -293.81 | -293.81 | 0 / 0 / 0 / 0 |
| ham_ham_JW24 | 24 | -414.66 | -274.36 | 0 | -460.47 | 0 / 0 / 13936 / 0 |
| hwb11.qasm | 15 | -708.67 | -691.94 | 0 | -698.49 | 0 / 0 / 40801 / 0 |
| hwb12.qasm | 20 | -1420.97 | -1399.97 | -1573.20 | -1375.86 | 0 / 0 / 0 / 0 |

## FakeKingston

- tests with all four arms finished: 106 of 106; failed jobs: QK2 0, QK3 0, PSFD 0, PSFR 0

| arm | tests with ops on failed elements | ops on failed elements | ESP = 0 (an op with error 1) | not in Target | q2 / QK2 (gmean, +1) | time (s, summed) |
|---|---|---|---|---|---|---|
| QK2 | 15 | 3842 | 15 | 25 | 1.000 | 63 |
| QK3 | 14 | 2393 | 14 | 25 | 0.979 | 108 |
| PSFD | 17 | 7186 | 17 | 25 | 1.054 | 219 |
| PSFR | 0 | 0 | 0 | 25 | 1.018 | 438 |

**Tests where the best arm's ESP >= 0.01: 65.** ESP ratios on them (PSFR / other; an ESP of 0 counts as a ratio over 10^6):

| other | median ratio | ratio >= 2 | ratio >= 10 | ratio <= 0.5 | ratio in (0.5, 2) |
|---|---|---|---|---|---|
| QK2 | 1.05 | 8 | 2 | 0 | 57 |
| QK3 | 1 | 2 | 2 | 2 | 61 |
| PSFD | 1.08 | 14 | 1 | 0 | 51 |

**Tests where the best arm's ESP >= 1e-06: 86.** ESP ratios on them (PSFR / other; an ESP of 0 counts as a ratio over 10^6):

| other | median ratio | ratio >= 2 | ratio >= 10 | ratio <= 0.5 | ratio in (0.5, 2) |
|---|---|---|---|---|---|
| QK2 | 1.07 | 23 | 14 | 2 | 61 |
| QK3 | 1 | 6 | 6 | 10 | 70 |
| PSFD | 1.23 | 33 | 19 | 0 | 53 |

Per test (log10 ESP; '0' = an op with error 1; failed = ops on failed elements):

| test | qubits | QK2 | QK3 | PSFD | PSFR | failed QK2 / QK3 / PSFD / PSFR |
|---|---|---|---|---|---|---|
| qpt.qasm | 1 | -0.00 | -0.00 | -0.00 | -0.00 | 0 / 0 / 0 / 0 |
| ham_enc_stdbinary_dvalues_d-level-4 | 2 | -0.00 | -0.00 | -0.01 | -0.00 | 0 / 0 / 0 / 0 |
| ham_mu_z_prime_enc_gray_dvalues_4-4-4-4 | 4 | -0.00 | -0.00 | -0.00 | -0.00 | 0 / 0 / 0 / 0 |
| ham_mu_y_prime_enc_unary_dvalues_4-4-4 | 4 | -0.01 | -0.00 | -0.01 | -0.00 | 0 / 0 / 0 / 0 |
| inverseqft1.qasm | 4 | -0.01 | -0.01 | -0.09 | -0.09 | 0 / 0 / 0 / 0 |
| inverseqft2.qasm | 4 | -0.01 | -0.01 | -0.09 | -0.09 | 0 / 0 / 0 / 0 |
| teleport.qasm | 3 | -0.01 | -0.01 | -0.09 | -0.01 | 0 / 0 / 0 / 0 |
| teleportv2.qasm | 3 | -0.01 | -0.01 | -0.09 | -0.01 | 0 / 0 / 0 / 0 |
| ham_mu_y_prime_enc_unary_dvalues_8-8-8-8-8-8 | 8 | -0.01 | -0.01 | -0.02 | -0.01 | 0 / 0 / 0 / 0 |
| ham_mu_y_prime_enc_stdbinary_dvalues_8-8-4-4 | 3 | -0.01 | -0.01 | -0.02 | -0.01 | 0 / 0 / 0 / 0 |
| ham_gnp-k_2_n-4_rinst-05 | 4 | -0.02 | -0.01 | -0.02 | -0.02 | 0 / 0 / 0 / 0 |
| tof_3.qasm | 5 | -0.03 | -0.02 | -0.02 | -0.02 | 0 / 0 / 0 / 0 |
| W-state.qasm | 3 | -0.02 | -0.02 | -0.10 | -0.02 | 0 / 0 / 0 / 0 |
| qec.qasm | 5 | -0.02 | -0.02 | -0.04 | -0.02 | 0 / 0 / 0 / 0 |
| ham_mu_y_prime_enc_stdbinary_dvalues_8-8-8-8 | 6 | -0.03 | -0.02 | -0.03 | -0.02 | 0 / 0 / 0 / 0 |
| barenco_tof_3.qasm | 5 | -0.05 | -0.02 | -0.03 | -0.02 | 0 / 0 / 0 / 0 |
| qft.qasm | 4 | -0.03 | -0.03 | -0.04 | -0.03 | 0 / 0 / 0 / 0 |
| tof_4.qasm | 7 | -0.05 | -0.03 | -0.03 | -0.03 | 0 / 0 / 0 / 0 |
| ham_mu_z_prime_enc_unary_dvalues_16-16-16-16 | 16 | -0.03 | -0.03 | -0.03 | -0.03 | 0 / 0 / 0 / 0 |
| ham_mu_y_prime_enc_stdbinary_dvalues_4-4-4-4 | 24 | -0.04 | -0.03 | -0.04 | -0.03 | 0 / 0 / 0 / 0 |
| ham_graph-1D-grid-pbc-qubitnodes_Lx-16_h-2 | 16 | -0.08 | -0.07 | -0.05 | -0.04 | 0 / 0 / 0 / 0 |
| ham_ham_parity-4 | 4 | -0.05 | -0.04 | -0.04 | -0.04 | 0 / 0 / 0 / 0 |
| ham_mu_x_prime_enc_stdbinary_dvalues_8-8-8-8 | 11 | -0.05 | -0.04 | -0.09 | -0.04 | 0 / 0 / 0 / 0 |
| ham_mu_y_prime_enc_stdbinary_dvalues_16-16-1 | 4 | -0.06 | -0.04 | -0.06 | -0.04 | 0 / 0 / 0 / 0 |
| tof_5.qasm | 9 | -0.06 | -0.04 | -0.05 | -0.04 | 0 / 0 / 0 / 0 |
| qft_4.qasm | 5 | -0.07 | -0.05 | -0.06 | -0.05 | 0 / 0 / 0 / 0 |
| barenco_tof_4.qasm | 7 | -0.08 | -0.05 | -0.05 | -0.05 | 0 / 0 / 0 / 0 |
| ham_7-uf20-0384.cnf-8-res | 8 | -0.08 | -0.06 | -0.07 | -0.05 | 0 / 0 / 0 / 0 |
| ham_mu_y_prime_enc_gray_dvalues_16-16-16 | 4 | -0.08 | -0.05 | -0.06 | -0.05 | 0 / 0 / 0 / 0 |
| ham_mu_y_prime_enc_unary_dvalues_16-16-16-16 | 32 | -0.08 | -0.06 | -0.09 | -0.06 | 0 / 0 / 0 / 0 |
| vbe_adder_3.qasm | 10 | -0.10 | -0.07 | -0.08 | -0.07 | 0 / 0 / 0 / 0 |
| adder.qasm | 10 | -0.12 | -0.09 | -0.14 | -0.08 | 0 / 0 / 0 / 0 |
| ham_gnp-k_4_n-6_rinst-15 | 6 | -0.10 | -0.09 | -0.11 | -0.09 | 0 / 0 / 0 / 0 |
| ham_tsp_rand-002_Ncity-4_enc-stdbinary | 8 | -0.10 | -0.09 | -0.11 | -0.09 | 0 / 0 / 0 / 0 |
| ham_graph-1D-grid-pbc-qubitnodes_Lx-26_h-6 | 26 | -0.13 | -0.10 | -0.18 | -0.11 | 0 / 0 / 0 / 0 |
| csla_mux_3.qasm | 15 | -0.15 | -0.13 | -0.27 | -0.10 | 0 / 0 / 0 / 0 |
| rc_adder_6.qasm | 14 | -0.17 | -0.11 | -0.21 | -0.11 | 0 / 0 / 0 / 0 |
| ham_enc_gray_dvalues_d-level-16 | 4 | -0.17 | -0.11 | -0.17 | -0.12 | 0 / 0 / 0 / 0 |
| tof_10.qasm | 19 | -0.15 | -0.15 | -0.15 | -0.12 | 0 / 0 / 0 / 0 |
| hwb6.qasm | 7 | -0.23 | -0.14 | -0.15 | -0.14 | 0 / 0 / 0 / 0 |
| ham_mu_x_prime_enc_unary_dvalues_16-16-16-16 | 64 | -0.16 | -0.16 | -0.18 | -0.15 | 0 / 0 / 0 / 0 |
| gf2^4_mult.qasm | 12 | -0.23 | -0.17 | -0.26 | -0.16 | 0 / 0 / 0 / 0 |
| ham_complbipart-n-10_a-5_b-5 | 10 | -0.20 | -0.21 | -0.34 | -0.19 | 0 / 0 / 0 / 0 |
| bigadder.qasm | 18 | -0.33 | -0.24 | -0.44 | -0.27 | 0 / 0 / 0 / 0 |
| gf2^5_mult.qasm | 15 | -0.47 | -0.30 | -0.43 | -0.33 | 0 / 0 / 0 / 0 |
| csum_mux_9.qasm | 30 | -0.58 | -0.36 | -0.51 | -0.45 | 0 / 0 / 0 / 0 |
| qcla_adder_10.qasm | 36 | -0.57 | -0.43 | -0.71 | -0.49 | 0 / 0 / 0 / 0 |
| gf2^7_mult.qasm | 21 | -0.98 | -0.64 | -1.01 | -0.77 | 0 / 0 / 0 / 0 |
| adder_8.qasm | 24 | -1.10 | -0.72 | -0.99 | -0.68 | 0 / 0 / 0 / 0 |
| ham15-low.qasm | 17 | -0.78 | -0.69 | -1.48 | -0.72 | 0 / 0 / 0 / 0 |
| qcla_mod_7.qasm | 26 | -0.86 | -0.79 | -1.20 | -0.83 | 0 / 0 / 0 / 0 |
| ham_gnp-k_5_n-24_rinst-04 | 24 | -0.87 | -0.83 | -1.29 | -0.85 | 0 / 0 / 0 / 0 |
| ham_ham_JW-8 | 8 | -1.29 | -0.86 | -1.30 | -0.86 | 0 / 0 / 0 / 0 |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-14_U- | 28 | -1.19 | -0.90 | -1.47 | -1.22 | 0 / 0 / 0 / 0 |
| ham15-med.qasm | 17 | -1.41 | -0.96 | -1.04 | -0.96 | 0 / 0 / 0 / 0 |
| ham_0-uuf100-0810.cnf-40-res | 40 | -1.38 | -1.31 | -2.02 | -1.33 | 0 / 0 / 0 / 0 |
| gf2^9_mult.qasm | 27 | -1.89 | -1.31 | -2.20 | -1.53 | 0 / 0 / 0 / 0 |
| ham_graph-2D-grid-nonpbc-qubitnodes_Lx-5_Ly- | 75 | 0 | 0 | 0 | -1.41 | 7 / 6 / 83 / 0 |
| ham_4-flat100-7.cnf-90-cc | 90 | 0 | 0 | -1.49 | -1.47 | 40 / 57 / 0 / 0 |
| ham_8-uuf100-0509.cnf-40-cc | 40 | -1.68 | -1.58 | -2.21 | -1.69 | 0 / 0 / 0 / 0 |
| gf2^10_mult.qasm | 30 | -2.14 | -1.68 | -2.44 | -1.61 | 0 / 0 / 0 / 0 |
| ham_tsp_prob-ulysses22_Ncity-8_enc-stdbinary | 24 | -2.00 | -1.67 | -2.23 | -1.83 | 0 / 0 / 0 / 0 |
| ham_tsp_prob-lin105_Ncity-7_enc-unary | 49 | -1.95 | -1.71 | -2.38 | -1.87 | 0 / 0 / 0 / 0 |
| ham_ham_JW10 | 10 | -2.31 | -1.74 | -2.21 | -1.75 | 0 / 0 / 0 / 0 |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-24_U- | 48 | -2.36 | -1.75 | -2.86 | -2.38 | 0 / 0 / 0 / 0 |
| ham_2-uf100-0396.cnf-46-res | 46 | -2.46 | -2.25 | -2.76 | -2.59 | 0 / 0 / 0 / 0 |
| ham_2-uf100-0601.cnf-46-res | 46 | -2.69 | -2.66 | -3.06 | -2.74 | 0 / 0 / 0 / 0 |
| ham_tsp_prob-d198_Ncity-8_enc-unary | 64 | -2.86 | -2.80 | -3.79 | -3.76 | 0 / 0 / 0 / 0 |
| ham_queen13_13,n-28,rinst-0 | 112 | 0 | 0 | 0 | -2.85 | 236 / 182 / 125 / 0 |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-4_U-4 | 12 | -4.43 | -3.39 | -5.08 | -3.18 | 0 / 0 / 0 / 0 |
| ham_ham_parity12 | 12 | -4.50 | -3.35 | -5.20 | -3.35 | 0 / 0 / 0 / 0 |
| mod_adder_1024.qasm | 28 | -8.52 | -3.42 | -5.89 | -4.21 | 0 / 0 / 0 / 0 |
| ham_ham_JW12 | 12 | -4.35 | -3.42 | -5.04 | -3.42 | 0 / 0 / 0 / 0 |
| ham_bh_graph-2D-triag-nonpbc-qubitnodes_Lx-3 | 28 | -4.14 | -3.72 | -7.51 | -4.09 | 0 / 0 / 0 / 0 |
| ham15-high.qasm | 20 | -5.64 | -3.98 | -6.62 | -4.31 | 0 / 0 / 0 / 0 |
| ham_graph-2D-triag-pbc-qubitnodes_Lx-13_Ly-1 | 91 | 0 | 0 | 0 | -4.24 | 70 / 93 / 259 / 0 |
| ham_ham_BK12 | 12 | -7.61 | -4.42 | -6.57 | -5.33 | 0 / 0 / 0 / 0 |
| ham_ham_JW14 | 14 | -6.02 | -4.60 | -8.31 | -4.52 | 0 / 0 / 0 / 0 |
| ham_fh-graph-1D-grid-pbc-qubitnodes_Lx-50_U- | 100 | 0 | 0 | 0 | -4.62 | 75 / 261 / 226 / 0 |
| ham_dsjc1000.1,n-36,rinst-4 | 72 | -5.82 | -4.89 | 0 | -5.30 | 0 / 0 / 81 / 0 |
| gf2^16_mult.qasm | 48 | -5.59 | -5.15 | -7.67 | -5.08 | 0 / 0 / 0 / 0 |
| ham_ham_parity-14 | 14 | -7.02 | -5.20 | -8.61 | -5.20 | 0 / 0 / 0 / 0 |
| ham_bh_graph-3D-grid-nonpbc-qubitnodes_Lx-2_ | 32 | -7.23 | -5.29 | -6.43 | -5.39 | 0 / 0 / 0 / 0 |
| ham_reg-4_n-90_rinst-07 | 90 | 0 | 0 | 0 | -5.50 | 146 / 12 / 399 / 0 |
| ham_gnp-k_4_n-40_rinst-07 | 40 | -5.88 | -6.09 | -8.32 | -6.44 | 0 / 0 / 0 / 0 |
| ham_reg-5_n-10_rinst-07 | 30 | -5.91 | -6.01 | -8.39 | -6.05 | 0 / 0 / 0 / 0 |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-46_U- | 92 | 0 | -6.88 | 0 | -8.06 | 123 / 0 / 382 / 0 |
| ham_enc_unary_dvalues_4-4-4 | 12 | -12.88 | -7.61 | -13.14 | -8.10 | 0 / 0 / 0 / 0 |
| ham_tsp_prob-pr76_Ncity-10_enc-unary | 100 | 0 | 0 | 0 | -7.96 | 12 / 10 / 451 / 0 |
| ham_tsp_prob-st70_Ncity-10_enc-unary | 100 | 0 | 0 | 0 | -7.96 | 6 / 68 / 450 / 0 |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-60_U- | 120 | 0 | 0 | 0 | -8.74 | 593 / 597 / 655 / 0 |
| ham_4-uf100-0246.cnf-70-res | 70 | -10.74 | -10.58 | -12.00 | -11.17 | 0 / 0 / 0 / 0 |
| ham_ham_BK-14 | 14 | -16.03 | -11.85 | -17.50 | -12.63 | 0 / 0 / 0 / 0 |
| ham_will199gpia,n-60,rinst-8 | 120 | 0 | 0 | 0 | -12.11 | 315 / 601 / 976 / 0 |
| hwb8.qasm | 12 | -20.08 | -13.22 | -21.41 | -16.38 | 0 / 0 / 0 / 0 |
| ham_tsp_prob-fl417_Ncity-16_enc-stdbinary | 64 | 0 | -19.81 | 0 | -22.96 | 18 / 0 / 18 / 0 |
| ham_bh_graph-2D-grid-nonpbc-qubitnodes_Lx-7_ | 98 | 0 | 0 | 0 | -19.83 | 608 / 263 / 6 / 0 |
| ham_tsp_prob-kroD100_Ncity-16_enc-stdbinary | 64 | -25.29 | -24.17 | 0 | -22.96 | 0 / 0 / 18 / 0 |
| ham_bh_graph-2D-triag-pbc-qubitnodes_Lx-5_Ly | 60 | -26.79 | -25.30 | 0 | -27.42 | 0 / 0 / 15 / 0 |
| gf2^32_mult.qasm | 96 | 0 | 0 | 0 | -28.28 | 1443 / 159 / 2258 / 0 |
| ham_bh_graph-1D-grid-pbc-qubitnodes_Lx-10_U- | 80 | 0 | 0 | -36.38 | -36.38 | 150 / 24 / 0 / 0 |
| ham_gnp-k_5_n-90_rinst-02 | 90 | -39.72 | 0 | 0 | -38.71 | 0 / 60 / 784 / 0 |
| ham_ham_BK22 | 22 | -149.82 | -105.15 | -152.31 | -127.35 | 0 / 0 / 0 / 0 |
| ham_ham_JW24 | 24 | -183.36 | -106.00 | -283.81 | -183.88 | 0 / 0 / 0 / 0 |
| hwb11.qasm | 15 | -337.82 | -279.80 | -285.37 | -285.37 | 0 / 0 / 0 / 0 |
| hwb12.qasm | 20 | -685.84 | -601.16 | -751.66 | -580.51 | 0 / 0 / 0 / 0 |

