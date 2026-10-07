# C23 against Qiskit level 2 (exploratory, Addendum 396)

CANCEL git_head 9c341a3; two-qubit count ratio (x + 1) / (y + 1); geometric means.

| sample | tests | C23 / QK | C22 / QK | REL / QK | C22 count as in its own run |
|---|---|---|---|---|---|
| BP-MOCK (git_head 86f992c) | 92 of 92 | 1.023 | 1.115 | 1.364 | 92 of 92 |
| BP-MOCK2 (git_head fffedd5) | 47 of 48 | 1.036 | 1.039 | 1.261 | 47 of 47 |
| both | 139 | 1.027 | 1.089 | 1.328 | |

C23 against QK on the 139 tests: fewer two-qubit gates on 18, as many on 59, more on 62; more by over 10% on 15, by over 25% on 4.

| stratum | tests | C23 / QK | REL / QK |
|---|---|---|---|
| 100-qubit, FakeTorino | 6 | 1.022 | 3.630 |
| Feynman, FakeTorino | 10 | 1.018 | 1.183 |
| HamLib, FakeTorino | 12 | 1.066 | 1.544 |
| HamLib, all-to-all | 8 | 1.001 | 1.286 |
| HamLib, heavy-hex | 8 | 0.994 | 1.736 |
| HamLib, linear | 8 | 1.062 | 1.496 |
| HamLib, square | 8 | 1.033 | 1.934 |
| QASMBench large, all-to-all | 6 | 1.000 | 1.068 |
| QASMBench large, heavy-hex | 6 | 1.053 | 1.310 |
| QASMBench large, linear | 5 | 0.994 | 1.428 |
| QASMBench large, square | 6 | 1.021 | 1.168 |
| QASMBench medium, all-to-all | 6 | 1.000 | 1.014 |
| QASMBench medium, heavy-hex | 6 | 1.083 | 1.343 |
| QASMBench medium, linear | 6 | 1.033 | 1.149 |
| QASMBench medium, square | 6 | 1.022 | 1.139 |
| QASMBench small, all-to-all | 8 | 1.058 | 1.058 |
| QASMBench small, heavy-hex | 8 | 1.009 | 1.070 |
| QASMBench small, linear | 8 | 0.999 | 1.005 |
| QASMBench small, square | 8 | 1.030 | 1.089 |

Largest C23 / QK:

| test | QK | C23 | ratio |
|---|---|---|---|
| test_QASMBench_small[basis_test_n4-all-to-all] | 6 | 10 | 1.57 |
| test_hamiltonians[ham_mu_x_prime_enc_stdbinary_dvalues_8-8-8-8-8-8-8-8-8-4-4-4-4-4-4-linear] | 56 | 74 | 1.32 |
| test_QASMBench_medium[bv_n19-heavy-hex] | 55 | 71 | 1.29 |
| test_QASMBench_small[wstate_n3-square] | 10 | 13 | 1.27 |
| test_QASMBench_small[pea_n5-linear] | 29 | 36 | 1.23 |
| test_QASMBench_medium[multiply_n13-heavy-hex] | 78 | 95 | 1.22 |
| test_hamlib_hamiltonians_transpile[ham_bh_graph-2D-triag-pbc-qubitnodes_Lx-10_Ly-10_U-30_enc-stdbinary_d-4] | 21619 | 25535 | 1.18 |
| test_feynman_transpile[mod_mult_55.qasm] | 81 | 94 | 1.16 |
| test_QASMBench_large[qugan_n39-heavy-hex] | 497 | 571 | 1.15 |
| test_hamiltonians[ham_bh_graph-2D-grid-nonpbc-qubitnodes_Lx-7_Ly-7_U-70_enc-gray_d-4-square] | 10486 | 11963 | 1.14 |

Smallest C23 / QK:

| test | QK | C23 | ratio |
|---|---|---|---|
| test_QASMBench_small[qaoa_n3-linear] | 9 | 7 | 0.80 |
| test_hamiltonians[ham_ham_JW12-heavy-hex] | 5547 | 5007 | 0.90 |
| test_QASMBench_small[qaoa_n6-square] | 54 | 50 | 0.93 |
| test_hamiltonians[ham_graph-2D-triag-nonpbc-qubitnodes_Lx-3_Ly-160_h-0.1-square] | 6594 | 6280 | 0.95 |
| test_feynman_transpile[mod_red_21.qasm] | 205 | 198 | 0.97 |
| test_hamiltonians[ham_tsp_prob-ulysses22_Ncity-8_enc-stdbinary-heavy-hex] | 1926 | 1870 | 0.97 |
| test_circSU2_89_transpile | 345 | 336 | 0.97 |
| test_feynman_transpile[grover_5.qasm] | 549 | 537 | 0.98 |
| test_QASMBench_large[bv_n140-linear] | 359 | 352 | 0.98 |
| test_QASMBench_large[swap_test_n83-linear] | 501 | 494 | 0.99 |

Compile time C23 / QK (different runs, indicative): median 3.87, geometric mean 3.72, range 0.10-19.02.
