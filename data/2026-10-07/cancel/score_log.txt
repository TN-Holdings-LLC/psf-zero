# CANCEL score

git_head 9c341a3, tests 140 (expected 140), jobs 420
P0 PASS: C23 failed where C22 finished, or invalid: 0; versions ok True; meta ok True
errors or timeouts per arm: {'C22': 0, 'C22B': 0, 'C23': 0}

tests finished by all arms: 140; cancellation tried on 32

| | prediction | value | verdict |
|---|---|---|---|
| K1 | no two-qubit gate cancels, C22 reproduces itself: C23 returns C22's circuit | 0 of 102 differ (6 where C22 did not reproduce) | **CONFIRMED** |
| K2 | some cancel: C23 has no more two-qubit gates than C22 | 0 of 32 have more | **CONFIRMED** |
| K4 | every checkable C23 output implements its input | 0 of 41 do not | **CONFIRMED** |
| K5 | no two-qubit gate cancels: median time C23 / C22 <= 1.15 | 1.023 on 108 | **CONFIRMED** |

Reported without prediction: the tests where cancellation was tried

| test | C22 two-qubit | C23 two-qubit | C23 kept | C23 / C22 time |
|---|---|---|---|---|
| test_QASMBench_small[error_correctiond3_n5-square] | 56 | 41 | cancelled | 1.11 |
| test_QASMBench_medium[factor247_n15-square] | 394739 | 393215 | cancelled | 2.21 |
| test_QASMBench_medium[factor247_n15-linear] | 425065 | 423541 | cancelled | 2.00 |
| test_QASMBench_large[qft_n160-heavy-hex] | 21361 | 15793 | cancelled | 1.24 |
| test_QASMBench_large[multiplier_n400-heavy-hex] | 787940 | 787940 | original | 2.03 |
| test_hamiltonians[ham_ham_parity-4-all-to-all] | 48 | 48 | original | 1.00 |
| test_hamiltonians[ham_4-uf100-0246.cnf-70-res-all-to-all] | 1262 | 1262 | original | 0.96 |
| test_hamiltonians[ham_reg-5_n-10_rinst-07-square] | 5367 | 5275 | cancelled | 3.70 |
| test_hamiltonians[ham_bh_graph-2D-grid-nonpbc-qubitnodes_Lx-7_Ly-7_U-70_enc-gray_d-4-square] | 12013 | 11963 | cancelled | 1.46 |
| test_hamiltonians[ham_tsp_prob-ulysses22_Ncity-8_enc-stdbinary-heavy-hex] | 1870 | 1870 | original | 1.70 |
| test_hamiltonians[ham_ham_JW12-heavy-hex] | 6175 | 5007 | cancelled | 1.50 |
| test_hamiltonians[ham_ham_JW-14-heavy-hex] | 15941 | 14364 | cancelled | 3.96 |
| test_hamiltonians[ham_reg-5_n-10_rinst-07-heavy-hex] | 6111 | 6111 | original | 1.94 |
| test_hamiltonians[ham_enc_unary_dvalues_4-4-4-linear] | 20982 | 14833 | cancelled | 1.67 |
| test_hamiltonians[ham_ham_BK22-linear] | 256735 | 207796 | cancelled | 1.78 |
| test_hamiltonians[ham_reg-5_n-10_rinst-07-linear] | 6403 | 6403 | original | 1.09 |
| test_hamiltonians[ham_bh_graph-2D-triag-nonpbc-qubitnodes_Lx-11_Ly-11_U-100_enc-gray_d-4-linear] | 64436 | 59389 | cancelled | 1.97 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-18] | 43491 | 39717 | cancelled | 3.09 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-10] | 2854 | 2484 | cancelled | 1.46 |
| test_hamlib_hamiltonians_transpile[ham_ham_parity10] | 2723 | 2543 | cancelled | 1.38 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-22] | 129070 | 120561 | cancelled | 1.87 |
| test_feynman_transpile[qcla_com_7.qasm] | 362 | 362 | original | 1.32 |
| test_BVlike_simplification_transpile | 392 | 0 | cancelled | 1.39 |
| test_QASMBench_medium[multiplier_n15-all-to-all] | 222 | 222 | original | 1.24 |
| test_QASMBench_large[bwt_n37-linear] | 3281757 | 3281757 | original | 2.05 |
| test_hamiltonians[ham_ham_parity-14-all-to-all] | 3666 | 3649 | cancelled | 1.41 |
| test_hamiltonians[ham_ham_parity-4-heavy-hex] | 63 | 63 | original | 1.02 |
| test_hamiltonians[ham_4-uf100-0246.cnf-70-res-linear] | 22901 | 22901 | original | 1.53 |
| test_hamiltonians[ham_enc_gray_dvalues_4-4-4-4-4-4-4-linear] | 4717 | 4717 | original | 1.45 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-14] | 13644 | 12927 | cancelled | 1.78 |
| test_hamlib_hamiltonians_transpile[ham_enc_gray_dvalues_8-8-8] | 14298 | 13261 | cancelled | 1.17 |
| test_feynman_transpile[hwb10.qasm] | 114203 | 113292 | cancelled | 2.13 |
