# REC-PROBE (rec_probe.json; exploratory)

git_head 724acbd, Benchpress b695f30, 20 tests, 60 jobs, 85.2 s

| test | QK (BP-MOCK) | R4 | R1 | D1 | R4 as in BP-MOCK | R1 cancel tried/kept | R1 recompiled | on failed R4/R1/D1 | R1/R4 time | valid, implements (R1) |
|---|---|---|---|---|---|---|---|---|---|---|
| test_hamlib_hamiltonians_transpile[ham_ham_JW-18] | 36583 | 58085 | 39717 | 39717 | True | 1/1 | 0 | 0/0/9533 | 1.85 | True, - |
| test_hamlib_hamiltonians_transpile[ham_bh_graph-2D-triag-pbc-qubitnodes_Lx-3_Ly-22_U-70_enc-unary_d-4] | 53035 | 91486 | 58199 | 58199 | True | 0/0 | 1 | 6542/3908/3908 | 2.06 | True, - |
| test_hamlib_hamiltonians_transpile[ham_bh_graph-2D-triag-pbc-qubitnodes_Lx-10_Ly-10_U-30_enc-stdbinary_d-4] | 21619 | 26221 | 25616 | 25535 | True | 0/0 | 1 | 1438/0/962 | 3.32 | True, - |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-10] | 2415 | 2371 | 2371 | 2484 | True | 2/2 | 0 | 0/0/373 | 0.53 | True, True |
| test_hamlib_hamiltonians_transpile[ham_ham_parity10] | 2438 | 2367 | 2367 | 2543 | True | 2/2 | 0 | 0/0/464 | 0.75 | True, True |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-22] | 113488 | 182362 | 120561 | 120561 | True | 1/1 | 0 | 0/0/9149 | 1.52 | True, - |
| test_hamlib_hamiltonians_transpile[ham_mu_x_prime_enc_unary_dvalues_4-4-4] | 18 | 18 | 18 | 18 | True | 0/0 | 0 | 0/0/0 | 0.86 | True, True |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-6] | 329 | 351 | 351 | 351 | True | 0/0 | 0 | 0/0/0 | 1.17 | True, True |
| test_feynman_transpile[mod_red_21.qasm] | 205 | 188 | 188 | 198 | True | 0/0 | 0 | 0/0/0 | 0.60 | True, True |
| test_feynman_transpile[qcla_com_7.qasm] | 365 | 803 | 362 | 362 | True | 1/0 | 0 | 0/0/19 | 3.41 | True, - |
| test_feynman_transpile[gf2^6_mult.qasm] | 551 | 710 | 561 | 561 | True | 0/0 | 0 | 0/0/0 | 1.09 | True, - |
| test_feynman_transpile[mod5_4.qasm] | 50 | 45 | 45 | 50 | True | 0/0 | 0 | 0/0/0 | 0.82 | True, True |
| test_feynman_transpile[gf2^8_mult.qasm] | 1096 | 1190 | 1190 | 1190 | True | 0/0 | 0 | 0/0/162 | 0.99 | True, - |
| test_feynman_transpile[barenco_tof_5.qasm] | 118 | 117 | 117 | 117 | True | 0/0 | 0 | 0/0/28 | 0.78 | True, True |
| test_circSU2_89_transpile | 345 | 1719 | 1344 | 336 | True | 0/0 | 1 | 140/0/24 | 20.78 | True, - |
| test_BVlike_simplification_transpile | 0 | 1071 | 0 | 0 | True | 1/1 | 0 | 0/0/0 | 0.75 | True, - |
| test_QAOA_100_transpile | 8750 | 9461 | 9461 | 8890 | True | 0/0 | 1 | 0/0/566 | 0.99 | True, - |
| test_BV_100_transpile | 196 | 580 | 580 | 200 | True | 0/0 | 1 | 0/0/16 | 0.97 | True, - |
| test_square_heisenberg_100_transpile | 1419 | 1758 | 1758 | 1587 | True | 0/0 | 1 | 0/0/114 | 1.00 | True, - |
| test_clifford_100_transpile | 58315 | 64650 | 64650 | 58949 | True | 0/0 | 1 | 0/0/2665 | 1.00 | True, - |
