# work_calib (exploratory, nothing predicted)

git head fbe95df, uncommitted tracked changes: none; candidate 2026-10-09.c26; 34 tests, 4 at a time; job timeout 600 s; budget 5400 s; 14 CPUs; Python 3.11.9

jobs: killed after 600 s 1, ok 33

## Per function: time = a * ops + b * amps (calls that ended)

| function | calls | never ended | total s | a (us per op) | b (ns per amplitude) | R^2 | largest k |
|---|---|---|---|---|---|---|---|
| excitation_cost | 56 | 1 | 568.8 | 19.88 | 56.962 | 0.998 | 16 |
| hybrid_cost | 39 | 0 | 81.4 | 33.83 | 46.132 | 0.620 | 15 |
| _implements | 25 | 0 | 52.6 | 24.31 | 4.793 | 0.758 | 15 |
| _same_action | 10 | 0 | 52.8 | 14.02 | 1.919 | 0.976 | 15 |
| transpile | 13 | 0 | 5.3 | - | - | - | - |
| _resynthesis_candidate | 28 | 0 | 9.6 | - | - | - | - |

## The 20 longest calls (a call that never ended is listed first)

| test | function | k | ops | amps | s |
|---|---|---|---|---|---|
| test_feynman_transpile[hwb10.qasm] | excitation_cost | 16 | 689252 | 7424311296 | never ended |
| test_feynman_transpile[hwb10.qasm] | excitation_cost | 16 | 503690 | 7424704512 | 432.63 |
| test_hamlib_hamiltonians_transpile[ham_enc_gray_dvalues_4-4- | excitation_cost | 15 | 22889 | 121765888 | 16.77 |
| test_hamlib_hamiltonians_transpile[ham_enc_gray_dvalues_4-4- | excitation_cost | 15 | 20503 | 137494528 | 16.72 |
| test_hamlib_hamiltonians_transpile[ham_enc_gray_dvalues_4-4- | excitation_cost | 15 | 22889 | 121765888 | 16.65 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-14] | _implements | 14 | 83014 | 1360101376 | 16.46 |
| test_hamlib_hamiltonians_transpile[ham_enc_gray_dvalues_4-4- | hybrid_cost | 15 | 22889 | 121765888 | 15.34 |
| test_hamlib_hamiltonians_transpile[ham_enc_gray_dvalues_4-4- | excitation_cost | 15 | 20503 | 137494528 | 14.47 |
| test_hamlib_hamiltonians_transpile[ham_enc_gray_dvalues_4-4- | hybrid_cost | 15 | 22889 | 121765888 | 13.87 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-14] | _same_action | 14 | 250594 | 4105732096 | 12.35 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-14] | excitation_cost | 14 | 58031 | 211795968 | 11.24 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-14] | _same_action | 14 | 250594 | 4105732096 | 11.13 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-14] | excitation_cost | 14 | 58031 | 211795968 | 10.33 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-14] | excitation_cost | 14 | 67266 | 192921600 | 9.89 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-14] | _implements | 14 | 134534 | 2204205056 | 9.73 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-14] | excitation_cost | 14 | 67266 | 192921600 | 9.37 |
| test_hamlib_hamiltonians_transpile[ham_enc_gray_dvalues_8-8- | hybrid_cost | 12 | 70208 | 48726016 | 7.39 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-14] | hybrid_cost | 14 | 67266 | 192921600 | 6.92 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-14] | hybrid_cost | 14 | 41506 | 177864704 | 6.79 |
| test_hamlib_hamiltonians_transpile[ham_enc_gray_dvalues_4-4- | _same_action | 15 | 86784 | 2843738112 | 6.52 |

## Calls a work cap would stop (counted calls only; work = ops * R + amps, in amplitudes; R from the pooled fit)

pooled fit over all counted calls: a 103.52 us per op, b 22.574 ns per amplitude, R^2 0.528; one op costs as much as 4586 amplitudes

| cap (work units) | calls over | of which never ended | their total s | tests touched |
|---|---|---|---|---|
| 2^20 | 111 | 1 | 755.6 | 13 |
| 2^22 | 86 | 1 | 754.9 | 11 |
| 2^24 | 62 | 1 | 751.9 | 8 |
| 2^26 | 51 | 1 | 743.1 | 6 |
| 2^28 | 31 | 1 | 624.4 | 6 |
| 2^30 | 12 | 1 | 517.1 | 4 |
| 2^32 | 4 | 1 | 456.1 | 2 |

## Whole compiles

| test | status | compile s | counted calls s | transpile s | resynthesis s |
|---|---|---|---|---|---|
| test_feynman_transpile[hwb10.qasm] | killed after 600 s | - | 432.6 | 0.0 | 5.4 |
| test_hamlib_hamiltonians_transpile[ham_enc_gray_dvalues_4-4- | ok | 134.2 | 121.9 | 0.8 | 0.6 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-14] | ok | 130.9 | 110.5 | 1.5 | 1.4 |
| test_hamlib_hamiltonians_transpile[ham_enc_gray_dvalues_8-8- | ok | 80.3 | 55.4 | 1.6 | 1.3 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-22] | ok | 66.4 | 0.0 | 0.0 | 0.0 |
| test_hamlib_hamiltonians_transpile[ham_bh_graph-2D-triag-pbc | ok | 31.1 | 0.0 | 0.0 | 0.0 |
| test_QV_100_transpile | ok | 30.1 | 0.0 | 0.0 | 0.0 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-10] | ok | 21.9 | 15.5 | 0.4 | 0.3 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-18] | ok | 21.5 | 0.0 | 0.0 | 0.0 |
| test_hamlib_hamiltonians_transpile[ham_ham_parity10] | ok | 21.4 | 15.3 | 0.3 | 0.3 |
| test_hamlib_hamiltonians_transpile[ham_bh_graph-2D-triag-pbc | ok | 20.9 | 0.0 | 0.0 | 0.0 |
| test_QFT_100_transpile | ok | 14.1 | 0.0 | 0.0 | 0.0 |
| test_circSU2_89_transpile | ok | 13.8 | 0.0 | 0.0 | 0.0 |
| test_clifford_100_transpile | ok | 9.7 | 0.0 | 0.0 | 0.0 |
| test_circSU2_100_transpile | ok | 9.2 | 0.0 | 0.0 | 0.0 |
| test_hamlib_hamiltonians_transpile[ham_gnp-k_5_n-60_rinst-19 | ok | 3.6 | 0.0 | 0.0 | 0.0 |
| test_hamlib_hamiltonians_transpile[ham_ham_JW-6] | ok | 3.3 | 1.1 | 0.1 | 0.1 |
| test_feynman_transpile[grover_5.qasm] | ok | 3.1 | 1.6 | 0.1 | 0.1 |
| test_feynman_transpile[mod_red_21.qasm] | ok | 2.7 | 1.1 | 0.1 | 0.0 |
| test_hamlib_hamiltonians_transpile[ham_bh_graph-1D-grid-nonp | ok | 2.6 | 0.0 | 0.0 | 0.0 |
| test_BV_100_transpile | ok | 2.3 | 0.0 | 0.0 | 0.0 |
| test_square_heisenberg_100_transpile | ok | 2.1 | 0.0 | 0.0 | 0.0 |
| test_hamlib_hamiltonians_transpile[ham_mu_x_prime_enc_unary_ | ok | 1.4 | 0.1 | 0.3 | 0.0 |
| test_BVlike_simplification_transpile | ok | 1.3 | 0.0 | 0.0 | 0.0 |
| test_hamlib_hamiltonians_transpile[ham_tsp_prob-ts225_Ncity- | ok | 1.1 | 0.0 | 0.0 | 0.0 |
| test_QAOA_100_transpile | ok | 1.1 | 0.0 | 0.0 | 0.0 |
| test_feynman_transpile[qcla_com_7.qasm] | ok | 1.1 | 0.0 | 0.0 | 0.0 |
| test_feynman_transpile[mod_mult_55.qasm] | ok | 1.0 | 0.2 | 0.0 | 0.0 |
| test_feynman_transpile[mod5_4.qasm] | ok | 1.0 | 0.1 | 0.0 | 0.0 |
| test_feynman_transpile[barenco_tof_5.qasm] | ok | 1.0 | 0.3 | 0.0 | 0.0 |
| test_feynman_transpile[barenco_tof_10.qasm] | ok | 0.8 | 0.0 | 0.0 | 0.0 |
| test_feynman_transpile[gf2^6_mult.qasm] | ok | 0.8 | 0.0 | 0.0 | 0.0 |
| test_feynman_transpile[gf2^8_mult.qasm] | ok | 0.7 | 0.0 | 0.0 | 0.0 |
| test_feynman_transpile[rb.qasm] | ok | 0.2 | 0.0 | 0.0 | 0.0 |
